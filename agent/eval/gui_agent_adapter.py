from __future__ import annotations

import base64
import hashlib
import io
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib import request
from urllib.parse import urlsplit

import requests

from agent.docker.worker_manager import DockerWorkerManager, InfrastructureError


REPO_ROOT = Path(__file__).resolve().parents[2]
GPT54_AGENT_PATH = REPO_ROOT / "agent" / "mm_agents" / "gpt54_agent.py"
SEED_AGENT_PATH = REPO_ROOT / "agent" / "mm_agents" / "seed_agent.py"
GUI_AGENT_PROMPT_PATH = REPO_ROOT / "agent" / "prompt" / "Guiagentprompt.json"
_SEED_AGENT_MODULE_LOCK = threading.Lock()
_SEED_REQUEST_LOG_LOCK = threading.Lock()


def _parse_image_size(raw: str, setting_name: str) -> Optional[Tuple[int, int]]:
    raw = raw.strip()
    if not raw:
        return None
    normalized = raw.lower().replace("×", "x")
    try:
        width_text, height_text = normalized.split("x", 1)
        width, height = int(width_text), int(height_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{setting_name} must use WIDTHxHEIGHT, for example 1280x720") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"{setting_name} width and height must be positive")
    return width, height


def _shared_gui_image_size() -> Optional[Tuple[int, int]]:
    return _parse_image_size(os.environ.get("GUIEVAL_IMAGE_SIZE", ""), "GUIEVAL_IMAGE_SIZE")


class SeedJudgeError(RuntimeError):
    """Raised when the Seed result-only judge exhausts request retries."""


def load_gpt54_agent_class():
    module_name = "software_gym_osworld_gpt54_agent"
    if module_name in sys.modules:
        patch_openai_client(sys.modules[module_name])
        return sys.modules[module_name].GPT54Agent
    spec = importlib.util.spec_from_file_location(module_name, GPT54_AGENT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load GPT54Agent from {GPT54_AGENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    patch_openai_client(module)
    return module.GPT54Agent


def load_seed_agent_module():
    """Lazy-load the unmodified reference SeedAgent only when seed mode is selected."""
    module_name = "software_gym_osworld_seed_agent"
    with _SEED_AGENT_MODULE_LOCK:
        cached = sys.modules.get(module_name)
        if cached is not None and hasattr(cached, "SeedAgent"):
            return cached
        # A failed or concurrent import may leave a partially initialized module
        # in sys.modules. Never expose it to another eval lane.
        sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_file_location(module_name, SEED_AGENT_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load SeedAgent from {SEED_AGENT_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        if not hasattr(module, "SeedAgent"):
            sys.modules.pop(module_name, None)
            raise ImportError(f"SeedAgent class is missing from {SEED_AGENT_PATH}")
        return module


def load_gui_agent_class():
    """Select the GUIEval backend agent via ``APPEVAL_AGENT_IMPL``.

    - ``gpt54`` (default): GPT54Agent using OpenAI Responses + computer-use;
      behavior is unchanged.
    - ``seed``: built-in bridge to the original SeedAgent; the Seed executor
      drives the GUI, and the same Seed model emits RESULT_JSON through a
      separate result-only judge request.
    """
    impl = (os.environ.get("APPEVAL_AGENT_IMPL") or "gpt54").strip().lower()
    if impl in {"seed", "doubao", "seed_gui"}:
        return SeedAgentBridge
    return load_gpt54_agent_class()


def patch_openai_client(module: Any) -> None:
    original_openai = getattr(module, "OpenAI", None)
    if original_openai is None or getattr(module, "_softwaregym_openai_patched", False):
        return

    class ResponsesShim:
        def __init__(self, api_key: str, base_url: str) -> None:
            self.api_key = api_key
            self.base_url = base_url.rstrip("/") + ("" if base_url.rstrip("/").endswith("/v1") else "/v1")

        def create(self, **kwargs: Any) -> Dict[str, Any]:
            req = request.Request(
                self.base_url + "/responses",
                data=json.dumps(kwargs).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            with request.urlopen(req, timeout=600) as response:
                raw = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"output": parsed}

    def openai_factory(*args: Any, **kwargs: Any) -> Any:
        # GUIAgent may use a dedicated endpoint/model (e.g. gpt-5.5) separate from the
        # codex design/coding stages. Dedicated GUIAGENT_* settings MUST take precedence
        # even over api_key/base_url the caller passes explicitly: GPT54Agent hardcodes
        # OpenAI(api_key=os.getenv("OPENAI_API_KEY")), so without this the global key
        # would be sent to the GUIAGENT base_url and fail with 401 (invalid key for that
        # endpoint). Always keep the GUIAGENT key+base as a matched pair.
        guiagent_key = (
            os.environ.get("GUIAGENT_OPENAI_API_KEY")
            or os.environ.get("GUIAGENT_API_KEY")
        )
        guiagent_base = (
            os.environ.get("GUIAGENT_OPENAI_BASE_URL")
            or os.environ.get("GUIAGENT_BASE_URL")
        )
        api_key = guiagent_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
        base_url = guiagent_base or os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL")

        def _norm(url: str) -> str:
            url = url.rstrip("/")
            return url if url.endswith("/v1") else url + "/v1"

        if guiagent_key:
            kwargs["api_key"] = guiagent_key
        elif api_key and "api_key" not in kwargs:
            kwargs["api_key"] = api_key
        if guiagent_base:
            kwargs["base_url"] = _norm(guiagent_base)
        elif base_url and "base_url" not in kwargs:
            kwargs["base_url"] = _norm(base_url)
        client = original_openai(*args, **kwargs)
        if not hasattr(client, "responses") and api_key and base_url:
            client.responses = ResponsesShim(api_key=api_key, base_url=base_url)
        return client

    module.OpenAI = openai_factory
    module._softwaregym_openai_patched = True


def collect_text(value: Any) -> str:
    parts: List[str] = []
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ["stdout", "output", "result", "text", "raw", "stderr"]:
            if key in value:
                parts.append(collect_text(value[key]))
        for key, item in value.items():
            if key not in {"stdout", "output", "result", "text", "raw", "stderr"}:
                nested = collect_text(item)
                if nested:
                    parts.append(nested)
    elif isinstance(value, list):
        for item in value:
            nested = collect_text(item)
            if nested:
                parts.append(nested)
    return "\n".join(part for part in parts if part)


INFRASTRUCTURE_ERROR_MARKERS = (
    "Could not execute command through OSWorld API",
    "Remote end closed connection without response",
    "Connection reset by peer",
    "timed out",
    "TimeoutError",
    "Display output is not active",
    "Could not extract PNG screenshot",
    "OSWorld /execute",
    "/execute",
    "/screenshot",
)


def looks_like_infrastructure_error(exc: Exception) -> bool:
    text = str(exc)
    return any(marker in text for marker in INFRASTRUCTURE_ERROR_MARKERS)


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, str) and len(value) > 2000:
        return value[:2000] + "...<truncated>"
    return value


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_for_json(record), ensure_ascii=False) + "\n")
        handle.flush()


def extract_base64_png(execute_result: Dict[str, Any]) -> bytes:
    text = collect_text(execute_result)
    candidates = re.findall(r"[A-Za-z0-9+/=]{100,}", text)
    for candidate in candidates:
        try:
            data = base64.b64decode(candidate, validate=True)
            if data.startswith(b"\x89PNG"):
                return data
        except Exception:
            continue
    raise InfrastructureError("Could not extract PNG screenshot from OSWorld /execute response")


def is_inactive_display_png(data: bytes) -> bool:
    """Detect the black noVNC/X11 failure screen shown as "Display output is not active"."""
    try:
        from PIL import Image
    except Exception:
        return False
    try:
        with Image.open(io.BytesIO(data)) as image:
            sample = image.convert("L").resize((160, 90))
            pixels = list(sample.getdata())
    except Exception:
        return False
    if not pixels:
        return False
    dark_ratio = sum(1 for value in pixels if value < 25) / len(pixels)
    bright_ratio = sum(1 for value in pixels if value > 200) / len(pixels)
    average = sum(pixels) / len(pixels)
    return dark_ratio > 0.88 and bright_ratio < 0.08 and average < 35


class OSWorldHTTPPyAutoGUIEnv:
    """Minimal env compatible with GPT54Agent.

    GPT54Agent already converts computer-use actions into pyautogui Python code.
    This env only sends that code through OSWorld `/execute` and obtains screenshots
    through pyautogui inside the VM. It deliberately does not reimplement action logic.
    """

    def __init__(self, worker: DockerWorkerManager, pause_after_action: float = 1.0) -> None:
        self.worker = worker
        self.pause_after_action = pause_after_action
        self.action_history: List[Any] = []

    def reset(self, task_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        if self.worker.dry_run:
            return {
                "screenshot": self._blank_png(),
                "accessibility_tree": None,
                "terminal": None,
            }
        screenshot = self._get_api_screenshot()
        if screenshot:
            if is_inactive_display_png(screenshot):
                raise InfrastructureError("Display output is not active")
            return {
                "screenshot": screenshot,
                "accessibility_tree": None,
                "terminal": None,
            }
        screenshot = self._get_vnc_screenshot()
        if screenshot:
            if is_inactive_display_png(screenshot):
                raise InfrastructureError("Display output is not active")
            return {
                "screenshot": screenshot,
                "accessibility_tree": None,
                "terminal": None,
            }
        screenshot_command = r"""
python3 - <<'PY'
import base64
import io
import pyautogui

img = pyautogui.screenshot()
buf = io.BytesIO()
img.save(buf, format="PNG")
print(base64.b64encode(buf.getvalue()).decode("ascii"))
PY
"""
        result = self.worker.execute(screenshot_command, timeout=15)
        return {
            "screenshot": extract_base64_png(result),
            "accessibility_tree": None,
            "terminal": None,
        }

    def _get_api_screenshot(self) -> Optional[bytes]:
        try:
            data = self.worker.http_get_bytes("/screenshot")
        except Exception as exc:
            self.worker.write_state("api-screenshot-error.json", {"error": str(exc)})
            return None
        if data.startswith(b"\x89PNG") and len(data) > 1024:
            return data
        self.worker.write_state(
            "api-screenshot-error.json",
            {"error": "invalid screenshot payload", "length": len(data), "prefix": data[:16].hex()},
        )
        return None

    def _container_ip(self) -> str:
        result = self.worker.run_command(
            [
                "docker",
                "inspect",
                self.worker.config.container_name,
                "--format",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            ],
            check=False,
        )
        return result.stdout.strip()

    def _get_vnc_screenshot(self) -> Optional[bytes]:
        container_ip = self._container_ip()
        if not container_ip:
            return None
        vncdo = os.environ.get("SOFTWAREGYM_VNCDO") or shutil.which("vncdo")
        if not vncdo or not Path(vncdo).exists():
            return None
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            screenshot_path = Path(handle.name)
        try:
            result = subprocess.run(
                [vncdo, "-s", f"{container_ip}::5900", "capture", str(screenshot_path)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
                check=False,
            )
            if result.returncode == 0 and screenshot_path.exists() and screenshot_path.stat().st_size > 0:
                data = screenshot_path.read_bytes()
                if data.startswith(b"\x89PNG"):
                    return data
            self.worker.write_state(
                "vnc-screenshot-error.json",
                {
                    "returncode": result.returncode,
                    "stdout": result.stdout[-1000:],
                    "stderr": result.stderr[-1000:],
                    "container_ip": container_ip,
                },
            )
        except Exception as exc:
            self.worker.write_state("vnc-screenshot-error.json", {"error": str(exc), "container_ip": container_ip})
        finally:
            try:
                screenshot_path.unlink()
            except OSError:
                pass
        return None

    @staticmethod
    def _blank_png() -> bytes:
        # 1x1 transparent PNG.
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )

    @staticmethod
    def _python_action_command(action: Any) -> str:
        code = str(action).strip()
        if code.startswith("```"):
            lines = code.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines).strip()
        return f"python3 - <<'PY'\n{code}\nPY\n"

    def step(self, action: Any, pause: float = 0.0) -> Tuple[Dict[str, Any], int, bool, Dict[str, Any]]:
        self.action_history.append(action)
        info: Dict[str, Any] = {}
        done = False
        if action in ["WAIT", "DONE", "FAIL"]:
            if action == "DONE":
                done = True
                info["done"] = True
            elif action == "FAIL":
                done = True
                info["fail"] = True
            else:
                time.sleep(pause or self.pause_after_action)
        else:
            command = self._python_action_command(action)
            result = self.worker.execute(command, timeout=30)
            info["execute_result"] = result
            time.sleep(pause or self.pause_after_action)
        return self._get_obs(), 0, done, info


@dataclass
class CaseRunResult:
    status: str
    evidence: str
    checkpoints: List[Dict[str, Any]]
    blocker: str
    repair_hint: str
    steps: List[Dict[str, Any]]
    raw_result: Dict[str, Any]


def render_template_lines(lines: List[str], variables: Dict[str, str]) -> str:
    rendered: List[str] = []
    for line in lines:
        text = line
        for key, value in variables.items():
            text = text.replace("{{" + key + "}}", value)
        rendered.append(text)
    return "\n".join(rendered)


def _case_prompt_projection(case: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only case fields needed by GUI execution and visual judging."""
    projected = {
        key: case[key]
        for key in (
            "id",
            "title",
            "goal",
            "initial_state",
            "gui_steps_hint",
            "expected_result",
        )
        if key in case
    }
    setup = case.get("setup")
    if isinstance(setup, Mapping) and str(setup.get("mechanism") or "none") != "none":
        projected["setup"] = dict(setup)
    checkpoints = case.get("checkpoints")
    if isinstance(checkpoints, list):
        projected["checkpoints"] = [
            {
                key: checkpoint[key]
                for key in ("name", "criteria")
                if key in checkpoint
            }
            for checkpoint in checkpoints
            if isinstance(checkpoint, Mapping)
        ]
    return projected


def build_case_instruction(
    case: Dict[str, Any],
    requirement_summary: str = "",
    role: str = "legacy",
) -> str:
    prompt_config = json.loads(GUI_AGENT_PROMPT_PATH.read_text(encoding="utf-8"))
    template_key = {
        "legacy": "template",
        "executor": "executor_template",
        "judge": "judge_template",
    }.get(role)
    if template_key is None:
        raise ValueError(f"Unknown GUIAgent prompt role: {role}")
    template = prompt_config.get(template_key, [])
    if not isinstance(template, list):
        raise ValueError(f"{GUI_AGENT_PROMPT_PATH} {template_key} must be a list")
    prompt_case = case if role == "legacy" else _case_prompt_projection(case)
    default_summary = (
        "No additional context; use the current case as the source of truth."
        if role == "legacy"
        else "No additional context; use the case contract above."
    )
    return render_template_lines(
        [str(line) for line in template],
        {
            "case_json": json.dumps(prompt_case, ensure_ascii=False, indent=2),
            "requirement_summary": requirement_summary or default_summary,
        },
    )


def build_step_budget_exhausted_instruction(base_instruction: str, max_steps: int) -> str:
    return (
        base_instruction
        + "\n\n## Step Budget Exhausted\n"
        + f"You have already used all {max_steps} allowed GUI action steps. "
        + "Do not perform any more computer actions. Do not click, type, scroll, wait, or request another screenshot.\n"
        + "You must now output the final RESULT_JSON based only on the current screenshot and the action history. "
        + "For each checkpoint that is visibly verified, mark it success. For each checkpoint that is visibly contradicted, "
        + "mark it fail. For every checkpoint or function that you did not verify, mark its status as unknown and explain why.\n"
        + "Return exactly one RESULT_JSON fenced code block and no GUI action."
    )


def parse_result_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = fenced or re.findall(r"(\{.*\})", text, flags=re.DOTALL)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "status" in value:
            return value
    return None


def case_goal_text(case: Mapping[str, Any]) -> str:
    for key in ("user_request", "task_goal", "title", "id"):
        value = str(case.get(key) or "").strip()
        if value:
            return value
    return "the requested GUI case"


def compact_action_text(action: Any, limit: int = 240) -> str:
    if isinstance(action, Mapping):
        raw = action.get("action", action)
    else:
        raw = action
    text = str(raw).replace("\n", "\\n")
    return text[:limit] + ("..." if len(text) > limit else "")


def fallback_step_results(steps: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    for step in steps:
        actions = step.get("actions")
        action_items = actions if isinstance(actions, list) else []
        action_text = " / ".join(compact_action_text(item) for item in action_items[:4])
        if len(action_items) > 4:
            action_text += f" / ... ({len(action_items)} actions total)"
        response = ""
        predict_info = step.get("predict_info")
        if isinstance(predict_info, Mapping):
            response = str(predict_info.get("response") or "").strip()
        screenshots = step.get("after_screenshots")
        screenshot_text = ""
        if isinstance(screenshots, list) and screenshots:
            screenshot_text = f"latest screenshot: {screenshots[-1]}"
        errors: List[str] = []
        for item in step.get("action_results") or []:
            if isinstance(item, Mapping) and item.get("error"):
                errors.append(str(item.get("error")))
        evidence_parts = []
        if response:
            evidence_parts.append("model response/reasoning: " + response[:500])
        if screenshot_text:
            evidence_parts.append(screenshot_text)
        if errors:
            evidence_parts.append("errors: " + " | ".join(errors)[:500])
        results.append(
            {
                "step": int(step.get("step") or len(results) + 1),
                "action": action_text or response or "No executable action recorded.",
                "observation": screenshot_text or "No screenshot was recorded for this step.",
                "expected": "Progress toward the case goal.",
                "status": "fail" if errors else "success",
                "evidence": " ; ".join(evidence_parts) or "Action was recorded in the trace.",
            }
        )
    return results


def fallback_checkpoints(case: Mapping[str, Any], blocker: str, evidence: str) -> List[Dict[str, str]]:
    raw_checkpoints = case.get("checkpoints")
    if raw_checkpoints is None:
        raw_checkpoints = case.get("step_checkpoints")
    checkpoints: List[Dict[str, str]] = []
    if isinstance(raw_checkpoints, list):
        for index, checkpoint in enumerate(raw_checkpoints, start=1):
            if not isinstance(checkpoint, Mapping):
                continue
            name = str(
                checkpoint.get("name")
                or checkpoint.get("checkpoint_id")
                or checkpoint.get("id")
                or f"checkpoint_{index:03d}"
            )
            criteria = str(
                checkpoint.get("criteria")
                or checkpoint.get("expected_visible_state")
                or checkpoint.get("expected_data_state")
                or checkpoint.get("description")
                or ""
            )
            checkpoints.append(
                {
                    "name": name,
                    "status": "unknown",
                    "evidence": evidence,
                    "reason": (
                        "Step budget was exhausted before GUIAgent produced RESULT_JSON. "
                        f"This checkpoint could not be conclusively verified. Expected: {criteria}"
                    ).strip(),
                }
            )
    if checkpoints:
        return checkpoints
    return [
        {
            "name": "Case completion",
            "status": "blocked",
            "evidence": evidence,
            "reason": f"GUIAgent did not produce a valid RESULT_JSON before stopping with {blocker}.",
        }
    ]


def fallback_blocked_result(
    case: Mapping[str, Any],
    status: str,
    blocker: str,
    evidence: str,
    steps: List[Dict[str, Any]],
    last_response: str = "",
    repair_hint: str = "",
) -> Dict[str, Any]:
    goal = case_goal_text(case)
    step_results = fallback_step_results(steps)
    latest_screenshot = ""
    for step in reversed(steps):
        screenshots = step.get("after_screenshots")
        if isinstance(screenshots, list) and screenshots:
            latest_screenshot = str(screenshots[-1])
            break
    last_actions: List[str] = []
    for step in steps[-3:]:
        actions = step.get("actions")
        if isinstance(actions, list):
            last_actions.extend(compact_action_text(item) for item in actions[:3])
    return {
        "status": status,
        "task_summary": f"{status}: attempted '{goal}' for {len(steps)} model step(s), but stopped with {blocker}.",
        "task_completion": {
            "completed": False,
            "reason": evidence,
        },
        "step_results": step_results,
        "checkpoints": fallback_checkpoints(case, blocker, evidence),
        "final_observation": (
            f"Latest recorded screenshot: {latest_screenshot}" if latest_screenshot else "No final screenshot was recorded."
        ),
        "failure_analysis": (
            f"Case goal: {goal}\n"
            f"Blocker: {blocker}\n"
            f"Evidence: {evidence}\n"
            f"Last response: {last_response[:1000]}\n"
            f"Last actions: {' / '.join(last_actions[-6:])}"
        ),
        "blocker": blocker,
        "repair_hint": repair_hint,
    }


_SEED_THINK_TOKEN = "think_never_used_51bce0c785ca2f68081bfa7d91973934"
_SEED_VALID_STATUSES = {"Pass", "Fail", "Blocked", "InfraError"}
_SEED_GUIEVAL_SUFFIX = r"""

## Seed response contract

Emit only Seed GUI tool calls while work remains. When stopping, call `finished` exactly once
with a concise note; never output RESULT_JSON or call `call_user`.
"""


def _seed_usage_from_response(response: Mapping[str, Any]) -> Dict[str, int]:
    usage = response.get("usage") if isinstance(response, Mapping) else {}
    usage = usage if isinstance(usage, Mapping) else {}
    prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    completion = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    input_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    input_details = input_details if isinstance(input_details, Mapping) else {}
    output_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    output_details = output_details if isinstance(output_details, Mapping) else {}
    cached = min(
        prompt,
        max(
            0,
            int(input_details.get("cached_tokens", 0) or 0),
            int(usage.get("cached_input_tokens", 0) or 0),
            int(usage.get("cache_read_input_tokens", 0) or 0),
            int(usage.get("prompt_cache_hit_tokens", 0) or 0),
        ),
    )
    reasoning = min(
        completion,
        max(
            0,
            int(output_details.get("reasoning_tokens", 0) or 0),
            int(usage.get("reasoning_output_tokens", 0) or 0),
        ),
    )
    return {
        "prompt_tokens": prompt,
        "cached_input_tokens": cached,
        "completion_tokens": completion,
        "reasoning_output_tokens": reasoning,
        "total_tokens": int(usage.get("total_tokens", 0) or 0) or prompt + completion,
    }


def _seed_merge_usage(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "model_time": 0.0,
        "prompt_tokens": 0,
        "cached_input_tokens": 0,
        "completion_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 0,
        "model_calls": 0,
    }
    for record in records:
        merged["model_time"] += float(record.get("model_time", 0.0) or 0.0)
        merged["prompt_tokens"] += int(record.get("prompt_tokens", 0) or 0)
        merged["cached_input_tokens"] += int(record.get("cached_input_tokens", 0) or 0)
        merged["completion_tokens"] += int(record.get("completion_tokens", 0) or 0)
        merged["reasoning_output_tokens"] += int(record.get("reasoning_output_tokens", 0) or 0)
        merged["total_tokens"] += int(record.get("total_tokens", 0) or 0)
        merged["model_calls"] += 1
    return merged


def _seed_json_objects(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    output: List[Dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            output.append(value)
    return output


def _seed_result_from_text(text: str) -> Optional[Dict[str, Any]]:
    for value in _seed_json_objects(text):
        if str(value.get("status", "")).strip():
            return value
    return None


def _normalize_seed_result(value: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    status_lookup = {status.lower(): status for status in _SEED_VALID_STATUSES}
    status = status_lookup.get(str(value.get("status", "")).strip().lower())
    if not status:
        return None
    result = dict(value)
    result["status"] = status
    result.setdefault("task_summary", "")
    completion = result.get("task_completion")
    if not isinstance(completion, dict):
        completion = {}
    completion.setdefault("completed", status == "Pass")
    completion.setdefault("reason", result.get("task_summary", ""))
    result["task_completion"] = completion
    for key in ("step_results", "checkpoints"):
        if not isinstance(result.get(key), list):
            result[key] = []
    result.setdefault("final_observation", "")
    result.setdefault("failure_analysis", "" if status == "Pass" else result.get("task_summary", ""))
    result.setdefault("blocker", "")
    result.setdefault("repair_hint", "")
    return result


def _render_seed_result(result: Dict[str, Any]) -> str:
    return "```json\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n```"


class SeedAgentBridge:
    """Seed GUI executor plus a separate, result-only judge."""

    def __init__(
        self,
        env: OSWorldHTTPPyAutoGUIEnv,
        platform: str = "ubuntu",
        model: Optional[str] = None,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        **_: Any,
    ) -> None:
        if action_space != "pyautogui":
            raise ValueError("SeedAgentBridge only supports pyautogui actions")
        if observation_type != "screenshot":
            raise ValueError("SeedAgentBridge only supports screenshot observations")

        self.env = env
        self.seed_module = load_seed_agent_module()
        self.exec_model = (os.environ.get("SEED_EXEC_MODEL") or "").strip()
        if not self.exec_model:
            raise RuntimeError("SEED_EXEC_MODEL is not set; configure the Seed executor model in the repo-root .env")
        self.exec_api_key = (
            os.environ.get("SEED_EXEC_API_KEY")
            or os.environ.get("DOUBAO_API_KEY")
            or os.environ.get("LEGACY_API_KEY")
            or ""
        )
        if not self.exec_api_key:
            raise RuntimeError("SEED_EXEC_API_KEY is not set; configure the Seed executor API key in the repo-root .env")
        self.exec_base_url = (
            os.environ.get("SEED_EXEC_BASE_URL")
            or os.environ.get("DOUBAO_API_URL")
            or os.environ.get("LEGACY_OPENAI_BASE_URL")
            or ""
        ).rstrip("/")
        if not self.exec_base_url:
            raise RuntimeError("SEED_EXEC_BASE_URL is not set; configure the Seed executor base URL in the repo-root .env")
        self.exec_reasoning_effort = os.environ.get("SEED_EXEC_REASONING_EFFORT", "high")
        self.request_timeout = int(os.environ.get("SEED_REQUEST_TIMEOUT", "180"))
        self.exec_request_retries = max(
            0,
            int(os.environ.get("SEED_EXEC_REQUEST_RETRIES", "2")),
        )
        self.exec_all_images = (
            os.environ.get("SEED_EXEC_ALL_IMAGES", "0").strip().lower()
            in {"1", "true", "yes"}
        )
        self.sleep_after_execution = float(os.environ.get("SEED_SLEEP_AFTER_EXEC", "1.0"))
        self.loop_detection_enabled = (
            os.environ.get("SEED_LOOP_DETECTION", "1").strip().lower()
            not in {"0", "false", "no"}
        )
        self.loop_same_action_repeats = max(
            2,
            int(os.environ.get("SEED_LOOP_SAME_ACTION_REPEATS", "2")),
        )
        self.loop_no_progress_steps = max(
            0,
            int(os.environ.get("SEED_LOOP_NO_PROGRESS_STEPS", "4")),
        )
        self.loop_min_visual_change = max(
            0.0,
            float(os.environ.get("SEED_LOOP_MIN_VISUAL_CHANGE", "0.15")),
        )
        self.judge_uses_exec_config = (
            os.environ.get("SEED_JUDGE_USE_EXEC_CONFIG", "1").strip().lower()
            not in {"0", "false", "no"}
        )
        if self.judge_uses_exec_config:
            self.judge_model = self.exec_model
            self.judge_api_key = self.exec_api_key
            self.judge_base_url = self.exec_base_url
        else:
            self.judge_model = os.environ.get("SEED_JUDGE_MODEL") or self.exec_model
            self.judge_api_key = (
                os.environ.get("SEED_JUDGE_API_KEY")
                or os.environ.get("GUIAGENT_OPENAI_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("API_KEY")
                or ""
            )
            self.judge_base_url = (
                os.environ.get("SEED_JUDGE_BASE_URL")
                or os.environ.get("GUIAGENT_OPENAI_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("BASE_URL")
                or ""
            ).rstrip("/")
        self.judge_max_tokens = int(os.environ.get("SEED_JUDGE_MAX_TOKENS", "8000"))
        self.judge_temperature = os.environ.get("SEED_JUDGE_TEMPERATURE")
        self.judge_reasoning_effort = os.environ.get(
            "SEED_JUDGE_REASONING_EFFORT",
            self.exec_reasoning_effort,
        )
        self.judge_attempts = max(1, int(os.environ.get("SEED_JUDGE_ATTEMPTS", "2")))
        self.judge_history_images = max(1, int(os.environ.get("SEED_JUDGE_HISTORY_IMAGES", "10")))
        self.judge_all_images = (
            os.environ.get("SEED_JUDGE_ALL_IMAGES", "0").strip().lower()
            in {"1", "true", "yes"}
        )
        self.judge_include_initial = (
            os.environ.get("SEED_JUDGE_INCLUDE_INITIAL", "0").strip().lower()
            in {"1", "true", "yes"}
        )
        self.judge_image_size = _parse_image_size(
            os.environ.get("SEED_JUDGE_IMAGE_SIZE", ""),
            "SEED_JUDGE_IMAGE_SIZE",
        )
        self.judge_timeout = int(os.environ.get("SEED_JUDGE_REQUEST_TIMEOUT", str(self.request_timeout)))
        self.judge_request_retries = max(
            0,
            int(os.environ.get("SEED_JUDGE_REQUEST_RETRIES", "2")),
        )
        self.judge_retry_backoff = max(
            0.0,
            float(os.environ.get("SEED_JUDGE_RETRY_BACKOFF_SECONDS", "2")),
        )
        self._inference_usage: List[Dict[str, Any]] = []
        self._base_instruction = ""
        self._judge_instruction = ""
        self._execution_log: List[str] = []
        self._judge_frames: List[Dict[str, Any]] = []
        self._judge_action_count = 0
        self._request_context: Dict[str, str] = {}
        self.request_log_path = Path(
            os.environ.get(
                "SEED_REQUEST_LOG",
                str(REPO_ROOT / "output" / "_pipeline_logs" / "seed_request_events.jsonl"),
            )
        )
        shared_image_size = _shared_gui_image_size()
        if shared_image_size is None:
            seed_resize_image = os.environ.get("SEED_RESIZE_IMAGE", "0") == "1"
            seed_image_width = int(os.environ.get("SEED_RESIZED_IMAGE_WIDTH", "1920"))
            seed_image_height = int(os.environ.get("SEED_RESIZED_IMAGE_HEIGHT", "1080"))
        else:
            seed_resize_image = True
            seed_image_width, seed_image_height = shared_image_size

        max_trajectory_length = int(os.environ.get("SEED_MAX_TRAJECTORY_LENGTH", "100"))
        history_n = max(1, int(os.environ.get("SEED_HISTORY_N", "5")))
        if self.exec_all_images:
            history_n = max(history_n, max_trajectory_length)

        self.reference = self.seed_module.SeedAgent(
            model=self.exec_model,
            model_type="seed",
            max_tokens=int(os.environ.get("SEED_EXEC_MAX_TOKENS", "4096")),
            top_p=float(os.environ.get("SEED_EXEC_TOP_P", "0.9")),
            temperature=float(os.environ.get("SEED_EXEC_TEMPERATURE", "0.0")),
            max_trajectory_length=max_trajectory_length,
            history_n=history_n,
            max_steps=int(os.environ.get("APPEVAL_MAX_STEPS", "60")),
            use_thinking=True,
            resize_image=seed_resize_image,
            resized_image_width=seed_image_width,
            resized_image_height=seed_image_height,
            inference_retries=self.exec_request_retries,
        )
        self.reference.platform = platform
        # Keep SeedAgent's prompts/history/action conversion, replacing only transport
        # so GUIEval gets non-streaming token usage from the same Seed model call.
        self.reference.inference_func = self._inference_with_usage

    @staticmethod
    def _chat_url(base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if not base.endswith("/v1"):
            base += "/v1"
        return base + "/chat/completions"

    def set_request_context(self, case_id: str, trace_dir: Optional[Path]) -> None:
        self._request_context = {
            "case_id": str(case_id or ""),
            "trace_dir": str(trace_dir.resolve()) if trace_dir is not None else "",
        }

    def set_judge_instruction(self, instruction: str) -> None:
        self._judge_instruction = str(instruction or "").strip()

    def _request_log_target(self) -> Path:
        configured = getattr(
            self,
            "request_log_path",
            REPO_ROOT / "output" / "_pipeline_logs" / "seed_request_events.jsonl",
        )
        suffix = configured.suffix or ".jsonl"
        stem = configured.stem if configured.suffix else configured.name
        return configured.with_name(f"{stem}.{os.getpid()}{suffix}")

    def _log_request_event(self, event: str, **fields: Any) -> None:
        record = {
            "time": datetime.now().astimezone().isoformat(),
            "event": event,
            "pid": os.getpid(),
            **getattr(self, "_request_context", {}),
            **fields,
        }
        try:
            with _SEED_REQUEST_LOG_LOCK:
                append_jsonl(self._request_log_target(), record)
        except Exception:
            pass

    @staticmethod
    def _payload_shape(payload: Mapping[str, Any]) -> Dict[str, Any]:
        image_count = 0
        image_chars = 0
        text_chars = 0
        messages = payload.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, Mapping):
                    continue
                content = message.get("content")
                items = content if isinstance(content, list) else [content]
                for item in items:
                    if isinstance(item, str):
                        text_chars += len(item)
                    elif isinstance(item, Mapping):
                        if item.get("type") == "text":
                            text_chars += len(str(item.get("text") or ""))
                        image_url = item.get("image_url")
                        if isinstance(image_url, Mapping):
                            url = str(image_url.get("url") or "")
                            if url:
                                image_count += 1
                                image_chars += len(url)
        return {
            "message_count": len(messages) if isinstance(messages, list) else 0,
            "text_chars": text_chars,
            "image_count": image_count,
            "image_data_chars": image_chars,
            "max_tokens": payload.get("max_tokens"),
            "reasoning_effort": payload.get("reasoning_effort"),
            "stream": payload.get("stream"),
        }

    @staticmethod
    def _response_request_ids(response: requests.Response) -> Dict[str, str]:
        result: Dict[str, str] = {}
        headers = getattr(response, "headers", {}) or {}
        for header in (
            "x-request-id",
            "request-id",
            "x-ms-request-id",
            "x-litellm-call-id",
            "openai-request-id",
        ):
            value = headers.get(header)
            if value:
                result[header.replace("-", "_")] = value
        return result

    def _post_seed_request(
        self,
        *,
        role: str,
        base_url: str,
        api_key: str,
        payload: Dict[str, Any],
        timeout: int,
        parent_request_id: str = "",
    ) -> Tuple[requests.Response, str]:
        request_id = "softwaregym-" + uuid.uuid4().hex
        endpoint = self._chat_url(base_url)
        parsed = urlsplit(endpoint)
        context_headers = {
            "X-SoftwareGym-Request-ID": request_id,
            "X-SoftwareGym-Role": role,
            "X-SoftwareGym-Case-ID": getattr(self, "_request_context", {}).get("case_id", ""),
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            **context_headers,
        }
        session = requests.Session()
        session.trust_env = False
        started_wall = datetime.now().astimezone()
        started = time.monotonic()
        common = {
            "request_id": request_id,
            "parent_request_id": parent_request_id,
            "role": role,
            "model": payload.get("model"),
            "endpoint": f"{parsed.scheme}://{parsed.netloc}{parsed.path}",
            "timeout_seconds": timeout,
            "request_shape": self._payload_shape(payload),
        }
        self._log_request_event(
            "seed_request_started",
            started_at=started_wall.isoformat(),
            **common,
        )
        try:
            response = session.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except Exception as exc:
            duration_ms = round((time.monotonic() - started) * 1000, 1)
            self._log_request_event(
                "seed_request_failed",
                duration_ms=duration_ms,
                exception_type=type(exc).__name__,
                error=str(exc)[:2000],
                cause_type=type(exc.__cause__).__name__ if exc.__cause__ is not None else "",
                cause=str(exc.__cause__)[:1000] if exc.__cause__ is not None else "",
                **common,
            )
            raise RuntimeError(
                f"Seed {role} request_id={request_id} {type(exc).__name__}: {exc}"
            ) from exc
        duration_ms = round((time.monotonic() - started) * 1000, 1)
        self._log_request_event(
            "seed_request_completed",
            duration_ms=duration_ms,
            status_code=response.status_code,
            response_request_ids=self._response_request_ids(response),
            **common,
        )
        return response, request_id

    def _inference_with_usage(self, messages: List[Dict[str, Any]]) -> str:
        payload: Dict[str, Any] = {
            "model": self.exec_model,
            "messages": messages,
            "max_tokens": self.reference.max_tokens,
            "temperature": self.reference.temperature,
            "top_p": self.reference.top_p,
            "stream": False,
        }
        if self.exec_reasoning_effort:
            payload["reasoning_effort"] = self.exec_reasoning_effort
        started = time.monotonic()
        response, request_id = self._post_seed_request(
            role="executor",
            base_url=self.exec_base_url,
            api_key=self.exec_api_key,
            payload=payload,
            timeout=self.request_timeout,
        )
        if (
            response.status_code == 400
            and "reasoning_effort" in payload
            and "reasoning_effort" in response.text
            and "UnsupportedParamsError" in response.text
        ):
            # Some Ark endpoints are routed through an OpenAI-compatible
            # proxy provider, which rejects reasoning_effort. Retry once without it.
            payload.pop("reasoning_effort", None)
            response, request_id = self._post_seed_request(
                role="executor_compat_retry",
                base_url=self.exec_base_url,
                api_key=self.exec_api_key,
                payload=payload,
                timeout=self.request_timeout,
                parent_request_id=request_id,
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"Seed request_id={request_id} HTTP {response.status_code}: {response.text[:500]}"
            )
        value = response.json()
        message = (value.get("choices") or [{}])[0].get("message") or {}
        content = str(message.get("content") or "")
        reasoning = str(message.get("reasoning_content") or "")
        usage = _seed_usage_from_response(value)
        self._inference_usage.append(
            {**usage, "model_time": time.monotonic() - started, "request_id": request_id}
        )
        return f"<{_SEED_THINK_TOKEN}>{reasoning}</{_SEED_THINK_TOKEN}>{content}"

    def reset(self, _logger: Any = None, **kwargs: Any) -> None:
        self.reference.reset(_logger=_logger, **kwargs)
        self._inference_usage = []
        self._base_instruction = ""
        self._judge_instruction = ""
        self._execution_log = []
        self._judge_frames = []
        self._judge_action_count = 0
        self._last_judge_frame: Optional[Dict[str, Any]] = None
        self._loop_previous_observation: Optional[Mapping[str, Any]] = None
        self._loop_last_action_signature = ""
        self._loop_same_action_count = 0
        self._loop_no_progress_count = 0
        self._loop_recovery_pending = False
        self._loop_trigger_reason = ""
        self._loop_recent_ineffective_actions: List[Dict[str, str]] = []

    @staticmethod
    def _thought(prediction: str) -> str:
        if f"</{_SEED_THINK_TOKEN}>" not in prediction:
            return prediction
        return (
            prediction.split(f"</{_SEED_THINK_TOKEN}>", 1)[0]
            .replace(f"<{_SEED_THINK_TOKEN}>", "")
            .strip()
        )

    @staticmethod
    def _wrap_actions(actions: List[Any]) -> List[Dict[str, Any]]:
        wrapped: List[Dict[str, Any]] = []
        for action in actions:
            code = str(action or "").strip()
            if not code or code in {"DONE", "FAIL", "WAIT"}:
                continue
            wrapped.append({"action_space": "pyautogui", "action": code, "batch_last": False})
        if wrapped:
            wrapped[-1]["batch_last"] = True
        return wrapped

    def _align_reference_trajectory(self) -> None:
        target = len(self.reference.observations)
        while len(self.reference.thoughts) < target:
            self.reference.thoughts.append("")
        while len(self.reference.actions) < target:
            self.reference.actions.append([])

    def _predict_reference(self, instruction: str, obs: Dict[str, Any]) -> Tuple[str, List[Any]]:
        history_before = len(self.reference.history_responses)
        try:
            prediction, actions = self.reference.predict(instruction, obs)
        except ValueError:
            if len(self.reference.history_responses) <= history_before:
                raise
            prediction = self.reference.history_responses[-1]
            actions = ["DONE"]
        self._align_reference_trajectory()
        return str(prediction or ""), list(actions or [])

    def _parsed_tool_calls(self, prediction: str) -> List[Dict[str, Any]]:
        try:
            calls = self.seed_module.parse_xml_action_v3(
                prediction,
                self.seed_module.GUI_TOOL_SCHEMAS,
            )
        except Exception:
            return []
        return [call for call in calls if isinstance(call, dict)]

    def _terminal_call(self, prediction: str) -> Tuple[str, str]:
        terminal_functions = {
            self.seed_module.FINISH_WORD,
            self.seed_module.ENV_FAIL_WORD,
            self.seed_module.CALL_USER,
            self.seed_module.INFEASIBLE,
        }
        for call in self._parsed_tool_calls(prediction):
            function = str(call.get("function", "")).strip()
            if function not in terminal_functions:
                continue
            parameters = call.get("parameters")
            parameters = parameters if isinstance(parameters, Mapping) else {}
            return function, str(parameters.get("content") or "")
        return "", ""

    def _result_from_prediction(self, prediction: str) -> Optional[Dict[str, Any]]:
        direct = _seed_result_from_text(prediction)
        normalized = _normalize_seed_result(direct) if direct else None
        if normalized:
            return normalized
        _, terminal_content = self._terminal_call(prediction)
        nested = _seed_result_from_text(terminal_content)
        return _normalize_seed_result(nested) if nested else None

    def _record_execution(self, prediction: str, raw_actions: List[Any]) -> None:
        calls = self._parsed_tool_calls(prediction)
        call_names = [str(call.get("function") or "?") for call in calls]
        if not call_names:
            call_names = [str(action)[:80] for action in raw_actions] or ["no_action"]
        thought = re.sub(r"\s+", " ", self._thought(prediction)).strip()
        self._execution_log.append(
            f"step {len(self._execution_log) + 1}: "
            f"actions={', '.join(call_names)}; executor_note={thought[:300] or '(none)'}"
        )

    def _observation_screenshot_b64(self, obs: Mapping[str, Any]) -> str:
        screenshot = obs.get("screenshot")
        if isinstance(screenshot, (bytes, bytearray)):
            image_bytes = bytes(screenshot)
        elif isinstance(screenshot, str):
            if screenshot.startswith("data:image/") and "," in screenshot:
                screenshot = screenshot.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(screenshot, validate=True)
            except Exception:
                return screenshot
        else:
            return ""

        target_size = getattr(self, "judge_image_size", None)
        reference = getattr(self, "reference", None)
        if target_size is None and getattr(reference, "resize_image", False):
            width = int(getattr(reference, "resized_image_width", 0) or 0)
            height = int(getattr(reference, "resized_image_height", 0) or 0)
            if width > 0 and height > 0:
                target_size = (width, height)
        if target_size is not None:
            try:
                from PIL import Image

                with Image.open(io.BytesIO(image_bytes)) as image:
                    if image.size != target_size:
                        resized = image.convert("RGB").resize(
                            target_size,
                            Image.Resampling.LANCZOS,
                        )
                        output = io.BytesIO()
                        resized.save(output, format="PNG")
                        image_bytes = output.getvalue()
            except Exception:
                pass
        return base64.b64encode(image_bytes).decode("ascii")

    @staticmethod
    def _screenshot_bytes(obs: Mapping[str, Any]) -> bytes:
        screenshot = obs.get("screenshot")
        if isinstance(screenshot, (bytes, bytearray)):
            return bytes(screenshot)
        if not isinstance(screenshot, str):
            return b""
        value = screenshot.split(",", 1)[1] if screenshot.startswith("data:image/") and "," in screenshot else screenshot
        try:
            return base64.b64decode(value, validate=True)
        except Exception:
            return value.encode("utf-8", errors="replace")

    @classmethod
    def _visual_change_score(
        cls,
        previous: Optional[Mapping[str, Any]],
        current: Mapping[str, Any],
    ) -> float:
        if previous is None:
            return 100.0
        try:
            from PIL import Image, ImageChops, ImageStat

            previous_bytes = cls._screenshot_bytes(previous)
            current_bytes = cls._screenshot_bytes(current)
            if not previous_bytes or not current_bytes:
                return 0.0
            with Image.open(io.BytesIO(previous_bytes)) as before:
                before_sample = before.convert("L").resize((64, 36))
            with Image.open(io.BytesIO(current_bytes)) as after:
                after_sample = after.convert("L").resize((64, 36))
            difference = ImageChops.difference(before_sample, after_sample)
            return round(float(ImageStat.Stat(difference).mean[0]) / 255.0 * 100.0, 3)
        except Exception:
            return 0.0

    @classmethod
    def _perceptual_hash(cls, obs: Mapping[str, Any]) -> Optional[int]:
        try:
            from PIL import Image

            image_bytes = cls._screenshot_bytes(obs)
            if not image_bytes:
                return None
            with Image.open(io.BytesIO(image_bytes)) as image:
                sample = image.convert("L").resize((9, 8))
                if hasattr(sample, "get_flattened_data"):
                    pixels = list(sample.get_flattened_data())
                else:
                    pixels = list(sample.getdata())
            value = 0
            for row in range(8):
                offset = row * 9
                for column in range(8):
                    value = (value << 1) | int(
                        pixels[offset + column] > pixels[offset + column + 1]
                    )
            return value
        except Exception:
            return None

    @staticmethod
    def _action_frame_metadata(code: str) -> Tuple[int, str]:
        normalized = code.lower()
        if "pyautogui.write(" in normalized:
            return 95, "text_input_result"
        if "pyautogui.hotkey(" in normalized or "pyautogui.press(" in normalized:
            return 90, "keyboard_or_submit_result"
        if "pyautogui.click(" in normalized:
            return 65, "click_result"
        if "pyautogui.scroll(" in normalized or "pyautogui.drag" in normalized:
            return 35, "navigation_result"
        if "time.sleep(" in normalized:
            return 10, "wait_result"
        return 30, "action_result"

    @staticmethod
    def _loop_action_signature(code: Any) -> str:
        value = str(code or "")
        value = re.sub(r"'''[\s\S]*?'''", " ", value)
        value = re.sub(r"(?m)^\s*(?:import|from)\s+.*$", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    @classmethod
    def _loop_action_label(cls, code: Any) -> str:
        signature = cls._loop_action_signature(code)
        return signature[:180] or str(code or "").strip()[:180] or "(empty action)"

    def _loop_recovery_directive(self) -> str:
        if not (
            getattr(self, "loop_detection_enabled", True)
            and getattr(self, "_loop_recovery_pending", False)
        ):
            return ""
        labels: List[str] = []
        for item in getattr(self, "_loop_recent_ineffective_actions", []):
            label = str(item.get("label") or "")
            if label and label not in labels:
                labels.append(label)
        recent = "\n".join(f"- {label}" for label in labels[-4:]) or "- unavailable"
        reason = getattr(self, "_loop_trigger_reason", "no visible progress")
        return (
            "\n\n## Loop Recovery Required\n"
            f"The loop guard detected {reason}. The recent actions below produced no "
            "material visible progress:\n"
            f"{recent}\n"
            "Do not repeat any of those actions or click the same target again. Inspect "
            "the newest screenshot and choose one materially different GUI path. Check "
            "for an open dialog, overlay, missing focus, or incorrect page before acting. "
            "If no different evidence-based path exists, call `finished` once and state "
            "which checkpoint is blocked."
        )

    def _loop_candidate_repeats_ineffective(self, raw_actions: List[Any]) -> bool:
        if not (
            getattr(self, "loop_detection_enabled", True)
            and getattr(self, "_loop_recovery_pending", False)
        ):
            return False
        forbidden = {
            str(item.get("signature") or "")
            for item in getattr(self, "_loop_recent_ineffective_actions", [])
            if item.get("signature")
        }
        candidates = [
            self._loop_action_signature(action)
            for action in raw_actions
            if str(action or "").strip() not in {"", "DONE", "FAIL", "WAIT"}
        ]
        return bool(candidates) and all(candidate in forbidden for candidate in candidates)

    def _update_loop_state(
        self,
        previous_obs: Optional[Mapping[str, Any]],
        current_obs: Mapping[str, Any],
        code: Any,
    ) -> Dict[str, Any]:
        if not getattr(self, "loop_detection_enabled", True):
            return {"enabled": False}
        change_score = self._visual_change_score(previous_obs, current_obs)
        threshold = max(0.0, float(getattr(self, "loop_min_visual_change", 0.15)))
        no_progress = previous_obs is not None and change_score < threshold
        signature = self._loop_action_signature(code)
        if no_progress:
            self._loop_no_progress_count = int(
                getattr(self, "_loop_no_progress_count", 0)
            ) + 1
            if signature and signature == getattr(self, "_loop_last_action_signature", ""):
                self._loop_same_action_count = int(
                    getattr(self, "_loop_same_action_count", 0)
                ) + 1
            else:
                self._loop_same_action_count = 1
            recent = getattr(self, "_loop_recent_ineffective_actions", [])
            recent.append(
                {
                    "signature": signature,
                    "label": self._loop_action_label(code),
                }
            )
            self._loop_recent_ineffective_actions = recent[-8:]
        else:
            self._loop_no_progress_count = 0
            self._loop_same_action_count = 0
            self._loop_recovery_pending = False
            self._loop_trigger_reason = ""
            self._loop_recent_ineffective_actions = []
        self._loop_last_action_signature = signature

        same_limit = max(2, int(getattr(self, "loop_same_action_repeats", 2)))
        no_progress_limit = max(0, int(getattr(self, "loop_no_progress_steps", 4)))
        if no_progress and self._loop_same_action_count >= same_limit:
            self._loop_recovery_pending = True
            self._loop_trigger_reason = (
                f"the same action repeating {self._loop_same_action_count} times "
                "without visible change"
            )
        elif (
            no_progress
            and no_progress_limit > 0
            and self._loop_no_progress_count >= no_progress_limit
        ):
            self._loop_recovery_pending = True
            self._loop_trigger_reason = (
                f"{self._loop_no_progress_count} consecutive actions without visible change"
            )
        return {
            "enabled": True,
            "visual_change_score": change_score,
            "minimum_visual_change": threshold,
            "no_progress": no_progress,
            "no_progress_count": self._loop_no_progress_count,
            "same_action_count": self._loop_same_action_count,
            "recovery_pending": self._loop_recovery_pending,
            "trigger_reason": self._loop_trigger_reason,
        }

    def _remember_judge_frame(
        self,
        obs: Mapping[str, Any],
        *,
        step: int,
        action: str,
        priority: int,
        reason: str,
    ) -> None:
        image_bytes = self._screenshot_bytes(obs)
        if not image_bytes:
            return
        digest = hashlib.sha256(image_bytes).hexdigest()
        perceptual_hash = self._perceptual_hash(obs)
        frames = getattr(self, "_judge_frames", None)
        if frames is None:
            frames = []
            self._judge_frames = frames
        if (
            getattr(self, "judge_all_images", False)
            and reason == "final"
            and frames
            and frames[-1]["digest"] == digest
        ):
            existing = frames[-1]
            self._last_judge_frame = existing
            existing["priority"] = max(100, int(existing.get("priority", 0)))
            if "final" not in str(existing.get("reason", "")):
                existing["reason"] = f"{existing.get('reason', 'history')}+final"
            return
        if not getattr(self, "judge_all_images", False):
            for existing in frames:
                if existing["digest"] != digest:
                    continue
                self._last_judge_frame = existing
                if reason == "final":
                    existing["priority"] = max(100, int(existing.get("priority", 0)))
                    if "final" not in str(existing.get("reason", "")):
                        existing["reason"] = f"{existing.get('reason', 'history')}+final"
                    return
                existing["priority"] = max(priority, int(existing.get("priority", 0)))
                return
        previous = frames[-1]["observation"] if frames else None
        change_score = self._visual_change_score(previous, obs)
        if change_score >= 8.0 and priority < 80:
            priority = max(priority, 80)
            reason = f"{reason}+visual_change"
        frame = {
            "step": step,
            "action": action,
            "priority": priority,
            "reason": reason,
            "change_score": change_score,
            "digest": digest,
            "perceptual_hash": perceptual_hash,
            "observation": obs,
        }
        frames.append(frame)
        self._last_judge_frame = frame

    @staticmethod
    def _evenly_spaced(items: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        if count <= 0 or not items:
            return []
        if len(items) <= count:
            return list(items)
        if count == 1:
            return [items[len(items) // 2]]
        last_index = len(items) - 1
        indices = [round(index * last_index / (count - 1)) for index in range(count)]
        return [items[index] for index in dict.fromkeys(indices)]

    def _judge_evidence_frames(self, obs: Mapping[str, Any]) -> List[Dict[str, Any]]:
        frames = getattr(self, "_judge_frames", None)
        if frames is None:
            frames = []
            self._judge_frames = frames
        if not frames:
            observations = [
                item
                for item in getattr(self.reference, "observations", [])
                if isinstance(item, Mapping)
            ]
            for index, observation in enumerate(observations):
                self._remember_judge_frame(
                    observation,
                    step=index,
                    action="historical observation",
                    priority=100 if index == 0 else 40,
                    reason="initial" if index == 0 else "history",
                )
        self._remember_judge_frame(
            obs,
            step=max(0, int(getattr(self, "_judge_action_count", 0))),
            action="final observed state",
            priority=100,
            reason="final",
        )
        frames = list(getattr(self, "_judge_frames", []))
        if not frames:
            return []
        if not getattr(self, "judge_include_initial", False):
            without_initial = [
                frame
                for frame in frames
                if not (
                    int(frame.get("step", 0)) == 0
                    and str(frame.get("reason", "")) == "initial"
                )
            ]
            # A no-action run still needs its final observed state as evidence.
            if without_initial:
                frames = without_initial
        # The GPT judge endpoint accepts at most 50 images. Keep this as a hard
        # cap even when ALL_IMAGES is enabled or a larger history limit is set.
        limit = (
            50
            if getattr(self, "judge_all_images", False)
            else min(50, max(1, int(getattr(self, "judge_history_images", 10))))
        )
        if len(frames) <= limit:
            selected = frames
        elif limit == 1:
            selected = [frames[-1]]
        else:
            mandatory = [frames[0]]
            if frames[-1]["digest"] != frames[0]["digest"]:
                mandatory.append(frames[-1])
            slots = max(0, limit - len(mandatory))
            middle = [
                frame
                for frame in frames[1:-1]
                if frame["digest"] not in {item["digest"] for item in mandatory}
            ]
            # Reserve half the budget for chronological coverage so short-lived
            # checkpoint surfaces are not crowded out by text-entry frames. Use
            # the other half for the strongest action/visual-change evidence.
            coverage_count = min(len(middle), max(1, slots // 2)) if slots else 0
            chosen = self._evenly_spaced(middle, coverage_count)
            slots -= len(chosen)
            if slots > 0:
                chosen_digests = {frame["digest"] for frame in chosen}
                remaining = [frame for frame in middle if frame["digest"] not in chosen_digests]
                ranked = sorted(
                    remaining,
                    key=lambda frame: (
                        int(frame.get("priority", 0)),
                        float(frame.get("change_score", 0.0)),
                        -int(frame.get("step", 0)),
                    ),
                    reverse=True,
                )
                chosen.extend(ranked[:slots])
            selected = mandatory + chosen
        selected = list(selected)
        selected.sort(
            key=lambda frame: (
                int(frame.get("step", 0)),
                "final" in str(frame.get("reason", "")),
            )
        )
        result: List[Dict[str, Any]] = []
        for frame in selected:
            encoded = self._observation_screenshot_b64(frame["observation"])
            if not encoded:
                continue
            result.append({**frame, "image": encoded})
        return result

    def _judge_screenshots(self, obs: Mapping[str, Any]) -> List[str]:
        """Compatibility helper returning only the selected image payloads."""
        return [frame["image"] for frame in self._judge_evidence_frames(obs)]

    def _write_judge_evidence_manifest(
        self,
        trigger: str,
        frames: List[Dict[str, Any]],
    ) -> None:
        trace_value = getattr(self, "_request_context", {}).get("trace_dir", "")
        if not trace_value:
            return
        trace_dir = Path(trace_value)
        try:
            trace_dir.mkdir(parents=True, exist_ok=True)
            selected_digests = {frame.get("digest") for frame in frames}

            def frame_metadata(frame: Mapping[str, Any], sequence: int) -> Dict[str, Any]:
                step = int(frame.get("step", 0) or 0)
                reason = str(frame.get("reason") or "")
                screenshot_path = str(frame.get("screenshot_path") or "")
                if not screenshot_path:
                    screenshot_path = (
                        "screenshots/initial.png"
                        if step <= 0 or "initial" in reason
                        else f"screenshots/step_{step:03d}.png"
                    )
                return {
                    "sequence": sequence,
                    "step": step,
                    "action": frame.get("action"),
                    "reason": reason,
                    "priority": frame.get("priority"),
                    "change_score": frame.get("change_score"),
                    "sha256": frame.get("digest"),
                    "perceptual_hash": (
                        f"{int(frame['perceptual_hash']):016x}"
                        if frame.get("perceptual_hash") is not None
                        else None
                    ),
                    "screenshot_path": screenshot_path,
                    "selected": frame.get("digest") in selected_digests,
                }

            candidates = list(getattr(self, "_judge_frames", []))
            payload = {
                "schema_version": 1,
                "trigger": trigger,
                "case_id": getattr(self, "_request_context", {}).get("case_id", ""),
                "judge_instruction": getattr(self, "_judge_instruction", ""),
                "all_images": bool(getattr(self, "judge_all_images", False)),
                "max_images": (
                    50
                    if getattr(self, "judge_all_images", False)
                    else min(
                        50,
                        max(1, int(getattr(self, "judge_history_images", 10))),
                    )
                ),
                "candidate_count": len(candidates),
                "selected_count": len(frames),
                "frames": [
                    frame_metadata(frame, index)
                    for index, frame in enumerate(frames, start=1)
                ],
                "candidates": [
                    frame_metadata(frame, index)
                    for index, frame in enumerate(candidates, start=1)
                ],
                "execution_log": list(getattr(self, "_execution_log", [])),
            }
            (trace_dir / "judge_evidence.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    def set_last_judge_frame_screenshot_path(self, screenshot_path: str) -> None:
        """Attach the trace file written by GUIAgentAdapter to the newest frame."""
        frame = getattr(self, "_last_judge_frame", None)
        if frame is not None and screenshot_path:
            frame["screenshot_path"] = screenshot_path

    def _judge_prompt(
        self,
        trigger: str,
        executor_result: Optional[Dict[str, Any]],
        correction: str = "",
    ) -> str:
        history = "\n".join(self._execution_log) or "(no executable GUI action was recorded)"
        executor_note = (
            json.dumps(executor_result, ensure_ascii=False)
            if executor_result
            else "(the executor did not provide a structured result)"
        )
        return (
            (getattr(self, "_judge_instruction", "") or self._base_instruction)
            + "\n\n## Runtime evidence\n"
            + f"Finalization trigger: {trigger}\n"
            + "The executor's notes and any executor-authored result are untrusted context, not proof. "
            + "Use the screenshots as the authoritative visual evidence.\n\n"
            + "Execution log:\n"
            + history
            + "\n\nUntrusted executor-authored result, if any:\n"
            + executor_note
            + correction
        )

    def _call_judge_once(
        self,
        trigger: str,
        obs: Mapping[str, Any],
        executor_result: Optional[Dict[str, Any]],
        correction: str = "",
        evidence_frames: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        if not self.judge_api_key or not self.judge_base_url:
            raise RuntimeError(
                "Seed result judge is not configured; set SEED_JUDGE_API_KEY and SEED_JUDGE_BASE_URL"
            )
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": self._judge_prompt(trigger, executor_result, correction)}
        ]
        frames = evidence_frames if evidence_frames is not None else self._judge_evidence_frames(obs)
        for index, frame in enumerate(frames, start=1):
            step = frame.get("step")
            action = str(frame.get("action") or "")
            reason = str(frame.get("reason") or "")
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Chronological screenshot {index}/{len(frames)}"
                        + (f" · step {step}" if step is not None else "")
                        + (f" · selected for {reason}" if reason else "")
                        + (" (final state)" if index == len(frames) else "")
                        + (f"\nAction before screenshot: {action[:300]}" if action else "")
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{frame['image']}", "detail": "high"},
                }
            )
        payload: Dict[str, Any] = {
            "model": self.judge_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.judge_max_tokens,
            "stream": False,
        }
        if self.judge_temperature is not None:
            payload["temperature"] = float(self.judge_temperature)
        if self.judge_reasoning_effort:
            payload["reasoning_effort"] = self.judge_reasoning_effort
        started = time.monotonic()
        response, request_id = self._post_seed_request(
            role="judge",
            base_url=self.judge_base_url,
            api_key=self.judge_api_key,
            payload=payload,
            timeout=self.judge_timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Seed judge request_id={request_id} HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        value = response.json()
        message = (value.get("choices") or [{}])[0].get("message") or {}
        text = str(message.get("content") or "")
        usage = _seed_usage_from_response(value)
        return text, {
            **usage,
            "model_time": time.monotonic() - started,
            "request_id": request_id,
        }

    def _judge_result(
        self,
        trigger: str,
        obs: Mapping[str, Any],
        executor_result: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        usage_records: List[Dict[str, Any]] = []
        last_text = ""
        evidence_frames = self._judge_evidence_frames(obs)
        self._write_judge_evidence_manifest(trigger, evidence_frames)
        request_retries = max(0, int(getattr(self, "judge_request_retries", 2)))
        retry_backoff = max(0.0, float(getattr(self, "judge_retry_backoff", 2.0)))
        for attempt in range(1, self.judge_attempts + 1):
            correction = ""
            if attempt > 1:
                correction = (
                    "\n\nYour previous response was not a valid RESULT_JSON. Correct only its "
                    "format and completeness; still output exactly one fenced json object.\n"
                    "Previous invalid response:\n"
                    + last_text[:2000]
                )
            last_error: Optional[Exception] = None
            usage: Optional[Dict[str, Any]] = None
            for request_attempt in range(request_retries + 1):
                try:
                    last_text, usage = self._call_judge_once(
                        trigger,
                        obs,
                        executor_result,
                        correction=correction,
                        evidence_frames=evidence_frames,
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    self._log_request_event(
                        "seed_judge_retry_scheduled"
                        if request_attempt < request_retries
                        else "seed_judge_retries_exhausted",
                        judge_attempt=attempt,
                        request_attempt=request_attempt + 1,
                        max_request_attempts=request_retries + 1,
                        error=str(exc)[:2000],
                    )
                    if request_attempt < request_retries and retry_backoff > 0:
                        time.sleep(retry_backoff * (request_attempt + 1))
            if usage is None:
                raise SeedJudgeError(
                    f"Seed judge failed after {request_retries + 1} request attempt(s): {last_error}"
                ) from last_error
            usage_records.append(usage)
            parsed = _seed_result_from_text(last_text)
            normalized = _normalize_seed_result(parsed) if parsed else None
            if normalized:
                return normalized, usage_records
        raise RuntimeError(
            f"Seed judge produced no valid RESULT_JSON after {self.judge_attempts} attempt(s): "
            f"{last_text[:500]}"
        )

    def _predict_info(
        self,
        response: str,
        role: str,
        state_correct: bool,
        judge_usage_records: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        executor_usage = _seed_merge_usage(self._inference_usage)
        judge_usage = _seed_merge_usage(judge_usage_records or [])
        usage = {
            key: executor_usage[key] + judge_usage[key]
            for key in (
                "model_time",
                "prompt_tokens",
                "cached_input_tokens",
                "completion_tokens",
                "reasoning_output_tokens",
                "total_tokens",
                "model_calls",
            )
        }

        def role_usage(value: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
            return {
                "prompt_tokens": value["prompt_tokens"],
                "cached_input_tokens": value["cached_input_tokens"],
                "completion_tokens": value["completion_tokens"],
                "reasoning_output_tokens": value["reasoning_output_tokens"],
                "total_tokens": value["total_tokens"],
                "model_calls": value["model_calls"],
                "request_ids": [
                    str(item["request_id"])
                    for item in records
                    if item.get("request_id")
                ],
            }

        return {
            "model_usage": usage,
            "usage_breakdown": {
                "executor": role_usage(executor_usage, self._inference_usage),
                "judge": role_usage(judge_usage, judge_usage_records or []),
            },
            "response": response,
            "role": role,
            "state_correct": state_correct,
            "implementation": "reference_seed_agent_separate_judge",
            "models": {"executor": self.exec_model, "judge": self.judge_model},
        }

    def predict(
        self,
        instruction: str,
        obs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        self._inference_usage = []
        self._loop_previous_observation = obs
        if not getattr(self, "_judge_frames", []):
            self._remember_judge_frame(
                obs,
                step=0,
                action="case initial state",
                priority=100,
                reason="initial",
            )
        budget_exhausted = "## Step Budget Exhausted" in instruction
        if not self._base_instruction:
            self._base_instruction = instruction.split("\n\n## Step Budget Exhausted", 1)[0]
        if budget_exhausted:
            result, judge_usage = self._judge_result("max_steps_exhausted", obs)
            return self._predict_info(
                _render_seed_result(result),
                "seed_judge_budget_exhausted",
                True,
                judge_usage,
            ), []

        recovery_was_pending = bool(getattr(self, "_loop_recovery_pending", False))
        operator_instruction = (
            instruction
            + _SEED_GUIEVAL_SUFFIX
            + self._loop_recovery_directive()
        )
        prediction, raw_actions = self._predict_reference(operator_instruction, obs)
        self._record_execution(prediction, raw_actions)

        executor_result = self._result_from_prediction(prediction)
        terminal_function, terminal_content = self._terminal_call(prediction)
        terminal_token = next((token for token in raw_actions if token in {"DONE", "FAIL"}), "")
        terminal = bool(terminal_function or terminal_token)
        if executor_result or terminal:
            trigger = terminal_function or terminal_token or "executor_result"
            result, judge_usage = self._judge_result(trigger, obs, executor_result)
            return self._predict_info(
                _render_seed_result(result),
                f"seed_judge_after_{trigger.lower()}",
                True,
                judge_usage,
            ), []

        if recovery_was_pending and self._loop_candidate_repeats_ineffective(raw_actions):
            rejected = ", ".join(
                self._loop_action_label(action)
                for action in raw_actions
                if str(action or "").strip() not in {"", "DONE", "FAIL", "WAIT"}
            )
            self._execution_log.append(
                "loop guard: rejected an unexecuted repeated action after recovery "
                f"instruction; proposed_action={rejected[:500] or '(none)'}"
            )
            result, judge_usage = self._judge_result(
                "executor_loop_detected",
                obs,
            )
            return self._predict_info(
                _render_seed_result(result),
                "seed_judge_after_loop_detected",
                True,
                judge_usage,
            ), []

        if raw_actions == ["WAIT"]:
            actions = [
                {
                    "action_space": "pyautogui",
                    "action": "import time\ntime.sleep(1.0)",
                    "batch_last": True,
                }
            ]
            return self._predict_info(self._thought(prediction), "seed_operator", True), actions

        actions = self._wrap_actions(raw_actions)
        if actions:
            return self._predict_info(self._thought(prediction), "seed_operator", bool(actions)), actions

        result, judge_usage = self._judge_result(
            "executor_no_actions",
            obs,
            executor_result,
        )
        return self._predict_info(
            _render_seed_result(result),
            "seed_judge_after_no_actions",
            True,
            judge_usage,
        ), []

    def step(
        self,
        action: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], int, bool, Dict[str, Any], Dict[str, Any]]:
        if not action:
            raise RuntimeError("Empty Seed GUI action")
        code = action.get("action") if isinstance(action, Mapping) else action
        previous_obs = getattr(self, "_loop_previous_observation", None)
        started = time.monotonic()
        obs, reward, terminated, info = self.env.step(code, self.sleep_after_execution)
        loop_detection = self._update_loop_state(previous_obs, obs, code)
        self._loop_previous_observation = obs
        self._judge_action_count = int(getattr(self, "_judge_action_count", 0)) + 1
        priority, reason = self._action_frame_metadata(str(code or ""))
        self._remember_judge_frame(
            obs,
            step=self._judge_action_count,
            action=re.sub(r"\s+", " ", str(code or "")).strip()[:500],
            priority=priority,
            reason=reason,
        )
        return obs, reward, terminated, info, {
            "step_time": time.monotonic() - started,
            "action": action,
            "loop_detection": loop_detection,
        }


class GUIAgentAdapter:
    def __init__(
        self,
        env: OSWorldHTTPPyAutoGUIEnv,
        model: Optional[str] = None,
        max_steps: int = 60,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        model = model or (
            os.environ.get("GUIAGENT_MODEL")
            or os.environ.get("APPEVAL_LLM_MODEL")
            or os.environ.get("MODEL")
        )
        if not model:
            raise RuntimeError(
                "GUI agent model is not set; configure GUIAGENT_MODEL (or APPEVAL_LLM_MODEL/MODEL) in the repo-root .env"
            )
        self.env = env
        self.model = model
        self.max_steps = max_steps
        self.logger = logger or logging.getLogger("software_gym.gui_agent")
        agent_cls = load_gui_agent_class()
        self.agent = agent_cls(
            env=env,
            platform="ubuntu",
            model=model,
            action_space="pyautogui",
            observation_type="screenshot",
        )

    def run_case(self, case: Dict[str, Any], requirement_summary: str = "", trace_dir: Optional[Path] = None) -> CaseRunResult:
        if hasattr(self.agent, "set_request_context"):
            self.agent.set_request_context(str(case.get("id") or ""), trace_dir)
        self.agent.reset(self.logger)
        obs = self.env.reset()
        split_seed_prompts = hasattr(self.agent, "set_judge_instruction")
        instruction = build_case_instruction(
            case,
            requirement_summary,
            role="executor" if split_seed_prompts else "legacy",
        )
        judge_instruction = ""
        if split_seed_prompts:
            judge_instruction = build_case_instruction(case, requirement_summary, role="judge")
            self.agent.set_judge_instruction(judge_instruction)
        steps: List[Dict[str, Any]] = []
        last_response = ""
        steps_trace_path = trace_dir / "steps.jsonl" if trace_dir is not None else None
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / "case.json").write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (trace_dir / "instruction.md").write_text(instruction, encoding="utf-8")
            if split_seed_prompts:
                (trace_dir / "executor_instruction.md").write_text(instruction, encoding="utf-8")
                (trace_dir / "judge_instruction.md").write_text(judge_instruction, encoding="utf-8")
        initial_screenshot = self._save_screenshot(trace_dir, "initial.png", obs)
        if steps_trace_path is not None:
            append_jsonl(
                steps_trace_path,
                {
                    "event": "case_started",
                    "case_id": case.get("id"),
                    "title": case.get("title"),
                    "initial_screenshot": initial_screenshot,
                    "max_steps": self.max_steps,
                },
            )

        for step_index in range(1, self.max_steps + 1):
            try:
                predict_info, actions = self.agent.predict(instruction, obs)
            except InfrastructureError:
                raise
            except SeedJudgeError as exc:
                if steps_trace_path is not None:
                    append_jsonl(
                        steps_trace_path,
                        {
                            "event": "judge_failed",
                            "step": step_index,
                            "error": str(exc),
                            "status": "JudgeInfraError",
                            "blocker": "judge_request_failed",
                        },
                    )
                return self._case_result_from_seed_judge_error(case, steps, exc)
            except Exception as exc:
                if steps_trace_path is not None:
                    append_jsonl(
                        steps_trace_path,
                        {
                            "event": "predict_failed",
                            "step": step_index,
                            "error": str(exc),
                            "status": "InfraError",
                            "blocker": "gui_agent_predict_exception",
                        },
                    )
                if looks_like_infrastructure_error(exc):
                    raise InfrastructureError(str(exc)) from exc
                return CaseRunResult(
                    status="InfraError",
                    evidence=f"GUIAgent predict failed: {exc}",
                    checkpoints=[],
                    blocker="gui_agent_predict_exception",
                    repair_hint="Do not feed this as app bug; rerun after model/API/OSWorld recovery.",
                    steps=steps,
                    raw_result={
                        "status": "InfraError",
                        "evidence": str(exc),
                        "failure_analysis": "GUIAgent failed before producing actions.",
                        "blocker": "gui_agent_predict_exception",
                    },
                )
            last_response = predict_info.get("response", "")
            step_record: Dict[str, Any] = {
                "step": step_index,
                "predict_info": sanitize_for_json(predict_info),
                "actions": sanitize_for_json(actions),
                "action_results": [],
                "before_screenshot": initial_screenshot if step_index == 1 else None,
                "after_screenshots": [],
            }
            if steps_trace_path is not None:
                append_jsonl(
                    steps_trace_path,
                    {
                        "event": "step_predicted",
                        "step": step_index,
                        "predict_info": step_record["predict_info"],
                        "actions": step_record["actions"],
                        "has_result_json": parse_result_json(last_response) is not None,
                    },
                )
            result_json = parse_result_json(last_response)
            if result_json is not None:
                steps.append(step_record)
                if steps_trace_path is not None:
                    append_jsonl(
                        steps_trace_path,
                        {
                            "event": "case_result",
                            "step": step_index,
                            "status": result_json.get("status"),
                            "result": result_json,
                        },
                    )
                return self._case_result_from_json(result_json, steps)

            if not actions:
                steps.append(step_record)
                if steps_trace_path is not None:
                    append_jsonl(
                        steps_trace_path,
                        {
                            "event": "case_blocked",
                            "step": step_index,
                            "status": "Blocked",
                            "blocker": "no_actions",
                            "evidence": last_response or "GUIAgent produced no executable actions and no RESULT_JSON.",
                        },
                    )
                evidence = last_response or "GUIAgent produced no executable actions and no RESULT_JSON."
                raw_result = fallback_blocked_result(
                    case,
                    status="Blocked",
                    blocker="no_actions",
                    evidence=evidence,
                    steps=steps,
                    last_response=last_response,
                    repair_hint="Check whether the target UI is visible and whether the case prompt is feasible.",
                )
                return CaseRunResult(
                    status="Blocked",
                    evidence=evidence,
                    checkpoints=raw_result["checkpoints"],
                    blocker="no_actions",
                    repair_hint="Check whether the target UI is visible and whether the case prompt is feasible.",
                    steps=steps,
                    raw_result=raw_result,
                )

            for action_index, action in enumerate(actions, start=1):
                if steps_trace_path is not None:
                    append_jsonl(
                        steps_trace_path,
                        {
                            "event": "action_started",
                            "step": step_index,
                            "action_index": action_index,
                            "action": sanitize_for_json(action),
                        },
                    )
                try:
                    obs, reward, terminated, info, step_meta = self.agent.step(action)
                except InfrastructureError:
                    raise
                except Exception as exc:
                    if steps_trace_path is not None:
                        append_jsonl(
                            steps_trace_path,
                            {
                                "event": "action_failed",
                                "step": step_index,
                                "action_index": action_index,
                                "action": sanitize_for_json(action),
                                "error": str(exc),
                                "status": "InfraError",
                                "blocker": "gui_agent_step_exception",
                            },
                        )
                    if looks_like_infrastructure_error(exc):
                        raise InfrastructureError(str(exc)) from exc
                    step_record["action_results"].append(
                        {
                            "error": str(exc),
                            "step_meta": {},
                        }
                    )
                    steps.append(step_record)
                    return CaseRunResult(
                        status="InfraError",
                        evidence=f"GUIAgent step failed: {exc}",
                        checkpoints=[],
                        blocker="gui_agent_step_exception",
                        repair_hint="Do not feed this as app bug; rerun after Docker/OSWorld execute recovery.",
                        steps=steps,
                        raw_result={
                            "status": "InfraError",
                            "evidence": str(exc),
                            "failure_analysis": "GUIAgent action execution failed through OSWorld.",
                            "blocker": "gui_agent_step_exception",
                        },
                    )
                screenshot_path = self._save_screenshot(
                    trace_dir,
                    f"step_{step_index:03d}.png",
                    obs,
                )
                if hasattr(self.agent, "set_last_judge_frame_screenshot_path"):
                    self.agent.set_last_judge_frame_screenshot_path(screenshot_path)
                if screenshot_path:
                    step_record["after_screenshots"].append(screenshot_path)
                action_result = {
                    "reward": reward,
                    "terminated": terminated,
                    "info": sanitize_for_json(info),
                    "step_meta": sanitize_for_json(step_meta),
                }
                step_record["action_results"].append(action_result)
                if steps_trace_path is not None:
                    append_jsonl(
                        steps_trace_path,
                        {
                            "event": "action_completed",
                            "step": step_index,
                            "action_index": action_index,
                            "action": sanitize_for_json(action),
                            "screenshot": screenshot_path,
                            **action_result,
                        },
                    )
                if terminated:
                    break
            steps.append(step_record)
            if steps_trace_path is not None:
                append_jsonl(
                    steps_trace_path,
                    {
                        "event": "step_completed",
                        "step": step_index,
                        "after_screenshots": step_record["after_screenshots"],
                        "action_results": step_record["action_results"],
                    },
                )

        final_result_response = ""
        final_result_json: Optional[Dict[str, Any]] = None
        final_instruction = build_step_budget_exhausted_instruction(instruction, self.max_steps)
        try:
            final_predict_info, final_actions = self.agent.predict(final_instruction, obs)
            final_result_response = str(final_predict_info.get("response") or "")
            final_result_json = parse_result_json(final_result_response)
            if steps_trace_path is not None:
                append_jsonl(
                    steps_trace_path,
                    {
                        "event": "final_result_requested",
                        "status": final_result_json.get("status") if final_result_json else "no_result_json",
                        "predict_info": sanitize_for_json(final_predict_info),
                        "ignored_actions": sanitize_for_json(final_actions),
                    },
                )
        except SeedJudgeError as exc:
            if steps_trace_path is not None:
                append_jsonl(
                    steps_trace_path,
                    {
                        "event": "final_result_request_failed",
                        "error": str(exc),
                        "status": "JudgeInfraError",
                        "blocker": "judge_request_failed",
                    },
                )
            return self._case_result_from_seed_judge_error(case, steps, exc)
        except Exception as exc:
            if steps_trace_path is not None:
                append_jsonl(
                    steps_trace_path,
                    {
                        "event": "final_result_request_failed",
                        "error": str(exc),
                    },
                )

        if final_result_json is not None:
            return self._case_result_from_json(final_result_json, steps)

        if steps_trace_path is not None:
            append_jsonl(
                steps_trace_path,
                {
                    "event": "case_blocked",
                    "status": "Blocked",
                    "blocker": "max_steps",
                    "evidence": f"Reached max_steps={self.max_steps} without RESULT_JSON.",
                    "last_response": final_result_response or last_response,
                },
            )
        evidence = f"Reached max_steps={self.max_steps} without RESULT_JSON. Last response: {final_result_response or last_response}"
        raw_result = fallback_blocked_result(
            case,
            status="Blocked",
            blocker="max_steps",
            evidence=evidence,
            steps=steps,
            last_response=final_result_response or last_response,
            repair_hint="Increase max_steps only if the trace shows steady progress; otherwise simplify the case or fix the UI blocker.",
        )
        return CaseRunResult(
            status="Blocked",
            evidence=evidence,
            checkpoints=raw_result["checkpoints"],
            blocker="max_steps",
            repair_hint="Increase max_steps only if the trace shows steady progress; otherwise simplify the case or fix the UI blocker.",
            steps=steps,
            raw_result=raw_result,
        )

    @staticmethod
    def _save_screenshot(trace_dir: Optional[Path], name: str, obs: Dict[str, Any]) -> str:
        if trace_dir is None:
            return ""
        screenshot = obs.get("screenshot")
        if not isinstance(screenshot, (bytes, bytearray)):
            return ""
        screenshot_dir = trace_dir / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        path = screenshot_dir / name
        path.write_bytes(bytes(screenshot))
        return str(path.relative_to(trace_dir))

    @staticmethod
    def _case_result_from_json(result: Dict[str, Any], steps: List[Dict[str, Any]]) -> CaseRunResult:
        status = str(result.get("status", "Blocked"))
        checkpoints = result.get("checkpoints")
        return CaseRunResult(
            status=status,
            evidence=str(result.get("evidence", "")),
            checkpoints=checkpoints if isinstance(checkpoints, list) else [],
            blocker=str(result.get("blocker", "")),
            repair_hint=str(result.get("repair_hint", "")),
            steps=steps,
            raw_result=result,
        )

    @staticmethod
    def _case_result_from_seed_judge_error(
        case: Dict[str, Any],
        steps: List[Dict[str, Any]],
        exc: SeedJudgeError,
    ) -> CaseRunResult:
        raw_result = {
            "status": "JudgeInfraError",
            "evidence": str(exc),
            "task_summary": "GUI execution completed, but the result-only Judge was unavailable.",
            "task_completion": {
                "completed": False,
                "reason": "judge_infrastructure",
            },
            "step_results": fallback_step_results(steps),
            "checkpoints": fallback_checkpoints(
                case,
                "judge_request_failed",
                str(exc),
            ),
            "final_observation": "The final GUI screenshot is preserved in the trace.",
            "failure_analysis": (
                "The GUI trajectory is complete and must not be rerun. "
                "Retry only the Judge using judge_evidence.json."
            ),
            "blocker": "judge_request_failed",
            "repair_hint": "Retry the result-only Judge from the saved evidence bundle.",
        }
        return CaseRunResult(
            status="JudgeInfraError",
            evidence=str(exc),
            checkpoints=raw_result["checkpoints"],
            blocker="judge_request_failed",
            repair_hint=raw_result["repair_hint"],
            steps=steps,
            raw_result=raw_result,
        )


__all__ = [
    "CaseRunResult",
    "GUIAgentAdapter",
    "OSWorldHTTPPyAutoGUIEnv",
    "SeedJudgeError",
    "build_case_instruction",
    "parse_result_json",
]
