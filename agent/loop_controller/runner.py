from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agent.loop_controller.avalonia_blueprint import (
    blueprint_digest,
    report_matches_blueprint,
)
from agent.loop_controller import usage as usage_mod
from agent.loop_controller.manifest import update_manifest_variant
from agent.loop_controller.resume import check_round_ready
from agent.loop_controller.validators import validate_cases
from agent.runtime_env import (
    activate_softwaregym_env_vars,
    ensure_softwaregym_for_runner,
    env_for_osworld,
    env_for_softwaregym,
    load_env_file,
    softwaregym_python,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
load_env_file(REPO_ROOT / ".env")


# Build artifacts must never be carried into a repair-round baseline: the coder
# only edits source, and a single .NET `bin/` (≈230MB) or Electron `release/`
# (≈96MB) tree balloons the seeded app to hundreds of MB. Codex then injects that
# directory tree into context and overflows the model's hard limit (400 invalid
# request), which no retry/backoff can recover. Stripping them keeps the seed at
# the source level (~1MB) so repair rounds actually run.
BUILD_ARTIFACT_DIRS = (
    "bin",
    "obj",
    "release",
    "dist",
    "build",
    "out",
    "node_modules",
    "target",
    ".git",
    ".vite",
    ".cache",
    ".next",
    "coverage",
    "__pycache__",
)
_seed_ignore = shutil.ignore_patterns(*BUILD_ARTIFACT_DIRS)


class StageError(RuntimeError):
    """Raised when a pipeline stage cannot safely continue."""


@dataclass(frozen=True)
class Stage:
    name: str
    work_dir: Path
    command_script: Path
    outputs: List[Path]
    kind: str = "codex"
    version: str = ""
    style_id: str = ""
    # Repair round: copy the previous round's source and spec into this round's
    # Coder output as a baseline before running.
    seed_app_src: Optional[Path] = None
    seed_feature_src: Optional[Path] = None
    seed_app_dest: Optional[Path] = None
    seed_feature_dest: Optional[Path] = None
    # Pre-seeding a baseline would make outputs_ready falsely report "done", so
    # use a completion marker to decide skipping.
    done_marker: Optional[Path] = None


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise StageError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def substyle_validation_feedback(work_dir: Path) -> str:
    """Return concise, current platform evidence for a failed SubStyle gate."""
    work_dir = Path(work_dir)
    blueprint_dir = work_dir / "ui-blueprint"
    report_path = work_dir / "avalonia_blueprint_validation_report.json"
    if not report_path.is_file() or not blueprint_dir.is_dir():
        return ""
    try:
        report = load_json(report_path)
        current_digest = blueprint_digest(blueprint_dir)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, StageError):
        return ""
    if (
        report.get("ok") is True
        or report.get("source_digest") != current_digest
    ):
        return ""

    lines = [
        f"Authoritative report: {report_path}",
        f"Blueprint digest: {current_digest}",
    ]
    errors = report.get("errors")
    if isinstance(errors, list) and errors:
        lines.append("Static/contract errors:")
        lines.extend(f"- {error}" for error in errors)
    probe = report.get("compile_probe")
    if isinstance(probe, dict):
        lines.append(f"Compile probe status: {probe.get('status', 'unknown')}")
        output_tail = str(probe.get("output_tail") or "").strip()
        if output_tail:
            lines.extend(
                [
                    "Compiler output (verbatim):",
                    "```text",
                    output_tail[-12000:],
                    "```",
                ]
            )
        reason = str(probe.get("reason") or "").strip()
        if reason:
            lines.append(f"Compile probe reason: {reason}")
    return "\n".join(lines)[-16000:]


def state_schema_validation_error(value: Any) -> str:
    """Return why a state-schema document is unsafe to reuse, or an empty string."""
    if not isinstance(value, dict) or not value:
        return "document must be a non-empty JSON object"
    version = value.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        return "schema_version must be a positive integer"
    for section in ("dump_state", "config_yaml", "sqlite"):
        if not isinstance(value.get(section), dict):
            return f"{section} must be a JSON object"
    usable = value.get("usable")
    if not isinstance(usable, dict):
        return "usable must be a JSON object"
    for key in (
        "dump_state_fields",
        "config_seed_top_level_keys",
        "sqlite_tables",
    ):
        entries = usable.get(key)
        if not isinstance(entries, list) or not all(
            isinstance(entry, str) for entry in entries
        ):
            return f"usable.{key} must be a list of strings"
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        return f"document is not strict JSON-serializable: {exc}"
    return ""


def append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def parse_styles(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def style_dirs(run_dir: Path, selected: Optional[Iterable[str]] = None) -> List[Path]:
    allowed = set(selected or [])
    root = run_dir / "software_variants"
    variants = sorted(path for path in root.glob("style_*") if path.is_dir())
    if allowed:
        variants = [path for path in variants if path.name in allowed]
    if not variants:
        raise StageError(f"No variants found under {root}")
    return variants


def file_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def directory_has_files(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def directory_has_blueprint_xaml(path: Path) -> bool:
    """Historical runs may contain .xaml; new contracts require .axaml below."""
    return path.exists() and path.is_dir() and any(
        item.is_file()
        for pattern in ("*.axaml", "*.xaml")
        for item in path.rglob(pattern)
    )


def directory_has_axaml(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(
        item.is_file() for item in path.rglob("*.axaml")
    )


def design_route_ready(path: Path) -> bool:
    if not file_nonempty(path):
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return "Pending SubStyleDesigner generation" not in text


def contract_required(work_dir: Path, name: str, version: int = 1) -> bool:
    """Whether this run was scaffolded with a named pipeline contract."""
    manifest_path = work_dir.parents[1] / "manifest.json"
    if not file_nonempty(manifest_path):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    contracts = manifest.get("contracts") if isinstance(manifest, dict) else None
    return isinstance(contracts, dict) and contracts.get(name) == version


def icon_slots_required(work_dir: Path) -> bool:
    return contract_required(work_dir, "icon_slots")


def avalonia_blueprint_required(work_dir: Path) -> bool:
    return contract_required(work_dir, "avalonia_blueprint")


def icon_report_ok(path: Path) -> bool:
    if not file_nonempty(path):
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("ok"))


def outputs_ready(stage: Stage) -> bool:
    if stage.name == "FunctionDesigner":
        return file_nonempty(stage.outputs[0])
    if stage.name == "IconDesigner":
        return file_nonempty(stage.outputs[0])
    if stage.name == "SubIconDesigner":
        # The validator already enforces this in codex-command.sh and writes the
        # report; report ok means SVGs are complete and valid.
        return icon_report_ok(stage.outputs[0])
    if stage.name == "StyleDesigner":
        return all(file_nonempty(path) for path in stage.outputs)
    if stage.name == "SubStyleDesigner":
        blueprint_required = avalonia_blueprint_required(stage.work_dir)
        base_ready = design_route_ready(stage.outputs[0]) and (
            directory_has_axaml(stage.outputs[1])
            if blueprint_required
            else directory_has_blueprint_xaml(stage.outputs[1])
        )
        if not base_ready:
            return False
        if blueprint_required and (
            len(stage.outputs) < 5
            or not report_matches_blueprint(stage.outputs[4], stage.outputs[1])
        ):
            return False
        if not icon_slots_required(stage.work_dir):
            return True
        return (
            len(stage.outputs) >= 4
            and file_nonempty(stage.outputs[2])
            and icon_report_ok(stage.outputs[3])
        )
    if stage.name == "Coder":
        return directory_has_files(stage.outputs[0]) and file_nonempty(stage.outputs[1])
    if stage.name == "DevCoder":
        return directory_has_files(stage.outputs[0]) and file_nonempty(stage.outputs[1]) and file_nonempty(stage.outputs[2])
    if stage.name == "EvalTaskGenerateAgent":
        if not file_nonempty(stage.outputs[0]):
            return False
        return not validate_cases(stage.outputs[0])
    if stage.name == "GUIEval":
        return file_nonempty(stage.outputs[0])
    return all(path.exists() for path in stage.outputs)


def ensure_stage_outputs(stage: Stage) -> None:
    if outputs_ready(stage):
        return
    missing = ", ".join(str(path) for path in stage.outputs)
    raise StageError(f"{stage.name} did not produce required outputs: {missing}")


class PipelineRunner:
    def __init__(
        self,
        run_dir: Path,
        styles: Optional[List[str]] = None,
        max_rounds: Optional[int] = None,
        dry_run: bool = False,
        eval_dry_run: bool = False,
        no_start_container: bool = False,
        case_attempts: int = 1,
        case_trajectory_reruns: int = 1,
        infra_case_retries: int = 3,
        max_steps: int = 50,
        codex_stage_retries: int = 2,
        codex_stage_timeout_seconds: int = 7200,
        coder_stage_timeout_seconds: int = 10800,
        codex_retry_backoff_base: float = 10.0,
        codex_retry_backoff_max: float = 120.0,
        coder_workers: int = 4,
        rerun_existing: bool = False,
        stop_on_pass: bool = False,
        parallel_styles: bool = False,
        parallel_workers: int = 5,
        substyle_workers: Optional[int] = None,
        stop_before_coder: bool = False,
        eval_window: int = 1,
        style_diversity_enabled: bool = True,
        style_diversity_tau: float = 0.60,
        style_diversity_max_rerolls: int = 3,
        style_diversity_mode: str = "corpus",
        visual_fidelity_threshold: float = 0.80,
        launch_smoke_enabled: bool = True,
        launch_smoke_max_repairs: int = 5,
        launch_smoke_infra_retries: int = 2,
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.manifest = load_json(self.run_dir / "manifest.json")
        self.software_type = str(self.manifest.get("software_type") or self.run_dir.name.split("_")[0])
        self.styles = styles
        self.max_rounds = int(max_rounds or self.manifest.get("max_rounds") or 3)
        self.dry_run = dry_run
        self.eval_dry_run = eval_dry_run
        self.no_start_container = no_start_container
        self.case_attempts = case_attempts
        self.case_trajectory_reruns = case_trajectory_reruns
        self.infra_case_retries = infra_case_retries
        self.max_steps = max_steps
        self.codex_stage_retries = max(0, codex_stage_retries)
        self.codex_stage_timeout_seconds = max(0, codex_stage_timeout_seconds)
        self.coder_stage_timeout_seconds = max(0, coder_stage_timeout_seconds)
        self.codex_retry_backoff_base = max(0.0, codex_retry_backoff_base)
        self.codex_retry_backoff_max = max(self.codex_retry_backoff_base, codex_retry_backoff_max)
        self.coder_workers = max(1, coder_workers)
        self._coder_pool = threading.BoundedSemaphore(self.coder_workers)
        self.rerun_existing = rerun_existing
        self.stop_on_pass = stop_on_pass
        self.parallel_styles = parallel_styles
        self.parallel_workers = max(1, parallel_workers)
        # Dedicated SubStyleDesigner parallelism (decoupled from coder/variant
        # parallelism). Used to fan out style generation quickly.
        self.substyle_workers = max(1, substyle_workers if substyle_workers is not None else parallel_workers)
        # design-only: stop after SubIconDesigner (before Coder) for offline
        # style-control/asset validation.
        self.stop_before_coder = stop_before_coder
        self.eval_window = max(1, eval_window)
        self.style_diversity_enabled = style_diversity_enabled
        self.style_diversity_tau = float(style_diversity_tau)
        self.style_diversity_max_rerolls = max(0, int(style_diversity_max_rerolls))
        self.style_diversity_mode = style_diversity_mode
        self.visual_fidelity_threshold = float(visual_fidelity_threshold)
        self.launch_smoke_enabled = launch_smoke_enabled
        self.launch_smoke_max_repairs = max(0, int(launch_smoke_max_repairs))
        self.launch_smoke_infra_retries = max(0, int(launch_smoke_infra_retries))
        self.events_path = self.run_dir / "runner-events.jsonl"
        self._event_lock = threading.Lock()

    def log_event(self, event: str, **details: Any) -> None:
        payload = {
            "event": event,
            "time": datetime.now().isoformat(timespec="seconds"),
            **details,
        }
        with self._event_lock:
            append_jsonl(self.events_path, payload)

    def codex_stage(
        self,
        name: str,
        work_dir: Path,
        outputs: List[Path],
        version: str = "",
        style_id: str = "",
        seed_app_src: Optional[Path] = None,
        seed_feature_src: Optional[Path] = None,
        seed_app_dest: Optional[Path] = None,
        seed_feature_dest: Optional[Path] = None,
        done_marker: Optional[Path] = None,
    ) -> Stage:
        return Stage(
            name=name,
            work_dir=work_dir,
            command_script=work_dir / "codex-command.sh",
            outputs=outputs,
            version=version,
            style_id=style_id,
            seed_app_src=seed_app_src,
            seed_feature_src=seed_feature_src,
            seed_app_dest=seed_app_dest,
            seed_feature_dest=seed_feature_dest,
            done_marker=done_marker,
        )

    def run_codex_stage(self, stage: Stage) -> None:
        if not stage.command_script.exists():
            raise StageError(f"{stage.name} command script is missing: {stage.command_script}")
        # For repair rounds with a pre-seeded baseline, use the completion marker
        # to decide "done" and avoid skipping due to pre-seeded code.
        already_done = stage.done_marker.exists() if stage.done_marker is not None else outputs_ready(stage)
        if already_done and not self.rerun_existing:
            self.log_event("stage_skipped_existing", stage=stage.name, work_dir=str(stage.work_dir))
            print(f"[skip] {stage.name}: outputs already exist")
            return
        if self.dry_run:
            self.log_event("stage_planned", stage=stage.name, work_dir=str(stage.work_dir), outputs=[str(path) for path in stage.outputs])
            print(f"[plan] {stage.name}: {stage.command_script}")
            return

        self._seed_repair_base(stage)

        if stage.name in {"Coder", "DevCoder"}:
            self.log_event(
                "coder_pool_wait",
                stage=stage.name,
                style_id=stage.style_id,
                version=stage.version,
                workers=self.coder_workers,
                work_dir=str(stage.work_dir),
            )
            self._coder_pool.acquire()
            self.log_event(
                "coder_pool_acquired",
                stage=stage.name,
                style_id=stage.style_id,
                version=stage.version,
                workers=self.coder_workers,
                work_dir=str(stage.work_dir),
            )
            try:
                return self._run_codex_stage_attempts(stage)
            finally:
                self._coder_pool.release()
                self.log_event(
                    "coder_pool_released",
                    stage=stage.name,
                    style_id=stage.style_id,
                    version=stage.version,
                    workers=self.coder_workers,
                    work_dir=str(stage.work_dir),
                )
        self._run_codex_stage_attempts(stage)

    def _seed_repair_base(self, stage: Stage) -> None:
        """Before a repair round runs, copy the previous round's source and spec into this round's Coder output as the incremental repair baseline.

        The baseline is refreshed every run (delete then copy) so reruns start
        clean; DevCoder is not pre-seeded here because it edits coder/app in place.
        """
        if stage.seed_app_src is None or stage.seed_app_dest is None:
            return
        src_app = stage.seed_app_src
        dest_app = stage.seed_app_dest
        if not src_app.exists():
            self.log_event(
                "repair_seed_missing",
                stage=stage.name,
                style_id=stage.style_id,
                version=stage.version,
                seed_app_src=str(src_app),
            )
            return
        if dest_app.exists():
            shutil.rmtree(dest_app)
        dest_app.parent.mkdir(parents=True, exist_ok=True)
        # Exclude build artifacts so the seeded baseline stays at source size and
        # codex does not overflow its context window on repair rounds.
        shutil.copytree(src_app, dest_app, ignore=_seed_ignore)
        if stage.seed_feature_src is not None and stage.seed_feature_dest is not None and stage.seed_feature_src.exists():
            stage.seed_feature_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(stage.seed_feature_src, stage.seed_feature_dest)
        self.log_event(
            "repair_seed_app",
            stage=stage.name,
            style_id=stage.style_id,
            version=stage.version,
            seed_app_src=str(src_app),
            seed_app_dest=str(dest_app),
        )

    def _capture_session_id(self, stage: Stage) -> None:
        """Extract `session id: <uuid>` from this stage's own codex.log and write it to work_dir/session_id.txt.

        This lets the same role resume with `codex exec resume` in the next round.
        Parsing the process log is naturally concurrency-safe; if extraction fails,
        no file is written and the next round falls back to a fresh session.
        """
        if stage.name not in {"Coder", "DevCoder", "SubStyleDesigner"}:
            return
        log_path = stage.work_dir / "codex.log"
        if not log_path.exists():
            return
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        match = re.search(r"session id:\s*([0-9a-fA-F-]{32,36})", text)
        if not match:
            self.log_event("session_id_capture_missing", stage=stage.name, style_id=stage.style_id, version=stage.version)
            return
        session_id = match.group(1).strip()
        (stage.work_dir / "session_id.txt").write_text(session_id + "\n", encoding="utf-8")
        self.log_event(
            "session_id_captured",
            stage=stage.name,
            style_id=stage.style_id,
            version=stage.version,
            session_id=session_id,
        )

    def _prepare_substyle_validation_retry(
        self,
        stage: Stage,
        attempt_index: int,
    ) -> bool:
        if stage.name != "SubStyleDesigner":
            return False
        feedback = substyle_validation_feedback(stage.work_dir)
        if not feedback:
            return False
        self._capture_session_id(stage)
        resume_file = stage.work_dir / "session_id.txt"
        if not file_nonempty(resume_file):
            self.log_event(
                "substyle_validation_repair_unavailable",
                stage=stage.name,
                style_id=stage.style_id,
                attempt=attempt_index,
                reason="failed attempt session id was not captured",
            )
            return False
        try:
            from agent.loop_controller.cli import LoopController

            LoopController().render_substyle_validation_repair(
                self.run_dir,
                stage.style_id or stage.work_dir.name,
                validation_feedback=feedback,
                resume_session_file=str(resume_file),
            )
        except Exception as exc:
            self.log_event(
                "substyle_validation_repair_error",
                stage=stage.name,
                style_id=stage.style_id,
                attempt=attempt_index,
                error=str(exc),
            )
            return False
        self.log_event(
            "substyle_validation_repair_prepared",
            stage=stage.name,
            style_id=stage.style_id,
            attempt=attempt_index,
            resume_session_file=str(resume_file),
            feedback_chars=len(feedback),
        )
        return True

    def _record_codex_stage_usage(
        self,
        stage: Stage,
        attempt_index: int,
        started_at: datetime,
        monotonic_start: float,
    ) -> None:
        """After success, parse this stage's codex.log, write usage.json (with duration), and emit a stage_usage event.

        Usage-statistics failures must never affect the main flow.
        """
        try:
            log_path = stage.work_dir / "codex.log"
            record = usage_mod.reconstruct_codex_record(log_path, self.run_dir)
            if record is None:
                record = {"stage": stage.name, "kind": "codex", "style_id": stage.style_id or None, "version": stage.version or None}
            record["attempts"] = attempt_index
            record["started_at"] = started_at.isoformat(timespec="seconds")
            record["ended_at"] = datetime.now().isoformat(timespec="seconds")
            record["duration_seconds"] = round(time.monotonic() - monotonic_start, 1)
            write_json(stage.work_dir / "usage.json", record)
            self.log_event(
                "stage_usage",
                stage=stage.name,
                style_id=stage.style_id,
                version=stage.version,
                model=record.get("model"),
                tokens=record.get("tokens"),
                session_cumulative_tokens=record.get("session_cumulative_tokens"),
                resumed=record.get("resumed"),
                duration_seconds=record.get("duration_seconds"),
            )
        except Exception as exc:
            self.log_event("stage_usage_error", stage=stage.name, style_id=stage.style_id, version=stage.version, error=str(exc))

    def _record_eval_stage_usage(
        self,
        eval_dir: Path,
        style_id: str,
        version: str,
        started_at: datetime,
        monotonic_start: float,
    ) -> None:
        """After GUIEval success: read GUI agent tokens from eval/cost.json, write eval/usage.json (with duration), and emit an event."""
        try:
            record: Dict[str, Any] = {
                "stage": "GUIEval",
                "kind": "gui_eval",
                "style_id": style_id,
                "version": version,
            }
            cost_path = eval_dir / "cost.json"
            if cost_path.exists():
                cost = load_json(cost_path)
                gui = (cost.get("components") or {}).get("gui_agent") or {}
                record["model"] = cost.get("model")
                record["tokens"] = int(
                    cost["effective_tokens"]
                    if "effective_tokens" in cost
                    else cost.get("total_tokens_without_retries", cost.get("total_tokens", 0))
                )
                record["tokens_without_retries"] = record["tokens"]
                record["tokens_with_retries"] = int(
                    cost["effective_tokens_with_retries"]
                    if "effective_tokens_with_retries" in cost
                    else cost.get("total_tokens", 0)
                )
                record["retry_overhead"] = max(
                    0,
                    record["tokens_with_retries"] - record["tokens"],
                )
                record["token_breakdown"] = {
                    "input_tokens": int(gui.get("input_tokens") or 0),
                    "cached_input_tokens": int(gui.get("cached_input_tokens") or 0),
                    "output_tokens": int(gui.get("output_tokens") or 0),
                    "reasoning_output_tokens": int(gui.get("reasoning_output_tokens") or 0),
                    "total_tokens": int(
                        gui.get("raw_total_tokens", gui.get("total_tokens", 0)) or 0
                    ),
                    "scope": "stage",
                }
                record["by_model"] = cost.get("by_model") or {}
            record["started_at"] = started_at.isoformat(timespec="seconds")
            record["ended_at"] = datetime.now().isoformat(timespec="seconds")
            record["duration_seconds"] = round(time.monotonic() - monotonic_start, 1)
            write_json(eval_dir / "usage.json", record)
            self.log_event(
                "stage_usage",
                stage="GUIEval",
                style_id=style_id,
                version=version,
                model=record.get("model"),
                tokens=record.get("tokens"),
                duration_seconds=record.get("duration_seconds"),
            )
        except Exception as exc:
            self.log_event("stage_usage_error", stage="GUIEval", style_id=style_id, version=version, error=str(exc))

    @staticmethod
    def _attempt_was_rate_limited(attempt_log_path: Path) -> bool:
        try:
            tail = attempt_log_path.read_text(encoding="utf-8", errors="ignore")[-8000:]
        except OSError:
            return False
        lowered = tail.lower()
        return (
            "429" in tail
            or "too many requests" in lowered
            or "rate limit" in lowered
            or "rate_limit" in lowered
        )

    def _retry_backoff_seconds(self, attempt_index: int, attempt_log_path: Path) -> tuple[float, bool]:
        """Exponential backoff with jitter; longer waits when the failure was a 429.

        Without a wait between attempts a rate-limit burst makes every retry fail
        instantly (exceeded retry limit) and crashes the whole runner. Backing off
        lets the rate-limit window clear before the next attempt.
        """
        rate_limited = self._attempt_was_rate_limited(attempt_log_path)
        base = self.codex_retry_backoff_base if rate_limited else 2.0
        if base <= 0:
            return 0.0, rate_limited
        delay = min(self.codex_retry_backoff_max, base * (2 ** (attempt_index - 1)))
        delay += random.uniform(0.0, delay * 0.25)  # jitter to de-synchronize parallel stages
        return delay, rate_limited

    def _sleep_before_retry(self, stage: Stage, attempt_index: int, attempt_log_path: Path) -> None:
        delay, rate_limited = self._retry_backoff_seconds(attempt_index, attempt_log_path)
        self.log_event(
            "stage_retry_backoff",
            stage=stage.name,
            attempt=attempt_index,
            backoff_seconds=round(delay, 1),
            rate_limited=rate_limited,
        )
        if delay > 0:
            time.sleep(delay)

    def _run_codex_stage_attempts(self, stage: Stage) -> None:
        max_attempts = self.codex_stage_retries + 1
        # Recovery runs must append their evidence rather than truncate the
        # original failed attempts.  Keep the retry budget local to this call
        # while assigning monotonically increasing on-disk attempt numbers.
        prior_attempts = []
        for path in stage.work_dir.glob("codex.attempt_*.log"):
            match = re.fullmatch(r"codex\.attempt_(\d+)\.log", path.name)
            if match:
                prior_attempts.append(int(match.group(1)))
        first_attempt_index = max(prior_attempts, default=0) + 1
        timeout_seconds = (
            self.coder_stage_timeout_seconds
            if stage.name in {"Coder", "DevCoder"}
            else self.codex_stage_timeout_seconds
        )
        timeout = timeout_seconds or None
        self.log_event(
            "stage_started",
            stage=stage.name,
            work_dir=str(stage.work_dir),
            max_attempts=max_attempts,
            timeout_seconds=timeout,
        )
        print(f"[run] {stage.name}: {stage.work_dir}")
        env = env_for_softwaregym(os.environ.copy())
        log_path = stage.work_dir / "codex.log"
        last_error = ""
        stage_started_at = datetime.now()
        stage_monotonic_start = time.monotonic()
        for local_attempt_index in range(1, max_attempts + 1):
            attempt_index = first_attempt_index + local_attempt_index - 1
            attempt_log_path = stage.work_dir / f"codex.attempt_{attempt_index:02d}.log"
            self.log_event(
                "stage_attempt_started",
                stage=stage.name,
                work_dir=str(stage.work_dir),
                attempt=attempt_index,
                max_attempts=max_attempts,
            )
            timed_out = False
            returncode = 0
            with attempt_log_path.open("w", encoding="utf-8") as log:
                try:
                    result = subprocess.run(
                        ["bash", str(stage.command_script)],
                        cwd=REPO_ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                        timeout=timeout,
                    )
                    returncode = result.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
                    returncode = 124
                    log.write(f"\n[TIMEOUT] {stage.name} exceeded {timeout} seconds on attempt {attempt_index}.\n")
            shutil.copyfile(attempt_log_path, log_path)
            if timed_out or returncode != 0:
                last_error = f"{stage.name} {'timed out' if timed_out else f'failed with exit code {returncode}'}"
                self.log_event(
                    "stage_attempt_failed",
                    stage=stage.name,
                    attempt=attempt_index,
                    returncode=returncode,
                    timed_out=timed_out,
                    log_path=str(attempt_log_path),
                )
                if local_attempt_index < max_attempts:
                    self._prepare_substyle_validation_retry(
                        stage,
                        attempt_index,
                    )
                    self.log_event(
                        "stage_retrying",
                        stage=stage.name,
                        next_attempt=attempt_index + 1,
                        max_attempts=max_attempts,
                    )
                    self._sleep_before_retry(stage, local_attempt_index, attempt_log_path)
                    continue
                self.log_event("stage_failed", stage=stage.name, returncode=returncode, log_path=str(log_path), attempts=max_attempts)
                raise StageError(f"{last_error}; see {log_path}")
            try:
                ensure_stage_outputs(stage)
            except StageError as exc:
                last_error = str(exc)
                self.log_event(
                    "stage_attempt_failed",
                    stage=stage.name,
                    attempt=attempt_index,
                    returncode=returncode,
                    output_check_failed=True,
                    error=last_error,
                    log_path=str(attempt_log_path),
                )
                if local_attempt_index < max_attempts:
                    self._prepare_substyle_validation_retry(
                        stage,
                        attempt_index,
                    )
                    self.log_event(
                        "stage_retrying",
                        stage=stage.name,
                        next_attempt=attempt_index + 1,
                        max_attempts=max_attempts,
                    )
                    self._sleep_before_retry(stage, local_attempt_index, attempt_log_path)
                    continue
                self.log_event("stage_failed", stage=stage.name, returncode=returncode, log_path=str(log_path), attempts=max_attempts)
                raise
            self._capture_session_id(stage)
            self._record_codex_stage_usage(stage, attempt_index, stage_started_at, stage_monotonic_start)
            if stage.done_marker is not None:
                stage.done_marker.parent.mkdir(parents=True, exist_ok=True)
                stage.done_marker.write_text(
                    json.dumps(
                        {"stage": stage.name, "version": stage.version, "style_id": stage.style_id, "attempt": attempt_index},
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            self.log_event(
                "stage_completed",
                stage=stage.name,
                outputs=[str(path) for path in stage.outputs],
                log_path=str(log_path),
                attempt=attempt_index,
            )
            return
        raise StageError(f"{stage.name} failed after {max_attempts} attempts: {last_error}; see {log_path}")

    def run_eval_stage(self, variant_dir: Path, version: str, style_id: str) -> Path:
        summary_path = variant_dir / "loop" / version / "eval" / "summary.json"
        stage = Stage(
            name="GUIEval",
            work_dir=variant_dir / "loop" / version / "eval",
            command_script=Path(""),
            outputs=[summary_path],
            kind="eval",
            version=version,
            style_id=style_id,
        )
        if outputs_ready(stage) and not self.rerun_existing:
            self.log_event("stage_skipped_existing", stage=stage.name, style_id=style_id, version=version)
            print(f"[skip] GUIEval {style_id} {version}: summary already exists")
            return summary_path
        if self.dry_run:
            self.log_event("stage_planned", stage=stage.name, style_id=style_id, version=version, output=str(summary_path))
            print(f"[plan] GUIEval {style_id} {version}: {summary_path}")
            return summary_path

        command = [
            softwaregym_python(),
            "-m",
            "agent.eval.run_eval",
            "--variant-dir",
            str(variant_dir),
            "--version",
            version,
            "--software-type",
            self.software_type,
            "--style-id",
            style_id,
            "--case-attempts",
            str(self.case_attempts),
            "--case-trajectory-reruns",
            str(self.case_trajectory_reruns),
            "--infra-case-retries",
            str(self.infra_case_retries),
            "--max-steps",
            str(self.max_steps),
            "--eval-window",
            str(self.eval_window),
        ]
        if self.eval_dry_run:
            command.append("--dry-run")
        if self.no_start_container:
            command.append("--no-start-container")

        log_path = stage.work_dir / "eval-runner.log"
        stage.work_dir.mkdir(parents=True, exist_ok=True)
        self.log_event("stage_started", stage=stage.name, style_id=style_id, version=version, command=command)
        print(f"[run] GUIEval {style_id} {version}")
        eval_started_at = datetime.now()
        eval_monotonic_start = time.monotonic()
        env = env_for_osworld(os.environ.copy())
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        if result.returncode != 0:
            self.log_event("stage_failed", stage=stage.name, style_id=style_id, version=version, returncode=result.returncode, log_path=str(log_path))
            raise StageError(f"GUIEval {style_id} {version} failed with exit code {result.returncode}; see {log_path}")
        ensure_stage_outputs(stage)
        self._record_eval_stage_usage(stage.work_dir, style_id, version, eval_started_at, eval_monotonic_start)
        self._record_visual_fidelity(variant_dir, version, style_id)
        self.log_event("stage_completed", stage=stage.name, style_id=style_id, version=version, output=str(summary_path), log_path=str(log_path))
        return summary_path

    def _record_visual_fidelity(self, variant_dir: Path, version: str, style_id: str) -> None:
        """After GUIEval finishes: compute visual fidelity for this round's artifact and write eval/visual_fidelity.json + .md.

        This is the "visual summary" feedback signal fed into the next Coder/DevCoder
        repair round. It is non-blocking; failures do not affect the main flow.
        """
        try:
            from agent.metrics import visual_fidelity as vf

            app_dir = variant_dir / "loop" / version / "coder" / "app"
            eval_dir = variant_dir / "loop" / version / "eval"
            row = vf.score_single(
                self.run_dir, style_id, app_dir, version,
                threshold=self.visual_fidelity_threshold,
            )
            write_json(eval_dir / "visual_fidelity.json", row)
            (eval_dir / "visual_fidelity.md").write_text(vf.render_feedback_md(row), encoding="utf-8")
            self.log_event(
                "stage_visual_fidelity", stage="GUIEval", style_id=style_id, version=version,
                score=row.get("score"), passed=row.get("pass"), flags=row.get("flags"),
            )
        except Exception as exc:  # noqa: BLE001
            self.log_event("stage_visual_fidelity_error", stage="GUIEval",
                           style_id=style_id, version=version, error=str(exc))

    def _gate_launch_before_eval(self, variant_dir: Path, version: str, style_id: str, round_dir: Path) -> bool:
        """Launch smoke gate between DevCoder and eval. Returns True to allow entry into eval; returns False when the repair budget is exhausted or the environment keeps failing, in which case the caller should skip this round's eval and proceed to the next round.

        - Pass: allow entry directly.
        - InfraError: environmental noise, not DevCoder's fault; retry the container
          up to a bound (does not consume the repair budget); if it still fails,
          give up and deny entry.
        - AppLaunchError: a real launch bug; feed evidence back to DevCoder for
          in-place repair, then re-run the smoke gate until fixed or the budget is
          exhausted.
        """
        infra_retries = 0
        for repair_attempt in range(self.launch_smoke_max_repairs + 1):
            gate = self._run_launch_smoke_gate(variant_dir, version, style_id)
            status = gate.get("status")
            self.log_event(
                "launch_smoke_result",
                style_id=style_id,
                version=version,
                attempt=repair_attempt,
                status=status,
                app_crash=gate.get("app_crash"),
            )
            if status == "Pass":
                return True
            if status == "InfraError":
                if infra_retries < self.launch_smoke_infra_retries:
                    infra_retries += 1
                    continue
                self.log_event("launch_smoke_infra_giveup", style_id=style_id, version=version)
                return False
            # AppLaunchError means the app failed to launch, which is a real bug.
            if repair_attempt >= self.launch_smoke_max_repairs:
                self.log_event(
                    "launch_smoke_budget_exhausted",
                    style_id=style_id,
                    version=version,
                    max_repairs=self.launch_smoke_max_repairs,
                )
                return False
            self.log_event(
                "launch_smoke_repair_started",
                style_id=style_id,
                version=version,
                next_attempt=repair_attempt + 1,
            )
            self._rerun_devcoder_for_launch(variant_dir, version, style_id, round_dir, gate, repair_attempt + 1)
        return False

    def _run_launch_smoke_gate(self, variant_dir: Path, version: str, style_id: str) -> Dict[str, Any]:
        """Run run_eval --smoke-only in a subprocess (same structure as run_eval_stage, using the OSWorld environment).

        The loop is driven entirely by the status field in launch_smoke/smoke.json.
        Gate failures are expected branches; the subprocess's non-zero exit code must
        never crash the pipeline.
        """
        smoke_dir = variant_dir / "loop" / version / "launch_smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        command = [
            softwaregym_python(),
            "-m",
            "agent.eval.run_eval",
            "--variant-dir",
            str(variant_dir),
            "--version",
            version,
            "--software-type",
            self.software_type,
            "--style-id",
            style_id,
            "--smoke-only",
        ]
        if self.no_start_container:
            command.append("--no-start-container")
        log_path = smoke_dir / "smoke-runner.log"
        env = env_for_osworld(os.environ.copy())
        gate_monotonic_start = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        gate_seconds = round(time.monotonic() - gate_monotonic_start, 1)
        # The smoke gate consumes no tokens (no GUIAgent/codex), so it is not
        # included in cost aggregation. Record container/deployment/probe timing
        # separately in launch_smoke/timing.json for observability (wall_clock
        # already includes it, but the per-stage view would otherwise miss it).
        self._record_launch_smoke_timing(smoke_dir, gate_seconds)
        smoke_json = smoke_dir / "smoke.json"
        if smoke_json.exists():
            try:
                return load_json(smoke_json)
            except Exception as exc:  # noqa: BLE001
                return {"status": "InfraError", "evidence": f"unreadable smoke.json: {exc}"}
        return {"status": "InfraError", "evidence": f"launch smoke produced no smoke.json; see {log_path}"}

    @staticmethod
    def _format_launch_feedback(gate: Dict[str, Any]) -> str:
        """Format evidence from smoke.json into compact feedback text for DevCoder (bounded truncation)."""
        lines: List[str] = []
        evidence = str(gate.get("evidence") or "").strip()
        if evidence:
            lines.append(f"- readiness/evidence: {evidence[:1500]}")
        stderr_tail = str(gate.get("app_stderr_tail") or "").strip()
        if stderr_tail:
            lines.append("- app stderr (tail):\n```\n" + stderr_tail[-2500:] + "\n```")
        runtime_errors = gate.get("runtime_errors") or []
        for rec in runtime_errors[:5]:
            if not isinstance(rec, dict):
                continue
            detail = rec.get("details") if isinstance(rec.get("details"), dict) else rec
            etype = str(detail.get("exceptionType") or detail.get("type") or "")[:200]
            msg = str(detail.get("message") or rec.get("message") or "")[:400]
            stack = str(detail.get("stackTrace") or detail.get("stack") or "")[:1500]
            lines.append(f"- unhandled_exception: {etype}: {msg}")
            if stack:
                lines.append("```\n" + stack + "\n```")
        hint = str(gate.get("repair_hint") or "").strip()
        if hint:
            lines.append(f"- repair_hint: {hint}")
        return "\n".join(lines) if lines else "- (no structured evidence captured; the window never mapped and no app log was found)"

    def _rerun_devcoder_for_launch(
        self,
        variant_dir: Path,
        version: str,
        style_id: str,
        round_dir: Path,
        gate: Dict[str, Any],
        attempt: int,
    ) -> None:
        """Re-run DevCoder with launch-failure evidence (resume this round's DevCoder session and repair coder/app in place)."""
        from agent.loop_controller.cli import LoopController

        devcoder_dir = round_dir / "devcoder"
        coder_dir = round_dir / "coder"
        coder_app_dir = coder_dir / "app"
        icons_variant_dir = variant_dir / "04_icons"
        style_manifest_path = icons_variant_dir / "icon_manifest.style.json"
        icon_usage_report_path = devcoder_dir / "icon_usage_report.json"
        smoke_report_path = variant_dir / "loop" / version / "launch_smoke" / "smoke.json"
        feedback = self._format_launch_feedback(gate)

        controller = LoopController(execute=False)
        session_file = devcoder_dir / "session_id.txt"
        resume_file = str(session_file) if session_file.exists() else None
        controller.render_devcoder_launch_repair(
            devcoder_dir=devcoder_dir,
            coder_app_dir=coder_app_dir,
            style_manifest_path=style_manifest_path,
            icon_usage_report_path=icon_usage_report_path,
            variables={
                "version": version,
                "software_type": self.software_type,
                "style_id": style_id,
                "coder_output_dir": coder_dir,
                "app_output_dir": coder_app_dir,
                "review_output_path": devcoder_dir / "review.md",
                "feature_matrix_output_path": devcoder_dir / "feature_matrix.md",
                "launch_smoke_report_path": smoke_report_path,
                "launch_repair_feedback": feedback,
                "attempt": attempt,
                "max_attempts": self.launch_smoke_max_repairs,
            },
            resume_session_file=resume_file,
        )
        # Token/cost accounting needs no special handling: DevCoder resumes the
        # same session, and the tail of codex.log is the session's cumulative
        # effective tokens. usage.py computes cross-version deltas, so all
        # launch-repair tokens in this round are automatically attributed to this
        # version without duplication or omission.
        # However, usage.json duration is overwritten by each repair, keeping only
        # the last one. Accumulate it back here so DevCoder time is not
        # under-reported.
        usage_path = devcoder_dir / "usage.json"
        prior_duration = 0.0
        prior_started: Optional[str] = None
        prior_attempts = 0
        prior_repairs = 0
        prior_mtime = usage_path.stat().st_mtime if usage_path.exists() else None
        if usage_path.exists():
            try:
                prior = load_json(usage_path)
                prior_duration = float(prior.get("duration_seconds") or 0.0)
                prior_started = prior.get("started_at")
                prior_attempts = int(prior.get("attempts") or 0)
                prior_repairs = int(prior.get("launch_repairs") or 0)
            except Exception:  # noqa: BLE001
                pass

        stage = self.codex_stage(
            "DevCoder",
            devcoder_dir,
            [coder_app_dir, devcoder_dir / "feature_matrix.md", devcoder_dir / "review.md"],
            version=version,
            style_id=style_id,
            done_marker=devcoder_dir / f".launch_repair_r{attempt}.json",
        )
        self.run_codex_stage(stage)

        # Only accumulate when usage.json was actually rewritten by this repair
        # (mtime changed, i.e. the stage really ran rather than being skipped), to
        # avoid double counting.
        try:
            new_mtime = usage_path.stat().st_mtime if usage_path.exists() else None
            if new_mtime is not None and new_mtime != prior_mtime:
                cur = load_json(usage_path)
                this_duration = float(cur.get("duration_seconds") or 0.0)
                cur["duration_seconds"] = round(prior_duration + this_duration, 1)
                if prior_started:
                    cur["started_at"] = prior_started  # keep the first DevCoder start time
                cur["attempts"] = prior_attempts + int(cur.get("attempts") or 1)
                cur["launch_repairs"] = prior_repairs + 1
                cur["launch_repair_seconds"] = round(
                    float(cur.get("launch_repair_seconds") or 0.0) + this_duration, 1
                )
                write_json(usage_path, cur)
        except Exception as exc:  # noqa: BLE001
            self.log_event("launch_repair_usage_error", style_id=style_id, version=version, error=str(exc))

    def _record_launch_smoke_timing(self, smoke_dir: Path, gate_seconds: float) -> None:
        """Accumulate the duration of each smoke-gate attempt into launch_smoke/timing.json (duration only, no tokens). Failures do not affect the main flow."""
        try:
            timing_path = smoke_dir / "timing.json"
            timing: Dict[str, Any] = {"attempts": [], "total_seconds": 0.0}
            if timing_path.exists():
                try:
                    loaded = load_json(timing_path)
                    if isinstance(loaded.get("attempts"), list):
                        timing = loaded
                except Exception:  # noqa: BLE001
                    pass
            timing["attempts"].append({"seconds": gate_seconds, "at": datetime.now().isoformat(timespec="seconds")})
            timing["total_seconds"] = round(sum(float(a.get("seconds") or 0.0) for a in timing["attempts"]), 1)
            timing["attempt_count"] = len(timing["attempts"])
            write_json(timing_path, timing)
        except Exception as exc:  # noqa: BLE001
            self.log_event("launch_smoke_timing_error", error=str(exc))

    def _write_launch_gate_summary(self, variant_dir: Path, version: str, style_id: str) -> None:
        """When the smoke gate fails and this round's eval is skipped, write a synthetic eval/summary.json.

        This satisfies the next round's check_round_ready (which requires a previous
        summary.json) and feeds the launch-failure evidence into the next
        Coder/DevCoder round.
        """
        eval_dir = variant_dir / "loop" / version / "eval"
        smoke_path = variant_dir / "loop" / version / "launch_smoke" / "smoke.json"
        smoke: Dict[str, Any] = {}
        if smoke_path.exists():
            try:
                smoke = load_json(smoke_path)
            except Exception:  # noqa: BLE001
                smoke = {}
        runtime_errors = smoke.get("runtime_errors") or []
        evidence = str(smoke.get("evidence") or "app failed the launch smoke gate")
        case_result = {
            "case_id": "launch_smoke_gate",
            "status": "AppLaunchError",
            "app_crash": bool(runtime_errors),
            "evidence": evidence,
            "task_summary": (
                "App launch failure: the app did NOT start in the OSWorld container even though the "
                "container/display were proven healthy. GUIEval was skipped for this round after "
                f"{self.launch_smoke_max_repairs} DevCoder launch-repair attempts. This is a REAL app bug "
                "(broken release/run.sh entry, missing runtime dependency, startup crash, or a window that "
                "never maps), not infra noise."
            ),
            "task_completion": {"completed": False, "reason": "app_launch_failed"},
            "step_results": [],
            "checkpoints": [],
            "final_observation": "",
            "failure_analysis": evidence,
            "blocker": "app_launch",
            "repair_hint": str(smoke.get("repair_hint") or ""),
            "runtime_errors": runtime_errors,
        }
        signatures: List[Dict[str, str]] = []
        seen: set = set()
        for rec in runtime_errors:
            if not isinstance(rec, dict):
                continue
            detail = rec.get("details") if isinstance(rec.get("details"), dict) else rec
            sig = {
                "exceptionType": str(detail.get("exceptionType") or detail.get("type") or "")[:200],
                "message": str(detail.get("message") or rec.get("message") or "")[:400],
                "source": str(detail.get("source") or "")[:160],
            }
            key = (sig["exceptionType"], sig["message"])
            if (sig["exceptionType"] or sig["message"]) and key not in seen:
                seen.add(key)
                signatures.append(sig)
        zero_rate = {"total": 0, "passed": 0, "failed_or_blocked": 0, "percent": 0.0}
        summary = {
            "version": version,
            "style_id": style_id,
            "variant_dir": str(variant_dir),
            "overall_status": "Fail",
            "gated_by": "launch_smoke",
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "cases_total": 0,
            "cases_pass": 0,
            "cases_fail": 0,
            "cases_blocked": 0,
            "cases_infra_error": 0,
            "cases_app_launch_error": 1,
            "cases_with_runtime_error": 1 if runtime_errors else 0,
            "cases_app_crash": 1 if runtime_errors else 0,
            "app_crash_signatures": signatures[:20],
            "task_level": dict(zero_rate),
            "checkpoint_level": dict(zero_rate),
            "step_level": dict(zero_rate),
            "case_results": [case_result],
            "failures": [case_result],
            "infra_errors": [],
            "note": f"GUIEval skipped: app failed the launch smoke gate; see loop/{version}/launch_smoke/smoke.json",
        }
        write_json(eval_dir / "summary.json", summary)
        self.log_event("launch_gate_summary_written", style_id=style_id, version=version, path=str(eval_dir / "summary.json"))

    def run_design_stages(self, variants: List[Path]) -> None:
        style_ids = [variant.name for variant in variants]
        function_stage = self.codex_stage(
            "FunctionDesigner",
            self.run_dir / "01_function_design",
            [self.run_dir / "01_function_design" / "requirement.md"],
        )
        icon_stage = self.codex_stage(
            "IconDesigner",
            self.run_dir / "02_icon_design",
            [self.run_dir / "02_icon_design" / "icon_manifest.json"],
        )
        style_stage = self.codex_stage(
            "StyleDesigner",
            self.run_dir / "02_style_design",
            [self.run_dir / "02_style_design" / "styles.json"]
            + [self.run_dir / "02_style_design" / f"{style_id}.md" for style_id in style_ids],
        )
        self.run_codex_stage(function_stage)
        # IconDesigner only depends on requirement.md and must produce the shared
        # semantic icon superset before SubIconDesigner.
        self.run_codex_stage(icon_stage)
        self.run_codex_stage(style_stage)
        substyle_stages = []
        for variant_dir in variants:
            style_id = variant_dir.name
            substyle_dir = self.run_dir / "03_substyle_design" / style_id
            substyle_stages.append(
                self.codex_stage(
                    "SubStyleDesigner",
                    substyle_dir,
                    [
                        substyle_dir / "design_route.md",
                        substyle_dir / "ui-blueprint",
                        substyle_dir / "icon_slots.json",
                        substyle_dir / "icon_slots_validation_report.json",
                        substyle_dir / "avalonia_blueprint_validation_report.json",
                    ],
                    style_id=style_id,
                )
            )
        # SubStyleDesigner uses its dedicated substyle_workers parallelism,
        # decoupled from --parallel-styles/coder.
        if self.substyle_workers > 1 and len(substyle_stages) > 1:
            self.run_codex_stages_parallel(
                substyle_stages, stage_group="SubStyleDesigner", workers=self.substyle_workers
            )
        else:
            for stage in substyle_stages:
                self.run_codex_stage(stage)

        # 1) Structural style-diversity gate (main): may force too-close
        #    SubStyleDesigners to diversify by layout_archetype.
        self._enforce_style_diversity(variants)

        # 2) Project the shared candidate superset into a per-style authoritative
        #    list based on the final blueprint's icon_slots.
        self._project_style_icon_manifests(variants)

        # 3) SubIconDesigner (kangj): depends on each style's ui-blueprint
        #    (SubStyleDesigner output) + style manifest, and must finish before
        #    Coder. Placed after the diversity gate so icons are generated from the
        #    final post-reroll blueprint.
        subicon_stages = []
        for variant_dir in variants:
            style_id = variant_dir.name
            icons_variant_dir = variant_dir / "04_icons"
            subicon_stages.append(
                self.codex_stage(
                    "SubIconDesigner",
                    icons_variant_dir,
                    [icons_variant_dir / "icon_validation_report.json"],
                    style_id=style_id,
                )
            )
        if self.substyle_workers > 1 and len(subicon_stages) > 1:
            self.run_codex_stages_parallel(
                subicon_stages, stage_group="SubIconDesigner", workers=self.substyle_workers
            )
        else:
            for stage in subicon_stages:
                self.run_codex_stage(stage)

    def _project_style_icon_manifests(self, variants: List[Path]) -> None:
        """Project GUI-selected icon slots before SubIconDesigner renders SVGs.

        Runs scaffolded before the icon-slot contract existed keep the historical
        full-superset projection. New runs fail early on missing or inconsistent
        slots so Coder never has to invent GUI placements after the fact.
        """
        if self.dry_run:
            self.log_event(
                "style_icon_manifest_projection_planned",
                styles=[variant.name for variant in variants],
            )
            return

        from agent.icons.validate_style_icons import (
            IconSlotError,
            project_style_manifest_file,
        )

        superset_path = self.run_dir / "02_icon_design" / "icon_manifest.json"
        require_slots = (
            isinstance(self.manifest.get("contracts"), dict)
            and self.manifest["contracts"].get("icon_slots") == 1
        )
        for variant_dir in variants:
            style_id = variant_dir.name
            icon_slots_path = self.run_dir / "03_substyle_design" / style_id / "icon_slots.json"
            projected_path = variant_dir / "04_icons" / "icon_manifest.style.json"
            try:
                projected = project_style_manifest_file(
                    superset_path=superset_path,
                    style_id=style_id,
                    projected_path=projected_path,
                    icon_slots_path=icon_slots_path,
                    require_icon_slots=require_slots,
                )
            except (OSError, json.JSONDecodeError, IconSlotError) as exc:
                raise StageError(f"Invalid icon slots for {style_id}: {exc}") from exc
            self.log_event(
                "style_icon_manifest_projected",
                style_id=style_id,
                source="icon_slots" if projected.get("slot_contract_version") == 1 else "legacy_superset",
                candidate_icon_count=projected.get("candidate_icon_count", len(projected.get("icons") or [])),
                selected_icon_count=len(projected.get("icons") or []),
                path=str(projected_path),
            )

    def _enforce_style_diversity(self, variants: List[Path]) -> None:
        """Measure structural heterogeneity of the generated blueprints and, if necessary, force too-close SubStyleDesigners to diversify by layout_archetype (reusing the repair template + session resume), with bounded retries.

        A 03_substyle_design/style_diversity.json telemetry file is always written
        once, even when rerolling is disabled.
        """
        if self.dry_run or len(variants) < 2:
            return
        try:
            from agent.metrics import style_diversity as sd
            from agent.loop_controller.cli import LoopController
        except Exception as exc:  # Metric unavailability must never break the main flow.
            self.log_event("style_diversity_unavailable", error=str(exc))
            return

        controller = LoopController(execute=False)
        max_attempts = self.style_diversity_max_rerolls if self.style_diversity_enabled else 0
        for attempt in range(max_attempts + 1):
            try:
                report = sd.run_gate(
                    self.run_dir, tau=self.style_diversity_tau, mode=self.style_diversity_mode
                )
            except Exception as exc:
                self.log_event("style_diversity_error", attempt=attempt, error=str(exc))
                return
            struct = report.get("structural", {})
            stats = struct.get("stats", {}) if isinstance(struct, dict) else {}
            declared = report.get("declared_archetypes", {})
            self.log_event(
                "style_diversity_measured",
                attempt=attempt,
                mode=self.style_diversity_mode,
                tau=self.style_diversity_tau,
                min_pairwise=stats.get("min_pairwise"),
                mean_pairwise=stats.get("mean_pairwise"),
                closest_pair=struct.get("closest_pair") if isinstance(struct, dict) else None,
                declared_min_hamming=declared.get("min_pairwise_hamming") if isinstance(declared, dict) else None,
                declared_passes=declared.get("passes") if isinstance(declared, dict) else None,
                reroll_targets=report.get("reroll_targets"),
            )
            targets = report.get("reroll_targets") or []
            if report.get("passes") or not targets or attempt >= max_attempts:
                if targets and attempt >= max_attempts:
                    self.log_event("style_diversity_budget_exhausted", remaining_targets=targets)
                return
            self.log_event("style_diversity_reroll_started", attempt=attempt + 1, targets=targets)
            for style_id in targets:
                substyle_dir = self.run_dir / "03_substyle_design" / style_id
                feedback = report.get("diversify_feedback", {}).get(style_id, "")
                session_file = substyle_dir / "session_id.txt"
                resume_file = str(session_file) if session_file.exists() else None
                controller.render_substyle_diversify(
                    self.run_dir, style_id, diversify_feedback=feedback, resume_session_file=resume_file
                )
                stage = self.codex_stage(
                    "SubStyleDesigner",
                    substyle_dir,
                    [
                        substyle_dir / "design_route.md",
                        substyle_dir / "ui-blueprint",
                        substyle_dir / "icon_slots.json",
                        substyle_dir / "icon_slots_validation_report.json",
                        substyle_dir / "avalonia_blueprint_validation_report.json",
                    ],
                    style_id=style_id,
                    done_marker=substyle_dir / f".diversify_r{attempt + 1}.json",
                )
                self.run_codex_stage(stage)

    def run_codex_stages_parallel(self, stages: List[Stage], stage_group: str, workers: Optional[int] = None) -> None:
        workers = min(workers if workers is not None else self.parallel_workers, len(stages))
        self.log_event("parallel_group_started", group=stage_group, workers=workers, stages=[str(stage.work_dir) for stage in stages])
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.run_codex_stage, stage): stage for stage in stages}
            for future in as_completed(futures):
                stage = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    self.log_event("parallel_stage_failed", group=stage_group, stage=stage.name, style_id=stage.style_id, error=str(exc))
                    raise
        self.log_event("parallel_group_completed", group=stage_group)

    def _write_state_schema_safe(self, variant_dir: Path, version: str, eval_task_dir: Path, style_id: str) -> None:
        """Probe the built app's real --dump-state / config.yaml / SQLite schema.

        Writes ``05_eval_tasks/state_schema.json`` which EvalTaskGenerate reads to author
        contract assertions against fields that truly exist. A whole-schema generation,
        serialization, or persistence failure is fatal so task generation never runs
        without this contract. Individual probes may still report ``ok: false``.
        """
        if self.dry_run:
            return
        schema_path = eval_task_dir / "state_schema.json"
        if schema_path.exists():
            try:
                existing = load_json(schema_path)
                invalid_reason = state_schema_validation_error(existing)
            except Exception as exc:  # noqa: BLE001
                invalid_reason = f"{type(exc).__name__}: {exc}"
            if not invalid_reason:
                self.log_event(
                    "state_schema_reused",
                    style_id=style_id,
                    version=version,
                    path=str(schema_path),
                )
                return
            self.log_event(
                "state_schema_invalid",
                style_id=style_id,
                version=version,
                path=str(schema_path),
                reason=invalid_reason,
            )
        app_dir = variant_dir / "loop" / version / "coder" / "app"
        try:
            if not (app_dir / "release" / "run.sh").exists():
                raise StageError(
                    f"cannot generate state schema: missing "
                    f"{app_dir / 'release' / 'run.sh'}"
                )
            from agent.eval.state_probe import build_state_schema

            schema = build_state_schema(app_dir, timeout=90)
            generated_error = state_schema_validation_error(schema)
            if generated_error:
                raise StageError(
                    f"generated state schema is invalid: {generated_error}"
                )
            eval_task_dir.mkdir(parents=True, exist_ok=True)
            write_json(schema_path, schema)
            persisted = load_json(schema_path)
            persisted_error = state_schema_validation_error(persisted)
            if persisted_error:
                raise StageError(
                    f"persisted state schema is invalid: {persisted_error}"
                )
            usable = schema.get("usable", {})
            self.log_event(
                "state_schema_probed",
                style_id=style_id,
                version=version,
                dump_state_ok=schema.get("dump_state", {}).get("ok"),
                dump_fields=len(usable.get("dump_state_fields", [])),
                config_keys=len(usable.get("config_seed_top_level_keys", [])),
                sqlite_tables=len(usable.get("sqlite_tables", [])),
            )
        except Exception as exc:  # noqa: BLE001
            self.log_event(
                "state_schema_probe_error",
                style_id=style_id,
                version=version,
                path=str(schema_path),
                error=f"{type(exc).__name__}: {exc}",
            )
            if isinstance(exc, StageError):
                raise
            raise StageError(
                f"state schema generation failed for {style_id} {version}: {exc}"
            ) from exc

    def _prune_cases_against_schema_safe(self, eval_task_dir: Path, style_id: str) -> None:
        """Drop infra_checks that reference fields/tables absent from the real snapshot."""
        if self.dry_run:
            return
        cases_path = eval_task_dir / "cases.json"
        schema_path = eval_task_dir / "state_schema.json"
        if not cases_path.exists() or not schema_path.exists():
            return
        try:
            from agent.eval.validate_cases import validate_cases as prune_cases_against_schema

            report = prune_cases_against_schema(cases_path, schema_path)
            if report.get("skipped"):
                self.log_event("eval_cases_prune_skipped", style_id=style_id, reason=report["skipped"])
            elif report.get("dropped"):
                write_json(eval_task_dir / "schema_validation.json", report)
                self.log_event(
                    "eval_cases_infra_pruned",
                    style_id=style_id,
                    dropped=len(report["dropped"]),
                    details=report["dropped"][:20],
                )
        except Exception as exc:  # noqa: BLE001
            self.log_event("eval_cases_validate_error", style_id=style_id, error=str(exc))

    def run_variant_rounds(self, variant_dir: Path) -> None:
        style_id = variant_dir.name
        if not self.dry_run:
            update_manifest_variant(self.run_dir, style_id, status="running", current_stage="runner")
        eval_task_done = False
        for round_index in range(1, self.max_rounds + 1):
            version = f"v{round_index}"
            if round_index > 1 and not self.dry_run:
                errors = check_round_ready(variant_dir, version)
                if errors:
                    raise StageError("; ".join(errors))

            round_dir = variant_dir / "loop" / version
            coder_dir = round_dir / "coder"
            devcoder_dir = round_dir / "devcoder"
            coder_seed_kwargs: Dict[str, Any] = {}
            if round_index > 1:
                # Repair round: Coder resumes its own previous session, and the
                # on-disk baseline also uses the previous round's coder/app
                # (consistent with session memory).
                prev_coder = variant_dir / "loop" / f"v{round_index - 1}" / "coder"
                coder_seed_kwargs = {
                    "seed_app_src": prev_coder / "app",
                    "seed_feature_src": prev_coder / "feature_matrix.md",
                    "seed_app_dest": coder_dir / "app",
                    "seed_feature_dest": coder_dir / "feature_matrix.md",
                    "done_marker": coder_dir / ".coder_complete.json",
                }
            self.run_codex_stage(
                self.codex_stage(
                    "Coder",
                    coder_dir,
                    [coder_dir / "app", coder_dir / "feature_matrix.md"],
                    version=version,
                    style_id=style_id,
                    **coder_seed_kwargs,
                )
            )
            self.run_codex_stage(
                self.codex_stage(
                    "DevCoder",
                    devcoder_dir,
                    [coder_dir / "app", devcoder_dir / "feature_matrix.md", devcoder_dir / "review.md"],
                    version=version,
                    style_id=style_id,
                )
            )

            if not self.dry_run and self.launch_smoke_enabled:
                if not self._gate_launch_before_eval(variant_dir, version, style_id, round_dir):
                    self._write_launch_gate_summary(variant_dir, version, style_id)
                    self.log_event("round_skipped_launch_gate", style_id=style_id, version=version)
                    continue

            if not eval_task_done:
                eval_task_dir = variant_dir / "05_eval_tasks"
                # Probe the just-built app's REAL runtime-state schema so EvalTaskGenerate
                # authors config_seed/infra_checks against fields that actually exist.
                self._write_state_schema_safe(variant_dir, version, eval_task_dir, style_id)
                self.run_codex_stage(
                    self.codex_stage(
                        "EvalTaskGenerateAgent",
                        eval_task_dir,
                        [eval_task_dir / "cases.json"],
                        style_id=style_id,
                    )
                )
                # Prune any infra_checks that reference fields/tables absent from the real
                # snapshot (safety net against false-negative contract downgrades).
                self._prune_cases_against_schema_safe(eval_task_dir, style_id)
                eval_task_done = True

            summary_path = self.run_eval_stage(variant_dir, version, style_id)
            if self.stop_on_pass and summary_path.exists():
                summary = load_json(summary_path)
                if summary.get("overall_status") == "Pass":
                    self.log_event("variant_stopped_on_pass", style_id=style_id, version=version)
                    break
        if not self.dry_run:
            update_manifest_variant(self.run_dir, style_id, status="completed", current_stage="runner")
            self._build_style_table_safe(style_id)

    def _build_style_table_safe(self, style_id: str) -> None:
        """After a single software synthesis completes, generate per-stage usage statistics JSON (time / per-model tokens / rounds) for this style and write it to software_variants/<style>/cost_by_stage.json. Failures must never break the main flow."""
        try:
            table = usage_mod.build_software_table_for_style(self.run_dir, style_id, write=True)
            totals = table.get("totals", {})
            self.log_event(
                "style_cost_table_written",
                style_id=style_id,
                path=str(self.run_dir / "software_variants" / style_id / "cost_by_stage.json"),
                effective_tokens=totals.get("effective_tokens"),
                duration_seconds=totals.get("duration_seconds"),
            )
        except Exception as exc:
            self.log_event("style_cost_table_error", style_id=style_id, error=str(exc))

    def run(self) -> Path:
        variants = style_dirs(self.run_dir, self.styles)
        plan = {
            "run_dir": str(self.run_dir),
            "software_type": self.software_type,
            "styles": [variant.name for variant in variants],
            "max_rounds": self.max_rounds,
            "dry_run": self.dry_run,
            "eval_dry_run": self.eval_dry_run,
            "parallel_styles": self.parallel_styles,
            "parallel_workers": self.parallel_workers,
            "coder_workers": self.coder_workers,
            "sequence": [
                "FunctionDesigner",
                "IconDesigner (shared semantic icon superset)",
                "StyleDesigner",
                "SubStyleDesigner[*] in parallel when --parallel-styles is set",
                "SubIconDesigner[*] in parallel when --parallel-styles is set (per-style SVG icons + validation)",
                "for each style in parallel when --parallel-styles is set: (Coder/DevCoder through shared coder pool -> EvalTaskGenerateAgent once -> GUIEval) x rounds",
            ],
        }
        write_json(self.run_dir / "runner-plan.json", plan)
        self.log_event("runner_started", **plan)
        try:
            self.run_design_stages(variants)
            if self.stop_before_coder:
                self.log_event("stop_before_coder", run_dir=str(self.run_dir), styles=[p.name for p in variants])
                self._build_cost_summary_safe()
                self._prune_anonymous_volumes_safe()
                self.log_event("runner_completed_design_only", run_dir=str(self.run_dir))
                return self.run_dir / "runner-plan.json"
            if self.parallel_styles and len(variants) > 1:
                workers = min(self.parallel_workers, len(variants))
                self.log_event("parallel_group_started", group="variant_rounds", workers=workers, styles=[path.name for path in variants])
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(self.run_variant_rounds, variant_dir): variant_dir for variant_dir in variants}
                    for future in as_completed(futures):
                        variant_dir = futures[future]
                        try:
                            future.result()
                        except Exception as exc:
                            self.log_event("parallel_variant_failed", style_id=variant_dir.name, error=str(exc))
                            raise
                self.log_event("parallel_group_completed", group="variant_rounds")
            else:
                for variant_dir in variants:
                    self.run_variant_rounds(variant_dir)
        except Exception as exc:
            self.log_event("runner_failed", error=str(exc))
            self._build_cost_summary_safe()
            self._prune_anonymous_volumes_safe()
            raise
        self._build_cost_summary_safe()
        self._prune_anonymous_volumes_safe()
        self.log_event("runner_completed", run_dir=str(self.run_dir))
        return self.run_dir / "runner-plan.json"

    def _prune_anonymous_volumes_safe(self) -> None:
        """Cleanup: reclaim OSWorld image leftovers in the form of 64-hex anonymous /storage volumes (~17GB each).

        Only dangling anonymous volumes are removed; named volumes (other project
        data) are never touched. Failures do not affect the main flow.
        """
        try:
            from agent.tools.cleanup import prune_anonymous_volumes

            removed = prune_anonymous_volumes()
            if removed:
                self.log_event("anonymous_volumes_pruned", count=len(removed))
        except Exception as exc:
            self.log_event("anonymous_volume_prune_error", error=str(exc))

    def _build_cost_summary_safe(self) -> None:
        """Cleanup: rebuild run-level token/time usage summary into cost_summary.json. Failures do not affect the main flow."""
        try:
            summary = usage_mod.build_cost_summary(self.run_dir, write=True)
            totals = summary.get("totals", {})
            self.log_event(
                "cost_summary_written",
                path=str(self.run_dir / "cost_summary.json"),
                tokens=totals.get("tokens"),
                codex_tokens=totals.get("codex_tokens"),
                gui_eval_tokens=totals.get("gui_eval_tokens"),
                wall_clock_seconds=totals.get("wall_clock_seconds"),
            )
        except Exception as exc:
            self.log_event("cost_summary_error", error=str(exc))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run software-gym stages in dependency order.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--styles", default=None, help="Comma-separated style ids, e.g. style_01,style_03.")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only write a plan and print stages; do not call Codex or eval.")
    parser.add_argument("--eval-dry-run", action="store_true", help="Run Codex stages normally but call GUI eval in dry-run mode.")
    parser.add_argument("--no-start-container", action="store_true", help="Pass --no-start-container to GUI eval.")
    parser.add_argument("--case-attempts", type=int, default=int(os.environ.get("APPEVAL_CASE_ATTEMPTS", "1")))
    parser.add_argument("--case-trajectory-reruns", type=int, default=int(os.environ.get("APPEVAL_CASE_TRAJECTORY_RERUNS", "1")))
    parser.add_argument("--infra-case-retries", type=int, default=int(os.environ.get("APPEVAL_INFRA_CASE_RETRIES", "3")))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("APPEVAL_MAX_STEPS", "50")))
    parser.add_argument("--codex-stage-retries", type=int, default=int(os.environ.get("SOFTWAREGYM_CODEX_STAGE_RETRIES", "2")))
    parser.add_argument(
        "--codex-stage-timeout-seconds",
        type=int,
        default=int(os.environ.get("SOFTWAREGYM_CODEX_STAGE_TIMEOUT_SECONDS", "7200")),
    )
    parser.add_argument(
        "--coder-stage-timeout-seconds",
        type=int,
        default=int(os.environ.get("SOFTWAREGYM_CODER_STAGE_TIMEOUT_SECONDS", "10800")),
        help="Timeout for Coder and DevCoder stages only. Other Codex stages use --codex-stage-timeout-seconds.",
    )
    parser.add_argument(
        "--codex-retry-backoff-base",
        type=float,
        default=float(os.environ.get("SOFTWAREGYM_CODEX_RETRY_BACKOFF_BASE", "10")),
        help="Base seconds for exponential backoff between codex stage retries (longer for 429/rate-limit failures).",
    )
    parser.add_argument(
        "--codex-retry-backoff-max",
        type=float,
        default=float(os.environ.get("SOFTWAREGYM_CODEX_RETRY_BACKOFF_MAX", "120")),
        help="Maximum seconds for the exponential backoff between codex stage retries.",
    )
    parser.add_argument("--rerun-existing", action="store_true", help="Rerun stages even if their expected outputs already exist.")
    parser.add_argument("--stop-on-pass", action="store_true", help="Stop a variant early when a round's GUI eval passes.")
    parser.add_argument("--parallel-styles", action="store_true", help="Run SubStyleDesigner and per-style implementation/eval chains in parallel.")
    parser.add_argument("--parallel-workers", type=int, default=int(os.environ.get("SOFTWAREGYM_PARALLEL_WORKERS", "5")))
    parser.add_argument(
        "--substyle-workers",
        type=int,
        default=int(os.environ.get("SOFTWAREGYM_SUBSTYLE_WORKERS", "5")),
        help="Dedicated parallelism for the SubStyleDesigner (and SubIconDesigner) stage, decoupled from --parallel-styles/coder. Default 5.",
    )
    parser.add_argument(
        "--stop-before-coder",
        action="store_true",
        help="Design-only: run through SubIconDesigner (function/icon/style/substyle/diversity/subicon) then stop before Coder. For offline style-control/asset validation.",
    )
    parser.add_argument("--coder-workers", type=int, default=int(os.environ.get("SOFTWAREGYM_CODER_WORKERS", "4")))
    parser.add_argument(
        "--eval-window",
        type=int,
        default=int(os.environ.get("APPEVAL_EVAL_WINDOW", "1")),
        help="Containers per style for GUIEval (sliding window). Each style fans its cases across this many OSWorld containers.",
    )
    parser.add_argument(
        "--no-style-diversity-gate",
        action="store_true",
        help="Disable the post-SubStyleDesigner structural diversity re-roll loop (still writes a telemetry report).",
    )
    parser.add_argument(
        "--style-diversity-tau",
        type=float,
        default=float(os.environ.get("SOFTWAREGYM_STYLE_DIVERSITY_TAU", "0.60")),
        help="Min normalized tree-edit distance between any two styles; closer pairs trigger a diversify re-roll. Default 0.60 sits at the lower edge of the healthy blueprint distribution (real good-run min ~0.60-0.82), so marginal/collapsed runs get caught while clearly-diverse runs pass.",
    )
    parser.add_argument(
        "--style-diversity-max-rerolls",
        type=int,
        default=int(os.environ.get("SOFTWAREGYM_STYLE_DIVERSITY_MAX_REROLLS", "3")),
        help="Max diversify re-roll attempts for too-similar styles (default 3). Each attempt re-rolls only the styles still below tau, then re-measures; budget exhaustion logs and proceeds (soft gate).",
    )
    parser.add_argument(
        "--style-diversity-mode",
        choices=["corpus", "main"],
        default=os.environ.get("SOFTWAREGYM_STYLE_DIVERSITY_MODE", "corpus"),
        help="Region-skeleton scope used for the diversity metric.",
    )
    parser.add_argument(
        "--visual-fidelity-threshold",
        type=float,
        default=float(os.environ.get("SOFTWAREGYM_VISUAL_FIDELITY_THRESHOLD", "0.80")),
        help="Pass threshold for the per-round visual-fidelity feedback report (built AppStyle tokens vs blueprint). Default 0.80. This is a feedback signal fed into the next repair round, not a blocking gate.",
    )
    parser.add_argument(
        "--no-launch-smoke-gate",
        action="store_true",
        help="Disable the DevCoder->eval launch smoke gate (deploy + no-arg start the app in OSWorld before eval).",
    )
    parser.add_argument(
        "--launch-smoke-max-repairs",
        type=int,
        default=int(os.environ.get("SOFTWAREGYM_LAUNCH_SMOKE_MAX_REPAIRS", "5")),
        help="Max DevCoder launch-repair re-runs when the app fails the launch smoke gate (per round). Default 5. On budget exhaustion the round skips GUIEval and moves on.",
    )
    parser.add_argument(
        "--launch-smoke-infra-retries",
        type=int,
        default=int(os.environ.get("SOFTWAREGYM_LAUNCH_SMOKE_INFRA_RETRIES", "2")),
        help="Max container retries when the launch smoke gate hits an InfraError (does NOT consume the DevCoder repair budget).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    ensure_softwaregym_for_runner()
    activate_softwaregym_env_vars()
    load_env_file(REPO_ROOT / ".env")
    args = build_parser().parse_args(argv)
    runner = PipelineRunner(
        run_dir=args.run_dir,
        styles=parse_styles(args.styles),
        max_rounds=args.max_rounds,
        dry_run=args.dry_run,
        eval_dry_run=args.eval_dry_run,
        no_start_container=args.no_start_container,
        case_attempts=args.case_attempts,
        case_trajectory_reruns=args.case_trajectory_reruns,
        infra_case_retries=args.infra_case_retries,
        max_steps=args.max_steps,
        codex_stage_retries=args.codex_stage_retries,
        codex_stage_timeout_seconds=args.codex_stage_timeout_seconds,
        coder_stage_timeout_seconds=args.coder_stage_timeout_seconds,
        codex_retry_backoff_base=args.codex_retry_backoff_base,
        codex_retry_backoff_max=args.codex_retry_backoff_max,
        rerun_existing=args.rerun_existing,
        stop_on_pass=args.stop_on_pass,
        parallel_styles=args.parallel_styles,
        parallel_workers=args.parallel_workers,
        substyle_workers=args.substyle_workers,
        stop_before_coder=args.stop_before_coder,
        coder_workers=args.coder_workers,
        eval_window=args.eval_window,
        style_diversity_enabled=not args.no_style_diversity_gate,
        style_diversity_tau=args.style_diversity_tau,
        style_diversity_max_rerolls=args.style_diversity_max_rerolls,
        style_diversity_mode=args.style_diversity_mode,
        visual_fidelity_threshold=args.visual_fidelity_threshold,
        launch_smoke_enabled=not args.no_launch_smoke_gate,
        launch_smoke_max_repairs=args.launch_smoke_max_repairs,
        launch_smoke_infra_retries=args.launch_smoke_infra_retries,
    )
    try:
        plan_path = runner.run()
    except StageError as exc:
        print(f"runner failed: {exc}", file=sys.stderr)
        return 1
    print(plan_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
