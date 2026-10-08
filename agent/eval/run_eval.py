from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import io
import json
import os
import queue
import re
import shlex
import sys
import tarfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.docker.worker_manager import (
    AppLaunchError,
    DockerWorkerManager,
    InfrastructureError,
    WorkerConfig,
    derive_worker_config,
    write_worker_env,
)
from agent.docker.cloudflared import restart_tunnel_if_running
from agent.eval.gui_agent_adapter import GUIAgentAdapter, OSWorldHTTPPyAutoGUIEnv
from agent.eval.cost import write_cost
from agent.loop_controller.manifest import update_variant_stage
from agent.runtime_env import activate_osworld_env_vars, ensure_not_base_for_eval, load_env_file


REPO_ROOT = Path(__file__).resolve().parents[2]
load_env_file(REPO_ROOT / ".env")


class AppReadinessError(AppLaunchError):
    """App launch failure with a machine-readable readiness classification."""

    def __init__(self, classification: str, result: Dict[str, Any]) -> None:
        self.classification = classification
        self.result = result
        super().__init__(
            f"App readiness failed ({classification}) in a healthy container/display: {result}"
        )


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clone_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def write_trace_bundle(
    trace_dir: Path,
    case: Dict[str, Any],
    result_payload: Dict[str, Any],
    steps: List[Dict[str, Any]],
    eval_dir: Path,
) -> Dict[str, Any]:
    if not (trace_dir / "case.json").exists():
        write_json(trace_dir / "case.json", case)
    if not (trace_dir / "instruction.md").exists():
        (trace_dir / "instruction.md").write_text("dry-run: GUIAgent instruction was not rendered\n", encoding="utf-8")
    screenshots = sorted(str(path.relative_to(trace_dir)) for path in (trace_dir / "screenshots").glob("*.png"))
    trajectory = {
        "case": case,
        "result": result_payload,
        "steps": steps,
        "screenshots": screenshots,
    }
    write_json(trace_dir / "trajectory.json", trajectory)
    write_json(trace_dir / "steps.json", {"steps": steps})
    manifest = {
        "case_id": str(case.get("id", "")),
        "trace_dir": str(trace_dir.relative_to(eval_dir)),
        "case_file": "case.json",
        "instruction_file": "instruction.md",
        "result_file": "RESULT_JSON",
        "trajectory_file": "trajectory.json",
        "steps_file": "steps.json",
        "screenshots_dir": "screenshots",
        "screenshots": screenshots,
    }
    write_json(trace_dir / "trace_manifest.json", manifest)
    return manifest


def load_cases(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{path} must contain top-level cases list")
    return [case for case in cases if isinstance(case, dict)]


# Regenerable build/restore caches that the RUNTIME never reads. Generated apps do
# not use one consistent spelling: besides .nuget/.dotnet, some builds place the
# same payload under app_data/nuget_packages and app_data/dotnet_home. Source/test
# bin/obj trees are also publish outputs duplicated by the self-contained release.
# Filtering all known variants keeps multi-lane uploads small without removing
# source files, tests, app data, or anything under release/.
_BUILD_CACHE_DIRS = {
    ".build-tools",
    ".dotnet",
    ".dotnet_home",
    ".nuget",
    "dotnet-home",
    "dotnet_home",
    "nuget-packages",
    "nuget_packages",
}
_BUILD_TREE_ROOTS = {"src", "tests"}
_BUILD_TREE_DIRS = {"bin", "obj"}


def _is_build_cache(name: str) -> bool:
    # name is an in-archive path, e.g. "app/src/bin/Release/..."; parts[0] == "app".
    relative_parts = [part for part in name.split("/")[1:] if part]
    if any(part in _BUILD_CACHE_DIRS for part in relative_parts):
        return True
    return (
        bool(relative_parts)
        and relative_parts[0] in _BUILD_TREE_ROOTS
        and any(part in _BUILD_TREE_DIRS for part in relative_parts[1:])
    )


def make_tarball_bytes(source_dir: Path) -> bytes:
    # Ship the app in full EXCEPT regenerable build/restore caches (.nuget/.dotnet),
    # which the runtime never reads (see _BUILD_CACHE_DIRS). Correctness of everything
    # shipped is guaranteed end-to-end by the sha256 check in upload_app, and the
    # transfer is made reliable via resumable chunks + retries.
    buffer = io.BytesIO()

    def _filter(tarinfo: tarfile.TarInfo):
        if _is_build_cache(tarinfo.name):
            return None
        return tarinfo

    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(source_dir, arcname="app", filter=_filter)
    return buffer.getvalue()


@dataclasses.dataclass(frozen=True)
class AppBundle:
    """Immutable, reusable upload payload for every lane of one style."""

    source_dir: Path
    tarball: bytes
    encoded: str
    sha256: str
    tarball_bytes: int
    encoded_bytes: int
    pack_seconds: float


def prepare_app_bundle(source_dir: Path) -> AppBundle:
    if not source_dir.exists():
        raise FileNotFoundError(f"app dir does not exist: {source_dir}")
    started = time.monotonic()
    tarball = make_tarball_bytes(source_dir)
    encoded = base64.b64encode(tarball).decode("ascii")
    return AppBundle(
        source_dir=source_dir,
        tarball=tarball,
        encoded=encoded,
        sha256=hashlib.sha256(tarball).hexdigest(),
        tarball_bytes=len(tarball),
        encoded_bytes=len(encoded),
        pack_seconds=time.monotonic() - started,
    )


def _exec_text(result: Dict[str, Any]) -> str:
    """Best-effort extraction of a command's stdout from an /execute response dict."""
    if not isinstance(result, dict):
        return str(result)
    for key in ("output", "stdout", "raw", "result"):
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    return json.dumps(result, ensure_ascii=False)


def upload_app(
    worker: DockerWorkerManager,
    app: Path | AppBundle,
    remote_app_dir: str,
) -> Dict[str, Any]:
    """Upload the FULL app dir into the guest VM over the OSWorld /execute channel.

    Robustness (no file pruning — the software is shipped whole):
      * sha256 of the tarball is verified inside the VM after reassembly, so a
        corrupt/partial transfer can never be mistaken for a good app;
      * chunks are appended, so a failed attempt RESUMES from the last whole chunk
        instead of re-sending everything (resumable upload);
      * 100 KB chunks cut the number of /execute round-trips ~4x vs the old
        24000-byte chunks (faster);
      * the whole assemble step has a 600s budget and the upload retries up to 5x.

    ``AppBundle`` lets every lane reuse one compression/base64 pass. Passing a Path
    remains supported for callers outside EvalRunner.
    """
    bundle = app if isinstance(app, AppBundle) else prepare_app_bundle(app)
    sha = bundle.sha256
    encoded = bundle.encoded
    total = len(encoded)
    remote_b64 = "/tmp/softwaregym_app.tgz.b64"
    remote_tgz = "/tmp/softwaregym_app.tgz"
    # 100 KB of base64 per /execute call. Must stay under Linux MAX_ARG_STRLEN (128 KiB
    # for a single `sh -c` argument); 100 KB leaves headroom yet cuts round-trips ~4x
    # vs the old 24000-byte chunks.
    chunk_size = 100000
    attempts = 5
    done = "__SOFTWAREGYM_UPLOAD_DONE__"
    mismatch = "__SOFTWAREGYM_SHA_MISMATCH__"
    assemble_cmd = " && ".join(
        [
            f"base64 -d {remote_b64} > {remote_tgz}",
            # Integrity gate: assembled bytes must match the local sha256, else abort.
            f"(printf '%s  %s\\n' {sha} {remote_tgz} | sha256sum -c - >/dev/null 2>&1 "
            f"|| (printf '{mismatch}'; exit 91))",
            f"rm -rf {remote_app_dir!r}",
            f"mkdir -p {remote_app_dir!r}",
            f"tar -xzf {remote_tgz} -C {remote_app_dir!r} --strip-components=1",
            f"chmod -R u+rwX {remote_app_dir!r}",
            # Entry point must exist after a full extract.
            f"test -x {remote_app_dir!r}/release/run.sh -o -f {remote_app_dir!r}/start.sh "
            f"-o -f {remote_app_dir!r}/package.json -o -f {remote_app_dir!r}/main.py",
            f"printf '{done}'",
        ]
    )
    last_text = ""
    started_at = time.monotonic()
    chunks_sent = 0
    resumed_bytes = 0
    for _attempt in range(1, attempts + 1):
        worker.execute(f"mkdir -p {remote_app_dir!r}")
        size_res = worker.execute(
            f"printf 'SZ:%s\\n' \"$(stat -c %s {remote_b64} 2>/dev/null || echo 0)\""
        )
        match = re.search(r"SZ:(\d+)", _exec_text(size_res))
        have = int(match.group(1)) if match else 0
        if have == 0 or have > total:
            worker.execute(f"rm -f {remote_b64}")
            start = 0
        else:
            # Drop any half-written tail chunk; sha256 still guards correctness.
            start = (have // chunk_size) * chunk_size
            worker.execute(f"truncate -s {start} {remote_b64} 2>/dev/null || rm -f {remote_b64}")
        resumed_bytes = max(resumed_bytes, start)
        for index in range(start, total, chunk_size):
            chunk = encoded[index : index + chunk_size]
            worker.execute(f"printf %s {chunk!r} >> {remote_b64}")
            chunks_sent += 1
        result = worker.execute(assemble_cmd, timeout=600)
        blob = json.dumps(result, ensure_ascii=False)
        last_text = _exec_text(result)
        if done in blob:
            worker.execute(f"rm -f {remote_b64} {remote_tgz}")
            return {
                "container": worker.config.container_name,
                "transport": "base64_execute",
                "sha256": sha,
                "tarball_bytes": bundle.tarball_bytes,
                "encoded_bytes": bundle.encoded_bytes,
                "pack_seconds": bundle.pack_seconds,
                "transfer_seconds": time.monotonic() - started_at,
                "attempts": _attempt,
                "chunks_sent": chunks_sent,
                "resumed_bytes": resumed_bytes,
                "chunk_size": chunk_size,
            }
        if mismatch in blob:
            # Corrupt assembly: discard and re-send from scratch next attempt.
            worker.execute(f"rm -f {remote_b64} {remote_tgz}")
        # Otherwise keep remote_b64 so the next attempt resumes from where it stopped.
    raise InfrastructureError(
        f"upload_app failed after {attempts} attempts "
        f"(sha256/extract/entry-point gate not satisfied): {last_text[:500]}"
    )


# Fixed, namespaced container-side root for per-case seed files uploaded by the
# `seed_open` setup mechanism. Kept OUTSIDE the app dir and wiped on every reset so
# nothing leaks across cases or survives a container restart.
SEED_INPUT_ROOT = "/tmp/softwaregym_seed_inputs"


def resolve_seed_dir(software_type: str) -> Optional[Path]:
    """Resolve the trusted host seed-file directory from pipeline_parameters.json.

    The directory comes from operator-controlled config (not the LLM), so cases may
    only reference files *inside* it (enforced separately by basename + containment).
    Returns None when not configured or missing.
    """
    resource_root = REPO_ROOT / "Resource"
    candidates = [resource_root / f"{software_type}_resources"]
    candidates.extend(sorted(resource_root.glob(f"{software_type}_resources*")))
    for resource_dir in candidates:
        params = resource_dir / "pipeline_parameters.json"
        if not params.is_file():
            continue
        try:
            data = json.loads(params.read_text(encoding="utf-8"))
        except Exception:
            continue
        seed_value = str(data.get("eval_task_seedfile") or "").strip()
        if not seed_value:
            return None
        seed_dir = Path(seed_value)
        if not seed_dir.is_absolute():
            seed_dir = resource_dir / seed_value
        seed_dir = seed_dir.resolve()
        return seed_dir if seed_dir.is_dir() else None
    return None


def upload_file(worker: DockerWorkerManager, src: Path, remote_path: str) -> None:
    """Upload a single binary file into the container via the base64-chunk primitive."""
    if not src.is_file():
        raise FileNotFoundError(f"seed file does not exist: {src}")
    encoded = base64.b64encode(src.read_bytes()).decode("ascii")
    remote_b64 = f"{remote_path}.b64"
    remote_dir = remote_path.rsplit("/", 1)[0]
    worker.execute(f"mkdir -p {shlex.quote(remote_dir)} && rm -f {shlex.quote(remote_b64)} {shlex.quote(remote_path)}")
    chunk_size = 24000
    for index in range(0, len(encoded), chunk_size):
        chunk = encoded[index : index + chunk_size]
        worker.execute(f"printf %s {chunk!r} >> {shlex.quote(remote_b64)}")
    worker.execute(
        " && ".join(
            [
                f"base64 -d {shlex.quote(remote_b64)} > {shlex.quote(remote_path)}",
                f"rm -f {shlex.quote(remote_b64)}",
                f"chmod u+rw {shlex.quote(remote_path)}",
            ]
        ),
        timeout=60,
    )


def clean_vm(worker: DockerWorkerManager, software_type: str, remote_app_dir: str) -> None:
    script = f"""
set -e
pkill -f {software_type!r} || true
pkill -f {remote_app_dir!r} || true
pkill -f softwaregym/apps || true
pkill -f pyautogui || true
rm -rf {remote_app_dir!r}
rm -f /home/user/Desktop/{software_type}.desktop
rm -rf /tmp/softwaregym_eval_exports /tmp/softwaregym_app.tgz /tmp/softwaregym_app.tgz.b64
rm -rf {SEED_INPUT_ROOT}
rm -rf /home/user/.cache/{software_type} /home/user/.config/{software_type} /home/user/.local/share/{software_type}
rm -f /home/user/.local/share/recently-used.xbel
rm -f /home/user/.config/autostart/{software_type}.desktop
wmctrl -c {software_type!r} 2>/dev/null || true
mkdir -p /tmp/softwaregym_eval_exports
"""
    worker.execute(script, timeout=30)


def close_app(worker: DockerWorkerManager, software_type: str, remote_app_dir: str) -> Dict[str, Any]:
    script = f"""
set +e
wmctrl -c {software_type!r} 2>/dev/null || true
pkill -f {remote_app_dir!r} || true
pkill -f {software_type!r} || true
sleep 1
printf __SOFTWAREGYM_APP_CLOSED__
"""
    return worker.execute(script, timeout=20)


# Container-side helper used by config_seed setup. Kept as a raw string (NOT an
# f-string) so braces/backslashes in the embedded Python stay literal. Inputs are
# passed via environment variables to avoid any shell-quoting hazards.
SEED_CONFIG_BODY = r'''
import json, os, re
from pathlib import Path
updates = json.loads(os.environ.get("SG_CONFIG_JSON", "{}"))
cfg = Path(os.environ["SG_APP_DIR"]) / "app_data" / "config.yaml"
cfg.parent.mkdir(parents=True, exist_ok=True)
try:
    import yaml  # type: ignore
    data = {}
    if cfg.exists():
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        data = {}
    data.update(updates)
    cfg.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
except Exception:
    # Fallback for containers without PyYAML: top-level scalar replace/append.
    text = cfg.read_text(encoding="utf-8") if cfg.exists() else ""
    lines = text.splitlines()
    for key, value in updates.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (int, float)):
            rendered = str(value)
        else:
            rendered = json.dumps(str(value), ensure_ascii=False)
        new_line = "{}: {}".format(key, rendered)
        replaced = False
        for index, line in enumerate(lines):
            if re.match(r"\s*" + re.escape(str(key)) + r"\s*:", line):
                lines[index] = new_line
                replaced = True
                break
        if not replaced:
            lines.append(new_line)
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
'''


# Container-side helper for the config_key infra check: reads one key (dot-path)
# from app_data/config.yaml and emits a base64-wrapped JSON result between markers,
# so it survives JSON transport without quote-escaping issues.
READ_CONFIG_BODY = r'''
import base64, json, os, re
from pathlib import Path
key = os.environ.get("SG_KEY", "")
cfg = Path(os.environ["SG_APP_DIR"]) / "app_data" / "config.yaml"
text = cfg.read_text(encoding="utf-8", errors="ignore") if cfg.exists() else ""
value = None
parsed = None
try:
    import yaml  # type: ignore
    parsed = yaml.safe_load(text)
except Exception:
    parsed = None
if isinstance(parsed, dict) and key:
    cur = parsed
    ok = True
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            ok = False
            break
    if ok:
        value = cur
if value is None and text and key:
    leaf = key.split(".")[-1]
    for line in text.splitlines():
        m = re.match(r"\s*" + re.escape(leaf) + r"\s*:\s*(.+?)\s*$", line)
        if m:
            value = m.group(1).strip().strip('"').strip("'")
            break
out = json.dumps({"value": value, "exists": cfg.exists()})
print("__SG_CFG_A__" + base64.b64encode(out.encode("utf-8")).decode("ascii") + "__SG_CFG_B__")
'''


def start_app(worker: DockerWorkerManager, remote_app_dir: str, launch_args: str = "") -> Dict[str, Any]:
    script = f"""
set -e
cd {remote_app_dir!r}
if [ -x ./release/run.sh ]; then
  nohup ./release/run.sh {launch_args} >/tmp/softwaregym_app_stdout.log 2>/tmp/softwaregym_app_stderr.log &
elif [ -x ./start.sh ]; then
  nohup ./start.sh {launch_args} >/tmp/softwaregym_app_stdout.log 2>/tmp/softwaregym_app_stderr.log &
elif [ -f package.json ]; then
  nohup npm start >/tmp/softwaregym_app_stdout.log 2>/tmp/softwaregym_app_stderr.log &
elif [ -f main.py ]; then
  nohup python3 main.py >/tmp/softwaregym_app_stdout.log 2>/tmp/softwaregym_app_stderr.log &
else
  echo "No known start entry found" >&2
  exit 7
fi
sleep 3
printf __SOFTWAREGYM_APP_STARTED__
"""
    result = worker.execute(script, timeout=45)
    if "__SOFTWAREGYM_APP_STARTED__" not in json.dumps(result, ensure_ascii=False):
        raise InfrastructureError(f"App start marker missing: {result}")
    return result


def verify_uploaded_app(worker: DockerWorkerManager, remote_app_dir: str) -> Dict[str, Any]:
    script = f"""
set -e
test -d {remote_app_dir!r}
find {remote_app_dir!r} -maxdepth 2 -type f | wc -l
test -x {remote_app_dir!r}/release/run.sh -o -f {remote_app_dir!r}/start.sh -o -f {remote_app_dir!r}/package.json -o -f {remote_app_dir!r}/main.py
printf __SOFTWAREGYM_UPLOAD_OK__
"""
    result = worker.execute(script, timeout=30)
    if "__SOFTWAREGYM_UPLOAD_OK__" not in json.dumps(result, ensure_ascii=False):
        raise InfrastructureError(f"Uploaded app validation failed: {result}")
    return result


def wait_for_app_readiness(
    worker: DockerWorkerManager,
    software_type: str,
    remote_app_dir: str,
    attempts: Optional[int] = None,
    allow_any_window: bool = False,
) -> Dict[str, Any]:
    if attempts is None:
        attempts = max(1, int(os.environ.get("OSWORLD_APP_READINESS_ATTEMPTS", "6")))
    retry_interval = max(
        0.0,
        float(os.environ.get("OSWORLD_APP_READINESS_INTERVAL_SECONDS", "2")),
    )
    last_result: Dict[str, Any] = {}
    readiness_script = f"""
set +e
APP_NAME="$(python3 - <<'PY' 2>/dev/null || true
import json, re
from pathlib import Path
base = Path({remote_app_dir!r}) / "app_data"
name = ""
yaml_path = base / "config.yaml"
if yaml_path.exists():
    for line in yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"\\s*appName\\s*:\\s*(.+?)\\s*$", line)
        if m:
            name = m.group(1).strip().strip('"').strip("'")
            break
if not name:
    json_path = base / "config.json"
    if json_path.exists():
        try:
            name = json.loads(json_path.read_text(encoding="utf-8")).get("appName", "")
        except Exception:
            name = ""
if name:
    print(name)
PY
)"
if {{ pgrep -af {remote_app_dir!r} || pgrep -af {software_type!r} || {{ [ -n "$APP_NAME" ] && pgrep -afi "$APP_NAME"; }}; }} | grep -v '__SOFTWAREGYM_' >/tmp/softwaregym_readiness_process.txt; then
  cat /tmp/softwaregym_readiness_process.txt
  awk '{{print $1}}' /tmp/softwaregym_readiness_process.txt | sort -u >/tmp/softwaregym_readiness_pids.txt
  python3 - <<'PY' >/tmp/softwaregym_readiness_pids_expanded.txt 2>/dev/null || true
from pathlib import Path

seed_file = Path("/tmp/softwaregym_readiness_pids.txt")
known = set(seed_file.read_text().split()) if seed_file.exists() else set()
parents = dict()
for stat_path in Path("/proc").glob("[0-9]*/stat"):
    try:
        fields = stat_path.read_text().rsplit(")", 1)[1].split()
        parents[stat_path.parent.name] = fields[1]
    except Exception:
        pass
changed = True
while changed:
    changed = False
    for pid, parent in parents.items():
        if parent in known and pid not in known:
            known.add(pid)
            changed = True
print("\\n".join(sorted(known, key=int)))
PY
  if [ -s /tmp/softwaregym_readiness_pids_expanded.txt ]; then
    mv /tmp/softwaregym_readiness_pids_expanded.txt /tmp/softwaregym_readiness_pids.txt
  fi
  printf __SOFTWAREGYM_PROCESS_OK__
fi
wmctrl -lp 2>/tmp/softwaregym_wmctrl.err | tee /tmp/softwaregym_readiness_windows.txt || true
if [ -s /tmp/softwaregym_readiness_windows.txt ]; then
  printf __SOFTWAREGYM_WINDOW_PRESENT__
fi
if [ -s /tmp/softwaregym_readiness_windows.txt ] && grep -Eiq {software_type!r} /tmp/softwaregym_readiness_windows.txt; then
  printf __SOFTWAREGYM_WINDOW_OK__
elif [ -n "$APP_NAME" ] && [ -s /tmp/softwaregym_readiness_windows.txt ] && grep -Fqi "$APP_NAME" /tmp/softwaregym_readiness_windows.txt; then
  printf __SOFTWAREGYM_WINDOW_OK__
fi
# A real top-level application window has a desktop column >= 0 (column 2 of
# `wmctrl -lp`). The sticky desktop/panel uses -1. Detecting a non-desktop mapped
# window lets us confirm the GUI is up even when the app uses a brand title that
# matches neither the software type nor app_data appName (generated apps brand
# themselves freely, e.g. "GridPilot" for a TableEditor).
if [ -s /tmp/softwaregym_readiness_windows.txt ] && awk 'NF>=2 && $2 != "-1" {{f=1}} END{{exit !f}}' /tmp/softwaregym_readiness_windows.txt; then
  printf __SOFTWAREGYM_REAL_WINDOW__
fi
if [ -s /tmp/softwaregym_readiness_pids.txt ] && [ -s /tmp/softwaregym_readiness_windows.txt ] && awk '
  NR==FNR {{p[$1]=1; next}}
  NF>=3 && $2 != "-1" && ($3 in p) {{f=1}}
  END {{exit !f}}
' /tmp/softwaregym_readiness_pids.txt /tmp/softwaregym_readiness_windows.txt; then
  printf __SOFTWAREGYM_APP_WINDOW_PID_OK__
fi
test -s /tmp/softwaregym_app_stderr.log && tail -n 50 /tmp/softwaregym_app_stderr.log || true
printf __SOFTWAREGYM_READINESS_PROBE__
"""
    window_only_result: Dict[str, Any] = {}
    saw_process = False
    saw_window_present = False
    saw_real_window = False
    for _ in range(attempts):
        last_result = worker.execute(readiness_script, timeout=15)
        text = json.dumps(last_result, ensure_ascii=False)
        probe_ran = "__SOFTWAREGYM_READINESS_PROBE__" in text
        process_ok = "__SOFTWAREGYM_PROCESS_OK__" in text
        window_present = "__SOFTWAREGYM_WINDOW_PRESENT__" in text
        window_ok = "__SOFTWAREGYM_WINDOW_OK__" in text
        real_window = "__SOFTWAREGYM_REAL_WINDOW__" in text
        app_window_pid_ok = "__SOFTWAREGYM_APP_WINDOW_PID_OK__" in text
        saw_process = saw_process or (probe_ran and process_ok)
        saw_window_present = saw_window_present or (probe_ran and window_present)
        saw_real_window = saw_real_window or (probe_ran and real_window)
        if probe_ran and process_ok and (window_ok or app_window_pid_ok):
            return last_result
        # Fallbacks (recorded, returned only if no strict match across attempts):
        #  - window_ok alone: a mapped window whose title matches the app, even if
        #    the process cmdline did not (rare branded/wrapper processes).
        #  - app_window_pid_ok: a real top-level window is owned by the matched app
        #    PID. Unlike the old process_ok + any-real-window fallback, this cannot
        #    accidentally accept an unrelated desktop window.
        if probe_ran and (window_ok or app_window_pid_ok):
            window_only_result = last_result
        time.sleep(retry_interval)
    if window_only_result:
        return window_only_result
    # No window/process markers across all attempts. Distinguish "the app never came
    # up" from "the environment itself is broken" AT THE SOURCE: run a final env
    # health probe (platform + /execute + screenshot). If the container/display are
    # provably healthy, this is the app's own launch failure (broken run.sh entry,
    # missing runtime dep, startup crash, window that never maps) -> AppLaunchError,
    # a REAL software bug that must NOT be filtered as infra noise. Only when the env
    # itself is unhealthy do we report InfrastructureError.
    try:
        env_healthy = worker.health_check_for_running_worker()
    except Exception:  # noqa: BLE001 — any probe failure means we cannot trust the env
        env_healthy = False
    if env_healthy:
        if saw_process and saw_real_window:
            classification = "process_running_window_mismatch"
        elif saw_process:
            classification = "process_running_no_app_window"
        elif saw_window_present:
            classification = "process_missing_unrelated_window_present"
        else:
            classification = "process_missing"
        raise AppReadinessError(classification, last_result)
    raise InfrastructureError(f"App readiness check failed: {last_result}")


def normalized_success(value: Any) -> bool:
    return str(value).strip().lower() in {"pass", "passed", "success", "succeeded", "ok", "true"}


def collect_status_items(case_results: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for result in case_results:
        value = result.get(key)
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))
    return items


def rate_payload(total: int, passed: int) -> Dict[str, Any]:
    return {
        "total": total,
        "passed": passed,
        "failed_or_blocked": max(total - passed, 0),
        "percent": round((passed / total) * 100, 2) if total else 0.0,
    }


def summarize_results(
    version: str,
    style_id: str,
    variant_dir: Path,
    case_results: List[Dict[str, Any]],
    infra_errors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    failures = [
        result
        for result in case_results
        if result.get("status") in {
            "Fail",
            "Blocked",
            "InfraError",
            "JudgeInfraError",
            "AppLaunchError",
        }
    ]
    # Inject absolute screenshot directories into each case so Coder/DevCoder in later
    # repair rounds can locate GUIAgent screenshots directly.
    # trace_path is relative to eval_dir; eval_dir = variant_dir/loop/<version>/eval.
    eval_dir = variant_dir / "loop" / version / "eval"
    for result in case_results:
        trace_path = result.get("trace_path")
        if not trace_path:
            continue
        screenshot_dir = (eval_dir / trace_path / "screenshots").resolve()
        result["screenshot_dir"] = str(screenshot_dir)
        result["trace_dir_abs"] = str((eval_dir / trace_path).resolve())
    task_total = len(case_results)
    task_pass = sum(1 for result in case_results if result.get("status") == "Pass")
    effective_infra_errors = list(infra_errors)
    if task_total == 0:
        effective_infra_errors.append(
            {
                "type": "NoCasesExecuted",
                "message": "GUIEval finished without executing any cases.",
            }
        )
    checkpoint_items = collect_status_items(case_results, "checkpoints")
    checkpoint_pass = sum(1 for item in checkpoint_items if normalized_success(item.get("status")))
    step_items = collect_status_items(case_results, "step_results")
    step_pass = sum(1 for item in step_items if normalized_success(item.get("status")))
    # Unified "app crash" signal: runtime_errors come from unhandled_exception records
    # written by the app's own global exception hook to log.json, so any case with
    # runtime_errors (even if judged InfraError/Blocked) indicates a real app crash
    # rather than environment noise. Summarize crash counts and deduplicated exception
    # signatures at the top level so repair rounds can immediately see what and where
    # the app threw, without digging case by case.
    def _crash_signature(rec: Dict[str, Any]) -> Dict[str, str]:
        detail = rec.get("details") if isinstance(rec.get("details"), dict) else rec
        return {
            "exceptionType": str(detail.get("exceptionType") or detail.get("type") or "")[:200],
            "message": str(detail.get("message") or rec.get("message") or "")[:400],
            "source": str(detail.get("source") or "")[:160],
        }
    app_crash_cases = [result for result in case_results if result.get("runtime_errors")]
    _crash_seen: set = set()
    app_crash_signatures: List[Dict[str, str]] = []
    for result in app_crash_cases:
        for rec in result.get("runtime_errors") or []:
            if not isinstance(rec, dict):
                continue
            sig = _crash_signature(rec)
            key = (sig["exceptionType"], sig["message"])
            if not (sig["exceptionType"] or sig["message"]) or key in _crash_seen:
                continue
            _crash_seen.add(key)
            app_crash_signatures.append(sig)
    return {
        "version": version,
        "style_id": style_id,
        "variant_dir": str(variant_dir),
        "overall_status": "Pass" if task_total > 0 and not failures and not effective_infra_errors else "Fail",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "cases_total": len(case_results),
        "cases_pass": sum(1 for result in case_results if result.get("status") == "Pass"),
        "cases_fail": sum(1 for result in case_results if result.get("status") == "Fail"),
        "cases_blocked": sum(1 for result in case_results if result.get("status") == "Blocked"),
        "cases_infra_error": sum(1 for result in case_results if result.get("status") == "InfraError") + len(effective_infra_errors),
        "cases_judge_error": sum(
            1 for result in case_results if result.get("status") == "JudgeInfraError"
        ),
        # App-launch failures are counted separately and not folded into cases_infra_error,
        # so repair rounds do not filter them as environment noise.
        "cases_app_launch_error": sum(1 for result in case_results if result.get("status") == "AppLaunchError"),
        "cases_with_runtime_error": sum(1 for result in case_results if result.get("runtime_errors")),
        "cases_app_crash": len(app_crash_cases),
        "app_crash_signatures": app_crash_signatures[:20],
        "task_level": rate_payload(task_total, task_pass),
        "checkpoint_level": rate_payload(len(checkpoint_items), checkpoint_pass),
        "step_level": rate_payload(len(step_items), step_pass),
        "case_results": case_results,
        "failures": failures,
        "infra_errors": effective_infra_errors,
        "cost_path": "cost.json",
    }


class _EvalLane:
    """One parallel eval lane: a dedicated OSWorld worker + its GUI agent.

    Lane 0 wraps the primary worker (identical to the legacy single-container
    path). Each lane pulls cases from a shared queue and owns its own recovery
    bookkeeping so a crash in one lane never corrupts another.
    """

    def __init__(self, index: int, worker: DockerWorkerManager, gui_agent: GUIAgentAdapter) -> None:
        self.index = index
        self.worker = worker
        self.gui_agent = gui_agent
        self.cases_since_recovery = 0
        self.healthy = True


class EvalRunner:
    def __init__(
        self,
        variant_dir: Path,
        version: str,
        software_type: str,
        style_id: str,
        dry_run: bool = False,
        case_attempts: int = 1,
        case_trajectory_reruns: int = 1,
        max_steps: int = 60,
        max_cases: Optional[int] = None,
        infra_case_retries: int = 3,
        eval_window: int = 1,
    ) -> None:
        self.variant_dir = variant_dir
        self.version = version
        self.software_type = software_type
        self.style_id = style_id
        self.dry_run = dry_run
        self.eval_window = max(1, eval_window)
        self.case_attempts = max(1, case_attempts)
        self.case_trajectory_reruns = max(1, case_trajectory_reruns)
        self.max_steps = max_steps
        self.max_cases = max_cases if max_cases is None else max(1, max_cases)
        self.infra_case_retries = max(0, infra_case_retries)
        self.version_dir = variant_dir / "loop" / version
        self.eval_dir = self.version_dir / "eval"
        # DevCoder modifies coder/app in place (no longer copied to devcoder/app);
        # coder/app is what gets evaluated.
        self.app_dir = self.version_dir / "coder" / "app"
        self.cases_file = variant_dir / "05_eval_tasks" / "cases.json"
        self.remote_app_dir = f"/home/user/softwaregym/apps/{software_type}_{style_id}_{version}"
        self._app_bundle: Optional[AppBundle] = None
        self._app_bundle_lock = threading.Lock()
        self._deploy_lock = threading.Lock()
        self._upload_state_lock = threading.Lock()
        self._upload_counts: Dict[str, int] = {}
        self.worker = DockerWorkerManager(
            derive_worker_config(variant_dir, software_type, style_id),
            state_dir=variant_dir / "docker",
            dry_run=dry_run,
        )
        # Sliding-window eval: lane 0 reuses the primary worker (identical to the
        # legacy single-container path); extra lanes get distinct container names,
        # overlays and (auto-fallback) ports so one style can fan its cases across
        # several OSWorld containers in parallel.
        self.workers: List[DockerWorkerManager] = self._build_worker_pool()

    def _build_worker_pool(self) -> List[DockerWorkerManager]:
        workers: List[DockerWorkerManager] = [self.worker]
        for index in range(1, self.eval_window):
            base: WorkerConfig = self.worker.config
            overlay = base.overlay_path
            lane_overlay = overlay.with_name(f"{overlay.stem}_w{index}{overlay.suffix}")
            lane_config = dataclasses.replace(
                base,
                worker_name=f"{base.worker_name}_w{index}",
                container_name=f"{base.container_name}_w{index}",
                overlay_path=lane_overlay,
                auto_fallback_ports=True,
            )
            workers.append(
                DockerWorkerManager(
                    lane_config,
                    state_dir=self.variant_dir / "docker" / f"w{index}",
                    dry_run=self.dry_run,
                )
            )
        return workers

    def sync_worker_state(self, label: str = "runtime", worker: Optional[DockerWorkerManager] = None) -> None:
        worker = worker or self.worker
        is_primary = worker is self.worker
        docker_dir = self.variant_dir / "docker"
        suffix = "" if is_primary else f".{worker.config.container_name}"
        if is_primary:
            write_worker_env(docker_dir / "worker.env", worker.config)
        write_json(self.eval_dir / f"worker_state.{label}{suffix}.json", worker.snapshot_state())
        if is_primary and (docker_dir / "cloudflared.pid").exists():
            try:
                tunnel = restart_tunnel_if_running(worker.config.vnc_port, docker_dir, timeout_seconds=20)
                write_json(self.eval_dir / f"cloudflared.{label}.json", tunnel)
            except Exception as exc:
                write_json(self.eval_dir / f"cloudflared.{label}.error.json", {"error": str(exc)})

    def ensure_display_ready(self, label: str, worker: Optional[DockerWorkerManager] = None) -> None:
        if self.dry_run:
            return
        worker = worker or self.worker
        env = OSWorldHTTPPyAutoGUIEnv(worker)
        obs = env.reset()
        screenshot = obs.get("screenshot")
        if not isinstance(screenshot, (bytes, bytearray)) or not bytes(screenshot).startswith(b"\x89PNG"):
            raise InfrastructureError(f"Display health check failed: {label}")

    def prepare_current_app_bundle(self) -> AppBundle:
        with self._app_bundle_lock:
            if self._app_bundle is None:
                self._app_bundle = prepare_app_bundle(self.app_dir)
                write_json(
                    self.eval_dir / "app_bundle.json",
                    {
                        "source_dir": str(self._app_bundle.source_dir),
                        "sha256": self._app_bundle.sha256,
                        "tarball_bytes": self._app_bundle.tarball_bytes,
                        "encoded_bytes": self._app_bundle.encoded_bytes,
                        "pack_seconds": self._app_bundle.pack_seconds,
                    },
                )
            return self._app_bundle

    def _record_upload(self, worker: DockerWorkerManager, stats: Dict[str, Any]) -> None:
        container = worker.config.container_name
        with self._upload_state_lock:
            sequence = self._upload_counts.get(container, 0) + 1
            self._upload_counts[container] = sequence
        write_json(
            self.eval_dir / f"upload_state.{container}.{sequence:02d}.json",
            stats,
        )

    def deploy_current_app(self, worker: Optional[DockerWorkerManager] = None) -> None:
        worker = worker or self.worker
        # A ready lane may recover while the next lane is still initializing. Keep
        # uploads serialized per style so those paths do not saturate /execute
        # concurrently; GUI case execution itself remains parallel.
        with self._deploy_lock:
            clean_vm(worker, self.software_type, self.remote_app_dir)
            stats = upload_app(worker, self.prepare_current_app_bundle(), self.remote_app_dir)
            self._record_upload(worker, stats)
            verify_uploaded_app(worker, self.remote_app_dir)
            close_app(worker, self.software_type, self.remote_app_dir)

    def remote_path(self, rel_or_abs: str) -> str:
        path = str(rel_or_abs or "").strip()
        if not path:
            return ""
        if path.startswith("/"):
            return path
        return f"{self.remote_app_dir.rstrip('/')}/{path.lstrip('/')}"

    def seed_config_yaml(self, config: Dict[str, Any], worker: Optional[DockerWorkerManager] = None) -> Dict[str, Any]:
        worker = worker or self.worker
        payload = json.dumps(config, ensure_ascii=False)
        script = (
            f"export SG_CONFIG_JSON={shlex.quote(payload)}\n"
            f"export SG_APP_DIR={shlex.quote(self.remote_app_dir)}\n"
            "python3 - <<'PY'\n" + SEED_CONFIG_BODY + "\nPY\n"
        )
        return worker.execute(script, timeout=30)

    def prepare_case_setup(self, case: Dict[str, Any], case_id: str, attempt_index: int, trace_dir: Path, worker: Optional[DockerWorkerManager] = None) -> str:
        """Apply opt-in per-case initialization (layer-2 contract setup).

        Returns launch args for start_app. Cases without a ``setup`` block keep the
        legacy no-argument launch behaviour. Setup failures are recorded as evidence
        (not raised) so a broken contract surfaces as a failed case rather than an
        infrastructure error.
        """
        worker = worker or self.worker
        setup = case.get("setup") if isinstance(case.get("setup"), dict) else {}
        mechanism = str(setup.get("mechanism", "none") or "none").strip().lower()
        record: Dict[str, Any] = {"case_id": case_id, "attempt": attempt_index, "mechanism": mechanism}
        launch_args = ""
        if mechanism in {"", "none"}:
            return ""
        try:
            if mechanism == "config_seed":
                config = setup.get("config") if isinstance(setup.get("config"), dict) else {}
                record["config"] = config
                if config:
                    record["result"] = self.seed_config_yaml(config, worker)
            elif mechanism in {"open", "new-from", "new_from"}:
                target = self.remote_path(setup.get("open_path", ""))
                mode = str(setup.get("open_mode", "") or "").strip().lower()
                use_new_from = mechanism in {"new-from", "new_from"} or mode in {"new-from", "new_from"}
                flag = "--new-from" if use_new_from else "--open"
                if target:
                    launch_args = f"{flag} {shlex.quote(target)}"
                record["launch_args"] = launch_args
            elif mechanism == "import":
                pkg = self.remote_path(setup.get("import_package", "app_data/samples/import_example"))
                import_mode = str(setup.get("import_mode", "merge") or "merge").strip().lower()
                if import_mode not in {"merge", "replace", "preview"}:
                    import_mode = "merge"
                cmd = (
                    f"cd {shlex.quote(self.remote_app_dir)} && "
                    f"./release/run.sh --import-data {shlex.quote(pkg)} --import-mode {import_mode}"
                )
                record["import_package"] = pkg
                record["import_mode"] = import_mode
                record["result"] = (worker or self.worker).execute(cmd, timeout=120)
            elif mechanism == "seed_open":
                # Upload a real seed file (image/media) from the trusted host seed dir
                # into a namespaced, per-case container path, then open it. The seed
                # source comes from operator config; the case may only pick a file
                # *inside* it (path-traversal hardened) — never an arbitrary host path.
                raw = str(setup.get("seed_file", "") or "").strip()
                name = os.path.basename(raw)
                record["requested_seed_file"] = raw
                seed_dir = resolve_seed_dir(self.software_type)
                if not seed_dir:
                    raise ValueError("seed_open requested but no eval_task_seedfile configured for this software type")
                src = (seed_dir / name).resolve()
                if not name or name in {".", ".."} or src.parent != seed_dir or not src.is_file():
                    raise ValueError(f"invalid or unknown seed_file: {raw!r}")
                mode = str(setup.get("open_mode", "new-from") or "new-from").strip().lower()
                flag = "--new-from" if mode in {"new-from", "new_from"} else "--open"
                remote = f"{SEED_INPUT_ROOT}/{case_id}/input/{name}"
                # Per-case wipe first (no leak across sequential cases on the same worker)
                (worker or self.worker).execute(f"rm -rf {shlex.quote(SEED_INPUT_ROOT)}/{shlex.quote(case_id)}")
                upload_file(worker or self.worker, src, remote)
                launch_args = f"{flag} {shlex.quote(remote)}"
                record["seed_source"] = str(src)
                record["remote_path"] = remote
                record["open_mode"] = "new-from" if flag == "--new-from" else "open"
                record["launch_args"] = launch_args
            else:
                record["warning"] = f"unknown setup mechanism: {mechanism}"
            record["status"] = "ok"
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
        write_json(trace_dir / "case_setup.json", record)
        return launch_args

    def open_case_app(self, case: Dict[str, Any], case_id: str, attempt_index: int, trace_dir: Path, worker: Optional[DockerWorkerManager] = None) -> None:
        worker = worker or self.worker
        close_app(worker, self.software_type, self.remote_app_dir)
        self.ensure_display_ready(f"before_start_{case_id}_attempt_{attempt_index}", worker)
        launch_args = self.prepare_case_setup(case, case_id, attempt_index, trace_dir, worker)
        start_app(worker, self.remote_app_dir, launch_args=launch_args)
        try:
            wait_for_app_readiness(worker, self.software_type, self.remote_app_dir)
        except AppReadinessError as exc:
            relaunch_retries = max(
                0,
                int(os.environ.get("OSWORLD_APP_RELAUNCH_RETRIES", "1")),
            )
            retryable = exc.classification in {
                "process_running_no_app_window",
                "process_running_window_mismatch",
            }
            if not retryable or relaunch_retries <= 0:
                raise
            last_error: AppReadinessError = exc
            for relaunch_index in range(1, relaunch_retries + 1):
                worker.write_state(
                    f"app-readiness-relaunch-{case_id}-{attempt_index}-{relaunch_index}.json",
                    {
                        "classification": last_error.classification,
                        "evidence": str(last_error),
                    },
                )
                close_app(worker, self.software_type, self.remote_app_dir)
                self.ensure_display_ready(
                    f"before_app_relaunch_{case_id}_attempt_{attempt_index}_{relaunch_index}",
                    worker,
                )
                start_app(worker, self.remote_app_dir, launch_args=launch_args)
                try:
                    wait_for_app_readiness(worker, self.software_type, self.remote_app_dir)
                    break
                except AppReadinessError as retry_exc:
                    last_error = retry_exc
            else:
                raise last_error
        self.ensure_display_ready(f"after_start_{case_id}_attempt_{attempt_index}", worker)

    def _capture_remote(self, command: str, timeout: float = 30.0, worker: Optional[DockerWorkerManager] = None) -> str:
        """Run a container command and return its stdout text reliably.

        Output is base64-wrapped between markers so it survives JSON transport
        without quote-escaping hazards.
        """
        worker = worker or self.worker
        marker_a, marker_b = "__SG_CAP_A__", "__SG_CAP_B__"
        wrapped = f"printf {marker_a!r}; {{ {command} ; }} 2>/dev/null | base64 -w0; printf {marker_b!r}"
        res = worker.execute(wrapped, timeout=timeout)
        text = json.dumps(res, ensure_ascii=False)
        match = re.search(re.escape(marker_a) + "(.*?)" + re.escape(marker_b), text)
        if not match:
            return ""
        try:
            return base64.b64decode(match.group(1)).decode("utf-8", errors="replace")
        except Exception:
            return ""

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        for candidate in (text, text[text.find("{"): text.rfind("}") + 1] if "{" in text and "}" in text else ""):
            if not candidate:
                continue
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    obj.setdefault("_parsed", True)
                    return obj
            except Exception:
                continue
        return {"_raw": text[:2000], "_parsed": False}

    def _load_dump_state(self, worker: Optional[DockerWorkerManager] = None) -> Dict[str, Any]:
        raw = self._capture_remote(f"cd {shlex.quote(self.remote_app_dir)} && ./release/run.sh --dump-state", worker=worker)
        if not raw:
            return {"_raw": "", "_parsed": False}
        return self._extract_json(raw)

    def _read_unhandled_exceptions(self, worker: Optional[DockerWorkerManager] = None) -> List[Dict[str, Any]]:
        """Read app_data/log.json inside the container and return all records with
        action=unhandled_exception. Fail-safe throughout: any read/parse failure returns
        whatever was parsed so far (or an empty list), never affecting case evaluation."""
        try:
            raw = self._capture_remote(
                f"cat {shlex.quote(self.remote_app_dir)}/app_data/log.json", worker=worker
            )
        except Exception:
            return []
        out: List[Dict[str, Any]] = []
        for line in (raw or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and str(obj.get("action", "")).strip().lower() == "unhandled_exception":
                out.append(obj)
        return out

    @staticmethod
    def _dig(obj: Any, path: str) -> Any:
        cur = obj
        if not path:
            return cur
        for part in path.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            elif isinstance(cur, list):
                try:
                    cur = cur[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
        return cur

    @staticmethod
    def _as_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in {"true", "1", "yes", "on"}:
            return True
        if s in {"false", "0", "no", "off"}:
            return False
        return None

    @staticmethod
    def _match_value(actual: Any, expect: Any, match: str) -> bool:
        if match == "exists":
            return actual is not None
        if actual is None:
            return False
        if match == "equals":
            # Bool-aware: JSON true/false vs "true"/1 etc. compare by truthiness.
            if isinstance(actual, bool) or isinstance(expect, bool):
                ab, eb = EvalRunner._as_bool(actual), EvalRunner._as_bool(expect)
                if ab is not None and eb is not None:
                    return ab == eb
            # Numeric-aware: 1 == 1.0, "14" == 14 (avoids str("1") != str("1.0")).
            try:
                if not isinstance(actual, bool) and not isinstance(expect, bool):
                    return float(actual) == float(expect)
            except (TypeError, ValueError):
                pass
            # Fall back to case-insensitive string compare (e.g. "dark" == "Dark").
            return str(actual).strip().lower() == str(expect).strip().lower()
        if match == "contains":
            a = str(actual).strip().lower()
            e = str(expect).strip().lower()
            if e in a:
                return True
            # Path-normalized: an app may store the basename ("open_example") while the
            # assertion used the full path ("app_data/samples/open_example"), or vice versa.
            a_base = os.path.basename(a.rstrip("/"))
            e_base = os.path.basename(e.rstrip("/"))
            return bool(e_base) and (e_base in a or e_base == a_base)
        if match in {"min", "gte"}:
            try:
                return float(actual) >= float(expect)
            except (TypeError, ValueError):
                return False
        return False

    def _check_dump_state(self, state: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
        field = str(spec.get("field", "")).strip()
        actual = self._dig(state, field)
        expect = spec.get("expect")
        match = str(spec.get("match", "contains")).strip().lower()
        return {
            "passed": self._match_value(actual, expect, match),
            "field": field,
            "actual": actual,
            "expect": expect,
            "match": match,
            "dump_parsed": state.get("_parsed", False),
        }

    def _check_sqlite_count(self, spec: Dict[str, Any], worker: Optional[DockerWorkerManager] = None) -> Dict[str, Any]:
        table = str(spec.get("table", "")).strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
            return {"passed": False, "error": f"invalid table name: {table}"}
        minimum = int(spec.get("min", 1))
        db_path = f"{self.remote_app_dir}/app_data/app.db"
        py = (
            f"import sqlite3; c=sqlite3.connect({db_path!r}); "
            f"print(c.execute('select count(*) from {table}').fetchone()[0])"
        )
        raw = self._capture_remote("python3 -c " + shlex.quote(py), worker=worker)
        try:
            count = int(raw.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"passed": False, "error": f"could not read count: {raw[:200]}", "min": minimum, "table": table}
        return {"passed": count >= minimum, "count": count, "min": minimum, "table": table}

    def _check_path_glob(self, spec: Dict[str, Any], worker: Optional[DockerWorkerManager] = None) -> Dict[str, Any]:
        glob = str(spec.get("glob", "")).strip()
        minimum = int(spec.get("min", 1))
        target = self.remote_path(glob)
        raw = self._capture_remote(f"ls -1d {target} 2>/dev/null | wc -l", worker=worker)
        try:
            count = int(raw.strip().splitlines()[-1])
        except (ValueError, IndexError):
            count = 0
        return {"passed": count >= minimum, "count": count, "min": minimum, "glob": glob}

    def run_infra_checks(self, case: Dict[str, Any], case_id: str, attempt_index: int, trace_dir: Path, worker: Optional[DockerWorkerManager] = None) -> Dict[str, Any]:
        """Deterministic protocol-boundary assertions for layer-2 contract cases.

        Opt-in: only runs when the case declares ``infra_checks``. The GUIAgent does
        not perform these; the runner evaluates them at the file/CLI boundary and the
        result gates pass/fail (a Pass GUI run with a failed infra check is downgraded
        to Fail). Cases without ``infra_checks`` are unaffected.
        """
        checks = case.get("infra_checks")
        report: Dict[str, Any] = {
            "case_id": case_id,
            "attempt": attempt_index,
            "applicable": isinstance(checks, list) and bool(checks),
            "passed": True,
            "checks": [],
        }
        if not report["applicable"]:
            return report
        dump_state_cache: Optional[Dict[str, Any]] = None
        for spec in checks:
            if not isinstance(spec, dict):
                continue
            ctype = str(spec.get("type", "")).strip().lower()
            entry: Dict[str, Any] = {"name": spec.get("name", ctype), "type": ctype, "passed": False}
            try:
                if ctype == "dump_state":
                    if dump_state_cache is None:
                        dump_state_cache = self._load_dump_state(worker)
                    entry.update(self._check_dump_state(dump_state_cache, spec))
                elif ctype == "sqlite_count":
                    entry.update(self._check_sqlite_count(spec, worker))
                elif ctype == "path_glob":
                    entry.update(self._check_path_glob(spec, worker))
                else:
                    entry["error"] = f"unknown infra check type: {ctype}"
            except Exception as exc:
                entry["error"] = str(exc)
            report["checks"].append(entry)
            if not entry.get("passed"):
                report["passed"] = False
        write_json(trace_dir / "infra_checks.json", report)
        return report

    def close_case_app(self, case_id: str, attempt_index: int, trace_dir: Path, worker: Optional[DockerWorkerManager] = None) -> None:
        worker = worker or self.worker
        try:
            result = close_app(worker, self.software_type, self.remote_app_dir)
            write_json(trace_dir / "app_close.json", {"case_id": case_id, "attempt": attempt_index, "result": result})
        except Exception as exc:
            write_json(trace_dir / "app_close_error.json", {"case_id": case_id, "attempt": attempt_index, "error": str(exc)})

    def _guiagent_model(self, lane_index: Optional[int] = None) -> str:
        """Session-level sticky channel for GUIEval (Operation Dynamo).

        GPT54Agent keeps a responses ``previous_response_id`` across the steps of a
        case; the backend binds that item to the resource that created it, so if an
        upstream load-balancer spreads one lane's steps across channels we get
        "created under a different resource" 400s. Pinning each lane to one
        channel alias (round-robin across lanes) keeps per-case affinity while still
        spreading load across resources. Opt-in via SOFTWAREGYM_GUIAGENT_CHANNEL_ALIASES
        (comma-separated); falls back to the plain model when unset.
        """
        base_model = (
            os.environ.get("GUIAGENT_MODEL")
            or os.environ.get("APPEVAL_LLM_MODEL")
            or os.environ.get("MODEL")
        )
        if not base_model:
            raise RuntimeError(
                "GUI agent model is not set; configure GUIAGENT_MODEL (or APPEVAL_LLM_MODEL/MODEL) in the repo-root .env"
            )
        raw = os.environ.get("SOFTWAREGYM_GUIAGENT_CHANNEL_ALIASES", "")
        aliases = [a.strip() for a in raw.split(",") if a.strip()]
        if aliases and lane_index is not None:
            return aliases[lane_index % len(aliases)]
        return base_model

    def make_gui_agent(self, worker: Optional[DockerWorkerManager] = None, lane_index: Optional[int] = None) -> GUIAgentAdapter:
        worker = worker or self.worker
        env = OSWorldHTTPPyAutoGUIEnv(worker)
        return GUIAgentAdapter(
            env=env,
            model=self._guiagent_model(lane_index),
            max_steps=self.max_steps,
        )

    def _start_worker(self, worker: DockerWorkerManager, start_container: bool) -> bool:
        """Bring one lane's container online, recovering once. Returns health."""
        if start_container:
            try:
                reset_overlay_default = "1" if worker.config.reset_overlay_on_recovery else "0"
                worker.start_container(
                    reset_container=os.environ.get("RESET_OSWORLD_CONTAINER", "1") != "0",
                    reset_overlay=os.environ.get("RESET_OSWORLD_OVERLAY", reset_overlay_default) == "1",
                )
                self.sync_worker_state("after_start", worker)
                return True
            except InfrastructureError as exc:
                recovery = worker.recover(f"initial worker start unhealthy: {exc}", force_overlay_reset=True)
                self.sync_worker_state("after_initial_start_recovery", worker)
                return recovery.healthy
        try:
            worker.wait_until_healthy()
            self.sync_worker_state("existing", worker)
            return True
        except InfrastructureError as exc:
            recovery = worker.recover(f"existing worker unhealthy: {exc}", force_overlay_reset=True)
            self.sync_worker_state("after_existing_worker_recovery", worker)
            return recovery.healthy

    def _deploy_worker(self, worker: DockerWorkerManager, infra_errors: List[Dict[str, Any]], lock: threading.Lock) -> bool:
        """Deploy the current app onto one lane's container, recovering once."""
        try:
            self.deploy_current_app(worker)
            return True
        except Exception as exc:
            recovery = worker.recover(f"deploy failed: {exc}", force_overlay_reset=False)
            self.sync_worker_state("after_deploy_recovery", worker)
            with lock:
                infra_errors.append({
                    "stage": "deploy",
                    "container": worker.config.container_name,
                    "error": str(exc),
                    "recovery": recovery.__dict__,
                })
            if not recovery.healthy:
                return False
            try:
                self.deploy_current_app(worker)
                return True
            except Exception as exc2:
                with lock:
                    infra_errors.append({
                        "stage": "deploy_retry",
                        "container": worker.config.container_name,
                        "error": str(exc2),
                    })
                return False

    def run(self, start_container: bool = True) -> Path:
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        update_variant_stage(self.variant_dir, self.version, "gui_eval", "running", trace_path=str(self.eval_dir))
        write_worker_env(self.variant_dir / "docker" / "worker.env", self.worker.config)
        write_json(self.eval_dir / "worker_state.initial.json", self.worker.snapshot_state())

        cases = load_cases(self.cases_file)
        if self.max_cases is not None:
            cases = cases[: self.max_cases]
        case_results: List[Dict[str, Any]] = []
        infra_errors: List[Dict[str, Any]] = []
        results_lock = threading.Lock()

        # Dry-run never touches containers; keep it single-lane and identical.
        if self.dry_run:
            for case_index, case in enumerate(cases, start=1):
                case_id = str(case.get("id") or f"case_{case_index:03d}")
                trace_dir = self.eval_dir / "traces" / case_id / "attempt_01"
                trace_dir.mkdir(parents=True, exist_ok=True)
                case_result = {
                    "case_id": case_id,
                    "status": "Blocked",
                    "evidence": "dry-run: GUIAgent was not invoked",
                    "task_summary": "dry-run: GUIAgent was not invoked",
                    "task_completion": {"completed": False, "reason": "dry_run"},
                    "step_results": [],
                    "checkpoints": [],
                    "final_observation": "",
                    "failure_analysis": "dry-run mode does not execute GUI tasks",
                    "blocker": "dry_run",
                    "repair_hint": "",
                    "trace_path": str(trace_dir.relative_to(self.eval_dir)),
                }
                write_json(trace_dir / "RESULT_JSON", case_result)
                trace_manifest = write_trace_bundle(trace_dir, case, case_result, [], self.eval_dir)
                case_result["trace_manifest_path"] = trace_manifest["trace_dir"] + "/trace_manifest.json"
                case_result["trajectory_path"] = trace_manifest["trace_dir"] + "/trajectory.json"
                case_result["screenshots"] = trace_manifest["screenshots"]
                case_results.append(case_result)
            summary = summarize_results(self.version, self.style_id, self.variant_dir, case_results, infra_errors)
            write_json(self.eval_dir / "summary.json", summary)
            write_json(self.eval_dir / "worker_state.final.json", self.worker.snapshot_state())
            write_cost(self.eval_dir, model="dry-run")
            update_variant_stage(self.variant_dir, self.version, "gui_eval", "completed", summary_path=str(self.eval_dir / "summary.json"))
            return self.eval_dir / "summary.json"

        # Compress/base64 the app once before any lane starts. Every initial deploy
        # and later recovery reuses this immutable payload.
        self.prepare_current_app_bundle()

        # Distribute cases across lanes via a shared work queue (sliding window).
        work_queue: "queue.Queue[tuple[int, Dict[str, Any]]]" = queue.Queue()
        for item in enumerate(cases, start=1):
            work_queue.put(item)

        lanes: List[_EvalLane] = []
        lanes_lock = threading.Lock()
        # Keep one cold-boot/upload in flight per style. Across parallel styles this
        # still permits style-level concurrency, while a ready lane can immediately
        # consume cases as the next lane initializes.
        startup_lock = threading.Lock()

        def lane_loop(lane: _EvalLane) -> None:
            while lane.healthy:
                try:
                    case_index, case = work_queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    lane.healthy = self._process_case(lane, case, case_index, results_lock, case_results, infra_errors)
                finally:
                    work_queue.task_done()

        def initialize_and_run_lane(index: int, worker: DockerWorkerManager) -> None:
            with startup_lock:
                # Ready lanes may have already claimed every case while this thread
                # waited. Avoid booting an unnecessary VM.
                if work_queue.empty():
                    return
                if not self._start_worker(worker, start_container):
                    with results_lock:
                        infra_errors.append({
                            "stage": "start",
                            "container": worker.config.container_name,
                            "error": "unhealthy after recovery",
                        })
                    return
                if not self._deploy_worker(worker, infra_errors, results_lock):
                    return
                lane = _EvalLane(index, worker, self.make_gui_agent(worker, lane_index=index))
                with lanes_lock:
                    lanes.append(lane)
            # Do not hold startup_lock while executing cases: GUI work remains fully
            # parallel and the next lane can start/deploy in the background.
            lane_loop(lane)

        workers_to_start = self.workers[: min(len(self.workers), len(cases))]
        threads = [
            threading.Thread(
                target=initialize_and_run_lane,
                args=(index, worker),
                name=f"eval-lane-{index}",
            )
            for index, worker in enumerate(workers_to_start)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        summary = summarize_results(self.version, self.style_id, self.variant_dir, case_results, infra_errors)
        summary["eval_window"] = self.eval_window
        lanes.sort(key=lambda lane: lane.index)
        summary["lanes"] = [
            {"index": lane.index, "container": lane.worker.config.container_name, "healthy": lane.healthy}
            for lane in lanes
        ]
        write_json(self.eval_dir / "summary.json", summary)
        for lane in lanes:
            tag = "" if lane.index == 0 else f".{lane.worker.config.container_name}"
            write_json(self.eval_dir / f"worker_state.final{tag}.json", lane.worker.snapshot_state())
        write_cost(self.eval_dir, model=os.environ.get("GUIAGENT_MODEL") or os.environ.get("APPEVAL_LLM_MODEL") or os.environ.get("MODEL"))
        update_variant_stage(
            self.variant_dir,
            self.version,
            "gui_eval",
            "completed" if summary.get("overall_status") == "Pass" else "failed",
            summary_path=str(self.eval_dir / "summary.json"),
            failure_type="infra_or_case" if summary.get("overall_status") != "Pass" else "",
        )
        return self.eval_dir / "summary.json"

    def smoke_check(self, start_container: bool = True) -> Path:
        """Launch smoke gate: prove the built app actually starts inside OSWorld.

        No GUIAgent, no cases — bring the lane-0 container up, deploy the current
        ``coder/app``, launch it with NO arguments and run the readiness probe once.
        Writes ``launch_smoke/smoke.json`` with a status of
        ``Pass`` | ``AppLaunchError`` | ``InfraError`` and returns its path.

        AppLaunchError (app never mapped a window in a proven-healthy
        container/display) is a REAL launch bug that the caller feeds back to
        DevCoder; InfraError is environment noise the caller retries.
        """
        smoke_dir = self.version_dir / "launch_smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        update_variant_stage(self.variant_dir, self.version, "launch_smoke", "running", trace_path=str(smoke_dir))
        worker = self.worker
        infra_errors: List[Dict[str, Any]] = []
        lock = threading.Lock()

        def _finish(status: str, evidence: str, runtime_errors: Optional[List[Dict[str, Any]]] = None, stderr_tail: str = "") -> Path:
            payload = {
                "status": status,  # Pass | AppLaunchError | InfraError
                "version": self.version,
                "style_id": self.style_id,
                "software_type": self.software_type,
                "evidence": evidence,
                "app_crash": bool(runtime_errors),
                "runtime_errors": runtime_errors or [],
                "app_stderr_tail": stderr_tail,
                "checked_at": datetime.now().isoformat(timespec="seconds"),
                "repair_hint": (
                    "App failed to launch while the OSWorld container/display were healthy. Fix the "
                    "launch path so `release/run.sh` (no arguments) brings up a visible window: check the "
                    "run.sh entry/paths, missing runtime dependencies, any startup crash/traceback in "
                    "runtime_errors/app_stderr_tail, and ensure the window title contains appName."
                    if status == "AppLaunchError"
                    else ""
                ),
            }
            write_json(smoke_dir / "smoke.json", payload)
            try:
                write_json(smoke_dir / "worker_state.smoke.json", worker.snapshot_state())
            except Exception:  # noqa: BLE001 — telemetry only, never mask the gate result
                pass
            update_variant_stage(
                self.variant_dir,
                self.version,
                "launch_smoke",
                "completed" if status == "Pass" else "failed",
                summary_path=str(smoke_dir / "smoke.json"),
                failure_type="" if status == "Pass" else status,
            )
            return smoke_dir / "smoke.json"

        if self.dry_run:
            return _finish("Pass", "dry-run: launch smoke skipped")

        # Bring lane-0 online and deploy the current app (both self-recover once).
        if not self._start_worker(worker, start_container):
            return _finish("InfraError", "primary worker unhealthy after recovery")
        if not self._deploy_worker(worker, infra_errors, lock):
            return _finish("InfraError", f"deploy failed: {infra_errors[-1:] or 'unknown'}")

        # No-argument launch + single readiness probe (equivalent to open_case_app
        # without any per-case setup). wait_for_app_readiness already splits app
        # launch failure (AppLaunchError) from env failure (InfrastructureError).
        try:
            close_app(worker, self.software_type, self.remote_app_dir)
            self.ensure_display_ready("smoke_before_start", worker)
            start_app(worker, self.remote_app_dir, launch_args="")
            wait_for_app_readiness(worker, self.software_type, self.remote_app_dir)
        except AppLaunchError as exc:
            runtime_errors = self._read_unhandled_exceptions(worker)
            stderr_tail = self._capture_remote("tail -n 80 /tmp/softwaregym_app_stderr.log", worker=worker)
            try:
                close_app(worker, self.software_type, self.remote_app_dir)
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
            return _finish("AppLaunchError", str(exc), runtime_errors=runtime_errors, stderr_tail=stderr_tail)
        except InfrastructureError as exc:
            return _finish("InfraError", str(exc))

        try:
            close_app(worker, self.software_type, self.remote_app_dir)
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass
        return _finish("Pass", "app launched and a window mapped in a healthy container")

    def _process_case(
        self,
        lane: _EvalLane,
        case: Dict[str, Any],
        case_index: int,
        lock: threading.Lock,
        case_results: List[Dict[str, Any]],
        infra_errors: List[Dict[str, Any]],
    ) -> bool:
        """Run one case end-to-end on ``lane``. Returns False if the lane died."""
        worker = lane.worker
        case_id = str(case.get("id") or f"case_{case_index:03d}")
        final_case_result: Optional[Dict[str, Any]] = None
        attempt_records: List[Dict[str, Any]] = []
        attempt_index = 1
        semantic_attempts_used = 0
        infra_retries_used = 0
        lane_dead = False
        while semantic_attempts_used < self.case_attempts:
            trace_dir = self.eval_dir / "traces" / case_id / f"attempt_{attempt_index:02d}"
            attempt_index += 1
            trace_dir.mkdir(parents=True, exist_ok=True)
            pre_unhandled = 0
            try:
                if not worker.health_check_for_running_worker():
                    recovery = worker.recover(f"pre-case unhealthy: {case_id}")
                    self.sync_worker_state(f"after_pre_case_recovery_{case_id}", worker)
                    if not recovery.healthy:
                        raise InfrastructureError(recovery.message)
                    lane.cases_since_recovery = 0
                    self.deploy_current_app(worker)
                    lane.gui_agent = self.make_gui_agent(worker, lane_index=lane.index)
                try:
                    self.ensure_display_ready(f"pre_case_{case_id}_attempt_{attempt_index}", worker)
                except InfrastructureError as display_exc:
                    recovery = worker.recover(f"display unhealthy before {case_id}: {display_exc}", force_overlay_reset=True)
                    self.sync_worker_state(f"after_display_recovery_{case_id}", worker)
                    if not recovery.healthy:
                        raise InfrastructureError(recovery.message)
                    lane.cases_since_recovery = 0
                    self.deploy_current_app(worker)
                    lane.gui_agent = self.make_gui_agent(worker, lane_index=lane.index)
                pre_unhandled = len(self._read_unhandled_exceptions(worker))
                self.open_case_app(case, case_id, attempt_index, trace_dir, worker)
                try:
                    result = lane.gui_agent.run_case(case, trace_dir=trace_dir)
                finally:
                    self.close_case_app(case_id, attempt_index, trace_dir, worker)
                if result.status == "InfraError":
                    evidence = result.evidence
                    if not evidence and isinstance(result.raw_result, dict):
                        evidence = str(result.raw_result.get("evidence", ""))
                    raise InfrastructureError(evidence or "GUIAgent returned InfraError")
                infra_check_report = self.run_infra_checks(case, case_id, attempt_index, trace_dir, worker)
                effective_status = result.status
                failure_analysis = result.raw_result.get("failure_analysis", "") if isinstance(result.raw_result, dict) else ""
                if infra_check_report.get("applicable") and result.status == "Pass" and not infra_check_report.get("passed", True):
                    effective_status = "Fail"
                    failed_checks = [c for c in infra_check_report.get("checks", []) if not c.get("passed")]
                    note = "Protocol-boundary infra checks failed: " + json.dumps(failed_checks, ensure_ascii=False)
                    failure_analysis = (failure_analysis + " | " + note) if failure_analysis else note
                case_result = {
                    "case_id": case_id,
                    "status": effective_status,
                    "gui_status": result.status,
                    "infra_checks": infra_check_report,
                    "evidence": result.evidence,
                    "task_summary": result.raw_result.get("task_summary", "") if isinstance(result.raw_result, dict) else "",
                    "task_completion": result.raw_result.get("task_completion", {}) if isinstance(result.raw_result, dict) else {},
                    "step_results": result.raw_result.get("step_results", []) if isinstance(result.raw_result, dict) else [],
                    "checkpoints": result.checkpoints,
                    "final_observation": result.raw_result.get("final_observation", "") if isinstance(result.raw_result, dict) else "",
                    "failure_analysis": failure_analysis,
                    "blocker": result.blocker,
                    "repair_hint": result.repair_hint,
                    "trace_path": str(trace_dir.relative_to(self.eval_dir)),
                    "attempt": attempt_index,
                    "lane": lane.index,
                }
                # Collect uncaught exceptions (including stack traces) from the app during
                # this case to help later repair rounds pinpoint the crash root cause.
                # Use offset-based attribution per case; if the log was reset (negative
                # delta), fall back to the full set.
                post_unhandled = self._read_unhandled_exceptions(worker)
                case_result["runtime_errors"] = (
                    post_unhandled[pre_unhandled:]
                    if len(post_unhandled) >= pre_unhandled
                    else post_unhandled
                )
                result_payload = result.raw_result or case_result
                write_json(trace_dir / "RESULT_JSON", result_payload)
                trace_manifest = write_trace_bundle(trace_dir, case, result_payload, result.steps, self.eval_dir)
                case_result["trace_manifest_path"] = trace_manifest["trace_dir"] + "/trace_manifest.json"
                case_result["trajectory_path"] = trace_manifest["trace_dir"] + "/trajectory.json"
                case_result["screenshots"] = trace_manifest["screenshots"]
                attempt_records.append(case_result)
                final_case_result = case_result
                semantic_attempts_used += 1
                if result.status in {"Pass", "Fail"}:
                    break
                if result.blocker not in {"no_actions", "max_steps", "dry_run"}:
                    break
            except AppLaunchError as exc:
                # The app failed to start in a proven-healthy container/display — a real
                # launch bug, not environment noise. AppLaunchError is a separate status
                # (not folded into InfraError) and: no recovery, no overlay rebuild, no
                # retry — the app deterministically fails to start, so retrying/switching
                # VMs only wastes compute; the container remains healthy, so no recovery is
                # needed. First rescue uncaught exceptions logged by the app's own hook
                # during this case (in case the startup crash wrote a log) as diagnostic
                # evidence.
                rescued_post = self._read_unhandled_exceptions(worker)
                rescued_runtime_errors = (
                    rescued_post[pre_unhandled:]
                    if len(rescued_post) >= pre_unhandled
                    else rescued_post
                )
                try:
                    close_app(worker, self.software_type, self.remote_app_dir)
                except Exception:  # noqa: BLE001 — best-effort cleanup, never mask the launch failure
                    pass
                case_result = {
                    "case_id": case_id,
                    "status": "AppLaunchError",
                    "app_crash": bool(rescued_runtime_errors),
                    "evidence": str(exc),
                    "task_summary": (
                        "App launch failure: the app did NOT start even though the container/display were "
                        "proven healthy. This is a REAL app bug (broken release/run.sh entry, missing runtime "
                        "dependency, startup crash/traceback, or a window that never maps), not infra noise."
                    ),
                    "task_completion": {"completed": False, "reason": "app_launch_failed"},
                    "step_results": [],
                    "checkpoints": [],
                    "final_observation": "",
                    "failure_analysis": str(exc),
                    "blocker": "app_launch",
                    "repair_hint": (
                        "The app failed to launch while the OSWorld container/display were healthy. Fix the "
                        "launch path so `release/run.sh` (no arguments) brings up a visible window: check the "
                        "run.sh entry/paths, missing runtime dependencies, any startup crash/traceback in "
                        "runtime_errors, and ensure the window title contains appName. Do NOT dismiss this as "
                        "infrastructure noise."
                    ),
                    "trace_path": str(trace_dir.relative_to(self.eval_dir)),
                    "attempt": attempt_index - 1,
                    "lane": lane.index,
                    "runtime_errors": rescued_runtime_errors,
                }
                write_json(trace_dir / "RESULT_JSON", case_result)
                trace_manifest = write_trace_bundle(trace_dir, case, case_result, [], self.eval_dir)
                case_result["trace_manifest_path"] = trace_manifest["trace_dir"] + "/trace_manifest.json"
                case_result["trajectory_path"] = trace_manifest["trace_dir"] + "/trajectory.json"
                attempt_records.append(case_result)
                final_case_result = case_result
                semantic_attempts_used += 1
                break
            except InfrastructureError as exc:
                # Rescue exception evidence: worker.recover / deploy_current_app will clear
                # log.json inside the container, so read out uncaught exceptions already
                # logged during this case first, to avoid losing evidence of a real crash
                # that might be misclassified as infra.
                rescued_post = self._read_unhandled_exceptions(worker)
                rescued_runtime_errors = (
                    rescued_post[pre_unhandled:]
                    if len(rescued_post) >= pre_unhandled
                    else rescued_post
                )
                recovery = worker.recover(f"case infra failure {case_id}: {exc}")
                self.sync_worker_state(f"after_case_recovery_{case_id}", worker)
                will_retry = recovery.healthy and infra_retries_used < self.infra_case_retries
                # Key discriminator: rescued_runtime_errors come from unhandled_exception
                # records written by the app's own global exception hook to log.json. If
                # non-empty, the app really threw an uncaught exception during startup/run
                # (e.g., startup crash), which is a real app bug, not evaluation-environment
                # noise — labels/hints must be written honestly so repair rounds do not
                # filter it as infra and leave it unfixed forever.
                app_crashed = bool(rescued_runtime_errors)
                case_result = {
                    "case_id": case_id,
                    "status": "InfraError",
                    "app_crash": app_crashed,
                    "evidence": str(exc),
                    "task_summary": (
                        "App crash: the app's own unhandled-exception hook logged a crash during startup/run (see runtime_errors). This is a REAL app bug, not environment noise."
                        if app_crashed
                        else "Infrastructure failure while executing GUI case."
                    ),
                    "task_completion": {"completed": False, "reason": "app_crash" if app_crashed else "infrastructure"},
                    "step_results": [],
                    "checkpoints": [],
                    "final_observation": "",
                    "failure_analysis": str(exc),
                    "blocker": "app_crash" if app_crashed else "infrastructure",
                    "repair_hint": (
                        "App logged unhandled exception(s) during startup/run (see runtime_errors: exceptionType/message/stackTrace/source). Fix the crashing code path and harden the global exception hooks so the GUI no longer dies on launch. Do NOT dismiss this as infra noise."
                        if app_crashed
                        else "Do not feed this as app bug; rerun after Docker/OSWorld recovery."
                    ),
                    "trace_path": str(trace_dir.relative_to(self.eval_dir)),
                    "attempt": attempt_index - 1,
                    "lane": lane.index,
                    "runtime_errors": rescued_runtime_errors,
                    "will_retry_current_case": will_retry,
                    "infra_retries_used": infra_retries_used,
                    "infra_case_retries": self.infra_case_retries,
                    "recovery": recovery.__dict__,
                }
                write_json(trace_dir / "RESULT_JSON", case_result)
                trace_manifest = write_trace_bundle(trace_dir, case, case_result, [], self.eval_dir)
                case_result["trace_manifest_path"] = trace_manifest["trace_dir"] + "/trace_manifest.json"
                case_result["trajectory_path"] = trace_manifest["trace_dir"] + "/trajectory.json"
                attempt_records.append(case_result)
                final_case_result = case_result
                if not recovery.healthy:
                    with lock:
                        infra_errors.append({"case_id": case_id, "container": worker.config.container_name, "error": str(exc), "recovery": recovery.__dict__})
                    lane_dead = True
                    break
                lane.cases_since_recovery = 0
                if will_retry:
                    infra_retries_used += 1
                    self.deploy_current_app(worker)
                    lane.gui_agent = self.make_gui_agent(worker, lane_index=lane.index)
                    continue
                semantic_attempts_used += 1
                self.deploy_current_app(worker)
        if final_case_result is not None:
            final_case_result["attempts"] = [clone_jsonable(record) for record in attempt_records]
            with lock:
                case_results.append(final_case_result)
            lane.cases_since_recovery += 1
        if lane_dead:
            return False
        batch_recovery = worker.after_case_batch(lane.cases_since_recovery)
        if not batch_recovery.healthy:
            with lock:
                infra_errors.append({"stage": "case_batch", "case_index": case_index, "container": worker.config.container_name, "recovery": batch_recovery.__dict__})
            return False
        if batch_recovery.action.startswith("restart"):
            self.sync_worker_state(f"after_batch_recovery_{case_index}", worker)
            self.deploy_current_app(worker)
            lane.gui_agent = self.make_gui_agent(worker, lane_index=lane.index)
            lane.cases_since_recovery = 0
        return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run software-gym GUI eval with Docker/OSWorld recovery.")
    parser.add_argument("--variant-dir", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--software-type", required=True)
    parser.add_argument("--style-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-start-container", action="store_true")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Launch smoke gate: deploy the built app and prove it starts in OSWorld, then exit (no cases).",
    )
    parser.add_argument("--case-attempts", type=int, default=int(os.environ.get("APPEVAL_CASE_ATTEMPTS", "1")))
    parser.add_argument("--case-trajectory-reruns", type=int, default=int(os.environ.get("APPEVAL_CASE_TRAJECTORY_RERUNS", "1")))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("APPEVAL_MAX_STEPS", "60")))
    parser.add_argument("--max-cases", type=int, default=int(os.environ["APPEVAL_MAX_CASES"]) if os.environ.get("APPEVAL_MAX_CASES") else None)
    parser.add_argument("--infra-case-retries", type=int, default=int(os.environ.get("APPEVAL_INFRA_CASE_RETRIES", "3")))
    parser.add_argument(
        "--eval-window",
        type=int,
        default=int(os.environ.get("APPEVAL_EVAL_WINDOW", "1")),
        help="Number of OSWorld containers this style fans its cases across (sliding window).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    ensure_not_base_for_eval()
    activate_osworld_env_vars()
    load_env_file(REPO_ROOT / ".env")
    print(f"[env] python={sys.executable} conda_env={os.environ.get('CONDA_DEFAULT_ENV')}")
    args = build_parser().parse_args(argv)
    runner = EvalRunner(
        variant_dir=args.variant_dir,
        version=args.version,
        software_type=args.software_type,
        style_id=args.style_id,
        dry_run=args.dry_run,
        case_attempts=args.case_attempts,
        case_trajectory_reruns=args.case_trajectory_reruns,
        max_steps=args.max_steps,
        max_cases=args.max_cases,
        infra_case_retries=args.infra_case_retries,
        eval_window=args.eval_window,
    )
    if args.smoke_only:
        smoke_path = runner.smoke_check(start_container=not args.no_start_container)
        status = str(load_json(smoke_path).get("status") or "InfraError")
        print(smoke_path)
        # Distinct exit codes so the caller can tell app-launch bug (3) from env
        # noise (4) even if smoke.json is unreadable; smoke.json remains the primary signal.
        return 0 if status == "Pass" else (3 if status == "AppLaunchError" else 4)
    summary = runner.run(start_container=not args.no_start_container)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
