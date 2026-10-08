from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


def start_tunnel(vnc_port: int, state_dir: Path, timeout_seconds: int = 30) -> Dict[str, object]:
    state_dir.mkdir(parents=True, exist_ok=True)
    log_path = state_dir / "cloudflared.log"
    pid_path = state_dir / "cloudflared.pid"
    command = ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{vnc_port}"]
    log_handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    url = ""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        match = re.search(r"https://[-a-zA-Z0-9.]+\.trycloudflare\.com", text)
        if match:
            url = match.group(0)
            break
        if proc.poll() is not None:
            break
        time.sleep(1)
    result = {"pid": proc.pid, "url": url, "log_path": str(log_path), "command": command, "ok": bool(url)}
    (state_dir / "cloudflared.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def stop_tunnel_if_running(state_dir: Path, timeout_seconds: int = 5) -> bool:
    pid_path = state_dir / "cloudflared.pid"
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        pid_path.unlink(missing_ok=True)
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        return False
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pid_path.unlink(missing_ok=True)
            return True
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    pid_path.unlink(missing_ok=True)
    return True


def restart_tunnel_if_running(vnc_port: int, state_dir: Path, timeout_seconds: int = 30) -> Dict[str, object]:
    if not (state_dir / "cloudflared.pid").exists():
        return {"ok": True, "restarted": False, "reason": "no_existing_tunnel"}
    stopped = stop_tunnel_if_running(state_dir)
    result = start_tunnel(vnc_port, state_dir, timeout_seconds=timeout_seconds)
    result["restarted"] = True
    result["stopped_previous"] = stopped
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expose a noVNC port through cloudflared.")
    parser.add_argument("--vnc-port", required=True, type=int)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = start_tunnel(args.vnc_port, args.state_dir, args.timeout_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
