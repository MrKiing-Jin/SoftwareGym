from __future__ import annotations

import argparse
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from agent.docker.worker_manager import DockerWorkerManager, derive_worker_config, load_worker_env


SUCCESS_MARKERS = ["Loop completed", "eval summary written", "summary.json"]
INFRA_FAILURE_MARKERS = [
    "OSWorld API not ready",
    "Connection reset by peer",
    "stable health check did not pass",
    "OSWorld recovery failed",
    "InfraError",
]


@dataclass
class WatchSpec:
    variant_dir: Path
    software_type: str
    style_id: str
    pid_file: Optional[Path] = None
    log_file: Optional[Path] = None
    current_log_file: Optional[Path] = None
    restart_command: Optional[str] = None


def read_pid(path: Optional[Path]) -> Optional[int]:
    if not path or not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def process_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_process_group(pid: Optional[int]) -> None:
    if not pid:
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def read_recent_log(path: Optional[Path], max_chars: int = 20000) -> str:
    if not path or not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def resolve_current_log(spec: WatchSpec) -> Optional[Path]:
    if spec.current_log_file and spec.current_log_file.exists():
        raw = spec.current_log_file.read_text(encoding="utf-8", errors="replace").strip()
        if raw:
            return Path(raw)
    return spec.log_file


class WorkerWatchdog:
    def __init__(
        self,
        spec: WatchSpec,
        check_interval_seconds: int = 3600,
        max_restarts: int = 3,
        unhealthy_threshold: int = 2,
        restart_on_failure: bool = True,
        restart_unhealthy_running: bool = True,
    ) -> None:
        self.spec = spec
        self.check_interval_seconds = check_interval_seconds
        self.max_restarts = max_restarts
        self.unhealthy_threshold = unhealthy_threshold
        self.restart_on_failure = restart_on_failure
        self.restart_unhealthy_running = restart_unhealthy_running
        self.restart_count = 0
        config = derive_worker_config(spec.variant_dir, spec.software_type, spec.style_id)
        config.max_restarts = max_restarts
        config.unhealthy_threshold = unhealthy_threshold
        self.worker = DockerWorkerManager(config, state_dir=spec.variant_dir / "docker")

    def check_once(self) -> Dict[str, object]:
        pid = read_pid(self.spec.pid_file)
        alive = process_alive(pid)
        log = read_recent_log(resolve_current_log(self.spec))
        completed = any(marker in log for marker in SUCCESS_MARKERS)
        infra_failed = any(marker in log for marker in INFRA_FAILURE_MARKERS)
        healthy = self.worker.health_check_for_running_worker()
        action = "none"

        if completed:
            return {"alive": alive, "healthy": healthy, "completed": True, "action": action}

        if infra_failed and self.restart_on_failure:
            action = self.restart("infra_failure_marker", pid)
        elif alive and not healthy and self.restart_unhealthy_running:
            if self.worker.consecutive_unhealthy >= self.unhealthy_threshold:
                action = self.restart("unhealthy_running_worker", pid)
            else:
                action = "wait_unhealthy_below_threshold"
        elif not alive and self.restart_on_failure:
            action = self.restart("process_dead", pid)

        return {
            "alive": alive,
            "healthy": healthy,
            "completed": completed,
            "infra_failed": infra_failed,
            "restart_count": self.restart_count,
            "action": action,
        }

    def restart(self, reason: str, pid: Optional[int]) -> str:
        if self.restart_count >= self.max_restarts:
            return f"give_up:{reason}"
        self.restart_count += 1
        kill_process_group(pid)
        recovery = self.worker.recover(reason)
        if self.spec.restart_command:
            # The command is intentionally explicit in the watch spec; this file
            # does not invent task restarts on its own.
            os.system(self.spec.restart_command)
            return f"restart_task_and_docker:{reason}:{recovery.action}"
        return f"restart_docker_only:{reason}:{recovery.action}"

    def loop(self) -> None:
        while True:
            result = self.check_once()
            print(result, flush=True)
            if result.get("completed"):
                return
            time.sleep(self.check_interval_seconds)


def parse_spec(path: Path) -> WatchSpec:
    values = load_worker_env(path)
    variant_dir = Path(values.get("VARIANT_DIR") or path.parent.parent)
    return WatchSpec(
        variant_dir=variant_dir,
        software_type=values.get("SOFTWARE_TYPE", "UnknownSoftware"),
        style_id=values.get("STYLE_ID", "style_01"),
        pid_file=Path(values["PID_FILE"]) if values.get("PID_FILE") else None,
        log_file=Path(values["LOG_FILE"]) if values.get("LOG_FILE") else None,
        current_log_file=Path(values["CURRENT_LOG_FILE"]) if values.get("CURRENT_LOG_FILE") else None,
        restart_command=values.get("RESTART_COMMAND") or None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch a software-gym Docker/GUI eval worker.")
    parser.add_argument("--worker-env", required=True, type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--check-interval-seconds", type=int, default=int(os.environ.get("CHECK_INTERVAL_SECONDS", "3600")))
    parser.add_argument("--max-restarts", type=int, default=int(os.environ.get("MAX_RESTARTS", "3")))
    parser.add_argument("--unhealthy-threshold", type=int, default=int(os.environ.get("UNHEALTHY_THRESHOLD", "2")))
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    watchdog = WorkerWatchdog(
        parse_spec(args.worker_env),
        check_interval_seconds=args.check_interval_seconds,
        max_restarts=args.max_restarts,
        unhealthy_threshold=args.unhealthy_threshold,
    )
    if args.once:
        print(watchdog.check_once())
    else:
        watchdog.loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
