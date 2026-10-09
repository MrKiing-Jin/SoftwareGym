from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from agent.runtime_env import env_for_osworld, softwaregym_python

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def update_worker_env(path: Path, updates: Dict[str, str]) -> None:
    values: Dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                values[key] = value
    values.update(updates)
    path.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")


def iter_variants(run_dir: Path, style_ids: Optional[Iterable[str]] = None) -> List[Path]:
    root = run_dir / "software_variants"
    allowed = set(style_ids or [])
    variants = sorted(path for path in root.glob("style_*") if path.is_dir())
    if allowed:
        variants = [path for path in variants if path.name in allowed]
    return variants


def launch_eval(
    run_dir: Path,
    variant_dir: Path,
    version: str,
    software_type: str,
    start_container: bool,
    dry_run: bool,
    case_attempts: int,
    infra_case_retries: int,
    max_steps: int,
) -> Dict[str, object]:
    run_dir = run_dir.resolve()
    variant_dir = variant_dir.resolve()
    daemon_dir = run_dir / "daemon" / datetime.now().strftime("%Y%m%d_%H%M%S") / variant_dir.name
    daemon_dir.mkdir(parents=True, exist_ok=True)
    log_path = daemon_dir / "eval.log"
    current_log_path = variant_dir / "docker" / f"{variant_dir.name}.current-log"
    pid_path = daemon_dir / "eval.pid"
    restart_env_path = daemon_dir / "restart.env"
    command = [
        softwaregym_python(),
        "-m",
        "agent.eval.run_eval",
        "--variant-dir",
        str(variant_dir),
        "--version",
        version,
        "--software-type",
        software_type,
        "--style-id",
        variant_dir.name,
        "--case-attempts",
        str(case_attempts),
        "--infra-case-retries",
        str(infra_case_retries),
        "--max-steps",
        str(max_steps),
    ]
    if not start_container:
        command.append("--no-start-container")
    if dry_run:
        command.append("--dry-run")
    restart_command = f"cd {shlex.quote(str(REPO_ROOT))} && {shlex.join(command)}"
    restart_env_path.write_text(
        "\n".join(
            [
                f"RUN_DIR={run_dir}",
                f"VARIANT_DIR={variant_dir}",
                f"VERSION={version}",
                f"RESTART_COMMAND={restart_command}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    current_log_path.write_text(str(log_path), encoding="utf-8")
    update_worker_env(
        variant_dir / "docker" / "worker.env",
        {
            "VARIANT_DIR": str(variant_dir),
            "PID_FILE": str(pid_path),
            "LOG_FILE": str(log_path),
            "CURRENT_LOG_FILE": str(current_log_path),
            "RESTART_ENV": str(restart_env_path),
            "RESTART_COMMAND": restart_command,
        },
    )
    env = env_for_osworld(os.environ.copy())
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    return {"variant": variant_dir.name, "pid": proc.pid, "log": str(log_path), "command": command}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch software-gym eval workers in parallel.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--software-type", default=None)
    parser.add_argument("--styles", default=None, help="Comma-separated style ids.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-start-container", action="store_true")
    parser.add_argument("--case-attempts", type=int, default=1)
    parser.add_argument("--infra-case-retries", type=int, default=int(os.environ.get("APPEVAL_INFRA_CASE_RETRIES", "3")))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("APPEVAL_MAX_STEPS", "50")))
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_json(args.run_dir / "manifest.json")
    software_type = args.software_type or str(manifest.get("software_type") or args.run_dir.name.split("_")[0])
    styles = [item.strip() for item in args.styles.split(",") if item.strip()] if args.styles else None
    launches = [
        launch_eval(
            args.run_dir,
            variant_dir,
            args.version,
            software_type,
            start_container=not args.no_start_container,
            dry_run=args.dry_run,
            case_attempts=args.case_attempts,
            infra_case_retries=args.infra_case_retries,
            max_steps=args.max_steps,
        )
        for variant_dir in iter_variants(args.run_dir, styles)
    ]
    launch_path = args.run_dir / "daemon" / "latest-launch.json"
    launch_path.parent.mkdir(parents=True, exist_ok=True)
    launch_path.write_text(json.dumps({"launches": launches}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"launches": launches, "launch_path": str(launch_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
