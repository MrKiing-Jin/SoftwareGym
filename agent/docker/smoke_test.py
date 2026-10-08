from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Optional

from agent.docker.worker_manager import (
    DEFAULT_BASE_IMAGE,
    DEFAULT_DOCKER_IMAGE,
    DEFAULT_OVERLAY_DIR,
    DockerWorkerManager,
    InfrastructureError,
    WorkerConfig,
)


def docker_image_exists(image: str) -> bool:
    result = subprocess.run(["docker", "image", "inspect", image], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return result.returncode == 0


def docker_pull(image: str) -> None:
    subprocess.run(["docker", "pull", image], check=True)


def run_smoke(state_dir: Path, base_image: Path, docker_image: str, pull: bool = False, dry_run: bool = False) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    config = WorkerConfig(
        software_type="SmokeApp",
        style_id="style_01",
        worker_name="smoke_style_01",
        container_name="osworld_smoke_style_01",
        api_port=5031,
        vnc_port=8031,
        http_port=8111,
        cdp_port=9251,
        overlay_path=DEFAULT_OVERLAY_DIR / "smoke_style_01.qcow2",
        base_image=base_image,
        docker_image=docker_image,
    )
    result = {
        "base_image": {"path": str(base_image), "exists": base_image.exists(), "readable": base_image.exists()},
        "docker_image": {"name": docker_image, "exists": dry_run or docker_image_exists(docker_image)},
        "health_checks": [],
        "novnc_url": config.novnc_url,
    }
    if not dry_run and pull and not result["docker_image"]["exists"]:  # type: ignore[index]
        docker_pull(docker_image)
        result["docker_image"]["exists"] = True  # type: ignore[index]
    manager = DockerWorkerManager(config, state_dir=state_dir, dry_run=dry_run)
    try:
        manager.start_container()
        for _ in range(3):
            result["health_checks"].append({"platform": manager.check_platform(), "execute": manager.check_execute()})  # type: ignore[index]
        result["healthy"] = all(item["platform"] and item["execute"] for item in result["health_checks"])  # type: ignore[index]
        result["worker"] = manager.snapshot_state()
    except InfrastructureError as exc:
        result["healthy"] = False
        result["error"] = str(exc)
    path = state_dir / "health.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a real OSWorld smoke test.")
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--base-image", type=Path, default=DEFAULT_BASE_IMAGE)
    parser.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    path = run_smoke(args.state_dir, args.base_image, args.docker_image, pull=args.pull, dry_run=args.dry_run)
    print(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return 0 if data.get("healthy") else 1


if __name__ == "__main__":
    raise SystemExit(main())
