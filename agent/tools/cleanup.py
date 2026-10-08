from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.docker.worker_manager import DEFAULT_OVERLAY_DIR


REPO_ROOT = Path(__file__).resolve().parents[2]

# OSWorld/qemu-docker mounts an anonymous volume for every container (each VM image is ~17 GB).
# These anonymous volumes are often not reclaimed after the container is removed, so logical
# usage can grow without bound (observed once at ~600 GB). Anonymous volumes have 64-character
# hex names; named volumes (e.g. agent-reach-xhs-data / rag-anything_*) must be preserved.
_ANON_VOLUME_RE = re.compile(r"^[0-9a-f]{64}$")


def disk_usage(path: Path) -> str:
    if not path.exists():
        return "0"
    result = subprocess.run(["du", "-sh", str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return result.stdout.split()[0] if result.stdout else "unknown"


def list_runs(output_dir: Path) -> List[Dict[str, str]]:
    if not output_dir.exists():
        return []
    runs = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if path.is_dir():
            runs.append({"path": str(path), "size": disk_usage(path)})
    return runs


def list_overlays(overlay_dir: Path) -> List[Dict[str, str]]:
    if not overlay_dir.exists():
        return []
    return [{"path": str(path), "size": disk_usage(path)} for path in sorted(overlay_dir.glob("*.qcow2"))]


def remove_failed_containers(prefix: str = "osworld_") -> List[str]:
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    removed: List[str] = []
    for name in result.stdout.splitlines():
        if name.startswith(prefix):
            subprocess.run(["docker", "rm", "-f", name], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            removed.append(name)
    return removed


def _docker(*args: str) -> str:
    result = subprocess.run(
        ["docker", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.stdout or ""


def docker_volume_reclaimable() -> str:
    """Return the human-readable RECLAIMABLE value from the Local Volumes line of ``docker system df``."""
    out = _docker("system", "df")
    for line in out.splitlines():
        if line.lower().startswith("local volumes"):
            # Format: "Local Volumes  27  2  636.9GB  568.1GB (89%)"
            tail = line.split()
            # RECLAIMABLE is usually the second-to-last token (the last one is the percentage).
            return " ".join(tail[-2:]) if len(tail) >= 2 else "unknown"
    return "unknown"


def dangling_anonymous_volumes() -> List[str]:
    """Return anonymous volumes that are dangling (referenced by no container) and have a 64-hex name.

    Named volumes are never returned, even if dangling, to avoid deleting persistent data such as
    agent-reach / rag-anything volumes.
    """
    out = _docker("volume", "ls", "-qf", "dangling=true")
    return [name for name in out.splitlines() if _ANON_VOLUME_RE.match(name.strip())]


def prune_anonymous_volumes(apply: bool) -> Dict[str, Any]:
    """Reclaim leaked anonymous OSWorld VM volumes.

    With ``apply=False`` only candidate count and reclaimable space are reported (dry-run);
    with ``apply=True`` the selected volumes are actually deleted.
    Only anonymous volumes selected by :func:`dangling_anonymous_volumes` are removed;
    named volumes are always preserved.
    """
    before = docker_volume_reclaimable()
    candidates = dangling_anonymous_volumes()
    removed: List[str] = []
    if apply and candidates:
        # Delete one by one. Named volumes are not in the candidate set by construction; a single
        # failure (e.g. transiently in use) does not abort the rest.
        for name in candidates:
            res = subprocess.run(
                ["docker", "volume", "rm", name],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if res.returncode == 0:
                removed.append(name)
    return {
        "reclaimable_before": before,
        "candidates": len(candidates),
        "removed": removed,
        "reclaimable_after": docker_volume_reclaimable() if apply else before,
        "applied": apply,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and safely clean software-gym generated artifacts.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output")
    parser.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY_DIR)
    parser.add_argument("--keep-runs", type=int, default=5)
    parser.add_argument("--delete-old-dry-runs", action="store_true")
    parser.add_argument("--remove-failed-containers", action="store_true")
    parser.add_argument(
        "--prune-volumes",
        action="store_true",
        help="Delete leaked anonymous OSWorld VM volumes (dangling with a 64-hex name); named volumes are left untouched. Without this flag only a dry-run report is printed.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    runs = list_runs(args.output_dir)
    overlays = list_overlays(args.overlay_dir)
    deleted: List[str] = []
    if args.delete_old_dry_runs:
        for run in runs[args.keep_runs :]:
            path = Path(run["path"])
            if "dry" in path.name.lower() or "smoke" in path.name.lower():
                shutil.rmtree(path, ignore_errors=True)
                deleted.append(str(path))
    containers = remove_failed_containers() if args.remove_failed_containers else []
    # Volume cleanup: always produce a dry-run report by default so leaks are visible;
    # only actually delete when --prune-volumes is passed.
    volumes = prune_anonymous_volumes(apply=args.prune_volumes)
    result = {
        "runs": runs,
        "overlays": overlays,
        "deleted": deleted,
        "removed_containers": containers,
        "anonymous_volumes": volumes,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
