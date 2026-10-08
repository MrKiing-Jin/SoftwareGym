from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Optional


def check_round_ready(variant_dir: Path, version: str) -> List[str]:
    errors: List[str] = []
    try:
        round_number = int(version.lstrip("v"))
    except ValueError:
        return [f"invalid version: {version}"]
    if round_number <= 1:
        return errors
    previous = variant_dir / "loop" / f"v{round_number - 1}"
    required = [
        previous / "eval" / "summary.json",
        previous / "devcoder" / "feature_matrix.md",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"cannot resume {version}: missing {path}")
    return errors


def hard_clean_variant(variant_dir: Path) -> None:
    for path in [variant_dir / "loop", variant_dir / "05_eval_tasks" / "cases.json"]:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate or clean software-gym resume state.")
    parser.add_argument("--variant-dir", required=True, type=Path)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--hard-clean", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.hard_clean:
        hard_clean_variant(args.variant_dir)
        print(f"cleaned {args.variant_dir}")
        return 0
    errors = check_round_ready(args.variant_dir, args.version)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"{args.variant_dir} {args.version}: resume ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
