from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def collect_status(run_dir: Path) -> Dict[str, Any]:
    manifest = load_json(run_dir / "manifest.json")
    variants = []
    for variant_dir in sorted((run_dir / "software_variants").glob("style_*")):
        variant_manifest = load_json(variant_dir / "variant_manifest.json")
        worker_env = variant_dir / "docker" / "worker.env"
        rounds = []
        for round_dir in sorted((variant_dir / "loop").glob("v*")):
            summary = load_json(round_dir / "eval" / "summary.json")
            rounds.append(
                {
                    "version": round_dir.name,
                    "summary_exists": bool(summary),
                    "overall_status": summary.get("overall_status"),
                    "cases_total": summary.get("cases_total"),
                    "cases_pass": summary.get("cases_pass"),
                    "cases_fail": summary.get("cases_fail"),
                    "cases_infra_error": summary.get("cases_infra_error"),
                }
            )
        variants.append(
            {
                "style_id": variant_dir.name,
                "status": variant_manifest.get("status"),
                "worker_env": str(worker_env),
                "rounds": rounds,
            }
        )
    return {"run_dir": str(run_dir), "software_type": manifest.get("software_type"), "variants": variants}


def print_markdown(status: Dict[str, Any]) -> None:
    print(f"# Status: {status.get('software_type')} ({status.get('run_dir')})")
    print()
    print("| variant | round | status | pass/fail/infra |")
    print("|---|---|---|---|")
    for variant in status.get("variants", []):
        for round_info in variant.get("rounds", []):
            counts = f"{round_info.get('cases_pass')}/{round_info.get('cases_fail')}/{round_info.get('cases_infra_error')}"
            print(f"| {variant['style_id']} | {round_info['version']} | {round_info.get('overall_status')} | {counts} |")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize software-gym run status.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    status = collect_status(args.run_dir)
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print_markdown(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
