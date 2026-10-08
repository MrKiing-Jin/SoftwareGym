#!/usr/bin/env python3
"""Validate GUI icon slots and project one style's authoritative icon manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# This script is invoked by absolute path after Codex may have changed cwd to a
# SubStyleDesigner work directory. Make the repository package importable.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.icons.validate_style_icons import IconSlotError, project_style_manifest_file


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate icon_slots.json and project icon_manifest.style.json.",
    )
    parser.add_argument("--superset-manifest", required=True)
    parser.add_argument("--icon-slots", required=True)
    parser.add_argument("--style-id", required=True)
    parser.add_argument("--projected-manifest", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    superset_path = Path(args.superset_manifest).expanduser().resolve()
    icon_slots_path = Path(args.icon_slots).expanduser().resolve()
    projected_path = Path(args.projected_manifest).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()

    def write_report(payload: Dict[str, Any]) -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    try:
        projected = project_style_manifest_file(
            superset_path=superset_path,
            style_id=args.style_id,
            projected_path=projected_path,
            icon_slots_path=icon_slots_path,
            require_icon_slots=True,
        )
    except (OSError, json.JSONDecodeError, IconSlotError) as exc:
        write_report(
            {
                "ok": False,
                "style_id": args.style_id,
                "icon_slots": str(icon_slots_path),
                "error": str(exc),
            }
        )
        print(f"icon slot projection failed for {args.style_id}: {exc}", file=sys.stderr)
        return 1

    write_report(
        {
            "ok": True,
            "style_id": args.style_id,
            "icon_slots": str(icon_slots_path),
            "projected_manifest": str(projected_path),
            "candidate_icon_count": projected.get("candidate_icon_count"),
            "selected_icon_count": projected.get("selected_icon_count"),
        }
    )
    print(
        f"icon slot projection ok for {args.style_id}: "
        f"{projected.get('selected_icon_count')}/{projected.get('candidate_icon_count')} candidates selected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
