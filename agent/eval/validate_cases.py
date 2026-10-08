"""Validate generated eval cases against the app's REAL runtime-state schema.

Second safety net for the layer-2 contract protocol: after EvalTaskGenerate writes
cases.json, drop any ``infra_checks`` entry whose ``field`` / ``table`` does not exist
in this app's actual ``--dump-state`` / SQLite (captured by state_probe). A guessed but
wrong assertion would otherwise deterministically fail — and downgrade — an otherwise
passing contract case (the #1 false-negative source). Pruning it leaves the case judged
by its remaining checks and GUI checkpoints.

This is descriptive and per-app: it never constrains how an app names its state, it only
removes assertions that provably cannot hold against the ground-truth snapshot.

CLI:
    python -m agent.eval.validate_cases --cases <cases.json> --schema <state_schema.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_cases(cases_path: Path, schema_path: Path) -> Dict[str, Any]:
    """Prune unsatisfiable infra_checks from cases.json in place.

    Returns a report dict. If the schema probe is unusable (app failed to dump state),
    nothing is pruned and legacy behaviour is preserved.
    """
    cases_path = Path(cases_path)
    schema_path = Path(schema_path)
    report: Dict[str, Any] = {"cases_path": str(cases_path), "dropped": [], "kept_checks": 0, "skipped": ""}

    try:
        schema = _load_json(schema_path)
    except Exception as exc:  # noqa: BLE001
        report["skipped"] = f"schema unreadable: {exc}"
        return report

    dump = schema.get("dump_state", {}) or {}
    sqlite = schema.get("sqlite", {}) or {}
    dump_ok = bool(dump.get("ok"))
    sqlite_ok = bool(sqlite.get("ok"))
    dump_fields = set((dump.get("paths") or {}).keys())
    sqlite_tables = set(sqlite.get("tables") or [])

    if not dump_ok and not sqlite_ok:
        report["skipped"] = "probe unusable (no dump_state, no sqlite); left cases untouched"
        return report

    try:
        doc = _load_json(cases_path)
    except Exception as exc:  # noqa: BLE001
        report["skipped"] = f"cases unreadable: {exc}"
        return report

    cases = doc if isinstance(doc, list) else doc.get("cases", [])
    changed = False
    for case in cases:
        checks = case.get("infra_checks")
        if not isinstance(checks, list) or not checks:
            continue
        kept: List[Dict[str, Any]] = []
        for chk in checks:
            if not isinstance(chk, dict):
                continue
            ctype = str(chk.get("type", "")).strip().lower()
            drop_reason = ""
            if ctype == "dump_state":
                # Only prune when we actually have a dump-state ground truth to check against.
                if dump_ok and str(chk.get("field", "")).strip() not in dump_fields:
                    drop_reason = f"field '{chk.get('field')}' absent from real --dump-state"
            elif ctype == "sqlite_count":
                if sqlite_ok and str(chk.get("table", "")).strip() not in sqlite_tables:
                    drop_reason = f"table '{chk.get('table')}' absent from real app.db"
            # path_glob and unknown types are runtime-resolved; keep them.
            if drop_reason:
                report["dropped"].append({
                    "case_id": case.get("id"),
                    "type": ctype,
                    "spec": chk,
                    "reason": drop_reason,
                })
                changed = True
            else:
                kept.append(chk)
        report["kept_checks"] += len(kept)
        if kept:
            case["infra_checks"] = kept
        else:
            case.pop("infra_checks", None)
            changed = True

    if changed:
        cases_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    report["changed"] = changed
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prune unsatisfiable infra_checks against the real state schema.")
    p.add_argument("--cases", required=True, type=Path)
    p.add_argument("--schema", required=True, type=Path)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_cases(args.cases, args.schema)
    if report.get("skipped"):
        print(f"skipped: {report['skipped']}")
    else:
        print(f"kept_checks={report['kept_checks']} dropped={len(report['dropped'])} changed={report.get('changed')}")
        for d in report["dropped"]:
            print(f"  - [{d['case_id']}] {d['type']}: {d['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
