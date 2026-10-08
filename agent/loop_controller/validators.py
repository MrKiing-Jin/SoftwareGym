from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.loop_controller.avalonia_blueprint import (
    report_matches_blueprint,
    validate_blueprint,
)


REQUIRED_FEATURE_SECTIONS = [
    "Software Startup and Overall Layout",
    "Feature Implementation Summary",
    "GUIAgent Evaluation Prompt",
]

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_cases(path: Path) -> List[str]:
    errors: List[str] = []
    if not path.exists():
        return errors
    data = load_json(path)
    cases = data.get("cases")
    if not isinstance(cases, list):
        return [f"{path}: top-level 'cases' must be a list"]
    required = {"id", "title", "goal", "initial_state", "gui_steps_hint", "checkpoints", "expected_result", "priority"}
    seen_ids: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            errors.append(f"{path}: cases[{index}] must be an object")
            continue
        case_id = str(case.get("id") or "")
        if not case_id:
            errors.append(f"{path}: cases[{index}] id must be non-empty")
        elif case_id in seen_ids:
            errors.append(f"{path}: duplicate case id {case_id}")
        seen_ids.add(case_id)
        missing = sorted(required - set(case))
        if missing:
            errors.append(f"{path}: cases[{index}] missing {missing}")
        if not isinstance(case.get("checkpoints"), list) or not case.get("checkpoints"):
            errors.append(f"{path}: cases[{index}].checkpoints must be a non-empty list")
        else:
            for checkpoint_index, checkpoint in enumerate(case["checkpoints"]):
                if not isinstance(checkpoint, dict):
                    errors.append(f"{path}: cases[{index}].checkpoints[{checkpoint_index}] must be an object")
                    continue
                if not checkpoint.get("name") or not checkpoint.get("criteria"):
                    errors.append(f"{path}: cases[{index}].checkpoints[{checkpoint_index}] requires name and criteria")
                if not (checkpoint.get("feature_ref") or checkpoint.get("source") or checkpoint.get("criteria")):
                    errors.append(f"{path}: cases[{index}].checkpoints[{checkpoint_index}] must be traceable to feature_matrix")
        if "allowed_shell_checks" in case and not isinstance(case.get("allowed_shell_checks"), list):
            errors.append(f"{path}: cases[{index}].allowed_shell_checks must be a list")
        if not case.get("priority"):
            errors.append(f"{path}: cases[{index}].priority must be set")
    return errors


def validate_feature_matrix(path: Path) -> List[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    errors: List[str] = []
    for section in REQUIRED_FEATURE_SECTIONS:
        if section not in text:
            errors.append(f"{path}: missing section '{section}'")
    required_columns = ["User Entry", "User Operation Path", "Data/State Changes", "Acceptance Points"]
    for column in required_columns:
        if column not in text:
            errors.append(f"{path}: missing feature table column '{column}'")
    return errors


def validate_run(run_dir: Path) -> List[str]:
    errors: List[str] = []
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return [f"{run_dir}: missing manifest.json"]
    manifest = load_json(manifest_path)
    variants = manifest.get("variants")
    if not isinstance(variants, dict) or not variants:
        errors.append(f"{manifest_path}: variants must be a non-empty object")
        return errors

    contracts = manifest.get("contracts")
    require_avalonia_blueprint = (
        isinstance(contracts, dict) and contracts.get("avalonia_blueprint") == 1
    )
    for style_id, variant in variants.items():
        if require_avalonia_blueprint:
            blueprint_dir = (
                run_dir / "03_substyle_design" / style_id / "ui-blueprint"
            )
            report_path = (
                blueprint_dir.parent / "avalonia_blueprint_validation_report.json"
            )
            blueprint_report = validate_blueprint(
                blueprint_dir,
                compile_probe="off",
            )
            errors.extend(
                f"{blueprint_dir}: {error}"
                for error in blueprint_report.get("errors", [])
            )
            if not report_matches_blueprint(report_path, blueprint_dir):
                errors.append(
                    f"{blueprint_dir}: missing, failed, or stale "
                    "avalonia_blueprint_validation_report.json"
                )
        rel = variant.get("variant_dir")
        if not rel:
            errors.append(f"{manifest_path}: variant {style_id} missing variant_dir")
            continue
        variant_dir = run_dir / rel
        errors.extend(validate_variant(variant_dir))
    return errors


def validate_variant(variant_dir: Path) -> List[str]:
    errors: List[str] = []
    for rel in [
        "variant_manifest.json",
        "selected_style/style_id.txt",
        "selected_style/design_route.md",
        "05_eval_tasks/prompt.rendered.md",
        "docker/worker.env",
    ]:
        if not (variant_dir / rel).exists():
            errors.append(f"{variant_dir}: missing {rel}")

    errors.extend(validate_cases(variant_dir / "05_eval_tasks" / "cases.json"))
    loop_dir = variant_dir / "loop"
    if not loop_dir.exists():
        errors.append(f"{variant_dir}: missing loop directory")
        return errors
    for version_dir in sorted(path for path in loop_dir.iterdir() if path.is_dir()):
        for rel in ["coder/prompt.rendered.md", "devcoder/prompt.rendered.md", "eval/eval_manifest.json"]:
            if not (version_dir / rel).exists():
                errors.append(f"{version_dir}: missing {rel}")
        errors.extend(validate_feature_matrix(version_dir / "devcoder" / "feature_matrix.md"))
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a software-gym pipeline run.")
    parser.add_argument("run_dir", type=Path)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    errors = validate_run(args.run_dir)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"{args.run_dir}: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
