#!/usr/bin/env python3
"""Project and validate one style's generated SVG icons.

IconDesigner produces a style-agnostic semantic superset
(``02_icon_design/icon_manifest.json``). SubIconDesigner then renders one SVG per
superset icon for a single software-gym style under ``<variant>/04_icons``.

This module is the platform-side quality gate that runs as a post-command of the
SubIconDesigner stage. It:

1. Loads the shared semantic candidate superset.
2. When ``icon_slots.json`` is present, validates that every candidate is either
   selected with concrete GUI placements or explicitly omitted, then projects
   only the selected icons into the authoritative per-style manifest. Legacy
   runs without an icon-slot contract retain the old full-superset projection.
3. Uses the in-repo validator (``agent.icons.svg_contract``) to check that every
   expected SVG exists and is valid/safe at the right size.
4. Writes the projected manifest and a JSON report, and exits non-zero when the
   icon set is incomplete or invalid so the runner retries the stage.

Only the Python standard library plus the in-repo ``svg_contract`` module is
used, so this gate has no external skill dependency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from agent.icons.svg_contract import (
    CATEGORY_DIRS,
    DEFAULT_SIZES,
    ManifestError,
    validate_manifest,
    validate_manifest_outputs,
)

ICON_SLOT_CONTRACT_VERSION = 1


class IconSlotError(ValueError):
    """Raised when a SubStyleDesigner icon-slot contract is inconsistent."""


def style_segment_for(style_id: str) -> str:
    """Map a software-gym style id to a kebab-case icon style segment.

    The skill validator enforces kebab-case style names, so ``style_04`` becomes
    ``style-04``.
    """
    return style_id.strip().replace("_", "-")


def _canonical_size(category: str, icon: Dict[str, Any], default_sizes: Dict[str, Any]) -> tuple[int, int]:
    """Force fixed-size categories to their canonical canvas; keep illustration's
    declared size when present and sane, otherwise fall back to the default."""
    default_w, default_h = default_sizes[category]
    if category == "illustration":
        width = icon.get("width")
        height = icon.get("height")
        if isinstance(width, int) and isinstance(height, int) and 120 <= width <= 1024 and 80 <= height <= 1024:
            return width, height
        return int(default_w), int(default_h)
    return int(default_w), int(default_h)


def validate_icon_slots(
    icon_slots: Dict[str, Any],
    superset: Dict[str, Any],
    style_id: str,
) -> Dict[tuple[str, str], Dict[str, Any]]:
    """Validate a style's GUI icon placements and return selected icons by key."""
    errors: List[str] = []
    if icon_slots.get("contract_version") != ICON_SLOT_CONTRACT_VERSION:
        errors.append(f"contract_version must be {ICON_SLOT_CONTRACT_VERSION}")
    declared_style = icon_slots.get("style_id")
    if declared_style != style_id:
        errors.append(f"style_id must be {style_id!r}, got {declared_style!r}")

    superset_keys: set[tuple[str, str]] = set()
    for index, icon in enumerate(superset.get("icons") or []):
        if not isinstance(icon, dict):
            continue
        category = str(icon.get("category") or "")
        name = str(icon.get("name") or "")
        if category and name:
            key = (category, name)
            if key in superset_keys:
                errors.append(f"superset duplicates {category}/{name}")
            superset_keys.add(key)

    selected_raw = icon_slots.get("icons")
    omitted_raw = icon_slots.get("omitted")
    if not isinstance(selected_raw, list) or not selected_raw:
        errors.append("icons must be a non-empty array")
        selected_raw = []
    if not isinstance(omitted_raw, list):
        errors.append("omitted must be an array")
        omitted_raw = []

    selected: Dict[tuple[str, str], Dict[str, Any]] = {}
    for index, item in enumerate(selected_raw):
        prefix = f"icons[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        category = str(item.get("category") or "")
        name = str(item.get("name") or "")
        key = (category, name)
        if key not in superset_keys:
            errors.append(f"{prefix} references unknown candidate {category}/{name}")
        if key in selected:
            errors.append(f"{prefix} duplicates selected icon {category}/{name}")
        placements = item.get("placements")
        if not isinstance(placements, list) or not placements:
            errors.append(f"{prefix}.placements must be a non-empty array")
            placements = []
        normalized_placements: List[Dict[str, str]] = []
        for placement_index, placement in enumerate(placements):
            placement_prefix = f"{prefix}.placements[{placement_index}]"
            if not isinstance(placement, dict):
                errors.append(f"{placement_prefix} must be an object")
                continue
            normalized: Dict[str, str] = {}
            for field in ("surface", "control_name", "placement"):
                value = placement.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{placement_prefix}.{field} must be a non-empty string")
                else:
                    normalized[field] = value.strip()
            if len(normalized) == 3:
                normalized_placements.append(normalized)
        selected[key] = {
            "required": bool(item.get("required", True)),
            "placements": normalized_placements,
            "rationale": str(item.get("rationale") or "").strip(),
        }

    omitted: set[tuple[str, str]] = set()
    for index, item in enumerate(omitted_raw):
        prefix = f"omitted[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        category = str(item.get("category") or "")
        name = str(item.get("name") or "")
        key = (category, name)
        if key not in superset_keys:
            errors.append(f"{prefix} references unknown candidate {category}/{name}")
        if key in omitted:
            errors.append(f"{prefix} duplicates omitted icon {category}/{name}")
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{prefix}.reason must be a non-empty string")
        omitted.add(key)

    overlap = sorted(set(selected) & omitted)
    if overlap:
        errors.append(f"icons and omitted overlap: {overlap}")
    unaccounted = sorted(superset_keys - set(selected) - omitted)
    if unaccounted:
        errors.append(f"candidate icons are neither selected nor omitted: {unaccounted}")
    if errors:
        raise IconSlotError("\n".join(errors))
    return selected


def project_superset_to_style_manifest(
    superset: Dict[str, Any],
    style_id: str,
    *,
    category_dirs: Dict[str, str],
    default_sizes: Dict[str, Any],
    icon_slots: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a per-style manifest from selected GUI icon slots.

    Colors are irrelevant to SVG validation (validate_svg only checks XML, size,
    viewBox, and safety), so we inject valid placeholder hex colors purely to
    satisfy the manifest schema validator. Without ``icon_slots`` the function
    preserves the legacy behavior of projecting every superset icon.
    """
    segment = style_segment_for(style_id)
    project = superset.get("project") or {}
    design_in = superset.get("design") or {}
    selected = validate_icon_slots(icon_slots, superset, style_id) if icon_slots is not None else None

    icons_out: List[Dict[str, Any]] = []
    for icon in superset.get("icons") or []:
        if not isinstance(icon, dict):
            continue
        category = str(icon.get("category") or "")
        name = str(icon.get("name") or "")
        slot = selected.get((category, name)) if selected is not None else None
        if selected is not None and slot is None:
            continue
        if category not in category_dirs or not name:
            # Leave malformed entries in place; validate_manifest will report them.
            width = icon.get("width") if isinstance(icon.get("width"), int) else 24
            height = icon.get("height") if isinstance(icon.get("height"), int) else 24
            path = f"{category_dirs.get(category, 'resources/icons/actions')}/{name or 'unnamed'}.svg"
        else:
            width, height = _canonical_size(category, icon, default_sizes)
            # The per-style 04_icons dir is already style-scoped, so paths live
            # directly under the category dir (no redundant per-style subfolder).
            path = f"{category_dirs[category]}/{name}.svg"
        projected_icon = {
            "category": category,
            "name": name,
            "purpose": icon.get("purpose") or f"{name} icon",
            "width": width,
            "height": height,
            "rationale": icon.get("rationale") or "Required by the shared icon superset.",
            "variants": [{"style": segment, "path": path}],
        }
        if slot is not None:
            projected_icon.update(
                {
                    "required": slot["required"],
                    "placements": slot["placements"],
                    "slot_rationale": slot["rationale"],
                }
            )
        icons_out.append(projected_icon)

    projected = {
        "manifest_version": 1,
        "project": {
            "name": str(project.get("name") or "Software Icons"),
            "slug": str(project.get("slug") or "software-icons"),
        },
        "design": {
            "style_summary": str(design_in.get("style_summary") or "Per-style icon set."),
            "primary_color": "#2563EB",
            "accent_color": "#22C55E",
            "background_color": "#EFF6FF",
            "notes": design_in.get("notes") if isinstance(design_in.get("notes"), list) else [],
        },
        "icons": icons_out,
    }
    if icon_slots is not None:
        projected.update(
            {
                "slot_contract_version": ICON_SLOT_CONTRACT_VERSION,
                "style_id": style_id,
                "candidate_icon_count": len(superset.get("icons") or []),
                "selected_icon_count": len(icons_out),
            }
        )
    return projected


def project_style_manifest_file(
    *,
    superset_path: Path,
    style_id: str,
    projected_path: Path,
    icon_slots_path: Path | None = None,
    require_icon_slots: bool = False,
) -> Dict[str, Any]:
    """Project and persist one style manifest for runner and CLI callers."""
    superset = json.loads(superset_path.read_text(encoding="utf-8"))
    if not isinstance(superset, dict):
        raise IconSlotError("superset manifest root must be a JSON object")

    icon_slots: Dict[str, Any] | None = None
    if icon_slots_path is not None and icon_slots_path.is_file():
        loaded_slots = json.loads(icon_slots_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_slots, dict):
            raise IconSlotError("icon_slots.json root must be a JSON object")
        icon_slots = loaded_slots
    elif require_icon_slots:
        raise IconSlotError(f"icon slot contract not found: {icon_slots_path}")

    projected = project_superset_to_style_manifest(
        superset,
        style_id,
        category_dirs=CATEGORY_DIRS,
        default_sizes=DEFAULT_SIZES,
        icon_slots=icon_slots,
    )
    projected_path.parent.mkdir(parents=True, exist_ok=True)
    projected_path.write_text(json.dumps(projected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return projected


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate one style's SVG icons against the shared semantic superset.",
    )
    parser.add_argument("--superset-manifest", required=True, help="Path to 02_icon_design/icon_manifest.json")
    parser.add_argument("--style-id", required=True, help="software-gym style id, e.g. style_04")
    parser.add_argument("--output-root", required=True, help="The style's 04_icons directory")
    parser.add_argument("--report", required=True, help="Where to write the JSON validation report")
    parser.add_argument(
        "--icon-slots",
        default=None,
        help="Optional SubStyleDesigner icon_slots.json; when present only selected GUI icons are projected",
    )
    parser.add_argument(
        "--require-icon-slots",
        action="store_true",
        help="Fail instead of using the legacy full-superset projection when icon_slots.json is missing",
    )
    parser.add_argument(
        "--projected-manifest",
        default=None,
        help="Optional path for the projected per-style manifest (defaults to <output-root>/icon_manifest.style.json)",
    )
    args = parser.parse_args(argv)

    superset_path = Path(args.superset_manifest).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    projected_path = (
        Path(args.projected_manifest).expanduser().resolve()
        if args.projected_manifest
        else output_root / "icon_manifest.style.json"
    )
    segment = style_segment_for(args.style_id)

    def fail(stage: str, message: str) -> int:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {"ok": False, "stage": stage, "style_id": args.style_id, "error": message},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"icon validation failed ({stage}): {message}", file=sys.stderr)
        return 1

    icon_slots_path = Path(args.icon_slots).expanduser().resolve() if args.icon_slots else None
    if not superset_path.is_file():
        return fail("load-superset", f"superset manifest not found: {superset_path}")
    try:
        projected = project_style_manifest_file(
            superset_path=superset_path,
            style_id=args.style_id,
            projected_path=projected_path,
            icon_slots_path=icon_slots_path,
            require_icon_slots=args.require_icon_slots,
        )
    except (OSError, json.JSONDecodeError, IconSlotError) as exc:
        return fail("project-style-manifest", str(exc))

    try:
        manifest = validate_manifest(projected, requested_styles=[segment])
    except ManifestError as exc:
        return fail("validate-manifest", f"projected manifest invalid: {exc}")

    result = validate_manifest_outputs(manifest=manifest, output_root=output_root)
    payload = result.to_jsonable()
    payload.update(
        {
            "style_id": args.style_id,
            "style_segment": segment,
            "superset_manifest": str(superset_path),
            "icon_slots": str(icon_slots_path) if icon_slots_path else None,
            "slot_contract_version": projected.get("slot_contract_version"),
            "projected_manifest": str(projected_path),
            "output_root": str(output_root),
            "expected_icon_count": len(manifest.icons),
            "generated_count": len(result.generated),
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not result.ok:
        print(
            f"icon validation failed for {args.style_id}: {len(result.issues)} issue(s); see {report_path}",
            file=sys.stderr,
        )
        return 1

    print(
        f"icon validation ok for {args.style_id}: {len(result.generated)} icons under {output_root}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
