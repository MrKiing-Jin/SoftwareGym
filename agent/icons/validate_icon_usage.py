#!/usr/bin/env python3
"""Validate that a Coder/DevCoder-built app actually *uses* its style icons.

SubIconDesigner generates a per-style SVG set and ``validate_style_icons`` gates
that the icons *exist* and are valid. That existence gate says nothing about
whether the Coder wired the icons into the GUI: historically the Coder copies the
whole icon set into ``app/resources/icons`` (so it is bundled) but only references
a handful of them in ``MainWindow.axaml``, leaving most generated icons unused.

This module is the platform-side *usage* gate. It:

1. Loads the authoritative per-style manifest
   (``<variant>/04_icons/icon_manifest.style.json``) to learn which icons exist
   and what each is *for* (the ``purpose`` field).
2. Scans the delivered app's source (``.axaml`` / ``.cs`` / ``.csproj`` / ...)
   for real references to each icon's resource path.
3. Computes an overall + per-category usage ratio. Slot-aware manifests gate all
   icons explicitly marked ``required``; legacy manifests keep the historical
   category filter. Both modes default to a 90% threshold.
4. Writes a JSON report listing the still-unused icons together with their
   ``purpose`` (so a retry can wire the right icon to the right control), and
   exits non-zero when usage is below the threshold so the runner retries the
   stage.

Only the Python standard library is used, so this gate has no external
dependency and mirrors ``validate_style_icons`` as a stage post-command.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Directories that never contain source references and are expensive to scan.
SKIP_DIR_NAMES = {"bin", "obj", ".git", "node_modules", ".vs", "packages"}
# Text-like files that may reference an icon resource path.
SCAN_SUFFIXES = {
    ".axaml",
    ".xaml",
    ".cs",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".resx",
}
# Icon categories that are expected to appear as clickable/visible GUI affordances
# and are therefore counted toward the usage gate. Illustrations (empty-state art)
# are reported but not gated by default because their surfaces are sparser.
DEFAULT_GATE_CATEGORIES = "app,action,navigation,status,content-type,entity"
DEFAULT_MIN_USAGE_RATIO = 0.9

# A semantic Tag only counts when source code contains a real, invoked dynamic
# SVG resolver and the tagged XAML element is a control type handled by it.
_DYNAMIC_TAG_TEST = re.compile(r"""StartsWith\s*\(\s*["']icon:""", re.IGNORECASE)
_DYNAMIC_RENDERER_MARKERS = ("Avalonia.Svg", "SKSvg", "SvgIcon")
_RESOLVER_CLASS = re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)")
_SUPPORTED_CONTROL_PATTERNS = (
    re.compile(r"\bcase\s+(?:global::)?(?:[A-Za-z_][\w]*\.)*([A-Z][A-Za-z0-9_]*)\b"),
    re.compile(r"\bis\s+(?:global::)?(?:[A-Za-z_][\w]*\.)*([A-Z][A-Za-z0-9_]*)\s+[A-Za-z_]"),
)
_XAML_ICON_TAG = re.compile(
    r"""<(?P<control>[A-Za-z_][\w:.-]*)\b[^<>]*?\bTag\s*=\s*(?P<quote>["'])(?P<tag>icon:[^"']+)(?P=quote)""",
    re.DOTALL,
)
_CONTROL_BASES = {
    "ToggleButton": {"Button"},
    "RadioButton": {"ToggleButton", "Button"},
    "CheckBox": {"ToggleButton", "Button"},
    "MenuItem": {"HeaderedContentControl"},
    "TabItem": {"HeaderedContentControl"},
    "GroupBox": {"HeaderedContentControl"},
}


def _iter_source_documents(app_dir: Path) -> List[Tuple[Path, str]]:
    """Return every scannable GUI/runtime source document under ``app_dir``.

    The bundled icon/illustration asset trees are skipped: we only want *code*
    references to icons, not the SVG assets themselves.
    """
    documents: List[Tuple[Path, str]] = []
    resources_root = (app_dir / "resources").resolve()
    release_resources_root = (app_dir / "release" / "resources").resolve()
    for root, dirs, files in os.walk(app_dir):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
        resolved = root_path.resolve()
        # Skip the bundled asset trees (the SVGs live here; not references).
        if resolved == resources_root or resolved == release_resources_root:
            dirs[:] = []
            continue
        for name in files:
            path = root_path / name
            if path.suffix.lower() not in SCAN_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            documents.append((path, text.replace("\\", "/")))
    return documents


def _referenced_paths(corpus: str, expected_paths: Set[str]) -> Set[str]:
    return {p for p in expected_paths if p in corpus}


def _control_is_supported(control: str, supported: Set[str]) -> bool:
    control = control.split(":")[-1]
    return control in supported or bool(_CONTROL_BASES.get(control, set()) & supported)


def _dynamic_semantic_references(
    documents: List[Tuple[Path, str]],
    semantic_to_path: Dict[str, str],
) -> tuple[Set[str], Dict[str, Any]]:
    """Resolve real ``Tag="icon:..."`` bindings used by dynamic SVG injectors.

    Tags alone are not usage: generated apps sometimes annotate arbitrary
    TextBlocks or layout containers without ever rendering an image. We only
    accept a tag when an invoked resolver parses ``icon:``, constructs ``.svg``
    resources through an SVG renderer, and handles that XAML control type.
    """
    corpus = "\n".join(text for _, text in documents)
    resolver_documents = [(path, text) for path, text in documents if _DYNAMIC_TAG_TEST.search(text)]
    resolver_classes = {
        class_name
        for _, text in resolver_documents
        for class_name in _RESOLVER_CLASS.findall(text)
    }
    resolver_invoked = any(f"{class_name}." in corpus for class_name in resolver_classes)
    renderer_present = ".svg" in corpus and any(marker in corpus for marker in _DYNAMIC_RENDERER_MARKERS)
    resolver_detected = bool(resolver_documents and resolver_invoked and renderer_present)

    supported_controls: Set[str] = set()
    if resolver_detected:
        for _, text in resolver_documents:
            for pattern in _SUPPORTED_CONTROL_PATTERNS:
                supported_controls.update(pattern.findall(text))

    referenced: Set[str] = set()
    unsupported: List[Dict[str, str]] = []
    for path, text in documents:
        if path.suffix.lower() not in {".axaml", ".xaml"}:
            continue
        for match in _XAML_ICON_TAG.finditer(text):
            semantic_key = match.group("tag")
            expected_path = semantic_to_path.get(semantic_key)
            if expected_path is None:
                continue
            control = match.group("control").split(":")[-1]
            if resolver_detected and _control_is_supported(control, supported_controls):
                referenced.add(expected_path)
            else:
                unsupported.append(
                    {
                        "semantic_key": semantic_key,
                        "control_type": control,
                        "source": str(path),
                    }
                )

    return referenced, {
        "resolver_detected": resolver_detected,
        "resolver_classes": sorted(resolver_classes),
        "supported_control_types": sorted(supported_controls),
        "unsupported_bindings": unsupported,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate that the delivered app references its generated style icons.",
    )
    parser.add_argument("--style-manifest", required=True, help="Path to <variant>/04_icons/icon_manifest.style.json")
    parser.add_argument("--app-dir", required=True, help="Delivered app root to scan (e.g. loop/<v>/coder/app)")
    parser.add_argument("--report", required=True, help="Where to write the JSON usage report")
    parser.add_argument(
        "--min-usage-ratio",
        type=float,
        default=None,
        help=(
            "Minimum gated usage ratio. CLI overrides SOFTWAREGYM_ICON_MIN_USAGE_RATIO; "
            "both slot-aware and legacy manifests default to 0.9."
        ),
    )
    parser.add_argument(
        "--gate-categories",
        default=os.environ.get("SOFTWAREGYM_ICON_GATE_CATEGORIES", DEFAULT_GATE_CATEGORIES),
        help="Comma-separated icon categories counted toward the gate.",
    )
    parser.add_argument(
        "--max-unused-listed",
        type=int,
        default=int(os.environ.get("SOFTWAREGYM_ICON_MAX_UNUSED_LISTED", "60")),
        help="Cap on how many unused icons to enumerate in the report.",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.style_manifest).expanduser().resolve()
    app_dir = Path(args.app_dir).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    gate_categories = {c.strip() for c in args.gate_categories.split(",") if c.strip()}

    def write_report(payload: Dict[str, Any]) -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def fail(stage: str, message: str) -> int:
        write_report({"ok": False, "stage": stage, "error": message})
        print(f"icon usage validation failed ({stage}): {message}", file=sys.stderr)
        return 1

    # Escape hatch: allow operators to disable the gate without editing the pipeline.
    if os.environ.get("SOFTWAREGYM_ICON_USAGE_GATE", "1").strip().lower() in {"0", "off", "false", "no"}:
        write_report({"ok": True, "skipped": True, "reason": "SOFTWAREGYM_ICON_USAGE_GATE disabled"})
        print("icon usage validation skipped (SOFTWAREGYM_ICON_USAGE_GATE disabled)")
        return 0

    if not manifest_path.is_file():
        return fail("load-manifest", f"style manifest not found: {manifest_path}")
    if not app_dir.is_dir():
        return fail("load-app", f"app dir not found: {app_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return fail("load-manifest", f"cannot parse style manifest: {exc}")

    icons = manifest.get("icons") if isinstance(manifest, dict) else None
    if not isinstance(icons, list) or not icons:
        return fail("load-manifest", "style manifest has no icons")

    # Build the expected icon table: resource path -> {category, name, purpose}.
    expected: Dict[str, Dict[str, Any]] = {}
    for icon in icons:
        if not isinstance(icon, dict):
            continue
        category = str(icon.get("category") or "")
        name = str(icon.get("name") or "")
        variants = icon.get("variants") or []
        path = ""
        if isinstance(variants, list) and variants and isinstance(variants[0], dict):
            path = str(variants[0].get("path") or "")
        if not path:
            continue
        expected[path] = {
            "category": category,
            "name": name,
            "path": path,
            "semantic_key": f"icon:{category}/{name}",
            "purpose": str(icon.get("purpose") or ""),
            "required": bool(icon.get("required", True)),
            "placements": icon.get("placements") if isinstance(icon.get("placements"), list) else [],
        }

    if not expected:
        return fail("load-manifest", "no icon variant paths found in style manifest")

    documents = _iter_source_documents(app_dir)
    corpus = "\n".join(text for _, text in documents)
    direct_referenced = _referenced_paths(corpus, set(expected))
    semantic_to_path = {meta["semantic_key"]: path for path, meta in expected.items()}
    dynamic_referenced, dynamic_details = _dynamic_semantic_references(documents, semantic_to_path)
    referenced = direct_referenced | dynamic_referenced

    # Per-category tallies (over every category, for visibility).
    by_category: Dict[str, Dict[str, int]] = {}
    for path, meta in expected.items():
        cat = meta["category"] or "unknown"
        bucket = by_category.setdefault(cat, {"expected": 0, "referenced": 0})
        bucket["expected"] += 1
        if path in referenced:
            bucket["referenced"] += 1
    for bucket in by_category.values():
        exp = bucket["expected"]
        bucket["ratio"] = round(bucket["referenced"] / exp, 4) if exp else 0.0

    slot_aware = manifest.get("slot_contract_version") == 1
    env_ratio = os.environ.get("SOFTWAREGYM_ICON_MIN_USAGE_RATIO")
    min_usage_ratio = (
        args.min_usage_ratio
        if args.min_usage_ratio is not None
        else float(env_ratio) if env_ratio is not None
        else DEFAULT_MIN_USAGE_RATIO
    )

    # A slot-aware manifest is already the GUI designer's selected set, so every
    # required icon is gated regardless of category. Legacy manifests retain the
    # historical category filter because they contain the entire candidate pool.
    if slot_aware:
        gated_paths = [p for p, m in expected.items() if m["required"]]
        gate_mode = "required-icon-slots"
    else:
        gated_paths = [p for p, m in expected.items() if m["category"] in gate_categories]
        gate_mode = "legacy-category-ratio"
    gated_referenced = [p for p in gated_paths if p in referenced]
    gated_total = len(gated_paths)
    usage_ratio = round(len(gated_referenced) / gated_total, 4) if gated_total else 1.0

    unused_gated = sorted(
        (expected[p] for p in gated_paths if p not in referenced),
        key=lambda m: (m["category"], m["name"]),
    )

    ok = usage_ratio >= min_usage_ratio
    payload: Dict[str, Any] = {
        "ok": ok,
        "app_dir": str(app_dir),
        "style_manifest": str(manifest_path),
        "gate_categories": sorted(gate_categories),
        "gate_mode": gate_mode,
        "slot_contract_version": manifest.get("slot_contract_version"),
        "min_usage_ratio": min_usage_ratio,
        "usage_ratio": usage_ratio,
        "gated_expected": gated_total,
        "gated_referenced": len(gated_referenced),
        "total_expected": len(expected),
        "total_referenced": len(referenced),
        "direct_referenced": len(direct_referenced),
        "dynamic_referenced": len(dynamic_referenced - direct_referenced),
        "dynamic_resolver": {
            "detected": dynamic_details["resolver_detected"],
            "classes": dynamic_details["resolver_classes"],
            "supported_control_types": dynamic_details["supported_control_types"],
        },
        "unsupported_dynamic_bindings": dynamic_details["unsupported_bindings"][: args.max_unused_listed],
        "by_category": dict(sorted(by_category.items())),
        "unused_count": len(unused_gated),
        "unused_icons": unused_gated[: args.max_unused_listed],
    }
    write_report(payload)

    if not ok:
        print(
            f"icon usage below threshold: {len(gated_referenced)}/{gated_total} gated icons referenced "
            f"(ratio {usage_ratio:.4f} < {min_usage_ratio:.4f}); {len(unused_gated)} unused; see {report_path}",
            file=sys.stderr,
        )
        return 1

    print(
        f"icon usage ok: {len(gated_referenced)}/{gated_total} gated icons referenced "
        f"(ratio {usage_ratio:.4f} >= {min_usage_ratio:.4f})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
