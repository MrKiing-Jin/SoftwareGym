"""Self-contained icon manifest + SVG validation contract.

This module vendors the parts of the external ``synthesize-software-icons``
skill (``icon_pipeline.manifest`` and ``icon_pipeline.validate_svg``) that the
SubIconDesigner quality gate relies on, so the pipeline no longer imports that
skill at runtime. Only the Python standard library is used.

What it provides:

* ``CATEGORY_DIRS`` / ``DEFAULT_SIZES`` -- the icon category contract.
* ``ManifestError`` -- raised when a (projected) manifest violates the contract.
* ``validate_manifest`` -- schema/shape validation of a per-style manifest.
* ``validate_manifest_outputs`` -- on-disk SVG existence, dimension, viewBox and
  safety checks for every icon variant in a validated manifest.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


MANIFEST_VERSION = 1

CATEGORY_DIRS = {
    "app": "resources/icons/app",
    "action": "resources/icons/actions",
    "navigation": "resources/icons/navigation",
    "status": "resources/icons/status",
    "content-type": "resources/icons/content-types",
    "entity": "resources/icons/entities",
    "avatar": "resources/avatars/defaults",
    "illustration": "resources/illustrations",
}

DEFAULT_SIZES: dict[str, tuple[int, int]] = {
    "app": (512, 512),
    "action": (24, 24),
    "navigation": (24, 24),
    "status": (24, 24),
    "content-type": (24, 24),
    "entity": (24, 24),
    "avatar": (256, 256),
    "illustration": (360, 240),
}

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

FIXED_SIZE_CATEGORIES = {
    "app",
    "action",
    "navigation",
    "status",
    "content-type",
    "entity",
    "avatar",
}

FORBIDDEN_TAGS = {"script", "foreignObject"}
EXTERNAL_VALUE_RE = re.compile(r"(?:https?://|javascript:|data:)", re.IGNORECASE)


class ManifestError(ValueError):
    """Raised when a manifest does not match the pipeline contract."""


@dataclass(frozen=True)
class IconVariant:
    path: str
    style: str


@dataclass(frozen=True)
class IconSpec:
    category: str
    name: str
    purpose: str
    width: int
    height: int
    rationale: str
    variants: list[IconVariant]


@dataclass(frozen=True)
class IconManifest:
    project_name: str
    project_slug: str
    design: dict[str, Any]
    icons: list[IconSpec]
    raw: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return self.raw


# --------------------------------------------------------------------------- #
# Manifest validation
# --------------------------------------------------------------------------- #
def validate_manifest(
    data: dict[str, Any],
    requested_styles: list[str] | None = None,
) -> IconManifest:
    """Validate and normalize a manifest dictionary."""

    errors: list[str] = []

    if data.get("manifest_version") != MANIFEST_VERSION:
        errors.append(f"manifest_version must be {MANIFEST_VERSION}")

    project = _expect_dict(data, "project", errors)
    project_name = _expect_non_empty_str(project, "name", errors)
    project_slug = _expect_non_empty_str(project, "slug", errors)
    if project_slug and not NAME_RE.fullmatch(project_slug):
        errors.append("project.slug must be kebab-case")

    design = _expect_dict(data, "design", errors)
    for key in ("style_summary", "primary_color", "accent_color", "background_color"):
        value = _expect_non_empty_str(design, key, errors)
        if key.endswith("_color") and value and not HEX_COLOR_RE.fullmatch(value):
            errors.append(f"design.{key} must be a #RRGGBB color")
    if "notes" in design and not isinstance(design["notes"], list):
        errors.append("design.notes must be an array when provided")

    icons_value = data.get("icons")
    if not isinstance(icons_value, list) or not icons_value:
        errors.append("icons must be a non-empty array")
        icons_value = []

    icons: list[IconSpec] = []
    seen_paths: set[str] = set()
    seen_keys: set[tuple[str, str]] = set()
    for index, item in enumerate(icons_value):
        prefix = f"icons[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue

        category = _expect_non_empty_str(item, "category", errors, prefix)
        name = _expect_non_empty_str(item, "name", errors, prefix)
        purpose = _expect_non_empty_str(item, "purpose", errors, prefix)
        rationale = _expect_non_empty_str(item, "rationale", errors, prefix)
        width = _expect_int(item, "width", errors, prefix)
        height = _expect_int(item, "height", errors, prefix)
        variants_value = item.get("variants")

        if category and category not in CATEGORY_DIRS:
            errors.append(f"{prefix}.category must be one of {sorted(CATEGORY_DIRS)}")
        if name and not NAME_RE.fullmatch(name):
            errors.append(f"{prefix}.name must be kebab-case")
        if category and width and height:
            _validate_size(category, width, height, errors, prefix)
        if category and name:
            key = (category, name)
            if key in seen_keys:
                errors.append(f"{prefix} duplicates category/name pair: {category}/{name}")
            seen_keys.add(key)

        variants: list[IconVariant] = []
        if not isinstance(variants_value, list) or not variants_value:
            errors.append(f"{prefix}.variants must be a non-empty array")
            variants_value = []

        seen_styles: set[str] = set()
        for variant_index, variant_item in enumerate(variants_value):
            variant_prefix = f"{prefix}.variants[{variant_index}]"
            if not isinstance(variant_item, dict):
                errors.append(f"{variant_prefix} must be an object")
                continue
            style = _expect_non_empty_str(variant_item, "style", errors, variant_prefix)
            path = _expect_non_empty_str(variant_item, "path", errors, variant_prefix)

            if style:
                if not NAME_RE.fullmatch(style):
                    errors.append(f"{variant_prefix}.style must be kebab-case")
                if requested_styles is not None and style not in requested_styles:
                    errors.append(f"{variant_prefix}.style must be one of {requested_styles}")
                if style in seen_styles:
                    errors.append(f"{variant_prefix}.style duplicates {style}")
                seen_styles.add(style)
            if path:
                _validate_icon_path(category, name, style, path, errors, variant_prefix)
                if path in seen_paths:
                    errors.append(f"{variant_prefix}.path duplicates another icon path: {path}")
                seen_paths.add(path)
            if style and path:
                variants.append(IconVariant(style=style, path=path))

        if requested_styles is not None:
            missing = [style for style in requested_styles if style not in seen_styles]
            extra = [style for style in seen_styles if style not in requested_styles]
            if missing:
                errors.append(f"{prefix}.variants missing requested styles: {missing}")
            if extra:
                errors.append(f"{prefix}.variants includes unknown styles: {extra}")

        if category and name and purpose and rationale and width and height and variants:
            icons.append(
                IconSpec(
                    category=category,
                    name=name,
                    purpose=purpose,
                    width=width,
                    height=height,
                    rationale=rationale,
                    variants=variants,
                )
            )

    if errors:
        raise ManifestError("\n".join(errors))

    return IconManifest(
        project_name=project_name,
        project_slug=project_slug,
        design=design,
        icons=icons,
        raw=data,
    )


def _expect_dict(root: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    value = root.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key} must be an object")
        return {}
    return value


def _expect_non_empty_str(
    root: dict[str, Any],
    key: str,
    errors: list[str],
    prefix: str | None = None,
) -> str:
    label = f"{prefix}.{key}" if prefix else key
    value = root.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must be a non-empty string")
        return ""
    return value.strip()


def _expect_int(root: dict[str, Any], key: str, errors: list[str], prefix: str) -> int:
    value = root.get(key)
    if not isinstance(value, int):
        errors.append(f"{prefix}.{key} must be an integer")
        return 0
    return value


def _validate_size(category: str, width: int, height: int, errors: list[str], prefix: str) -> None:
    if width <= 0 or height <= 0:
        errors.append(f"{prefix}.width and {prefix}.height must be positive")
        return

    default_width, default_height = DEFAULT_SIZES[category]
    if category in FIXED_SIZE_CATEGORIES:
        if (width, height) != (default_width, default_height):
            errors.append(
                f"{prefix} size for {category} must be {default_width}x{default_height}"
            )
        return

    # Empty-state illustrations may vary, but keep them within UI asset bounds.
    if width < 120 or height < 80 or width > 1024 or height > 1024:
        errors.append(f"{prefix} illustration size is outside the allowed range")


def _validate_icon_path(
    category: str,
    name: str,
    style: str,
    path: str,
    errors: list[str],
    prefix: str,
) -> None:
    if "\\" in path:
        errors.append(f"{prefix}.path must use forward slashes")
        return

    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts:
        errors.append(f"{prefix}.path must be relative and cannot contain '..'")
    if pure.suffix != ".svg":
        errors.append(f"{prefix}.path must end with .svg")
    # The per-style output root is already style-scoped (one variant dir per
    # style), so icon paths live directly under the category dir without a
    # redundant per-style subfolder, i.e. ``<category-dir>/<name>.svg``.
    if category in CATEGORY_DIRS:
        expected_dir = CATEGORY_DIRS[category]
        if not path.startswith(f"{expected_dir}/"):
            errors.append(f"{prefix}.path must be under {expected_dir}/")
    if name and pure.name != f"{name}.svg":
        errors.append(f"{prefix}.path filename must be {name}.svg")


# --------------------------------------------------------------------------- #
# On-disk SVG validation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SvgValidationIssue:
    path: str
    message: str


@dataclass(frozen=True)
class SvgValidationResult:
    ok: bool
    generated: list[dict[str, Any]]
    issues: list[SvgValidationIssue]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "generated": self.generated,
            "issues": [
                {"path": issue.path, "message": issue.message} for issue in self.issues
            ],
        }


def validate_manifest_outputs(*, manifest: IconManifest, output_root: Path) -> SvgValidationResult:
    issues: list[SvgValidationIssue] = []
    generated: list[dict[str, Any]] = []

    root = output_root.resolve()
    for icon in manifest.icons:
        for variant in icon.variants:
            svg_path = (root / variant.path).resolve()
            if not _is_relative_to(svg_path, root):
                issues.append(SvgValidationIssue(variant.path, "path resolves outside output root"))
                continue
            if not svg_path.exists():
                issues.append(SvgValidationIssue(variant.path, "SVG file was not generated"))
                continue

            file_issues = _validate_svg_file(svg_path=svg_path, icon=icon, variant=variant)
            issues.extend(file_issues)
            if not file_issues:
                generated.append(
                    {
                        "path": variant.path,
                        "category": icon.category,
                        "name": icon.name,
                        "style": variant.style,
                        "width": icon.width,
                        "height": icon.height,
                    }
                )

    return SvgValidationResult(ok=not issues, generated=generated, issues=issues)


def _validate_svg_file(*, svg_path: Path, icon: IconSpec, variant: IconVariant) -> list[SvgValidationIssue]:
    issues: list[SvgValidationIssue] = []
    label = variant.path

    try:
        tree = ET.parse(svg_path)
    except ET.ParseError as exc:
        return [SvgValidationIssue(label, f"invalid XML: {exc}")]
    except OSError as exc:
        return [SvgValidationIssue(label, f"cannot read SVG: {exc}")]

    root = tree.getroot()
    if _local_name(root.tag) != "svg":
        issues.append(SvgValidationIssue(label, "root element must be <svg>"))
        return issues

    _validate_dimensions(root, icon, issues, label)
    _validate_view_box(root, icon, issues, label)
    _validate_safe_svg(root, issues, label)
    return issues


def _validate_dimensions(root: ET.Element, icon: IconSpec, issues: list[SvgValidationIssue], label: str) -> None:
    width = _numeric_attr(root, "width")
    height = _numeric_attr(root, "height")
    if width != icon.width:
        issues.append(SvgValidationIssue(label, f"width must be {icon.width}"))
    if height != icon.height:
        issues.append(SvgValidationIssue(label, f"height must be {icon.height}"))


def _validate_view_box(root: ET.Element, icon: IconSpec, issues: list[SvgValidationIssue], label: str) -> None:
    view_box = root.attrib.get("viewBox")
    if not view_box:
        issues.append(SvgValidationIssue(label, "viewBox is required"))
        return

    parts = view_box.replace(",", " ").split()
    if len(parts) != 4:
        issues.append(SvgValidationIssue(label, "viewBox must have four numbers"))
        return
    try:
        min_x, min_y, width, height = (float(part) for part in parts)
    except ValueError:
        issues.append(SvgValidationIssue(label, "viewBox contains non-numeric values"))
        return

    if (min_x, min_y) != (0.0, 0.0):
        issues.append(SvgValidationIssue(label, "viewBox must start at 0 0"))
    if width != float(icon.width) or height != float(icon.height):
        issues.append(SvgValidationIssue(label, f"viewBox must match {icon.width}x{icon.height}"))


def _validate_safe_svg(root: ET.Element, issues: list[SvgValidationIssue], label: str) -> None:
    for element in root.iter():
        tag = _local_name(element.tag)
        if tag in FORBIDDEN_TAGS:
            issues.append(SvgValidationIssue(label, f"<{tag}> is not allowed"))
        for attr_name, attr_value in element.attrib.items():
            if EXTERNAL_VALUE_RE.search(attr_value):
                issues.append(
                    SvgValidationIssue(label, f"external or unsafe reference in attribute {attr_name}")
                )


def _numeric_attr(root: ET.Element, name: str) -> int | None:
    value = root.attrib.get(name)
    if value is None:
        return None
    match = re.fullmatch(r"(\d+)(?:px)?", value.strip())
    if not match:
        return None
    return int(match.group(1))


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


__all__ = [
    "CATEGORY_DIRS",
    "DEFAULT_SIZES",
    "ManifestError",
    "IconManifest",
    "IconSpec",
    "IconVariant",
    "SvgValidationIssue",
    "SvgValidationResult",
    "validate_manifest",
    "validate_manifest_outputs",
]
