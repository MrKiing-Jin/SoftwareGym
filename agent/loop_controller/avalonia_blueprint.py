"""Hard validation gate for newly generated Avalonia 11 UI blueprints.

The static checks are authoritative and always run.  The new-run pipeline also
requires an Avalonia XAML compiler probe to build a temporary project; missing
.NET/NuGet/Avalonia restore is a gate failure, not a pass.  The probe removes
only ``x:Class``/compiled-binding metadata and common code-behind event
attributes: those require application classes or handlers that a standalone
blueprint cannot provide.  Controls, properties, resources, styles, bindings,
and the visual tree are left intact, so platform-XAML errors still fail the
gate.  ``auto`` remains available for non-gating diagnostics, but only a
``passed`` compiler report can satisfy the pipeline completion contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Optional


CONTRACT_VERSION = 1
AVALONIA_NAMESPACE = "https://github.com/avaloniaui"
XAML_NAMESPACE = "http://schemas.microsoft.com/winfx/2006/xaml"
WPF_PRESENTATION_NAMESPACE = "http://schemas.microsoft.com/winfx/2006/xaml/presentation"
REQUIRED_BLUEPRINTS = ("AppStyle.axaml", "MainWindow.axaml")

_XAML_KEY = f"{{{XAML_NAMESPACE}}}Key"
_STATIC_RESOURCE_RE = re.compile(
    r"\{StaticResource\s+(?:ResourceKey\s*=\s*)?([^,}\s]+)",
    re.IGNORECASE,
)
_WPF_NAMED_STATIC_RESOURCE_RE = re.compile(
    r"\{StaticResource\s+ResourceKey\s*=",
    re.IGNORECASE,
)
_WPF_MARKUP_RE = re.compile(r"\{x:(?:Static|Type)\b", re.IGNORECASE)
_XMLNS_X_RE = re.compile(
    r"""\bxmlns:x\s*=\s*["']http://schemas\.microsoft\.com/winfx/2006/xaml["']""",
    re.IGNORECASE,
)
_NORMALIZED_XAML_ATTR_RE = re.compile(
    r"""\s+x:(?:Class|DataType|CompileBindings)\s*=\s*(?:"[^"]*"|'[^']*')""",
    re.IGNORECASE,
)

# WPF property/trigger forms which do not have Avalonia 11 equivalents in this
# shape.  Avalonia styles use Selector plus Classes/pseudo-classes.
_FORBIDDEN_ATTRIBUTES = {
    "Visibility",
    "Style",
    "BasedOn",
    "TargetType",
    "ResourceKey",
}
_FORBIDDEN_ELEMENTS = {
    "Trigger",
    "DataTrigger",
    "MultiTrigger",
    "MultiDataTrigger",
    "EventSetter",
    "Storyboard",
}
# Removing these handler attributes is the standalone-probe boundary.  Their
# behavior belongs to Coder's implementation; the blueprint gate still checks
# the complete control/property/resource surface.
_CODE_BEHIND_EVENTS = {
    "AttachedToVisualTree",
    "Click",
    "Closed",
    "Closing",
    "DetachedFromVisualTree",
    "DoubleTapped",
    "Drop",
    "GotFocus",
    "KeyDown",
    "KeyUp",
    "Loaded",
    "LostFocus",
    "Opened",
    "PointerEntered",
    "PointerExited",
    "PointerMoved",
    "PointerPressed",
    "PointerReleased",
    "SelectionChanged",
    "TextChanged",
    "Tapped",
}


def _local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1]


def _namespace(name: str) -> str:
    if name.startswith("{") and "}" in name:
        return name[1:].split("}", 1)[0]
    return ""


def _blueprint_sources(blueprint_dir: Path) -> list[Path]:
    return sorted(
        {
            *blueprint_dir.rglob("*.axaml"),
            *blueprint_dir.rglob("*.xaml"),
        },
        key=lambda path: path.relative_to(blueprint_dir).as_posix(),
    )


def blueprint_digest(blueprint_dir: Path) -> str:
    """Hash filenames and bytes so a stale success report cannot satisfy runner."""
    digest = hashlib.sha256()
    for path in _blueprint_sources(Path(blueprint_dir)):
        rel = path.relative_to(blueprint_dir).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(4, "big"))
        digest.update(rel)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def report_matches_blueprint(report_path: Path, blueprint_dir: Path) -> bool:
    if not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        current_digest = blueprint_digest(blueprint_dir)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    metadata_matches = (
        isinstance(report, dict)
        and report.get("contract_version") == CONTRACT_VERSION
        and report.get("ok") is True
        and report.get("source_digest") == current_digest
        and isinstance(report.get("compile_probe"), dict)
        and report["compile_probe"].get("status") == "passed"
    )
    if not metadata_matches:
        return False
    # Never let a handcrafted/stale JSON report replace the authoritative
    # static gate on runner resume/skip-existing paths.
    return bool(validate_blueprint(blueprint_dir, compile_probe="off").get("ok"))


def _cached_avalonia_version() -> Optional[str]:
    package_root = Path(
        os.environ.get("NUGET_PACKAGES", Path.home() / ".nuget" / "packages")
    )
    avalonia_root = package_root / "avalonia"
    if not avalonia_root.is_dir():
        return None

    def version_key(value: str) -> tuple[int, ...]:
        numbers = re.findall(r"\d+", value)
        return tuple(int(number) for number in numbers)

    versions = [
        path.name
        for path in avalonia_root.iterdir()
        if path.is_dir() and path.name.startswith("11.")
    ]
    return max(versions, key=version_key) if versions else None


def _dotnet_path() -> Optional[str]:
    located = shutil.which("dotnet")
    if located:
        return located
    candidates = []
    env_path = os.environ.get("SOFTWAREGYM_DOTNET")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.home() / ".local" / "dotnet8" / "dotnet")
    candidates.append(Path("/tmp/dotnet8/dotnet"))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _normalize_for_compile(text: str) -> str:
    normalized = _NORMALIZED_XAML_ATTR_RE.sub("", text)
    for event_name in sorted(_CODE_BEHIND_EVENTS):
        normalized = re.sub(
            rf"""\s+{re.escape(event_name)}\s*=\s*(?:"[^"]*"|'[^']*')""",
            "",
            normalized,
        )
    return normalized


def _run_compile_probe(
    blueprint_dir: Path,
    axaml_files: Iterable[Path],
    *,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    dotnet = _dotnet_path()
    cached_version = _cached_avalonia_version()
    # An exact cached version is deterministic and avoids network work.  On a
    # fresh host, NuGet's supported floating 11.* range resolves the latest
    # Avalonia 11 release without silently crossing to Avalonia 12.
    version = cached_version or os.environ.get(
        "SOFTWAREGYM_AVALONIA_PROBE_VERSION",
        "11.*",
    )
    base = {
        "mode": "normalized-standalone",
        "normalization": [
            "remove x:Class/x:DataType/x:CompileBindings",
            "remove common code-behind event-handler attributes",
            "preserve controls, properties, resources, styles, bindings, and visual tree",
        ],
        "dotnet": dotnet,
        "avalonia_version": version,
        "avalonia_version_source": "cache" if cached_version else "nuget-11.x",
    }
    if not dotnet:
        return {**base, "status": "skipped", "reason": "dotnet SDK not available"}

    with tempfile.TemporaryDirectory(prefix="softwaregym-avalonia-probe-") as tmp:
        root = Path(tmp)
        probe_dir = root / "Blueprints"
        probe_dir.mkdir()
        for source in axaml_files:
            relative = source.relative_to(blueprint_dir)
            destination = probe_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                _normalize_for_compile(source.read_text(encoding="utf-8")),
                encoding="utf-8",
            )

        project = root / "BlueprintProbe.csproj"
        project.write_text(
            "\n".join(
                [
                    '<Project Sdk="Microsoft.NET.Sdk">',
                    "  <PropertyGroup>",
                    "    <TargetFramework>net8.0</TargetFramework>",
                    "    <OutputType>Library</OutputType>",
                    "    <Nullable>enable</Nullable>",
                    "    <AvaloniaUseCompiledBindingsByDefault>false</AvaloniaUseCompiledBindingsByDefault>",
                    "  </PropertyGroup>",
                    "  <ItemGroup>",
                    f'    <PackageReference Include="Avalonia" Version="{version}" />',
                    "  </ItemGroup>",
                    "</Project>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["DOTNET_CLI_HOME"] = str(root / ".dotnet-home")
        env["DOTNET_NOLOGO"] = "1"
        try:
            restore = subprocess.run(
                [
                    dotnet,
                    "restore",
                    str(project),
                    "--ignore-failed-sources",
                    "--nologo",
                    "--verbosity",
                    "quiet",
                ],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {**base, "status": "skipped", "reason": f"restore unavailable: {exc}"}
        if restore.returncode != 0:
            return {
                **base,
                "status": "skipped",
                "reason": "Avalonia probe dependencies could not be restored",
                "output_tail": restore.stdout[-4000:],
            }

        try:
            build = subprocess.run(
                [
                    dotnet,
                    "build",
                    str(project),
                    "--no-restore",
                    "--nologo",
                    "--verbosity",
                    "minimal",
                ],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {**base, "status": "failed", "reason": f"build failed: {exc}"}
        return {
            **base,
            "status": "passed" if build.returncode == 0 else "failed",
            "returncode": build.returncode,
            "output_tail": build.stdout[-6000:],
        }


def validate_blueprint(
    blueprint_dir: Path,
    *,
    compile_probe: str = "required",
) -> dict[str, Any]:
    """Validate one new-run blueprint directory and return a JSON-safe report."""
    blueprint_dir = Path(blueprint_dir).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    files = _blueprint_sources(blueprint_dir) if blueprint_dir.is_dir() else []
    axaml_files = [path for path in files if path.suffix.lower() == ".axaml"]
    legacy_xaml = [path for path in files if path.suffix.lower() == ".xaml"]

    if not blueprint_dir.is_dir():
        errors.append(f"blueprint directory does not exist: {blueprint_dir}")
    for filename in REQUIRED_BLUEPRINTS:
        if not (blueprint_dir / filename).is_file():
            errors.append(f"missing required Avalonia blueprint: {filename}")
    if legacy_xaml:
        joined = ", ".join(path.relative_to(blueprint_dir).as_posix() for path in legacy_xaml)
        errors.append(f"new blueprint contract forbids .xaml files; use .axaml: {joined}")
    if not axaml_files:
        errors.append("no .axaml files found")

    resource_keys: set[str] = set()
    static_references: list[tuple[str, str]] = []
    parsed: dict[Path, ET.Element] = {}
    file_reports: list[dict[str, Any]] = []

    for path in axaml_files:
        relative = path.relative_to(blueprint_dir).as_posix()
        entry: dict[str, Any] = {"path": relative, "ok": True}
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"{relative}: cannot read UTF-8 XML: {exc}")
            entry["ok"] = False
            file_reports.append(entry)
            continue
        try:
            root = ET.fromstring(text)
        except ET.ParseError as exc:
            errors.append(f"{relative}: invalid XML: {exc}")
            entry["ok"] = False
            file_reports.append(entry)
            continue

        parsed[path] = root
        root_namespace = _namespace(root.tag)
        root_name = _local_name(root.tag)
        entry.update({"root": root_name, "namespace": root_namespace})
        if root_namespace != AVALONIA_NAMESPACE:
            errors.append(
                f"{relative}: root namespace must be {AVALONIA_NAMESPACE!r}, "
                f"got {root_namespace or '(none)'!r}"
            )
        if not _XMLNS_X_RE.search(text):
            errors.append(f"{relative}: missing Avalonia-compatible xmlns:x declaration")
        if WPF_PRESENTATION_NAMESPACE in text:
            errors.append(f"{relative}: WPF presentation namespace is forbidden")
        if _WPF_MARKUP_RE.search(text):
            errors.append(f"{relative}: WPF x:Static/x:Type markup is forbidden")
        if _WPF_NAMED_STATIC_RESOURCE_RE.search(text):
            errors.append(
                f"{relative}: WPF StaticResource ResourceKey= syntax is forbidden"
            )
        if "pack://application:,,," in text.lower() or ";component/" in text.lower():
            errors.append(f"{relative}: WPF pack/component resource URI is forbidden")

        for element in root.iter():
            element_name = _local_name(element.tag)
            if element_name in _FORBIDDEN_ELEMENTS:
                errors.append(f"{relative}: WPF-only element <{element_name}> is forbidden")
            if element_name == "Style" and "Selector" not in {
                _local_name(name) for name in element.attrib
            }:
                errors.append(
                    f"{relative}: Avalonia <Style> requires Selector; "
                    "WPF key/TargetType styles are not accepted"
                )
            key = element.attrib.get(_XAML_KEY)
            if key:
                resource_keys.add(key.strip())
            for raw_name, value in element.attrib.items():
                attribute_name = _local_name(raw_name)
                if attribute_name in _FORBIDDEN_ATTRIBUTES:
                    replacement = (
                        "IsVisible"
                        if attribute_name == "Visibility"
                        else "Avalonia Selector/Classes syntax"
                    )
                    errors.append(
                        f"{relative}: WPF-only attribute {attribute_name!r} is forbidden; "
                        f"use {replacement}"
                    )
                if attribute_name == "Property" and value.rsplit(".", 1)[-1] in {
                    "Style",
                    "Visibility",
                }:
                    errors.append(
                        f"{relative}: WPF-only setter property {value!r} is forbidden"
                    )
                if (
                    attribute_name == "Source"
                    and value.split("?", 1)[0].lower().endswith(".xaml")
                ):
                    errors.append(
                        f"{relative}: resource/view Source must reference .axaml, not .xaml"
                    )
                for reference in _STATIC_RESOURCE_RE.findall(value):
                    static_references.append((relative, reference.strip()))

        entry["ok"] = not any(error.startswith(f"{relative}:") for error in errors)
        file_reports.append(entry)

    app_style = parsed.get(blueprint_dir / "AppStyle.axaml")
    if app_style is not None and _local_name(app_style.tag) not in {
        "Styles",
        "ResourceDictionary",
    }:
        errors.append(
            "AppStyle.axaml: root must be Avalonia <Styles> or <ResourceDictionary>"
        )
    main_window = parsed.get(blueprint_dir / "MainWindow.axaml")
    if main_window is not None and _local_name(main_window.tag) != "Window":
        errors.append("MainWindow.axaml: root must be Avalonia <Window>")
    app_style_resource_keys = (
        {
            element.attrib[_XAML_KEY].strip()
            for element in app_style.iter()
            if element.attrib.get(_XAML_KEY)
        }
        if app_style is not None
        else set()
    )
    if app_style is not None and not app_style_resource_keys:
        errors.append(
            "AppStyle.axaml: blueprint must define Avalonia resources with x:Key"
        )

    for relative, reference in static_references:
        if reference not in resource_keys:
            errors.append(
                f"{relative}: StaticResource {reference!r} has no x:Key definition "
                "in the blueprint set"
            )

    probe: dict[str, Any]
    if compile_probe == "off" or errors:
        probe = {
            "status": "not-run",
            "reason": "disabled" if compile_probe == "off" else "static validation failed",
        }
    else:
        probe = _run_compile_probe(blueprint_dir, axaml_files)
        if probe.get("status") == "failed":
            errors.append(
                "Avalonia XAML compiler probe failed; see compile_probe.output_tail"
            )
        elif compile_probe == "required" and probe.get("status") != "passed":
            errors.append(
                f"Avalonia XAML compiler probe was required but {probe.get('status')}: "
                f"{probe.get('reason', 'unknown reason')}"
            )
        elif probe.get("status") == "skipped":
            warnings.append(str(probe.get("reason") or "compiler probe unavailable"))

    return {
        "contract_version": CONTRACT_VERSION,
        "ok": not errors,
        "blueprint_dir": str(blueprint_dir),
        "source_digest": blueprint_digest(blueprint_dir)
        if blueprint_dir.is_dir()
        else None,
        "required_files": list(REQUIRED_BLUEPRINTS),
        "files": file_reports,
        "resource_keys": sorted(resource_keys),
        "static_resource_references": [
            {"file": relative, "key": key} for relative, key in static_references
        ],
        "compile_probe": probe,
        "warnings": warnings,
        "errors": errors,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a new-run Avalonia 11 .axaml UI blueprint."
    )
    parser.add_argument("--blueprint-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--compile-probe",
        choices=("auto", "off", "required"),
        default="required",
        help="Require normalized real Avalonia compilation (default: required).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_blueprint(
        args.blueprint_dir,
        compile_probe=args.compile_probe,
    )
    write_report(args.report, report)
    if report["ok"]:
        probe_status = report["compile_probe"].get("status")
        print(
            f"{args.blueprint_dir}: valid Avalonia blueprint "
            f"(compile probe: {probe_status})"
        )
        return 0
    for error in report["errors"]:
        print(error)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
