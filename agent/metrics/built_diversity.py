"""Structural style-diversity *and* blueprint-fidelity measured on the BUILT
apps, not just the design blueprints.

Motivation
----------
``style_diversity`` scores the ``03_substyle_design/*/ui-blueprint/*.axaml``
design blueprints. That guarantees the *designs* diverge, but the agent
ultimately learns from the *shipped* software. If the Coder reimplements the UI
freely, "diverse blueprints" does not imply "diverse products". This module
closes that gap by running the exact same region-skeleton + Zhang-Shasha TED
machinery on each style's built Avalonia app (the ``.axaml`` views under
``software_variants/<style>/loop/v*/coder/app``).

Two views:

* ``analyze_built_apps`` -> heterogeneity matrix across the *built* styles
  (are the shipped products structurally distinct?).
* ``fidelity`` -> per-style blueprint-vs-built TED + interpretable feature
  diff (did the build stay faithful to its blueprint?).

Hard prerequisite: the build must implement its UI in ``.axaml`` (enforced via
the Coder prompt). Styles whose UI is hand-built in C# expose no parseable view
and are reported under ``skipped`` rather than silently dropped.

CLI:
    python -m agent.metrics.built_diversity <run_dir> [--mode corpus|main]
        [--fidelity] [--json out.json]

Zero third-party dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .blueprint_tree import (
    BlueprintParser,
    Node,
    _is_dialog_file,
    build_skeleton,
    feature_summary,
)
from .style_diversity import compute_matrix, normalized_ted, _print_report


_SKIP_DIR_MARKERS = ("/obj/", "/bin/", "/.nuget/")


# --- locating built views ---------------------------------------------------

def _app_axaml(app_dir: Path) -> list[Path]:
    """All real view ``.axaml`` files in a built app (excludes build output,
    nuget cache, and the App.axaml application shell)."""
    out: list[Path] = []
    for p in app_dir.rglob("*.axaml"):
        s = str(p)
        if any(marker in s for marker in _SKIP_DIR_MARKERS):
            continue
        if p.stem.lower() == "app":
            continue
        out.append(p)
    return out


def find_built_app(run_dir, style_id: str) -> tuple[Optional[Path], Optional[str]]:
    """Return (app_dir, version) for the latest loop version of *style_id*
    whose ``coder/app`` actually ships ``.axaml`` views, else (None, None)."""
    base = Path(run_dir) / "software_variants" / style_id / "loop"
    if not base.is_dir():
        return None, None
    versions = sorted(
        base.glob("v*"),
        key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else -1,
        reverse=True,
    )
    for v in versions:
        app = v / "coder" / "app"
        if app.is_dir() and _app_axaml(app):
            return app, v.name
    return None, None


def _find_main_view(files: list[Path]) -> Optional[Path]:
    for f in files:
        if f.stem.lower() == "mainwindow":
            return f
    for f in files:
        low = f.stem.lower()
        if "main" in low or "shell" in low:
            return f
    return files[0] if files else None


def build_app_skeleton(app_dir: Path, mode: str = "corpus") -> Optional[Node]:
    """Normalize a built app's ``.axaml`` views into one region-skeleton tree,
    mirroring blueprint_tree.build_skeleton's main/corpus semantics."""
    files = _app_axaml(app_dir)
    if not files:
        return None
    # Built apps load views from code, so parse each .axaml file directly.
    parser = BlueprintParser(app_dir)

    if mode == "main":
        main = _find_main_view(files)
        return parser.parse_file(main) if main else None

    if mode == "corpus":
        children: list[Node] = []
        for p in sorted(files):
            if _is_dialog_file(p):
                continue
            sub = parser.parse_file(p)
            if sub is not None:
                children.append(Node(label=f"VIEW:{p.stem}", children=sub.children))
        return Node(label="ROOT", children=children) if children else None

    raise ValueError(f"unknown mode: {mode}")


# --- built-app heterogeneity matrix -----------------------------------------

def analyze_built_apps(run_dir, mode: str = "corpus") -> dict:
    """Heterogeneity matrix across the built styles of a run."""
    run_dir = Path(run_dir)
    variants = run_dir / "software_variants"
    style_ids = (
        sorted(d.name for d in variants.glob("style_*") if d.is_dir())
        if variants.is_dir()
        else []
    )

    trees: dict[str, Node] = {}
    features: dict[str, dict] = {}
    coverage: dict[str, str] = {}
    skipped: list[dict] = []

    for sid in style_ids:
        app, ver = find_built_app(run_dir, sid)
        if app is None:
            skipped.append({
                "style": sid,
                "reason": "no built .axaml views (C#-coded UI or no successful build)",
            })
            continue
        tree = build_app_skeleton(app, mode=mode)
        if tree is None:
            skipped.append({"style": sid, "reason": "axaml parsed to empty tree"})
            continue
        trees[sid] = tree
        features[sid] = feature_summary(tree)
        coverage[sid] = ver

    base = {
        "scope": "built_app",
        "mode": mode,
        "n_styles": len(trees),
        "n_total": len(style_ids),
        "coverage": coverage,
        "skipped": skipped,
    }
    if len(trees) < 2:
        base["error"] = "need >=2 built XAML apps to compare"
        return base

    result = compute_matrix(trees)
    result.update(base)
    result["features"] = features
    return result


# --- blueprint vs built fidelity --------------------------------------------

_FIDELITY_FEATURES = ("max_grid_cols", "n_collections", "n_dockpanels", "depth")


def fidelity(run_dir, mode: str = "main") -> dict:
    """Per-style blueprint-vs-built comparison.

    TED is reported for reference, but it is inflated by the gap in abstraction
    level (a blueprint shell vs a fully implemented app), so the interpretable
    macro-feature diff (pane count, collection paradigms, dock regions) is the
    more reliable fidelity signal.
    """
    run_dir = Path(run_dir)
    substyle = run_dir / "03_substyle_design"
    rows: list[dict] = []
    if not substyle.is_dir():
        return {"scope": "fidelity", "mode": mode, "rows": rows, "error": "no 03_substyle_design"}

    for d in sorted(substyle.glob("style_*")):
        sid = d.name
        bp_dir = d / "ui-blueprint"
        if not bp_dir.is_dir():
            continue
        app, ver = find_built_app(run_dir, sid)
        if app is None:
            rows.append({"style": sid, "built": False})
            continue
        bt = build_skeleton(bp_dir, mode=mode)
        at = build_app_skeleton(app, mode=mode)
        if bt is None or at is None:
            rows.append({"style": sid, "built": True, "version": ver, "ted": None})
            continue
        bf = feature_summary(bt)
        af = feature_summary(at)
        rows.append({
            "style": sid,
            "built": True,
            "version": ver,
            "ted": round(normalized_ted(bt, at), 4),
            "feature_diff": {
                k: {"blueprint": bf[k], "built": af[k]} for k in _FIDELITY_FEATURES
            },
        })
    return {"scope": "fidelity", "mode": mode, "rows": rows}


# --- pretty printing --------------------------------------------------------

def _print_built(result: dict) -> None:
    cov = result.get("coverage", {})
    print(f"  scope=built_app  mode={result['mode']}  built={result['n_styles']}/{result.get('n_total','?')}")
    if cov:
        print("  versions: " + ", ".join(f"{k}={v}" for k, v in sorted(cov.items())))
    if result.get("skipped"):
        for sk in result["skipped"]:
            print(f"    [skip] {sk['style']}: {sk['reason']}")
    if result.get("error"):
        print(f"  [!] {result['error']}")
        return
    _print_report(result)


def _print_fidelity(result: dict) -> None:
    print(f"\n=== blueprint -> built fidelity  [{result['mode']}] ===")
    if result.get("error"):
        print(f"  [!] {result['error']}")
        return
    hdr = f"  {'style':<9}{'built':<7}{'ver':<5}{'ted':>6}   " + "  ".join(
        f"{k}(bp/app)" for k in _FIDELITY_FEATURES
    )
    print(hdr)
    for r in result["rows"]:
        if not r.get("built"):
            print(f"  {r['style']:<9}{'no':<7}")
            continue
        ted = r.get("ted")
        ted_s = f"{ted:.2f}" if isinstance(ted, (int, float)) else "-"
        diffs = "  ".join(
            f"{r['feature_diff'][k]['blueprint']}/{r['feature_diff'][k]['built']}"
            for k in _FIDELITY_FEATURES
        ) if r.get("feature_diff") else ""
        print(f"  {r['style']:<9}{'yes':<7}{str(r.get('version','')):<5}{ted_s:>6}   {diffs}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="path to a run output directory (contains software_variants/)")
    ap.add_argument("--mode", choices=["main", "corpus", "both"], default="corpus")
    ap.add_argument("--fidelity", action="store_true", help="also print blueprint->built fidelity")
    ap.add_argument("--json", help="optional path to dump full result as JSON")
    args = ap.parse_args(argv)

    run_name = Path(args.run_dir).name
    modes = ["main", "corpus"] if args.mode == "both" else [args.mode]
    out: dict = {}
    for mode in modes:
        print(f"\n=== {run_name}  built-app diversity  [{mode}] ===")
        res = analyze_built_apps(args.run_dir, mode=mode)
        _print_built(res)
        out[f"built_{mode}"] = res

    if args.fidelity:
        fid = fidelity(args.run_dir, mode="main")
        _print_fidelity(fid)
        out["fidelity"] = fid

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
