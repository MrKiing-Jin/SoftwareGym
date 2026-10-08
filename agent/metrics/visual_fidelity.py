"""Visual-fidelity gate: did the BUILT app keep the blueprint's design tokens?

Motivation
----------
``style_diversity`` / ``built_diversity`` measure *layout structure* (where the
panels go). They say nothing about whether the shipped app kept the blueprint's
*visual system*: color palette, corner radii, spacing, typography and control
sizing. The Coder prompt (item 5c / 8b) now hard-requires importing the
blueprint ``AppStyle.axaml`` tokens verbatim and binding all visuals through
``StaticResource``/``DynamicResource`` instead of hardcoding. This module is the
automated counterpart of that contract: it parses the design tokens from each
style's blueprint ``AppStyle.axaml`` and from the built app's resource
dictionaries, then scores how faithfully the build reproduced them.

What it checks (all interpretable):

* ``palette``   - Jaccard overlap of the distinct color *values* (token renaming
                  is fine; we compare values, not key names).
* ``primary``   - the blueprint's primary/accent color value survives in the
                  built palette.
* ``radius``    - Jaccard overlap of the corner-radius scalar set.
* ``font``      - the blueprint's main (non-mono) font family survives.
* ``binding``   - share of view visuals bound via Static/DynamicResource vs.
                  inline hardcoded ``#RRGGBB`` (detects token bypass).

A weighted ``score`` in [0,1] plus a ``pass`` gate (default threshold 0.8 and
primary color required) and human-readable ``flags`` (e.g. ``radius_drift``).
The 0.8 bar is deliberately strict: a faithful AppStyle import scores ~0.9-1.0,
while a build that drifts BOTH corner radii and fonts caps out at 0.70 even with
a perfect palette -- so it cannot pass without also keeping radii/typography.

Hard prerequisite (same as built_diversity): the build must implement UI in
``.axaml``. C#-coded UIs expose no resource dictionary and are reported as
``built=False``.

CLI:
    python -m agent.metrics.visual_fidelity <run_dir> [--threshold 0.8]
        [--json out.json]

Zero third-party dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from .built_diversity import _app_axaml, find_built_app

# --- regexes ---------------------------------------------------------------
_RE_COLOR = re.compile(r'<Color\s+x:Key="([^"]+)"\s*>\s*(#[0-9A-Fa-f]{6,8})\s*<', re.I)
_RE_BRUSH_INLINE = re.compile(r'<SolidColorBrush\b[^>]*\bColor="(#[0-9A-Fa-f]{6,8})"', re.I)
_RE_RADIUS = re.compile(r'<CornerRadius\s+x:Key="[^"]+"\s*>([^<]+)</CornerRadius>', re.I)
_RE_THICK = re.compile(r'<Thickness\s+x:Key="[^"]+"\s*>([^<]+)</Thickness>', re.I)
_RE_FONT = re.compile(r'<FontFamily\s+x:Key="([^"]+)"\s*>([^<]+)</FontFamily>', re.I)
_RE_NUM = re.compile(r'[-+]?\d*\.?\d+')
# view-level usage signals
_RE_RES_REF = re.compile(r'\{(?:Static|Dynamic)Resource\b', re.I)
_RE_INLINE_HEX_ATTR = re.compile(
    r'(?:Background|Foreground|BorderBrush|Fill|Stroke|Color)\s*=\s*"(#[0-9A-Fa-f]{6,8})"', re.I)
_RE_STYLE_BLOCK = re.compile(r'<Style\b[^>]*\bSelector="([^"]+)"[^>]*>(.*?)</Style>', re.I | re.S)
_RE_SETTER = re.compile(r'<Setter\s+Property="([^"]+)"\s+Value="([^"]+)"', re.I)

_PRIMARY_HINT = re.compile(r'(primary|accent|brand)', re.I)
_MONO_HINT = re.compile(r'(mono|code|consol|cascadia|jetbrains)', re.I)


def _norm_hex(h: str) -> str:
    """Normalize ARGB/RGB hex to upper RRGGBB (drop leading alpha)."""
    h = h.lstrip("#").upper()
    if len(h) == 8:  # AARRGGBB
        h = h[2:]
    elif len(h) == 4:  # ARGB short
        h = h[1:]
    if len(h) == 3:  # RGB short -> expand
        h = "".join(c * 2 for c in h)
    return h


def _nums(s: str) -> list[float]:
    return [float(x) for x in _RE_NUM.findall(s)]


def extract_tokens(text: str) -> dict:
    """Parse the design-token surface from one XAML/AXAML text."""
    color_map: dict[str, str] = {}
    for k, v in _RE_COLOR.findall(text):
        color_map[k] = _norm_hex(v)
    colors = set(color_map.values()) | {_norm_hex(c) for c in _RE_BRUSH_INLINE.findall(text)}

    radii: set[float] = set()
    for m in _RE_RADIUS.findall(text):
        radii.update(_nums(m))
    spacing: set[float] = set()
    for m in _RE_THICK.findall(text):
        spacing.update(_nums(m))

    fonts_main: set[str] = set()
    fonts_mono: set[str] = set()
    for key, fam in _RE_FONT.findall(text):
        first = fam.split(",")[0].strip()
        if not first:
            continue
        (fonts_mono if (_MONO_HINT.search(key) or _MONO_HINT.search(first)) else fonts_main).add(first)

    # primary/accent color value(s), by key hint
    primary = {v for k, v in color_map.items() if _PRIMARY_HINT.search(k)}

    # control sizing per selector (Button MinHeight etc.) + any inline CornerRadius setters
    controls: dict[str, dict] = {}
    for sel, body in _RE_STYLE_BLOCK.findall(text):
        setters = {p: v for p, v in _RE_SETTER.findall(body)}
        controls[sel.strip()] = setters
        for p, v in setters.items():
            if p == "CornerRadius":
                radii.update(_nums(v))
    return {
        "colors": colors,
        "color_map": color_map,
        "primary": primary,
        "radii": radii,
        "spacing": spacing,
        "fonts_main": fonts_main,
        "fonts_mono": fonts_mono,
        "controls": controls,
    }


def _button_min_height(controls: dict) -> Optional[float]:
    for sel, setters in controls.items():
        if sel.split(".")[0].strip().lower() == "button" and "MinHeight" in setters:
            try:
                return float(setters["MinHeight"])
            except ValueError:
                return None
    return None


# --- collection -------------------------------------------------------------

def _merge(a: dict, b: dict) -> dict:
    for key in ("colors", "primary", "radii", "spacing", "fonts_main", "fonts_mono"):
        a[key] |= b[key]
    a["color_map"].update(b["color_map"])
    a["controls"].update(b["controls"])
    return a


def _empty_tokens() -> dict:
    return {"colors": set(), "color_map": {}, "primary": set(), "radii": set(),
            "spacing": set(), "fonts_main": set(), "fonts_mono": set(), "controls": {}}


def blueprint_tokens(bp_dir: Path) -> Optional[dict]:
    """Tokens from AppStyle.axaml, with read-only fallback for historical .xaml."""
    cands = list(bp_dir.glob("AppStyle*.axaml"))
    if not cands:
        cands = list(bp_dir.glob("AppStyle*.xaml"))
    if not cands:
        sources = list(bp_dir.glob("*.axaml")) or list(bp_dir.glob("*.xaml"))
        cands = [
            p
            for p in sources
            if len(
                _RE_COLOR.findall(
                    p.read_text(encoding="utf-8", errors="ignore")
                )
            )
            >= 3
        ]
    if not cands:
        return None
    tok = _empty_tokens()
    for p in cands:
        _merge(tok, extract_tokens(p.read_text(encoding="utf-8", errors="ignore")))
    return tok


def built_tokens(app_dir: Path) -> tuple[Optional[dict], dict]:
    """Aggregate tokens across the built app's resource-dictionary .axaml, plus
    view-level binding stats (resource refs vs inline hardcoded colors)."""
    files = _app_axaml(app_dir)
    # App.axaml is excluded by _app_axaml; include it back as it may merge/define tokens
    files = files + [p for p in app_dir.rglob("App.axaml")
                     if "/obj/" not in str(p) and "/bin/" not in str(p)]
    if not files:
        return None, {}
    tok = _empty_tokens()
    has_tokens = False
    res_refs = 0
    inline_hex: list[str] = []
    for p in files:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        t = extract_tokens(txt)
        if len(t["color_map"]) >= 3:
            has_tokens = True
        _merge(tok, t)
        res_refs += len(_RE_RES_REF.findall(txt))
        inline_hex += [_norm_hex(h) for h in _RE_INLINE_HEX_ATTR.findall(txt)]
    stats = {
        "resource_refs": res_refs,
        "inline_hex_count": len(inline_hex),
        "inline_hex_values": sorted(set(inline_hex)),
        "binding_ratio": round(res_refs / (res_refs + len(inline_hex)), 4) if (res_refs + len(inline_hex)) else 1.0,
        "has_token_dict": has_tokens,
    }
    return (tok if has_tokens or tok["colors"] else None), stats


# --- scoring ----------------------------------------------------------------

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


_WEIGHTS = {"palette": 0.40, "primary": 0.15, "radius": 0.20, "font": 0.10, "binding": 0.15}


def score_style(bp: dict, bt: dict, stats: dict) -> dict:
    palette = round(_jaccard(bp["colors"], bt["colors"]), 4)
    # primary survives: any blueprint primary value present in built palette
    bp_primary = bp["primary"] or set()
    primary_present = 1.0 if (bp_primary and bp_primary & bt["colors"]) else (
        1.0 if not bp_primary else 0.0)
    # Do not penalize radius if the blueprint defined no radius tokens. Matches the
    # primary/font behavior: a missing blueprint spec is treated as satisfied, to
    # avoid false-positive radius_drift on dimensions the blueprint did not specify.
    radius = 1.0 if not bp["radii"] else round(_jaccard(bp["radii"], bt["radii"]), 4)
    bp_font = bp["fonts_main"]
    font_present = 1.0 if (bp_font and bp_font & bt["fonts_main"]) else (
        1.0 if not bp_font else 0.0)
    binding = float(stats.get("binding_ratio", 1.0))

    dims = {"palette": palette, "primary": primary_present, "radius": radius,
            "font": font_present, "binding": binding}
    overall = round(sum(_WEIGHTS[k] * dims[k] for k in _WEIGHTS), 4)

    flags = []
    if palette < 0.8:
        flags.append("palette_drift")
    if not primary_present:
        flags.append("primary_missing")
    if radius < 0.7:
        flags.append("radius_drift")
    if not font_present:
        flags.append("font_drift")
    if binding < 0.9:
        flags.append("hardcoded_colors")

    bp_bmh = _button_min_height(bp["controls"])
    bt_bmh = _button_min_height(bt["controls"])
    if bp_bmh is not None and bt_bmh is not None and abs(bp_bmh - bt_bmh) > 2:
        flags.append("button_size_drift")

    return {
        "dimensions": dims,
        "score": overall,
        "flags": flags,
        "detail": {
            "palette": {"blueprint_n": len(bp["colors"]), "built_n": len(bt["colors"]),
                        "shared_n": len(bp["colors"] & bt["colors"]),
                        "missing_from_built": sorted(bp["colors"] - bt["colors"]),
                        "extra_in_built": sorted(bt["colors"] - bp["colors"])},
            "primary": {"blueprint": sorted(bp_primary), "present": bool(primary_present)},
            "radius": {"blueprint": sorted(bp["radii"]), "built": sorted(bt["radii"])},
            "font_main": {"blueprint": sorted(bp_font), "built": sorted(bt["fonts_main"])},
            "binding": {"resource_refs": stats.get("resource_refs"),
                        "inline_hex_count": stats.get("inline_hex_count"),
                        "inline_hex_values": stats.get("inline_hex_values", [])[:12]},
            "button_min_height": {"blueprint": bp_bmh, "built": bt_bmh},
        },
    }


# --- run-level analysis -----------------------------------------------------

def analyze(run_dir, threshold: float = 0.8) -> dict:
    run_dir = Path(run_dir)
    substyle = run_dir / "03_substyle_design"
    rows: list[dict] = []
    if not substyle.is_dir():
        return {"scope": "visual_fidelity", "threshold": threshold, "rows": rows,
                "error": "no 03_substyle_design"}

    scored = 0
    passed = 0
    for d in sorted(substyle.glob("style_*")):
        sid = d.name
        bp_dir = d / "ui-blueprint"
        if not bp_dir.is_dir():
            continue
        bp = blueprint_tokens(bp_dir)
        if bp is None:
            rows.append({"style": sid, "built": None, "error": "no blueprint AppStyle tokens"})
            continue
        app, ver = find_built_app(run_dir, sid)
        if app is None:
            rows.append({"style": sid, "built": False})
            continue
        bt, stats = built_tokens(app)
        if bt is None:
            # Built UI ships no design-token dictionary at all: the blueprint
            # AppStyle was never imported -> hardest violation of the 5c
            # contract. Count it as a scored failure, not a silent skip.
            scored += 1
            rows.append({"style": sid, "built": True, "version": ver,
                         "score": 0.0, "pass": False, "flags": ["no_token_dict"],
                         "dimensions": {k: 0.0 for k in _WEIGHTS},
                         "detail": {"binding": stats}})
            continue
        sc = score_style(bp, bt, stats)
        ok = sc["score"] >= threshold and "primary_missing" not in sc["flags"]
        scored += 1
        passed += int(ok)
        rows.append({"style": sid, "built": True, "version": ver,
                     "score": sc["score"], "pass": ok, "flags": sc["flags"],
                     "dimensions": sc["dimensions"], "detail": sc["detail"]})

    return {
        "scope": "visual_fidelity",
        "threshold": threshold,
        "weights": _WEIGHTS,
        "n_scored": scored,
        "n_pass": passed,
        "mean_score": round(sum(r["score"] for r in rows if r.get("score") is not None) / scored, 4) if scored else None,
        "rows": rows,
    }


# --- per-style scoring (loop feedback) --------------------------------------

def score_single(run_dir, style_id: str, app_dir, version: Optional[str] = None,
                 threshold: float = 0.8) -> dict:
    """Score a single style/version build for visual fidelity (called each repair loop).

    Returns one row dict with the same shape as the elements of analyze() rows,
    plus threshold/style/version.
    """
    run_dir = Path(run_dir)
    app_dir = Path(app_dir)
    row: dict = {"scope": "visual_fidelity", "style": style_id,
                 "version": version, "threshold": threshold}
    bp_dir = run_dir / "03_substyle_design" / style_id / "ui-blueprint"
    bp = blueprint_tokens(bp_dir)
    if bp is None:
        row.update({"built": True, "score": None, "pass": None,
                    "error": "no blueprint AppStyle tokens"})
        return row
    if not app_dir.is_dir() or not _app_axaml(app_dir):
        row.update({"built": False, "score": None, "pass": None,
                    "error": "no built .axaml views"})
        return row
    bt, stats = built_tokens(app_dir)
    if bt is None:
        row.update({"built": True, "score": 0.0, "pass": False,
                    "flags": ["no_token_dict"],
                    "dimensions": {k: 0.0 for k in _WEIGHTS},
                    "detail": {"binding": stats}})
        return row
    sc = score_style(bp, bt, stats)
    ok = sc["score"] >= threshold and "primary_missing" not in sc["flags"]
    row.update({"built": True, "score": sc["score"], "pass": ok,
                "flags": sc["flags"], "dimensions": sc["dimensions"],
                "detail": sc["detail"]})
    return row


_FLAG_FIX = {
    "palette_drift": "The built palette diverges too far from the blueprint AppStyle: revert the <Color> values in the built resource dictionaries (Styles.axaml, App.axaml, etc.) to the AppStyle.axaml palette (we compare values, not key names), and do not introduce new colors not present in the blueprint.",
    "primary_missing": "The blueprint primary/accent color disappeared from the build: make sure that value survives in the built palette and is referenced by primary buttons/accent elements via token.",
    "radius_drift": "Corner-radius tokens do not match the blueprint: align the built CornerRadius set back to the blueprint (see blueprint vs built below), and do not invent custom radius values.",
    "font_drift": "Main font family is missing or replaced: restore the main FontFamily specified by the blueprint, and do not fall back to the system default.",
    "hardcoded_colors": "Many inline hard-coded colors (#RRGGBB) in views bypass the resource dictionary: switch to {StaticResource ...}/{DynamicResource ...} bindings and raise the binding ratio to >=0.9.",
    "button_size_drift": "Button size (MinHeight) drifts from the blueprint control-size token: align it back to the blueprint.",
}


def render_feedback_md(row: dict) -> str:
    """Render a single style's visual-fidelity result as markdown feedback for the Coder/DevCoder repair loop."""
    sid = row.get("style", "?")
    ver = row.get("version", "?")
    lines = [f"# Visual fidelity report ({sid} / {ver})", ""]
    if row.get("error"):
        lines += [f"> Cannot score: {row['error']} (skip visual feedback this round and focus on functional fixes)", ""]
        return "\n".join(lines)
    score = row.get("score")
    passed = row.get("pass")
    thr = row.get("threshold", 0.8)
    verdict = "PASS ✅" if passed else "FAIL ❌"
    lines += [
        f"- Overall score **{score}** (threshold {thr}) → **{verdict}**",
        "- This is an automatic comparison of the built UI against its blueprint AppStyle design tokens (palette/primary/radius/font/binding ratio). "
        "**Use it only as a reference signal for visual fixes; it does not block the pipeline and must not override working functionality.**",
        "",
    ]
    dims = row.get("dimensions") or {}
    if dims:
        lines += ["## Dimension scores",
                  f"- palette: {dims.get('palette')}",
                  f"- primary: {dims.get('primary')}",
                  f"- radius: {dims.get('radius')}",
                  f"- font: {dims.get('font')}",
                  f"- token binding ratio: {dims.get('binding')}",
                  ""]
    flags = row.get("flags") or []
    if passed and not flags:
        lines += ["## Summary", "This build already reproduces the blueprint visually; **no visual-only changes are needed**.", ""]
        return "\n".join(lines)
    if flags:
        lines += ["## Targeted visual fixes (by flag)"]
        for f in flags:
            lines.append(f"- **{f}**: {_FLAG_FIX.get(f, 'Align this dimension with the blueprint AppStyle.')}")
        lines.append("")
    det = row.get("detail") or {}
    pal = det.get("palette") or {}
    if pal.get("missing_from_built"):
        lines += [f"- Colors present in blueprint but missing from build: `{pal['missing_from_built']}`"]
    rad = det.get("radius") or {}
    if rad:
        lines += [f"- Radius comparison: blueprint=`{rad.get('blueprint')}` built=`{rad.get('built')}`"]
    fnt = det.get("font_main") or {}
    if fnt:
        lines += [f"- Main font comparison: blueprint=`{fnt.get('blueprint')}` built=`{fnt.get('built')}`"]
    lines += ["",
              "> Fix principle: change only the **resource-dictionary token values** in the build to align with the blueprint. "
              "Do not alter functional logic or hard-code values just to match visuals; keep all currently passing functionality from regressing."]
    return "\n".join(lines)


# --- pretty printing --------------------------------------------------------

def _print(result: dict) -> None:
    print(f"  scope=visual_fidelity  threshold={result['threshold']}  "
          f"scored={result['n_scored']}  pass={result['n_pass']}  mean={result.get('mean_score')}")
    if result.get("error"):
        print(f"  [!] {result['error']}")
        return
    hdr = (f"  {'style':<9}{'built':<6}{'ver':<5}{'score':>6}{'pass':>6}   "
           f"{'pal':>5}{'prim':>5}{'rad':>5}{'font':>5}{'bind':>6}   flags")
    print(hdr)
    for r in result["rows"]:
        if r.get("built") is None:
            print(f"  {r['style']:<9}{'-':<6}{'':<5}{'':>6}{'':>6}   {r.get('error','')}")
            continue
        if not r.get("built"):
            print(f"  {r['style']:<9}{'no':<6}")
            continue
        if "score" not in r:
            print(f"  {r['style']:<9}{'yes':<6}{str(r.get('version','')):<5}   {r.get('error','')}")
            continue
        d = r["dimensions"]
        print(f"  {r['style']:<9}{'yes':<6}{str(r.get('version','')):<5}"
              f"{r['score']:>6.2f}{('OK' if r['pass'] else 'FAIL'):>6}   "
              f"{d['palette']:>5.2f}{d['primary']:>5.1f}{d['radius']:>5.2f}"
              f"{d['font']:>5.1f}{d['binding']:>6.2f}   {','.join(r['flags'])}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="run output dir (contains 03_substyle_design/ and software_variants/)")
    ap.add_argument("--threshold", type=float, default=0.8, help="pass threshold for the overall score (default 0.8: a build that drifts both radii and fonts caps at 0.70 and cannot pass)")
    ap.add_argument("--style", help="only analyze a single style id (e.g. style_09)")
    ap.add_argument("--json", help="optional path to dump full result as JSON")
    args = ap.parse_args(argv)

    run_name = Path(args.run_dir).name
    print(f"\n=== {run_name}  visual fidelity (blueprint AppStyle -> built tokens) ===")
    res = analyze(args.run_dir, threshold=args.threshold)
    if args.style:
        res["rows"] = [r for r in res["rows"] if r.get("style") == args.style]
    _print(res)

    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
