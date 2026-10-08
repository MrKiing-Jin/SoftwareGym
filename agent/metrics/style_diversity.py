"""Structural style-diversity scoring for a set of UI-blueprint variants.

Pipeline:
    ui-blueprint/*.axaml --(blueprint_tree)-->  region-skeleton tree per style
    pair of trees        --(Zhang-Shasha TED)-->  edit distance
    distance / max(|T1|,|T2|)                  ->  normalized in [0, 1]

Aggregated into a heterogeneity matrix plus min/mean pairwise scores. The
*minimum* pairwise distance is the gate-relevant number: it tells us whether the
two most similar styles are still meaningfully different.

CLI:
    python -m agent.metrics.style_diversity <03_substyle_design dir> [--mode main|corpus] [--json out.json]

Zero third-party dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .blueprint_tree import Node, build_skeleton, feature_summary
from . import vocab


# --- Zhang-Shasha tree edit distance ----------------------------------------

class _Indexed:
    """Postorder-indexed view of a tree for Zhang-Shasha."""

    def __init__(self, root: Node):
        self.labels: list[str] = []
        self.lmds: list[int] = []  # leftmost-leaf-descendant postorder id
        self._build(root)
        self.keyroots = self._keyroots()

    def _build(self, root: Node) -> int:
        # returns postorder id of root; fills labels + lmds
        def rec(node: Node) -> int:
            first_lmd: Optional[int] = None
            for i, c in enumerate(node.children):
                cid = rec(c)
                if i == 0:
                    first_lmd = self.lmds[cid]
            idx = len(self.labels)
            self.labels.append(node.label)
            self.lmds.append(idx if first_lmd is None else first_lmd)
            return idx

        rec(root)
        return len(self.labels) - 1

    def _keyroots(self) -> list[int]:
        seen: dict[int, int] = {}
        for i in range(len(self.labels) - 1, -1, -1):
            if self.lmds[i] not in seen:
                seen[self.lmds[i]] = i
        return sorted(seen.values())

    def __len__(self) -> int:
        return len(self.labels)


def tree_edit_distance(a: Node, b: Node) -> int:
    """Unit-cost Zhang-Shasha TED (insert=delete=relabel-on-mismatch=1)."""
    A = _Indexed(a)
    B = _Indexed(b)
    n, m = len(A), len(B)
    TD = [[0] * m for _ in range(n)]

    for ki in A.keyroots:
        for kj in B.keyroots:
            li, lj = A.lmds[ki], B.lmds[kj]
            rows = ki - li + 2
            cols = kj - lj + 2
            fd = [[0] * cols for _ in range(rows)]
            for x in range(1, rows):
                fd[x][0] = fd[x - 1][0] + 1
            for y in range(1, cols):
                fd[0][y] = fd[0][y - 1] + 1
            for x in range(1, rows):
                ai = li + x - 1
                for y in range(1, cols):
                    bj = lj + y - 1
                    if A.lmds[ai] == li and B.lmds[bj] == lj:
                        cost = 0 if A.labels[ai] == B.labels[bj] else 1
                        fd[x][y] = min(
                            fd[x - 1][y] + 1,
                            fd[x][y - 1] + 1,
                            fd[x - 1][y - 1] + cost,
                        )
                        TD[ai][bj] = fd[x][y]
                    else:
                        p = A.lmds[ai] - li
                        q = B.lmds[bj] - lj
                        fd[x][y] = min(
                            fd[x - 1][y] + 1,
                            fd[x][y - 1] + 1,
                            fd[p][q] + TD[ai][bj],
                        )
    return TD[n - 1][m - 1]


def normalized_ted(a: Node, b: Node) -> float:
    raw = tree_edit_distance(a, b)
    denom = max(a.size(), b.size())
    if denom == 0:
        return 0.0
    return min(1.0, raw / denom)


# --- matrix / aggregation ---------------------------------------------------

def compute_matrix(trees: dict[str, Node]) -> dict:
    names = sorted(trees)
    n = len(names)
    matrix = [[0.0] * n for _ in range(n)]
    pairs: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = normalized_ted(trees[names[i]], trees[names[j]])
            matrix[i][j] = matrix[j][i] = round(d, 4)
            pairs.append((names[i], names[j], round(d, 4)))

    flat = [d for _, _, d in pairs]
    stats = {
        "min_pairwise": round(min(flat), 4) if flat else None,
        "mean_pairwise": round(sum(flat) / len(flat), 4) if flat else None,
        "max_pairwise": round(max(flat), 4) if flat else None,
    }
    closest = min(pairs, key=lambda p: p[2]) if pairs else None
    return {
        "names": names,
        "matrix": matrix,
        "pairs": pairs,
        "stats": stats,
        "closest_pair": closest,
    }


def analyze_blueprints(substyle_dir, mode: str = "main") -> dict:
    substyle_dir = Path(substyle_dir)
    style_dirs = sorted(
        d for d in substyle_dir.glob("style_*") if (d / "ui-blueprint").is_dir()
    )
    trees: dict[str, Node] = {}
    features: dict[str, dict] = {}
    skipped: list[str] = []
    for d in style_dirs:
        tree = build_skeleton(d / "ui-blueprint", mode=mode)
        if tree is None:
            skipped.append(d.name)
            continue
        trees[d.name] = tree
        features[d.name] = feature_summary(tree)

    if len(trees) < 2:
        return {
            "mode": mode,
            "n_styles": len(trees),
            "skipped": skipped,
            "error": "need >=2 parseable styles",
        }

    result = compute_matrix(trees)
    result["mode"] = mode
    result["n_styles"] = len(trees)
    result["skipped"] = skipped
    result["features"] = features
    return result


# --- declared-archetype side (from styles.json) -----------------------------

def load_declared_archetypes(run_dir) -> Dict[str, dict]:
    """Read each style's declared `layout_archetype` from 02_style_design/styles.json.

    Returns {style_id: archetype_dict}. Empty when styles.json is missing or has
    no archetypes (e.g. a run produced before this field existed)."""
    styles_json = Path(run_dir) / "02_style_design" / "styles.json"
    if not styles_json.is_file():
        return {}
    try:
        data = json.loads(styles_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: Dict[str, dict] = {}
    for entry in data.get("styles", []) or []:
        sid = str(entry.get("id") or "").strip()
        arch = entry.get("layout_archetype")
        if sid and isinstance(arch, dict):
            out[sid] = arch
    return out


# --- gate: pick re-roll targets + build per-style feedback -------------------

def _mean_distance_to_others(name: str, names: list, matrix: list) -> float:
    i = names.index(name)
    others = [matrix[i][j] for j in range(len(names)) if j != i]
    return sum(others) / len(others) if others else 0.0


def select_reroll_targets(struct: dict, tau: float) -> list:
    """For every structurally too-close pair (TED < tau), pick the more redundant
    member (lower mean distance to all others) as the re-roll target."""
    if struct.get("error"):
        return []
    names = struct["names"]
    matrix = struct["matrix"]
    targets: list[str] = []
    for s1, s2, d in struct["pairs"]:
        if d >= tau:
            continue
        m1 = _mean_distance_to_others(s1, names, matrix)
        m2 = _mean_distance_to_others(s2, names, matrix)
        pick = s1 if m1 <= m2 else s2
        if pick not in targets:
            targets.append(pick)
    return targets


def build_diversify_feedback(
    target: str,
    struct: dict,
    archetypes: Dict[str, dict],
    tau: float,
) -> str:
    """Human-readable feedback block injected into SubStyleDesigner's repair
    template ({{diversify_feedback}}). Names the closest sibling, the target's
    assigned archetype, and which dimensions to push apart."""
    names = struct["names"]
    matrix = struct["matrix"]
    i = names.index(target)
    # closest sibling by structural distance
    sib_dist = sorted(
        ((names[j], matrix[i][j]) for j in range(len(names)) if j != i),
        key=lambda kv: kv[1],
    )
    lines: list[str] = []
    if sib_dist:
        sib, d = sib_dist[0]
        lines.append(
            f"    - Structural heterogeneity alert: the UI control tree of this style ({target}) is too similar to sibling style {sib}; "
            f"normalized tree-edit distance is only {d:.2f} (threshold {tau:.2f}; closer to 0 means more alike)."
        )
        my_arch = vocab.normalize_tuple(archetypes.get(target, {}))
        sib_arch = vocab.normalize_tuple(archetypes.get(sib, {}))
        if any(my_arch):
            arch_str = ", ".join(
                f"{dim}={val or '<unspecified>'}" for dim, val in zip(vocab.DIMENSION_ORDER, my_arch)
            )
            lines.append(f"    - This style's assigned layout_archetype: {arch_str}. It must be genuinely realized in MainWindow.")
        if any(sib_arch):
            diff_dims = [
                dim
                for dim, a, b in zip(vocab.DIMENSION_ORDER, my_arch, sib_arch)
                if a and b and a != b
            ]
            same_dims = [
                dim
                for dim, a, b in zip(vocab.DIMENSION_ORDER, my_arch, sib_arch)
                if a and b and a == b
            ]
            if diff_dims:
                lines.append(
                    f"    - You and {sib} should differ on these dimensions: {', '.join(diff_dims)}; "
                    "please make these differences substantial and obvious in the control structure."
                )
            if same_dims:
                lines.append(
                    f"    - You and {sib} are the same on these dimensions: {', '.join(same_dims)}; "
                    "identical dimensions are allowed, but structural differences on the remaining dimensions must be large enough to avoid overall sameness."
                )
    else:
        lines.append(f"    - This style ({target}) needs to differentiate its structure from other styles.")
    return "\n".join(lines)


def run_gate(run_dir, tau: float = 0.60, mode: str = "corpus") -> dict:
    """Compute the realized structural diversity + declared-archetype distinctness
    for a run, decide re-roll targets, and write 03_substyle_design/style_diversity.json.
    Returns the full report dict."""
    run_dir = Path(run_dir)
    substyle_dir = run_dir / "03_substyle_design"
    struct = analyze_blueprints(substyle_dir, mode=mode)

    archetypes = load_declared_archetypes(run_dir)
    declared = vocab.assignment_report(archetypes) if archetypes else {"available": False}

    targets = select_reroll_targets(struct, tau) if not struct.get("error") else []
    feedback = {t: build_diversify_feedback(t, struct, archetypes, tau) for t in targets}

    report = {
        "tau": tau,
        "mode": mode,
        "structural": struct,
        "declared_archetypes": declared,
        "reroll_targets": targets,
        "diversify_feedback": feedback,
        "passes": (not targets) and (not struct.get("error")),
    }
    out_path = substyle_dir / "style_diversity.json"
    try:
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return report


# --- pretty printing --------------------------------------------------------

def _print_report(result: dict) -> None:
    if result.get("error"):
        print(f"  [!] {result['error']} (parsed {result['n_styles']}, skipped {result.get('skipped')})")
        return
    names = result["names"]
    matrix = result["matrix"]
    short = [n.replace("style_", "s") for n in names]

    print(f"  mode={result['mode']}  styles={result['n_styles']}  skipped={result['skipped']}")
    header = "        " + " ".join(f"{s:>5}" for s in short)
    print(header)
    for i, row in enumerate(matrix):
        cells = " ".join(f"{v:5.2f}" for v in row)
        print(f"  {short[i]:>5} {cells}")

    s = result["stats"]
    print(f"  min={s['min_pairwise']}  mean={s['mean_pairwise']}  max={s['max_pairwise']}")
    cp = result["closest_pair"]
    if cp:
        print(f"  closest pair: {cp[0]} ~ {cp[1]}  (dist={cp[2]})")

    print("  per-style structural features:")
    feats = result["features"]
    cols = ["nodes", "depth", "max_grid_cols", "n_collections", "has_tabcontrol", "has_treeview", "n_afford_regions", "n_content_hosts"]
    print("        " + " ".join(f"{c[:9]:>10}" for c in cols))
    for n in names:
        f = feats[n]
        vals = " ".join(f"{str(f[c]):>10}" for c in cols)
        print(f"  {n.replace('style_','s'):>5} {vals}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("substyle_dir", help="path to a run's 03_substyle_design directory")
    ap.add_argument("--mode", choices=["main", "corpus", "both"], default="both")
    ap.add_argument("--json", help="optional path to dump full result as JSON")
    args = ap.parse_args(argv)

    modes = ["main", "corpus"] if args.mode == "both" else [args.mode]
    out = {}
    for mode in modes:
        print(f"\n=== {Path(args.substyle_dir).parent.name}  [{mode}] ===")
        res = analyze_blueprints(args.substyle_dir, mode=mode)
        _print_report(res)
        out[mode] = res

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
