"""Software-agnostic UI layout-archetype vocabulary (single source of truth).

This is the "diversity by construction" backbone for style heterogeneity. Rather
than only *measuring* how different styles turned out (which can fail late), we
ask StyleDesigner to commit each style to a distinct combination across a few
universal structural dimensions, then verify the commitment.

Two independent signals keep styles apart:

1. Declared archetype tuple (this module) -- cheap, reliable distinctness check
   on what StyleDesigner *intended* (min pairwise Hamming over the 4 dimensions).
2. Realized structural TED (style_diversity.py) -- did the blueprints SubStyle-
   Designer produced actually diverge.

The dimensions below are deliberately domain-neutral so the same catalog applies
to a table editor, a chat app, a video player, a code editor, etc. Each value is
a coarse, observable structural choice -- NOT a visual-theme tweak.

`catalog_markdown()` renders the canonical human-readable catalog that is embedded
verbatim in the StyleDesigner / SubStyleDesigner prompts, so prompt text and the
validator can never drift.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence, Tuple


# Each dimension -> ordered list of allowed values, with a one-line gloss.
DIMENSIONS: Dict[str, Dict[str, str]] = {
    "nav_paradigm": {
        "_desc": "How the user primarily reaches commands and sections.",
        "menu_bar": "Classic top menu bar drives most commands.",
        "top_toolbar": "Prominent top button/toolbar strip is the main command surface.",
        "left_icon_rail": "Narrow left activity rail of icon buttons switches modes.",
        "left_nav_list": "Left navigation list/tree of sections or objects.",
        "command_palette": "Command-palette / quick-open first; chrome stays minimal.",
    },
    "spatial_layout": {
        "_desc": "Macro arrangement of the main window.",
        "single_pane": "One focused work surface, little permanent chrome.",
        "two_pane": "Two-pane list+detail (master/detail).",
        "three_pane": "Three columns (nav + list + detail/content).",
        "rail_plus_workspace": "Activity rail + collapsible side panel(s) + central workspace.",
        "tabbed_documents": "Tabbed document/workspace area as the central metaphor.",
        "dashboard_tiles": "Tiled dashboard / card grid of modules.",
    },
    "object_browsing": {
        "_desc": "How the primary objects are listed and switched between.",
        "tree": "Hierarchical tree of objects.",
        "flat_list": "Flat scrollable list of objects.",
        "tabs": "Tab strip across open objects.",
        "cards_grid": "Grid of cards/thumbnails.",
        "drawer": "Objects live in a toggled drawer/overlay, hidden by default.",
    },
    "action_locus": {
        "_desc": "Where the primary actions for the current object live.",
        "top_toolbar": "Actions sit in a top contextual toolbar.",
        "inline_row": "Actions are inline on rows/items.",
        "right_inspector": "Actions live in a right-hand inspector/properties panel.",
        "floating_button": "A floating primary action button / speed dial.",
        "bottom_bar": "A bottom command/transport bar holds primary actions.",
    },
}

DIMENSION_ORDER: Tuple[str, ...] = tuple(DIMENSIONS.keys())

# Minimum number of dimensions (out of 4) that must differ between ANY two
# styles. 2 means "no two styles may share 3+ of the 4 structural choices".
MIN_PAIRWISE_HAMMING = 2


def allowed_values(dimension: str) -> List[str]:
    return [k for k in DIMENSIONS[dimension] if k != "_desc"]


def is_valid_value(dimension: str, value: str) -> bool:
    return dimension in DIMENSIONS and value in allowed_values(dimension)


def normalize_tuple(archetype: Dict[str, str]) -> Tuple[str, ...]:
    """Project a declared archetype dict onto the canonical dimension order.
    Missing/invalid values become '' so distinctness math still works."""
    out = []
    for dim in DIMENSION_ORDER:
        val = str(archetype.get(dim, "")).strip()
        out.append(val if is_valid_value(dim, val) else "")
    return tuple(out)


def tuple_hamming(a: Sequence[str], b: Sequence[str]) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def assignment_report(archetypes: Dict[str, Dict[str, str]]) -> Dict[str, object]:
    """Given {style_id: archetype_dict}, report distinctness + any violations.

    Returns min pairwise Hamming, the closest (most similar) pair, per-style
    invalid/missing fields, and whether the assignment satisfies the construction
    rule (all pairwise Hamming >= MIN_PAIRWISE_HAMMING and no invalid fields)."""
    tuples = {sid: normalize_tuple(a) for sid, a in archetypes.items()}

    invalid: Dict[str, List[str]] = {}
    for sid, a in archetypes.items():
        bad = []
        for dim in DIMENSION_ORDER:
            val = str(a.get(dim, "")).strip()
            if not is_valid_value(dim, val):
                bad.append(f"{dim}={val or '<missing>'}")
        if bad:
            invalid[sid] = bad

    pairs = []
    for s1, s2 in combinations(sorted(tuples), 2):
        h = tuple_hamming(tuples[s1], tuples[s2])
        pairs.append((s1, s2, h))

    min_h = min((h for _, _, h in pairs), default=None)
    closest = min(pairs, key=lambda p: p[2]) if pairs else None
    too_similar = [(s1, s2, h) for s1, s2, h in pairs if h < MIN_PAIRWISE_HAMMING]

    return {
        "dimension_order": list(DIMENSION_ORDER),
        "tuples": {sid: list(t) for sid, t in tuples.items()},
        "min_pairwise_hamming": min_h,
        "closest_pair": closest,
        "invalid_fields": invalid,
        "too_similar_pairs": too_similar,
        "passes": (not invalid) and (min_h is not None) and min_h >= MIN_PAIRWISE_HAMMING,
        "min_required_hamming": MIN_PAIRWISE_HAMMING,
    }


def catalog_markdown() -> str:
    """Canonical catalog text embedded verbatim in the prompts."""
    lines: List[str] = []
    for dim in DIMENSION_ORDER:
        spec = DIMENSIONS[dim]
        lines.append(f"- `{dim}` — {spec['_desc']}")
        for val in allowed_values(dim):
            lines.append(f"    - `{val}`: {spec[val]}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(catalog_markdown())
