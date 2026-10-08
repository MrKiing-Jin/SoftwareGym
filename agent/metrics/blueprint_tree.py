"""Normalize Avalonia/XAML UI blueprints into a software-agnostic "region
skeleton" tree.

The goal is to keep only the signal that defines *layout and interaction
structure* (how regions are arranged, which navigation/collection paradigms are
used) and to discard everything that is functionally driven or cosmetic
(menu commands, colors, text, sizes, x:Name, icons, individual buttons).

Design decisions (see calibration notes in style_diversity.py):

* Top menu bars (``Menu`` / ``ContextMenu``) are collapsed to a single
  ``MENUBAR`` leaf. They are dominated by the per-software command list and
  would otherwise mask real stylistic differences.
* ``Grid`` nodes are labeled with a normalized row/column signature
  (``A`` = Auto, ``F`` = fixed px, ``*`` = star), because the number and shape
  of panes is the strongest software-agnostic layout signal.
* ``DockPanel`` children carry their ``Dock`` edge as a label suffix.
* Collection controls (ListBox/TreeView/DataGrid/TabControl/...) become a
  single ``COLLECTION:<type>`` leaf -- their item content is data, not layout.
* Runs of interactive leaf controls (buttons/sliders/inputs) collapse to a
  single ``AFFORD`` marker per container: "this region has actions" without
  counting individual commands.
* ``ContentControl`` / ``ContentPresenter`` with ``Tag="SomeView.axaml"`` is
  inlined by parsing the referenced sibling blueprint, so a shell that hosts
  views by reference yields one coherent tree regardless of file count.

The module is deliberately dependency-free (xml.etree + stdlib only).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# --- control-type vocabulary ------------------------------------------------

# Layout containers we keep as structural nodes.
_LAYOUT_CONTAINERS = {
    "Grid",
    "DockPanel",
    "StackPanel",
    "WrapPanel",
    "Canvas",
    "UniformGrid",
    "RelativePanel",
    "ScrollViewer",
    "Border",
    "Panel",
    "SplitView",
    "Expander",
    "GroupBox",
    "HeaderedContentControl",
    "Viewbox",
    "LayoutTransformControl",
    "TransitioningContentControl",
}

# Content hosts that may reference a sub-view via Tag.
_CONTENT_HOSTS = {"ContentControl", "ContentPresenter"}

# Collection / paradigm controls -> collapsed leaf, NOT recursed.
_COLLECTIONS = {
    "ListBox",
    "ListView",
    "TreeView",
    "DataGrid",
    "ItemsControl",
    "ItemsRepeater",
    "ItemsPresenter",
    "TabControl",
    "TabStrip",
    "Carousel",
    "ComboBox",
    "ListView",
}

# Interactive leaves: presence => the parent region has an action surface.
_AFFORDANCES = {
    "Button",
    "ToggleButton",
    "RepeatButton",
    "RadioButton",
    "CheckBox",
    "Slider",
    "TextBox",
    "AutoCompleteBox",
    "NumericUpDown",
    "MaskedTextBox",
    "DatePicker",
    "TimePicker",
    "CalendarDatePicker",
    "ColorPicker",
    "SplitButton",
    "DropDownButton",
    "ToggleSwitch",
}

# Menu bars: collapse whole subtree.
_MENUBARS = {"Menu", "ContextMenu", "MenuFlyout"}

# Pure decoration / text / glyphs -> ignored entirely.
_DECORATION = {
    "TextBlock",
    "SelectableTextBlock",
    "Image",
    "PathIcon",
    "Path",
    "Separator",
    "ProgressBar",
    "Ellipse",
    "Rectangle",
    "Line",
    "Polygon",
    "Polyline",
    "Svg",
    "AccessText",
    "Label",
}

# Roots / wrappers that should be made transparent (recurse into children).
_TRANSPARENT_ROOTS = {"Window", "UserControl", "Page"}


def _local(tag: str) -> str:
    """Strip namespace prefix from a tag/attr name ('x:Name' -> 'Name').

    Attached-property dots are preserved ('DockPanel.Dock' stays intact).
    """
    if not tag:
        return ""
    return tag.split(":")[-1]


# --- tolerant tag-stream parser ---------------------------------------------
#
# The blueprints are *generated* near-XML and routinely break strict parsers
# (duplicate attributes after prefix stripping, unescaped & / < in body text,
# stray entities). We only need the element tree + a few layout attributes, so
# we tokenize tags directly and ignore body text entirely. This is strictly
# more robust than xml.etree for our purpose.


class _LiteElement:
    """Minimal ElementTree-compatible element (tag, attrib, children)."""

    __slots__ = ("tag", "attrib", "_children")

    def __init__(self, tag: str, attrib: Optional[dict] = None):
        self.tag = tag
        self.attrib = attrib or {}
        self._children: list["_LiteElement"] = []

    def get(self, key: str, default=None):
        return self.attrib.get(key, default)

    def __iter__(self):
        return iter(self._children)

    def append(self, child: "_LiteElement"):
        self._children.append(child)


_TAG_RE = re.compile(r"<(/?)([A-Za-z_][\w.\-:]*)([^>]*?)(/?)>", re.DOTALL)
_ATTR_RE = re.compile(r"([\w.\-:]+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')", re.DOTALL)
_VOID_RE = re.compile(r"<!--.*?-->|<!\[CDATA\[.*?\]\]>|<![^>]*>|<\?.*?\?>", re.DOTALL)


def _parse_lenient(text: str) -> Optional[_LiteElement]:
    """Build a tree from a near-XML blueprint, tolerating malformed markup."""
    text = _VOID_RE.sub("", text)
    root = _LiteElement("#doc")
    stack = [root]
    for m in _TAG_RE.finditer(text):
        closing, raw_name, attr_str, self_close = m.groups()
        name = _local(raw_name)
        if closing:
            # pop to the matching open tag if present, else ignore
            for depth in range(len(stack) - 1, 0, -1):
                if stack[depth].tag == name:
                    del stack[depth:]
                    break
            continue
        attrib: dict[str, str] = {}
        for am in _ATTR_RE.finditer(attr_str):
            key = _local(am.group(1))
            val = am.group(2) if am.group(2) is not None else (am.group(3) or "")
            attrib[key] = val  # duplicate -> last wins
        elem = _LiteElement(name, attrib)
        stack[-1].append(elem)
        if not self_close:
            stack.append(elem)
    # the document's first real child is the logical root (Window/UserControl)
    for child in root:
        return child
    return None


@dataclass
class Node:
    """A node in the normalized region-skeleton tree."""

    label: str
    children: list["Node"] = field(default_factory=list)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)


# --- grid / dock signatures -------------------------------------------------

_SIZE_TOKEN = re.compile(r"^\s*(\d+(\.\d+)?|\*|\d*\*|Auto)\s*$", re.IGNORECASE)


def _normalize_definitions(spec: str) -> str:
    """Turn 'Auto,*,2*,260' into 'A,*,*,F' (Auto / star / fixed)."""
    tokens = []
    for raw in spec.split(","):
        t = raw.strip()
        if not t:
            continue
        low = t.lower()
        if low == "auto":
            tokens.append("A")
        elif low.endswith("*"):
            tokens.append("*")
        else:
            tokens.append("F")
    return ",".join(tokens)


def _grid_signature(elem: "_LiteElement") -> str:
    """Build a normalized 'Grid r=... c=...' label from a Grid element."""
    rows = elem.get("RowDefinitions")
    cols = elem.get("ColumnDefinitions")

    # long-form <Grid.RowDefinitions> / <Grid.ColumnDefinitions>
    if rows is None or cols is None:
        for child in list(elem):
            ltag = _local(child.tag)
            if ltag.endswith("RowDefinitions") and rows is None:
                rows = ",".join(
                    (rd.get("Height") or "*") for rd in child if _local(rd.tag).endswith("RowDefinition")
                )
            elif ltag.endswith("ColumnDefinitions") and cols is None:
                cols = ",".join(
                    (cd.get("Width") or "*") for cd in child if _local(cd.tag).endswith("ColumnDefinition")
                )

    parts = ["Grid"]
    if rows:
        parts.append("r=" + _normalize_definitions(rows))
    if cols:
        parts.append("c=" + _normalize_definitions(cols))
    return " ".join(parts)


def _dock_suffix(elem: "_LiteElement") -> str:
    dock = elem.get("DockPanel.Dock") or elem.get("Dock")
    if dock:
        return "@" + dock.strip().lower()
    return ""


# --- core builder -----------------------------------------------------------

class BlueprintParser:
    """Parses one ui-blueprint directory worth of XAML into skeleton trees."""

    def __init__(self, blueprint_dir: Path, max_inline_depth: int = 4):
        self.dir = Path(blueprint_dir)
        self.max_inline_depth = max_inline_depth
        # filename (lowercased, without extension) -> path
        self._index: dict[str, Path] = {}
        for p in _blueprint_files(self.dir):
            self._index[p.stem.lower()] = p

    # -- public -------------------------------------------------------------

    def parse_file(self, path: Path, _inline_depth: int = 0, _seen: Optional[set] = None) -> Optional[Node]:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        return self.parse_text(text, _inline_depth=_inline_depth, _seen=_seen or set())

    def parse_text(self, text: str, _inline_depth: int = 0, _seen: Optional[set] = None) -> Optional[Node]:
        root = _parse_lenient(text)
        if root is None:
            return None
        seen = _seen if _seen is not None else set()
        children = self._convert_children(root, _inline_depth, seen)
        if not children:
            return None
        return Node(label="ROOT", children=children)

    # -- internal -----------------------------------------------------------

    def _convert_children(self, elem: "_LiteElement", inline_depth: int, seen: set) -> list[Node]:
        """Convert the meaningful child controls of *elem* into skeleton nodes,
        collapsing affordance runs into a single AFFORD marker."""
        out: list[Node] = []
        has_afford = False
        for child in list(elem):
            tag = _local(child.tag)
            if "." in tag:
                # property element (e.g. Grid.RowDefinitions, Border.Background,
                # TreeViewItem.Header) -> not a control; skip its subtree.
                continue
            node = self._convert_element(child, inline_depth, seen)
            if node is _AFFORD_SENTINEL:
                has_afford = True
                continue
            if node is not None:
                out.append(node)
        if has_afford:
            out.append(Node(label="AFFORD"))
        return out

    def _convert_element(self, elem: "_LiteElement", inline_depth: int, seen: set):
        tag = _local(elem.tag)

        if tag in _DECORATION:
            return None
        if tag in _AFFORDANCES:
            return _AFFORD_SENTINEL
        if tag in _MENUBARS:
            return Node(label="MENUBAR")
        if tag in _COLLECTIONS:
            return Node(label=f"COLLECTION:{tag}" + _dock_suffix(elem))

        if tag in _CONTENT_HOSTS:
            inlined = self._maybe_inline(elem, inline_depth, seen)
            label = "ContentHost" + _dock_suffix(elem)
            if inlined is not None:
                return Node(label=label, children=[inlined])
            return Node(label=label)

        if tag in _LAYOUT_CONTAINERS or tag in _TRANSPARENT_ROOTS:
            if tag == "Grid":
                label = _grid_signature(elem) + _dock_suffix(elem)
            else:
                label = tag + _dock_suffix(elem)
            children = self._convert_children(elem, inline_depth, seen)
            if tag in _TRANSPARENT_ROOTS:
                # a Window/UserControl wrapper: keep a stable ROOT-ish node
                return Node(label=tag, children=children)
            return Node(label=label, children=children)

        # Unknown / custom control: treat as an opaque region node but still
        # recurse (it may host layout). Drop the namespace-y class noise.
        children = self._convert_children(elem, inline_depth, seen)
        if children:
            return Node(label="Region" + _dock_suffix(elem), children=children)
        return Node(label="Region" + _dock_suffix(elem))

    def _maybe_inline(self, elem: "_LiteElement", inline_depth: int, seen: set) -> Optional[Node]:
        tag_attr = elem.get("Tag") or ""
        m = re.search(r"([A-Za-z0-9_]+)\.(?:axaml|xaml)", tag_attr)
        if not m:
            return None
        if inline_depth >= self.max_inline_depth:
            return None
        stem = m.group(1).lower()
        target = self._index.get(stem)
        if target is None or stem in seen:
            return None
        seen = seen | {stem}
        sub = self.parse_file(target, _inline_depth=inline_depth + 1, _seen=seen)
        return sub


# Sentinel object distinct from None / Node used to flag affordance leaves.
_AFFORD_SENTINEL = object()


# --- high-level entry points ------------------------------------------------

_DIALOG_HINTS = (
    "dialog",
    "wizard",
    "overlay",
    "confirm",
    "about",
    "help",
    "settings",
    "onboarding",
    "welcome",
    "recovery",
    "export",
    "import",
    "keyboard",
    "shortcut",
    "popup",
    "flyout",
    "toast",
)


def _is_dialog_file(path: Path) -> bool:
    name = path.stem.lower()
    return any(h in name for h in _DIALOG_HINTS)


def _blueprint_files(blueprint_dir: Path) -> list[Path]:
    """Prefer new .axaml blueprints while retaining offline legacy .xaml reads."""
    by_stem = {
        path.stem.lower(): path for path in Path(blueprint_dir).glob("*.xaml")
    }
    by_stem.update(
        {
            path.stem.lower(): path
            for path in Path(blueprint_dir).glob("*.axaml")
        }
    )
    return sorted(by_stem.values(), key=lambda path: path.name.lower())


def _find_main(blueprint_dir: Path) -> Optional[Path]:
    files = {p.stem.lower(): p for p in _blueprint_files(blueprint_dir)}
    for key in ("mainwindow", "shell", "appshell", "mainview"):
        if key in files:
            return files[key]
    # fallback: any window-ish file
    for stem, p in files.items():
        if "main" in stem or "window" in stem:
            return p
    return None


def build_skeleton(blueprint_dir, mode: str = "main") -> Optional[Node]:
    """Build a single region-skeleton tree for one style's ui-blueprint dir.

    mode="main"   -> MainWindow shell (inlines Tag="View.axaml" references).
    mode="corpus" -> forest of every non-style, non-dialog blueprint merged
                     under a synthetic ROOT (filename-sorted for determinism).
    """
    blueprint_dir = Path(blueprint_dir)
    parser = BlueprintParser(blueprint_dir)

    if mode == "main":
        main = _find_main(blueprint_dir)
        if main is None:
            return None
        return parser.parse_file(main)

    if mode == "corpus":
        files = sorted(
            p
            for p in _blueprint_files(blueprint_dir)
            if p.stem.lower() not in ("appstyle", "styles", "theme", "themes")
            and not _is_dialog_file(p)
        )
        children: list[Node] = []
        for p in files:
            sub = parser.parse_file(p)
            if sub is not None:
                children.append(Node(label=f"VIEW:{p.stem}", children=sub.children))
        if not children:
            return None
        return Node(label="ROOT", children=children)

    raise ValueError(f"unknown mode: {mode}")


# --- interpretable feature vector (diagnostics) -----------------------------

def feature_summary(tree: Node) -> dict:
    """Extract a few interpretable, software-agnostic structural features used
    for human-readable diagnostics alongside the TED matrix."""
    labels: list[str] = []

    def walk(n: Node, depth: int):
        labels.append(n.label)
        for c in n.children:
            walk(c, depth + 1)

    def max_depth(n: Node) -> int:
        if not n.children:
            return 1
        return 1 + max(max_depth(c) for c in n.children)

    walk(tree, 0)

    grid_cols = []
    for lab in labels:
        m = re.search(r"c=([A-Za-z*,]+)", lab)
        if m:
            grid_cols.append(len(m.group(1).split(",")))

    return {
        "nodes": tree.size(),
        "depth": max_depth(tree),
        "max_grid_cols": max(grid_cols) if grid_cols else 0,
        "n_grids": sum(1 for l in labels if l.startswith("Grid")),
        "n_dockpanels": sum(1 for l in labels if l.startswith("DockPanel")),
        "n_collections": sum(1 for l in labels if l.startswith("COLLECTION")),
        "has_tabcontrol": any("COLLECTION:TabControl" in l for l in labels),
        "has_treeview": any("COLLECTION:TreeView" in l for l in labels),
        "n_afford_regions": sum(1 for l in labels if l == "AFFORD"),
        "n_content_hosts": sum(1 for l in labels if l.startswith("ContentHost")),
    }
