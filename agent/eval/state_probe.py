"""Probe a built app's REAL runtime-state schema for eval-task generation.

Runs the delivered app's headless `./release/run.sh --dump-state` and reads
`app_data/config.yaml`, then emits a machine-readable snapshot of the fields that
ACTUALLY exist (real dot-paths + sample values + types), plus the SQLite tables and
import/sample globs. EvalTaskGenerate consumes this snapshot so it can author
`config_seed` / `infra_checks` against fields that truly exist in THIS app, instead of
guessing paths from prose. This removes dump-state field-path false-negatives without
constraining how each app names its own state.

The snapshot is descriptive, not prescriptive: every app is free to name its state
however it likes; the probe simply reports the ground truth per app.

CLI:
    python -m agent.eval.state_probe --app-dir <app> [--out state_schema.json]
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import re
import subprocess
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if value is None:
        return "null"
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"
    if isinstance(value, time):
        return "time"
    if isinstance(value, bytes):
        return "bytes"
    if isinstance(value, bytearray):
        return "bytearray"
    if isinstance(value, set):
        return "set"
    if isinstance(value, frozenset):
        return "frozenset"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _json_mapping_key(value: Any, path: str) -> str:
    """Return a stable JSON object key without erasing a non-string key's type."""
    if isinstance(value, str):
        return value
    if value is None:
        encoded = "null"
    elif isinstance(value, bool):
        encoded = "true" if value else "false"
    elif isinstance(value, int):
        encoded = str(value)
    elif isinstance(value, float):
        if math.isnan(value):
            encoded = "NaN"
        elif math.isinf(value):
            encoded = "Infinity" if value > 0 else "-Infinity"
        else:
            encoded = repr(value)
    elif isinstance(value, datetime):
        encoded = value.isoformat()
    elif isinstance(value, (date, time)):
        encoded = value.isoformat()
    elif isinstance(value, (bytes, bytearray)):
        encoded = base64.b64encode(bytes(value)).decode("ascii")
    elif isinstance(value, tuple):
        encoded = json.dumps(
            _json_safe_value(value, path),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        raise TypeError(
            f"{path}: unsupported mapping-key type "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )
    return f"<{_type_name(value)}:{encoded}>"


def _json_safe_value(value: Any, path: str = "$") -> Any:
    """Recursively normalize a known value category to deterministic strict JSON.

    The caller records the original Python type separately. Known YAML-native
    extensions get explicit, stable encodings; arbitrary objects are rejected rather
    than silently stringified.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"non_finite_float": "NaN"}
        if math.isinf(value):
            return {
                "non_finite_float": "Infinity" if value > 0 else "-Infinity"
            }
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return {
            "encoding": "base64",
            "data": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_json_safe_value(item, f"{path}[]") for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if isinstance(value, (list, tuple)):
        return [
            _json_safe_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        normalized_dict: Dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = _json_mapping_key(key, f"{path}.<key>")
            if normalized_key in normalized_dict:
                raise TypeError(
                    f"{path}: mapping keys collide after JSON normalization: "
                    f"{normalized_key!r}"
                )
            normalized_dict[normalized_key] = _json_safe_value(
                item, f"{path}.{normalized_key}"
            )
        return normalized_dict
    raise TypeError(
        f"{path}: unsupported value type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _flatten_scalars(obj: Any, prefix: str = "", out: Optional[Dict[str, Any]] = None, max_depth: int = 8) -> Dict[str, Any]:
    """Flatten a JSON/YAML object to leaf dot-paths -> value.

    Dot-path semantics match run_eval._dig: dict keys and list integer indices. For
    lists we descend only into element 0 (as ``path.0...``) to keep a representative
    sample; the list itself is also recorded as a leaf when it holds scalars/empty.
    """
    if out is None:
        out = {}
    if max_depth < 0:
        return out
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_text = _json_mapping_key(key, f"{prefix or '$'}.<key>")
            child = f"{prefix}.{key_text}" if prefix else key_text
            if isinstance(value, dict) and value:
                _flatten_scalars(value, child, out, max_depth - 1)
            elif isinstance(value, list):
                if value and isinstance(value[0], (dict, list)):
                    _flatten_scalars(value[0], f"{child}.0", out, max_depth - 1)
                    out[child] = {"type": "list", "len": len(value), "value": None}
                else:
                    out[child] = {
                        "type": "list",
                        "len": len(value),
                        "value": _json_safe_value(value[:5], child),
                    }
            else:
                out[child] = {
                    "type": _type_name(value),
                    "value": _json_safe_value(value, child),
                }
    elif isinstance(obj, list):
        if obj and isinstance(obj[0], (dict, list)):
            _flatten_scalars(obj[0], f"{prefix}.0", out, max_depth - 1)
        out[prefix or "$"] = {"type": "list", "len": len(obj), "value": None}
    else:
        sample_path = prefix or "$"
        out[sample_path] = {
            "type": _type_name(obj),
            "value": _json_safe_value(obj, sample_path),
        }
    return out


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON object extraction from CLI stdout (may include stray lines)."""
    text = (text or "").strip()
    candidates = [text]
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{"): text.rfind("}") + 1])
    for cand in candidates:
        if not cand:
            continue
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return {}


def probe_dump_state(app_dir: Path, timeout: int = 60) -> Dict[str, Any]:
    """Run the app's headless --dump-state and return {ok, paths, raw, error}."""
    run_sh = app_dir / "release" / "run.sh"
    result: Dict[str, Any] = {"ok": False, "paths": {}, "raw": "", "error": ""}
    if not run_sh.exists():
        result["error"] = f"missing {run_sh}"
        return result
    try:
        proc = subprocess.run(
            ["bash", "./release/run.sh", "--dump-state"],
            cwd=str(app_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        raw = proc.stdout or ""
        obj = _extract_json(raw)
        if not obj:
            result["error"] = f"unparseable dump-state (exit={proc.returncode}); stderr={proc.stderr[:300]}"
            result["raw"] = raw[:2000]
            return result
        result["paths"] = _flatten_scalars(obj)
        result["raw"] = json.dumps(obj, ensure_ascii=False)[:6000]
        result["ok"] = True
    except subprocess.TimeoutExpired:
        result["error"] = f"--dump-state timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def read_config_yaml(app_dir: Path) -> Dict[str, Any]:
    """Parse app_data/config.yaml and return {ok, top_level_keys, paths, error}.

    ``top_level_keys`` is what the config_seed mechanism can shallow-merge safely
    (SEED_CONFIG_BODY does a top-level dict.update); nested keys are reported under
    ``paths`` for visibility but flagged as not directly config_seed-able.
    """
    cfg = app_dir / "app_data" / "config.yaml"
    result: Dict[str, Any] = {
        "ok": False,
        "top_level_keys": [],
        "unsupported_top_level_keys": [],
        "paths": {},
        "error": "",
    }
    if not cfg.exists():
        result["error"] = f"missing {cfg}"
        return result
    text = cfg.read_text(encoding="utf-8", errors="replace")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            result["error"] = "config.yaml is not a mapping"
            return result
        result["top_level_keys"] = sorted(
            key for key in data.keys() if isinstance(key, str)
        )
        result["unsupported_top_level_keys"] = sorted(
            _json_mapping_key(key, "$.<key>")
            for key in data.keys()
            if not isinstance(key, str)
        )
        result["paths"] = _flatten_scalars(data)
        result["ok"] = True
        return result
    except ImportError:
        # PyYAML absent (mirrors SEED_CONFIG_BODY's fallback). JSON is valid YAML and
        # some apps emit config.yaml as JSON, so try a JSON parse first; only fall back
        # to the indentation line-parser for genuine block-style YAML.
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                result["top_level_keys"] = sorted(data.keys())
                result["paths"] = _flatten_scalars(data)
                result["error"] = "PyYAML unavailable; parsed as JSON"
                result["ok"] = True
                return result
        except Exception:  # noqa: BLE001
            pass
        result.update(_fallback_parse_yaml(text))
        result["error"] = "PyYAML unavailable; used indentation fallback parser"
        return result
    except Exception as exc:  # noqa: BLE001
        result["ok"] = False
        result["top_level_keys"] = []
        result["paths"] = {}
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


_YAML_LINE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)\s*:\s*(?P<val>.*?)\s*$")


def _coerce_scalar(raw: str) -> Any:
    if raw == "" or raw in {"~", "null"}:
        return None
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw.strip('"').strip("'")


def _fallback_parse_yaml(text: str) -> Dict[str, Any]:
    """Indentation-based parse of the simple machine-generated config.yaml.

    Handles top-level scalars and one level of nested mappings (the shapes the
    pipeline actually emits). Returns {ok, top_level_keys, paths}.
    """
    top_keys: List[str] = []
    paths: Dict[str, Any] = {}
    current_parent: Optional[str] = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#") or line.lstrip().startswith("- "):
            continue
        m = _YAML_LINE.match(line)
        if not m:
            continue
        indent, key, val = len(m.group("indent")), m.group("key"), m.group("val")
        if indent == 0:
            if val == "":  # opens a nested mapping
                current_parent = key
                if key not in top_keys:
                    top_keys.append(key)
            else:
                current_parent = None
                if key not in top_keys:
                    top_keys.append(key)
                paths[key] = {"type": "scalar", "value": _coerce_scalar(val)}
        elif current_parent is not None and val != "":
            paths[f"{current_parent}.{key}"] = {"type": "scalar", "value": _coerce_scalar(val)}
    return {"ok": bool(top_keys), "top_level_keys": sorted(top_keys), "paths": paths}


def probe_sqlite_tables(app_dir: Path) -> Dict[str, Any]:
    db = app_dir / "app_data" / "app.db"
    result: Dict[str, Any] = {"ok": False, "tables": [], "error": ""}
    if not db.exists():
        result["error"] = "no app_data/app.db"
        return result
    try:
        import sqlite3

        conn = sqlite3.connect(str(db))
        rows = conn.execute("select name from sqlite_master where type='table' order by name").fetchall()
        conn.close()
        result["ok"] = True
        result["tables"] = [r[0] for r in rows if not str(r[0]).startswith("sqlite_")]
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def probe_paths(app_dir: Path) -> Dict[str, Any]:
    """List import/sample globs available for path_glob checks."""
    out: Dict[str, Any] = {}
    for rel in ("app_data/imports", "app_data/samples/import_example", "app_data/samples/open_example"):
        p = app_dir / rel
        out[rel] = sorted([c.name for c in p.iterdir()]) if p.is_dir() else None
    return out


def build_state_schema(app_dir: Path, timeout: int = 60) -> Dict[str, Any]:
    app_dir = Path(app_dir)
    dump = probe_dump_state(app_dir, timeout=timeout)
    cfg = read_config_yaml(app_dir)
    sqlite = probe_sqlite_tables(app_dir)
    paths = probe_paths(app_dir)
    schema = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "app_dir": str(app_dir),
        "dump_state": dump,
        "config_yaml": cfg,
        "sqlite": sqlite,
        "fs_paths": paths,
        "usable": {
            "dump_state_fields": sorted(dump.get("paths", {}).keys()),
            "config_seed_top_level_keys": cfg.get("top_level_keys", []),
            "sqlite_tables": sqlite.get("tables", []),
        },
        "notes": [
            "dump_state_fields: exact dot-paths that exist in THIS app's --dump-state output; infra_checks[type=dump_state].field MUST be chosen from here.",
            "config_seed_top_level_keys: keys the config_seed mechanism can shallow-merge into config.yaml (top-level only); prefer these for setup.config.",
            "sqlite_tables: real table names for infra_checks[type=sqlite_count].table.",
            "This snapshot is descriptive ground truth per app; it does not constrain how the app names state.",
        ],
    }
    # Fail here, close to the probe, if a future field bypasses the explicit
    # normalization contract. Runner treats whole-schema failures as fatal.
    json.dumps(schema, ensure_ascii=False, allow_nan=False)
    return schema


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Probe a built app's real runtime-state schema.")
    p.add_argument("--app-dir", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None, help="Write schema JSON here (default: <app-dir>/../state_schema.json is NOT used; prints to stdout if omitted).")
    p.add_argument("--timeout", type=int, default=60)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    schema = build_state_schema(args.app_dir, timeout=args.timeout)
    text = json.dumps(schema, ensure_ascii=False, indent=2, allow_nan=False)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        u = schema["usable"]
        print(f"wrote {args.out}")
        print(f"  dump_state ok={schema['dump_state']['ok']} fields={len(u['dump_state_fields'])}")
        print(f"  config_yaml ok={schema['config_yaml']['ok']} top_keys={len(u['config_seed_top_level_keys'])}")
        print(f"  sqlite tables={u['sqlite_tables']}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
