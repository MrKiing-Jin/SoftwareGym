from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

_MANIFEST_LOCK = threading.Lock()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def update_variant_stage(variant_dir: Path, version: str, stage: str, status: str, **details: Any) -> None:
    path = variant_dir / "variant_manifest.json"
    with _MANIFEST_LOCK:
        manifest = load_json(path)
        manifest.setdefault("rounds", {})
        manifest["current_round"] = version
        manifest["status"] = status
        round_state = manifest["rounds"].setdefault(version, {})
        round_state[stage] = {"status": status, "updated_at": datetime.now().isoformat(timespec="seconds"), **details}
        write_json(path, manifest)


def update_manifest_variant(run_dir: Path, style_id: str, **details: Any) -> None:
    path = run_dir / "manifest.json"
    with _MANIFEST_LOCK:
        manifest = load_json(path)
        variants = manifest.setdefault("variants", {})
        state = variants.setdefault(style_id, {})
        state.update(details)
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(path, manifest)
