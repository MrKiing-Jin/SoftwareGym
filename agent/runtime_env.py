from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, MutableMapping


REPO_ROOT = Path(__file__).resolve().parents[1]
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_env_value(raw_value: str) -> str:
    """Parse a simple dotenv value and discard unquoted trailing comments."""
    value = raw_value.strip()
    if not value:
        return ""
    if value[0] in {"\"", "'"}:
        quote = value[0]
        end = value.find(quote, 1)
        if end >= 0:
            return value[1:end]
        return value[1:]
    # In an unquoted value, a # preceded by whitespace starts a comment.
    # Keep embedded hashes such as URL fragments (for example, token#part).
    comment_at = next(
        (index for index, char in enumerate(value) if char == "#" and index > 0 and value[index - 1].isspace()),
        None,
    )
    if comment_at is not None:
        value = value[:comment_at]
    return value.rstrip()


def load_env_file(path: Path | None = None, override: bool = False) -> Dict[str, str]:
    env_path = path or (REPO_ROOT / ".env")
    loaded: Dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _parse_env_value(value)
        if not key:
            continue
        force_override = override or _truthy(os.environ.get("SOFTWAREGYM_ENV_OVERRIDE"))
        if force_override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


def softwaregym_python() -> str:
    return (
        os.environ.get("SOFTWAREGYM_PYTHON")
        or os.environ.get("APPEVAL_PYTHON")
        or current_python()
    )


def current_python() -> str:
    return sys.executable


def _prepend_path(env: MutableMapping[str, str], directory: Path) -> None:
    current = env.get("PATH", "")
    if not current or "$PATH" in current:
        current = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    env["PATH"] = str(directory) + (os.pathsep + current if current else "")


def _proxy_disabled(env: MutableMapping[str, str]) -> bool:
    return _truthy(env.get("SOFTWAREGYM_NO_PROXY"))


def _apply_proxy_policy(env: MutableMapping[str, str]) -> None:
    if not _proxy_disabled(env):
        return
    for key in PROXY_ENV_KEYS:
        env.pop(key, None)
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"


def env_for_softwaregym(base_env: MutableMapping[str, str] | None = None) -> Dict[str, str]:
    env = dict(base_env or os.environ)
    python_path = Path(softwaregym_python()).absolute()
    env_root = python_path.parent.parent
    env["VIRTUAL_ENV"] = str(env_root)
    env["PYTHONNOUSERSITE"] = "1"
    for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PROMPT_MODIFIER", "CONDA_PYTHON_EXE"):
        env.pop(key, None)
    _prepend_path(env, python_path.parent)
    _apply_proxy_policy(env)
    return env


def env_for_osworld(base_env: MutableMapping[str, str] | None = None) -> Dict[str, str]:
    env = dict(base_env or os.environ)
    python_path = Path(softwaregym_python()).absolute()
    env_root = python_path.parent.parent
    env["CONDA_PREFIX"] = str(env_root)
    env["CONDA_DEFAULT_ENV"] = env_root.name
    env["CONDA_PYTHON_EXE"] = str(python_path)
    env["CONDA_PROMPT_MODIFIER"] = f"({env_root.name}) "
    env.pop("PYTHONNOUSERSITE", None)
    env.pop("VIRTUAL_ENV", None)
    _prepend_path(env, python_path.parent)
    _apply_proxy_policy(env)
    return env


def activate_softwaregym_env_vars() -> None:
    base_env = dict(os.environ)
    os.environ.clear()
    os.environ.update(env_for_softwaregym(base_env))


def activate_osworld_env_vars() -> None:
    base_env = dict(os.environ)
    os.environ.clear()
    os.environ.update(env_for_osworld(base_env))


def ensure_not_base_for_eval() -> None:
    executable = Path(sys.executable).resolve()
    expected = Path(softwaregym_python()).resolve()
    if executable != expected:
        raise RuntimeError(f"GUI eval must run with softwaregym python: expected {expected}, got {executable}")


def ensure_softwaregym_for_runner() -> None:
    executable = Path(sys.executable).resolve()
    expected = Path(softwaregym_python()).resolve()
    if executable != expected:
        raise RuntimeError(f"Loop runner must run with softwaregym python: expected {expected}, got {executable}")


__all__ = [
    "activate_osworld_env_vars",
    "activate_softwaregym_env_vars",
    "current_python",
    "env_for_osworld",
    "env_for_softwaregym",
    "ensure_not_base_for_eval",
    "ensure_softwaregym_for_runner",
    "load_env_file",
        "softwaregym_python",
]
