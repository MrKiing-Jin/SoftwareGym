from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from agent.docker.worker_manager import DEFAULT_BASE_IMAGE, iter_port_groups, port_group_free
from agent.runtime_env import env_for_osworld, env_for_softwaregym, load_env_file, softwaregym_python


REPO_ROOT = Path(__file__).resolve().parents[2]
load_env_file(REPO_ROOT / ".env")


def check_command(name: str) -> Dict[str, object]:
    path = shutil.which(name)
    return {"name": name, "ok": path is not None, "path": path}


def check_python_env(name: str, path: str, require_openai_responses: bool = False, env: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    executable = Path(path)
    result: Dict[str, object] = {"name": name, "path": str(executable), "exists": executable.exists(), "ok": False}
    if not executable.exists():
        return result
    snippet = "import sys; print(sys.executable)"
    if require_openai_responses:
        snippet += "; import openai; print(getattr(openai, '__version__', 'unknown'), hasattr(openai.OpenAI(api_key='x'), 'responses'))"
    probe = subprocess.run(
        [
            str(executable),
            "-c",
            snippet,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    result["returncode"] = probe.returncode
    result["stdout"] = probe.stdout.strip()
    result["stderr"] = probe.stderr.strip()[-500:]
    result["ok"] = probe.returncode == 0
    return result


def check_ports() -> Dict[str, object]:
    checked = 0
    for group in iter_port_groups():
        checked += 1
        if port_group_free(group):
            return {
                "groups": [{"ports": group, "free": True}],
                "checked_groups": checked,
                "any_free": True,
            }
    return {"groups": [], "checked_groups": checked, "any_free": False}


def run_check(base_image: Path = DEFAULT_BASE_IMAGE) -> Dict[str, object]:
    commands = [check_command(name) for name in ["docker", "qemu-img", "cloudflared", "codex", "python3"]]
    env = {
        "API_KEY": "set" if os.environ.get("API_KEY") else "missing",
        "OPENAI_API_KEY": "set" if os.environ.get("OPENAI_API_KEY") else "missing",
        "BASE_URL": os.environ.get("BASE_URL", "missing"),
        "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", "missing"),
        "MODEL": os.environ.get("MODEL", "missing"),
    }
    python_envs = [
        check_python_env("softwaregym", softwaregym_python(), env=env_for_softwaregym(os.environ)),
        check_python_env("softwaregym-eval", softwaregym_python(), require_openai_responses=True, env=env_for_osworld(os.environ)),
    ]
    return {
        "commands": commands,
        "python_envs": python_envs,
        "base_image": {"path": str(base_image), "exists": base_image.exists(), "readable": os.access(base_image, os.R_OK)},
        "ports": check_ports(),
        "env": env,
        "ok": all(item["ok"] for item in commands if item["name"] in {"docker", "qemu-img", "python3"})
        and all(item["ok"] for item in python_envs)
        and base_image.exists()
        and check_ports()["any_free"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check software-gym runtime environment.")
    parser.add_argument("--base-image", type=Path, default=DEFAULT_BASE_IMAGE)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_check(args.base_image)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for command in result["commands"]:  # type: ignore[index]
            print(f"{command['name']}: {'ok' if command['ok'] else 'missing'} {command.get('path') or ''}")
        for pyenv in result["python_envs"]:  # type: ignore[index]
            print(f"{pyenv['name']}_python: {'ok' if pyenv['ok'] else 'failed'} {pyenv['path']}")
        base = result["base_image"]  # type: ignore[index]
        print(f"base_image: {'ok' if base['exists'] and base['readable'] else 'missing'} {base['path']}")
        print(f"ports_any_free: {result['ports']['any_free']}")  # type: ignore[index]
        print(f"overall: {'ok' if result['ok'] else 'failed'}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
