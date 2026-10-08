from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from agent.docker.worker_manager import DEFAULT_OVERLAY_DIR, port_group_for_offset
from agent.runtime_env import activate_softwaregym_env_vars, ensure_softwaregym_for_runner, load_env_file


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "agent"
PROMPT_DIR = AGENT_ROOT / "prompt"
SKILL_DIR = AGENT_ROOT / "skill"
RUNTIME_CONFIG = AGENT_ROOT / "config" / "agent-runtime.json"
OUTPUT_ROOT = REPO_ROOT / "output"

load_env_file(REPO_ROOT / ".env")


PROMPT_FILES = {
    "FunctionDesigner": "Functiondesignerprompt.json",
    "IconDesigner": "Icondesignerprompt.json",
    "StyleDesigner": "Styledesignerprompt.json",
    "SubStyleDesigner": "Substyledesignerprompt.json",
    "SubIconDesigner": "Subicondesignerprompt.json",
    "Coder": "Coderprompt.json",
    "DevCoder": "Devcoderprompt.json",
    "EvalTaskGenerateAgent": "Evaltaskgenerateprompt.json",
}


SKILL_DIRS = {
    "FunctionDesigner": "Functiondesignerskill",
    "IconDesigner": "Icondesignerskill",
    "StyleDesigner": "Styledesignerskill",
    "SubStyleDesigner": "Substyledesignerskill",
    "SubIconDesigner": "Subicondesignerskill",
    "Coder": "Coderskill",
    "DevCoder": "Devcoderskill",
    "EvalTaskGenerateAgent": "Evaltaskgenerateskill",
}


@dataclass(frozen=True)
class CodexCommand:
    agent_name: str
    work_dir: Path
    prompt_path: Path
    log_path: Path
    command: List[str]
    env: Dict[str, str]
    # Repair round: resume the previous same-role session by reading the session id
    # from this file at runtime; if empty, fall back to a fresh session.
    resume_session_file: Optional[str] = None
    resume_rest: Optional[List[str]] = None
    # Extra commands appended after the main codex command succeeds (e.g. icon validation).
    # Any failure fails the whole script and triggers a stage retry.
    post_commands: tuple[tuple[str, ...], ...] = ()

    def shell_script(self) -> str:
        env_lines = [f"export {key}={shlex.quote(value)}" for key, value in sorted(self.env.items())]
        fresh_line = shlex.join(self.command)
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", *env_lines]
        if self.resume_session_file and self.resume_rest is not None:
            rid_file = shlex.quote(self.resume_session_file)
            work_dir_q = shlex.quote(str(self.work_dir))
            # codex exec resume does not support --cd; cd into work_dir first to match the fresh branch cwd semantics.
            resume_line = 'codex exec resume "$RID" ' + shlex.join(self.resume_rest)
            lines += [
                'RID=""',
                f'if [ -f {rid_file} ]; then RID="$(tr -d "[:space:]" < {rid_file} || true)"; fi',
                'if [ -n "$RID" ]; then',
                '  echo "[resume] continuing codex session $RID"',
                f'  cd {work_dir_q}',
                f'  {resume_line}',
                'else',
                '  echo "[resume] no previous session id found; starting a fresh session"',
                f'  {fresh_line}',
                'fi',
            ]
        else:
            lines += [fresh_line]
        for post in self.post_commands:
            lines.append(shlex.join(post))
        lines.append("")
        return "\n".join(lines)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def link_path(target: Path, link: Path, target_is_directory: bool = False) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    relative_target = os.path.relpath(target, link.parent)
    link.symlink_to(relative_target, target_is_directory=target_is_directory)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def resource_dir_for_software(repo_root: Path, software_type: str) -> Path:
    resource_root = repo_root / "Resource"
    preferred = resource_root / f"{software_type}_resources"
    if preferred.exists():
        return preferred
    matches = sorted(resource_root.glob(f"{software_type}_resources*"))
    if matches:
        return matches[0]
    return preferred


def resource_positioning_for_software(resource_dir: Path) -> str:
    parameters_path = resource_dir / "pipeline_parameters.json"
    if parameters_path.exists() and parameters_path.is_file():
        try:
            parameters = load_json(parameters_path)
        except Exception:
            parameters = {}
        description = parameters.get("software_description") or parameters.get("product_positioning")
        if description:
            return str(description).strip()
        positioning_file = parameters.get("product_positioning_file")
        if positioning_file:
            candidate = resource_dir / str(positioning_file)
            if candidate.exists() and candidate.is_file():
                return candidate.read_text(encoding="utf-8").strip()
    for candidate in [
        resource_dir / "product_positioning.md",
        resource_dir / "positioning.md",
        # Legacy Chinese-named positioning file kept as a fallback for resource
        # drops produced before the English filename convention.
        resource_dir / "产品定位.md",
    ]:
        if candidate.exists() and candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    return ""


def eval_seedfile_for_software(resource_dir: Path) -> tuple[str, str]:
    """Resolve the optional eval seed-file directory + a human-readable manifest.

    Returns (seed_dir_abs_or_empty, manifest_text). When `eval_task_seedfile` is not
    configured or the directory is missing/empty, returns ("", "") and the eval-task
    prompt simply keeps its default (no seeded-open) behaviour.
    """
    parameters_path = resource_dir / "pipeline_parameters.json"
    if not (parameters_path.exists() and parameters_path.is_file()):
        return "", ""
    try:
        parameters = load_json(parameters_path)
    except Exception:
        return "", ""
    seed_value = str(parameters.get("eval_task_seedfile") or "").strip()
    if not seed_value:
        return "", ""
    seed_dir = Path(seed_value)
    if not seed_dir.is_absolute():
        seed_dir = (resource_dir / seed_value)
    seed_dir = seed_dir.resolve()
    if not seed_dir.is_dir():
        return "", ""
    image_ext = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff", ".svg"}
    video_ext = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v", ".ogv", ".wmv", ".flv"}
    audio_ext = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"}
    table_ext = {".csv", ".tsv", ".xlsx", ".xls", ".ods"}
    # Presentation / document seed formats (opened as the editing target via
    # --new-from/--open). Needed for SlidesEditor-style software whose seed_open
    # target is a real .pptx/.slx/.odp presentation rather than a media asset.
    doc_ext = {".pptx", ".ppt", ".slx", ".odp", ".key", ".pdf", ".docx", ".odt"}
    # Mail / message seed formats (opened as the current message via --new-from/--open).
    # Needed for MailClient-style software whose seed_open target is a real .eml.
    mail_ext = {".eml", ".mbox"}
    code_ext = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".c", ".h", ".cpp", ".cc", ".hpp",
        ".java", ".go", ".rs", ".rb", ".php", ".cs", ".kt", ".swift", ".sh", ".bash",
        ".sql", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml",
        ".html", ".htm", ".css", ".scss", ".less", ".md", ".rst", ".txt", ".log",
    }
    allowed_ext = image_ext | video_ext | audio_ext | table_ext | code_ext | doc_ext | mail_ext
    files = sorted(p for p in seed_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_ext)
    if not files:
        return "", ""
    # Seed metadata may live in a per-family manifest in the resource dir. Index every
    # entry by filename so each media/data family renders rich, predictable hints.
    dims: Dict[str, Dict[str, Any]] = {}
    for manifest_name, keys in (
        ("images_manifest.json", ("images",)),
        ("videos_manifest.json", ("videos",)),
        ("sheets_manifest.json", ("files",)),
        ("codefiles_manifest.json", ("files",)),
        ("emails_manifest.json", ("files", "emails", "resources")),
        ("media_manifest.json", ("media", "items", "files")),
        ("manifest_seedfile.json", ("resources", "items", "files")),
    ):
        manifest_json = resource_dir / manifest_name
        if not manifest_json.exists():
            continue
        try:
            data = load_json(manifest_json)
            for key in keys:
                for item in data.get(key, []) or []:
                    if isinstance(item, dict) and item.get("file"):
                        dims[str(item["file"])] = item
        except Exception:
            continue
    lines = []
    for p in files:
        meta = dims.get(p.name, {})
        ext = p.suffix.lower()
        fmt = str(meta.get("format") or meta.get("codec") or meta.get("container") or p.suffix.lstrip(".")).lower()
        if ext in video_ext:
            size = f"{meta['width']}x{meta['height']}" if meta.get("width") and meta.get("height") else "?"
            dur = meta.get("duration_sec")
            dur_s = f"{dur}s" if dur else "?s"
            # Surface audio/subtitle track counts so the eval-task agent can pick
            # files that actually exercise audio (volume/mute/multi-audio/channel
            # select/MP3 export) and embedded-subtitle features; most video seeds
            # are video-only, so without this the agent cannot tell which files
            # have audio or subtitle tracks to design cases around.
            audio_n = meta.get("audio_tracks")
            sub_n = meta.get("subtitle_tracks")
            extras = []
            if audio_n is not None:
                extras.append(f"{audio_n} audio")
            if sub_n is not None:
                extras.append(f"{sub_n} subs")
            extra_s = (" | " + ", ".join(extras)) if extras else ""
            lines.append(f"- {p.name} | {fmt} | {size} | {dur_s}{extra_s}")
        elif ext in audio_ext:
            dur = meta.get("duration_sec")
            dur_s = f"{dur}s" if dur else "?s"
            lines.append(f"- {p.name} | {fmt} | {dur_s}")
        elif ext in image_ext:
            size = f"{meta['width']}x{meta['height']}" if meta.get("width") and meta.get("height") else "?"
            alpha = "alpha" if meta.get("alpha") else "no-alpha"
            lines.append(f"- {p.name} | {fmt} | {size} | {alpha}")
        elif ext in table_ext:
            ncols = meta.get("n_columns")
            nrows = meta.get("n_data_rows")
            shape = []
            if ncols is not None:
                shape.append(f"{ncols} cols")
            if nrows is not None:
                shape.append(f"{nrows} rows")
            cols = meta.get("columns") or []
            head = (": " + ", ".join(str(c) for c in cols[:8])) if cols else ""
            lines.append(f"- {p.name} | {fmt} | {' '.join(shape) or '?'}{head}")
        elif ext in code_ext:
            lang = meta.get("language") or fmt
            nl = meta.get("n_lines")
            suffix = f" | {nl} lines" if nl else ""
            lines.append(f"- {p.name} | {lang}{suffix}")
        elif ext in mail_ext:
            # Surface subject/from/attachment/link/cc so EvalTaskGenerate can pick
            # seeds that exercise read, reply, filter, attachment, and export cases.
            sender = meta.get("from") or meta.get("sender") or "?"
            subject = meta.get("subject") or "?"
            extras = []
            if meta.get("has_attachment"):
                n_att = meta.get("n_attachments")
                extras.append(f"{n_att} att" if n_att is not None else "attachment")
            if meta.get("has_html"):
                extras.append("html")
            if meta.get("has_link"):
                extras.append("link")
            if meta.get("has_cc") or meta.get("cc"):
                extras.append("cc")
            category = meta.get("category")
            if category:
                extras.append(str(category))
            extra_s = (" | " + ", ".join(extras)) if extras else ""
            lines.append(f"- {p.name} | {fmt} | from={sender} | subj={subject}{extra_s}")
        else:
            lines.append(f"- {p.name} | {fmt}")
    return str(seed_dir), "\n".join(lines)


def render_value(value: Any, variables: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        rendered = value
        for key, replacement in variables.items():
            rendered = rendered.replace("{{" + key + "}}", str(replacement))
        return rendered
    if isinstance(value, list):
        return [render_value(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: render_value(item, variables) for key, item in value.items()}
    return value


def resolve_template(rendered: Mapping[str, Any], template_key: str) -> Any:
    """Resolve a template entry, supporting an ``extends`` mechanism so a repair
    template can inherit the base ``template`` and only add a delta. A template
    can be a list of lines, or a dict like
    ``{"extends": "template", "prepend": [...], "append": [...]}``.
    """
    template = rendered.get(template_key, [])
    if isinstance(template, dict) and "extends" in template:
        base = rendered.get(template["extends"], [])
        base_lines = list(base) if isinstance(base, list) else [str(base)]
        prepend = list(template.get("prepend", []))
        append = list(template.get("append", []))
        return prepend + base_lines + append
    return template


def format_prompt(
    prompt_config: Mapping[str, Any],
    skill_text: str,
    variables: Mapping[str, Any],
    template_key: str = "template",
) -> str:
    rendered = render_value(prompt_config, variables)
    template = resolve_template(rendered, template_key)
    if isinstance(template, list):
        body = "\n".join(str(line) for line in template)
    else:
        body = str(template)
    return "\n".join(
        [
            "# Agent Skill",
            "",
            skill_text.strip(),
            "",
            "# Rendered Task",
            "",
            body.strip(),
            "",
        ]
    )


def snake_case(value: str) -> str:
    chars: List[str] = []
    for index, char in enumerate(value):
        if char.isupper() and index > 0 and value[index - 1].islower():
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


def docker_safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    safe = safe.strip("_.-")
    return safe.lower() or "run"


class RuntimePolicy:
    def __init__(self, path: Path = RUNTIME_CONFIG) -> None:
        self.path = path
        self.config = load_json(path)
        self.default = self.config.get("default_codex_config", {})
        self.agents = self.config.get("agents", {})
        self.aliases: Dict[str, str] = {}
        for name, policy in self.agents.items():
            self.aliases[name.lower()] = name
            for alias in policy.get("aliases", []):
                self.aliases[str(alias).lower()] = name

    def resolve(self, agent_name: str) -> str:
        resolved = self.aliases.get(agent_name.lower())
        if not resolved:
            raise KeyError(f"Unknown agent: {agent_name}")
        return resolved

    def web_search(self, agent_name: str) -> str:
        resolved = self.resolve(agent_name)
        policy = self.agents.get(resolved, {})
        return str(policy.get("web_search", self.default.get("web_search", "disabled")))

    def model(self) -> str:
        model = (os.environ.get("MODEL") or "").strip()
        if not model:
            raise RuntimeError("MODEL is not set; configure the Codex model in the repo-root .env")
        return model

    def base_url(self) -> str:
        base_url = (os.environ.get("BASE_URL") or "").strip()
        if not base_url:
            raise RuntimeError("BASE_URL is not set; configure the Codex base URL in the repo-root .env")
        return base_url

    def sandbox(self) -> str:
        return str(self.default.get("sandbox") or "workspace-write")


class LoopController:
    def __init__(self, repo_root: Path = REPO_ROOT, execute: bool = False) -> None:
        self.repo_root = repo_root
        self.execute = execute
        self.runtime = RuntimePolicy(repo_root / "agent" / "config" / "agent-runtime.json")

    def scaffold_run(
        self,
        software_type: str,
        style_count: int,
        max_rounds: int,
        software_description: str = "",
        run_id: Optional[str] = None,
        style_allowlist: Optional[Iterable[str]] = None,
    ) -> Path:
        timestamp = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.repo_root / "output" / f"{software_type}_{timestamp}"
        styles = list(style_allowlist) if style_allowlist else [f"style_{index:02d}" for index in range(1, style_count + 1)]
        resource_dir = resource_dir_for_software(self.repo_root, software_type)
        resource_positioning = resource_positioning_for_software(resource_dir)
        description_text = software_description.strip()
        run_dir.mkdir(parents=True, exist_ok=False)

        manifest: Dict[str, Any] = {
            "software_type": software_type,
            "software_description": software_description.strip(),
            "resource_dir": str(resource_dir),
            "resource_positioning_source": str(resource_dir / "pipeline_parameters.json") if (resource_dir / "pipeline_parameters.json").exists() else "",
            "run_id": timestamp,
            "style_count": len(styles),
            "max_rounds": max_rounds,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "runtime_config": str(RUNTIME_CONFIG.relative_to(self.repo_root)),
            "contracts": {"icon_slots": 1, "avalonia_blueprint": 1},
            "execute": self.execute,
            "variants": {},
            "stages": {},
        }

        function_dir = run_dir / "01_function_design"
        icons_dir = run_dir / "02_icon_design"
        style_dir = run_dir / "02_style_design"
        substyle_root = run_dir / "03_substyle_design"
        function_dir.mkdir()
        icons_dir.mkdir()
        style_dir.mkdir()
        substyle_root.mkdir()

        self.render_agent(
            "FunctionDesigner",
            function_dir,
            {
                "run_dir": run_dir,
                "manifest_path": run_dir / "manifest.json",
                "function_design_dir": function_dir,
                "software_type": software_type,
                "software_description": description_text,
                "resource_dir": resource_dir,
                "resource_positioning": resource_positioning or "No resource directory positioning description provided.",
                "requirement_output_path": function_dir / "requirement.md",
                "cost_output_path": function_dir / "cost.json",
            },
        )
        self.render_agent(
            "IconDesigner",
            icons_dir,
            {
                "run_dir": run_dir,
                "manifest_path": run_dir / "manifest.json",
                "icons_dir": icons_dir,
                "software_type": software_type,
                "software_description": description_text or "(none provided)",
                "requirement_path": function_dir / "requirement.md",
                "icon_manifest_output_path": icons_dir / "icon_manifest.json",
                "cost_output_path": icons_dir / "cost.json",
            },
        )
        self.render_agent(
            "StyleDesigner",
            style_dir,
            {
                "run_dir": run_dir,
                "manifest_path": run_dir / "manifest.json",
                "style_design_dir": style_dir,
                "requirement_path": function_dir / "requirement.md",
                "style_count": len(styles),
                "styles_json_path": style_dir / "styles.json",
                "style_markdown_pattern": style_dir / "style_XX.md",
                "cost_output_path": style_dir / "cost.json",
            },
        )

        for style_id in styles:
            substyle_dir = substyle_root / style_id
            substyle_dir.mkdir(parents=True)
            (substyle_dir / "ui-blueprint").mkdir(parents=True, exist_ok=True)
            (substyle_dir / "design_route.md").write_text(
                f"# {style_id} design route\n\nTo be generated by SubStyleDesigner.\n",
                encoding="utf-8",
            )
            self.render_agent(
                "SubStyleDesigner",
                substyle_dir,
                {
                    "run_dir": run_dir,
                    "manifest_path": run_dir / "manifest.json",
                    "substyle_root_dir": substyle_root,
                    "substyle_dir": substyle_dir,
                    "style_id": style_id,
                    "requirement_path": function_dir / "requirement.md",
                    "style_spec_path": style_dir / f"{style_id}.md",
                    "icon_candidate_manifest_path": icons_dir / "icon_manifest.json",
                    "ui_blueprint_dir": substyle_dir / "ui-blueprint",
                    "avalonia_blueprint_report_path": substyle_dir
                    / "avalonia_blueprint_validation_report.json",
                    "design_route_output_path": substyle_dir / "design_route.md",
                    "icon_slots_output_path": substyle_dir / "icon_slots.json",
                    "cost_output_path": substyle_dir / "cost.json",
                },
                post_commands=[
                    *self.avalonia_blueprint_post_commands(
                        substyle_dir / "ui-blueprint",
                        substyle_dir / "avalonia_blueprint_validation_report.json",
                    ),
                    *self.icon_slot_projection_post_commands(
                        icons_dir / "icon_manifest.json",
                        substyle_dir / "icon_slots.json",
                        style_id,
                        run_dir
                        / "software_variants"
                        / style_id
                        / "04_icons"
                        / "icon_manifest.style.json",
                        substyle_dir / "icon_slots_validation_report.json",
                    ),
                ],
            )
            variant_dir = run_dir / "software_variants" / style_id
            self.scaffold_variant(variant_dir, software_type, style_id, function_dir, substyle_dir, max_rounds)
            manifest["variants"][style_id] = {
                "variant_dir": str(variant_dir.relative_to(run_dir)),
                "status": "scaffolded",
                "max_rounds": max_rounds,
            }

        write_json(run_dir / "manifest.json", manifest)
        write_json(run_dir / "cost.json", {"total_tokens": 0, "components": {}})
        return run_dir

    def scaffold_variant(
        self,
        variant_dir: Path,
        software_type: str,
        style_id: str,
        function_dir: Path,
        substyle_dir: Path,
        max_rounds: int,
    ) -> None:
        selected_style = variant_dir / "selected_style"
        eval_task_dir = variant_dir / "05_eval_tasks"
        icons_variant_dir = variant_dir / "04_icons"
        loop_dir = variant_dir / "loop"
        docker_dir = variant_dir / "docker"
        resource_dir = resource_dir_for_software(self.repo_root, software_type)
        run_dir = variant_dir.parents[1]
        superset_manifest_path = run_dir / "02_icon_design" / "icon_manifest.json"
        run_tag = docker_safe_name(variant_dir.parents[1].name)
        style_suffix = style_id.replace("style_", "")
        style_index = int(style_suffix) if style_suffix.isdigit() else 1
        port_group = port_group_for_offset(max(0, style_index - 1))
        worker_prefix = f"{docker_safe_name(software_type)}_{run_tag}_{docker_safe_name(style_id)}"
        overlay_path = DEFAULT_OVERLAY_DIR / f"{worker_prefix}.qcow2"
        for path in [selected_style, eval_task_dir, icons_variant_dir, loop_dir, docker_dir]:
            path.mkdir(parents=True, exist_ok=True)
        (selected_style / "style_id.txt").write_text(style_id + "\n", encoding="utf-8")
        link_path(substyle_dir / "ui-blueprint", selected_style / "ui-blueprint", target_is_directory=True)
        link_path(substyle_dir / "design_route.md", selected_style / "design_route.md")
        link_path(substyle_dir / "icon_slots.json", selected_style / "icon_slots.json")

        icon_validation_report = icons_variant_dir / "icon_validation_report.json"
        style_manifest_path = icons_variant_dir / "icon_manifest.style.json"
        icon_slots_path = substyle_dir / "icon_slots.json"
        self.render_agent(
            "SubIconDesigner",
            icons_variant_dir,
            {
                "run_dir": run_dir,
                "variant_dir": variant_dir,
                "software_type": software_type,
                "style_id": style_id,
                "style_segment": style_id.replace("_", "-"),
                "superset_manifest_path": superset_manifest_path,
                "style_manifest_path": style_manifest_path,
                "icon_slots_path": icon_slots_path,
                "style_spec_path": run_dir / "02_style_design" / f"{style_id}.md",
                "ui_blueprint_dir": substyle_dir / "ui-blueprint",
                "design_route_path": substyle_dir / "design_route.md",
                "icons_output_dir": icons_variant_dir,
                "validation_report_path": icon_validation_report,
                "cost_output_path": icons_variant_dir / "cost.json",
            },
            post_commands=[
                [
                    sys.executable,
                    "-m",
                    "agent.icons.validate_style_icons",
                    "--superset-manifest",
                    str(superset_manifest_path.resolve()),
                    "--icon-slots",
                    str(icon_slots_path.resolve()),
                    "--style-id",
                    style_id,
                    "--output-root",
                    str(icons_variant_dir.resolve()),
                    "--report",
                    str(icon_validation_report.resolve()),
                ]
            ],
        )

        write_json(
            variant_dir / "variant_manifest.json",
            {
                "software_type": software_type,
                "style_id": style_id,
                "status": "scaffolded",
                "current_round": None,
                "max_rounds": max_rounds,
                "docker": {},
                "rounds": {},
            },
        )
        (docker_dir / "worker.env").write_text(
            "\n".join(
                [
                    f"SOFTWARE_TYPE={software_type}",
                    f"STYLE_ID={style_id}",
                    f"VARIANT_DIR={variant_dir}",
                    f"WORKER_NAME={worker_prefix}",
                    f"OSWORLD_CONTAINER=kangj_{docker_safe_name(software_type)}_{docker_safe_name(style_id)}",
                    f"OSWORLD_API_PORT={port_group[0]}",
                    f"OSWORLD_VNC_PORT={port_group[1]}",
                    f"OSWORLD_HTTP_PORT={port_group[2]}",
                    f"OSWORLD_CDP_PORT={port_group[3]}",
                    f"OSWORLD_OVERLAY={overlay_path}",
                    "OSWORLD_CASE_BATCH_SIZE=1",
                    "OSWORLD_RECOVERY_HEALTH_PASSES=3",
                    "OSWORLD_RESET_OVERLAY_ON_RECOVERY=1",
                    "OSWORLD_RESTART_EVERY_CASE_BATCH=1",
                    "OSWORLD_AUTO_FALLBACK_PORTS=1",
                    "MAX_RESTARTS=0",
                    "UNHEALTHY_THRESHOLD=2",
                    "PID_FILE=",
                    "LOG_FILE=",
                    "RESTART_COMMAND=",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        for round_index in range(1, max_rounds + 1):
            version = f"v{round_index}"
            round_dir = loop_dir / version
            coder_dir = round_dir / "coder"
            devcoder_dir = round_dir / "devcoder"
            eval_dir = round_dir / "eval"
            # DevCoder edits coder/app in place; there is no separate devcoder/app.
            # devcoder/ only holds review/feature_matrix and similar metadata.
            for path in [coder_dir / "app", devcoder_dir, eval_dir / "traces"]:
                path.mkdir(parents=True, exist_ok=True)

            prev_round_dir = round_dir.parent / f"v{round_index - 1}"
            previous_summary = "" if round_index == 1 else prev_round_dir / "eval" / "summary.json"
            previous_feature = "" if round_index == 1 else prev_round_dir / "devcoder" / "feature_matrix.md"
            previous_review = "" if round_index == 1 else prev_round_dir / "devcoder" / "review.md"
            # Visual-fidelity feedback written by the runner after the previous round's GUIEval
            # (eval/visual_fidelity.md): fed to repair rounds as a "visual summary".
            previous_visual_fidelity = "" if round_index == 1 else prev_round_dir / "eval" / "visual_fidelity.md"
            # Coder resumes its own session; baseline and fallback sources both use its own
            # previous coder/app (same as the runner's seed source).
            previous_app = "" if round_index == 1 else prev_round_dir / "coder" / "app"
            # Repair rounds: each role resumes its own previous codex session
            # (Coder chain and DevCoder chain are independent).
            coder_resume_file = None if round_index == 1 else self.path_to_string(prev_round_dir / "coder" / "session_id.txt")
            devcoder_resume_file = None if round_index == 1 else self.path_to_string(prev_round_dir / "devcoder" / "session_id.txt")
            previous_round_inputs = ""
            round_implementation_rule = (
                "This is v1: implement the complete software from scratch, "
                "and finish all functionality required by the requirements and UI blueprint in one pass."
            )
            # v1 uses the generation template; v2+ use the incremental repair template
            # (fix based on the previous source, no rewrite).
            stage_template_key = "template" if round_index == 1 else "repair_template"
            if round_index > 1:
                previous_round_inputs = "\n".join(
                    [
                        f"- Previous round GUI evaluation summary: {self.path_to_string(previous_summary)}",
                        f"- Previous round feature implementation spec: {self.path_to_string(previous_feature)}",
                    ]
                )
                round_implementation_rule = (
                    f"This is v{round_index}: read the previous round's summary and feature_matrix, "
                    "only fix issues supported by evidence, keep already-passing functionality stable, "
                    "and fill in any functionality that still falls short."
                )
            coder_icon_usage_report = coder_dir / "icon_usage_report.json"
            devcoder_icon_usage_report = devcoder_dir / "icon_usage_report.json"
            self.render_agent(
                "Coder",
                coder_dir,
                {
                    "run_dir": variant_dir.parents[1],
                    "variant_dir": variant_dir,
                    "variant_manifest_path": variant_dir / "variant_manifest.json",
                    "selected_style_dir": selected_style,
                    "round_dir": round_dir,
                    "coder_dir": coder_dir,
                    "software_type": software_type,
                    "style_id": style_id,
                    "version": version,
                    "requirement_path": function_dir / "requirement.md",
                    "ui_blueprint_dir": selected_style / "ui-blueprint",
                    "resource_dir": str(resource_dir.resolve()),
                    "style_icons_dir": icons_variant_dir,
                    "icon_slots_path": selected_style / "icon_slots.json",
                    "app_output_dir": coder_dir / "app",
                    "feature_matrix_output_path": coder_dir / "feature_matrix.md",
                    "cost_output_path": coder_dir / "cost.json",
                    "icon_usage_report_path": coder_icon_usage_report,
                    "previous_round_inputs": previous_round_inputs,
                    "round_implementation_rule": round_implementation_rule,
                    "previous_summary_path": previous_summary,
                    "previous_feature_matrix_path": previous_feature,
                    "previous_review_path": previous_review,
                    "previous_visual_fidelity_path": previous_visual_fidelity,
                    "previous_app_dir": previous_app,
                },
                template_key=stage_template_key,
                resume_session_file=coder_resume_file,
                post_commands=self.icon_usage_post_commands(
                    style_manifest_path, coder_dir / "app", coder_icon_usage_report
                ),
            )
            self.render_agent(
                "DevCoder",
                devcoder_dir,
                {
                    "run_dir": variant_dir.parents[1],
                    "variant_dir": variant_dir,
                    "variant_manifest_path": variant_dir / "variant_manifest.json",
                    "selected_style_dir": selected_style,
                    "round_dir": round_dir,
                    "devcoder_dir": devcoder_dir,
                    "software_type": software_type,
                    "style_id": style_id,
                    "version": version,
                    "coder_output_dir": coder_dir,
                    "requirement_path": function_dir / "requirement.md",
                    "ui_blueprint_dir": selected_style / "ui-blueprint",
                    "resource_dir": str(resource_dir.resolve()),
                    "style_icons_dir": icons_variant_dir,
                    "icon_slots_path": selected_style / "icon_slots.json",
                    "app_output_dir": coder_dir / "app",
                    "feature_matrix_output_path": devcoder_dir / "feature_matrix.md",
                    "review_output_path": devcoder_dir / "review.md",
                    "cost_output_path": devcoder_dir / "cost.json",
                    "icon_usage_report_path": devcoder_icon_usage_report,
                    "previous_summary_path": previous_summary,
                    "previous_feature_matrix_path": previous_feature,
                    "previous_review_path": previous_review,
                    "previous_visual_fidelity_path": previous_visual_fidelity,
                },
                template_key=stage_template_key,
                resume_session_file=devcoder_resume_file,
                post_commands=self.icon_usage_post_commands(
                    style_manifest_path, coder_dir / "app", devcoder_icon_usage_report
                ),
            )
            write_json(eval_dir / "eval_manifest.json", {"version": version, "style_id": style_id, "status": "pending"})

        seed_dir, seed_manifest = eval_seedfile_for_software(resource_dir)
        self.render_agent(
            "EvalTaskGenerateAgent",
            eval_task_dir,
            {
                "run_dir": variant_dir.parents[1],
                "variant_dir": variant_dir,
                "variant_manifest_path": variant_dir / "variant_manifest.json",
                "software_type": software_type,
                "eval_task_dir": eval_task_dir,
                "feature_matrix_path": loop_dir / "v1" / "devcoder" / "feature_matrix.md",
                "cases_output_path": eval_task_dir / "cases.json",
                "cost_output_path": eval_task_dir / "cost.json",
                "state_schema_path": eval_task_dir / "state_schema.json",
                "eval_task_seedfile_dir": seed_dir,
                "eval_task_seedfile_manifest": seed_manifest or "(none available — do NOT use the seed_open mechanism)",
                "eval_task_seedfile_available": "yes" if seed_dir else "no",
            },
        )
        write_json(eval_task_dir / "cost.json", {"total_tokens": 0})

    def render_substyle_diversify(
        self,
        run_dir: Path,
        style_id: str,
        diversify_feedback: str,
        resume_session_file: Optional[str] = None,
    ) -> CodexCommand:
        """Re-render a single style's SubStyleDesigner command using the diversify
        `repair_template`, so the runner can re-roll a blueprint that collapsed
        toward a sibling. Paths mirror scaffold_run exactly; only template_key,
        the injected feedback, and the optional resume session differ."""
        run_dir = Path(run_dir)
        function_dir = run_dir / "01_function_design"
        style_dir = run_dir / "02_style_design"
        substyle_root = run_dir / "03_substyle_design"
        substyle_dir = substyle_root / style_id
        return self.render_agent(
            "SubStyleDesigner",
            substyle_dir,
            {
                "run_dir": run_dir,
                "manifest_path": run_dir / "manifest.json",
                "substyle_root_dir": substyle_root,
                "substyle_dir": substyle_dir,
                "style_id": style_id,
                "requirement_path": function_dir / "requirement.md",
                "style_spec_path": style_dir / f"{style_id}.md",
                "icon_candidate_manifest_path": run_dir / "02_icon_design" / "icon_manifest.json",
                "ui_blueprint_dir": substyle_dir / "ui-blueprint",
                "avalonia_blueprint_report_path": substyle_dir
                / "avalonia_blueprint_validation_report.json",
                "design_route_output_path": substyle_dir / "design_route.md",
                "icon_slots_output_path": substyle_dir / "icon_slots.json",
                "cost_output_path": substyle_dir / "cost.json",
                "diversify_feedback": diversify_feedback,
            },
            template_key="repair_template",
            resume_session_file=resume_session_file,
            post_commands=[
                *self.avalonia_blueprint_post_commands(
                    substyle_dir / "ui-blueprint",
                    substyle_dir / "avalonia_blueprint_validation_report.json",
                ),
                *self.icon_slot_projection_post_commands(
                    run_dir / "02_icon_design" / "icon_manifest.json",
                    substyle_dir / "icon_slots.json",
                    style_id,
                    run_dir
                    / "software_variants"
                    / style_id
                    / "04_icons"
                    / "icon_manifest.style.json",
                    substyle_dir / "icon_slots_validation_report.json",
                ),
            ],
        )

    def render_substyle_validation_repair(
        self,
        run_dir: Path,
        style_id: str,
        validation_feedback: str,
        resume_session_file: str,
    ) -> CodexCommand:
        """Render a focused compile-gate repair into the existing SubStyle session.

        Ordinary stage retries previously replayed the full generation prompt in a
        fresh session. This focused message instead feeds the authoritative platform
        report back to the session that produced the invalid blueprint, preserving
        its context while requiring minimal, evidence-driven XAML fixes.
        """
        run_dir = Path(run_dir)
        substyle_dir = run_dir / "03_substyle_design" / style_id
        prompt_config = load_json(PROMPT_DIR / PROMPT_FILES["SubStyleDesigner"])
        variables = {
            "style_id": style_id,
            "ui_blueprint_dir": substyle_dir / "ui-blueprint",
            "design_route_output_path": substyle_dir / "design_route.md",
            "icon_slots_output_path": substyle_dir / "icon_slots.json",
            "avalonia_blueprint_report_path": substyle_dir
            / "avalonia_blueprint_validation_report.json",
            "validation_feedback": validation_feedback,
        }
        normalized_vars = {
            key: self.path_to_string(value) for key, value in variables.items()
        }
        lines = render_value(
            resolve_template(prompt_config, "validation_repair"),
            normalized_vars,
        )
        body = (
            "\n".join(str(line) for line in lines)
            if isinstance(lines, list)
            else str(lines)
        )
        prompt_text = "\n".join(["# Rendered Task", "", body.strip(), ""])
        prompt_path = substyle_dir / "prompt.validation_repair.md"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        command = self.build_codex_command(
            "SubStyleDesigner",
            substyle_dir,
            prompt_path,
            resume_session_file=resume_session_file,
            post_commands=[
                *self.avalonia_blueprint_post_commands(
                    substyle_dir / "ui-blueprint",
                    substyle_dir / "avalonia_blueprint_validation_report.json",
                ),
                *self.icon_slot_projection_post_commands(
                    run_dir / "02_icon_design" / "icon_manifest.json",
                    substyle_dir / "icon_slots.json",
                    style_id,
                    run_dir
                    / "software_variants"
                    / style_id
                    / "04_icons"
                    / "icon_manifest.style.json",
                    substyle_dir / "icon_slots_validation_report.json",
                ),
            ],
        )
        (substyle_dir / "codex-command.sh").write_text(
            command.shell_script(),
            encoding="utf-8",
        )
        if self.execute:
            self.execute_command(command)
        return command

    def render_devcoder_launch_repair(
        self,
        devcoder_dir: Path,
        coder_app_dir: Path,
        style_manifest_path: Path,
        icon_usage_report_path: Path,
        variables: Mapping[str, Any],
        resume_session_file: Optional[str] = None,
    ) -> CodexCommand:
        """Render a focused DevCoder re-run to fix a launch-smoke failure.

        Sends ONLY the self-contained ``launch_repair`` instruction block (rendered
        with the smoke-gate evidence) as the new message. On a resumed session the
        model still holds this round's full review context, so we deliberately do
        NOT resend the whole repair template. The command is regenerated in place
        (``codex-command.sh``) and keeps the icon-usage gate as a post-command so a
        launch fix cannot silently drop icon usage below threshold.
        """
        devcoder_dir = Path(devcoder_dir)
        prompt_config = load_json(PROMPT_DIR / PROMPT_FILES["DevCoder"])
        normalized_vars = {key: self.path_to_string(value) for key, value in variables.items()}
        lines = render_value(resolve_template(prompt_config, "launch_repair"), normalized_vars)
        body = "\n".join(str(line) for line in lines) if isinstance(lines, list) else str(lines)
        prompt_text = "\n".join(["# Rendered Task", "", body.strip(), ""])
        prompt_path = devcoder_dir / "prompt.launch_repair.md"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        command = self.build_codex_command(
            "DevCoder",
            devcoder_dir,
            prompt_path,
            resume_session_file=resume_session_file,
            post_commands=self.icon_usage_post_commands(style_manifest_path, coder_app_dir, icon_usage_report_path),
        )
        (devcoder_dir / "codex-command.sh").write_text(command.shell_script(), encoding="utf-8")
        if self.execute:
            self.execute_command(command)
        return command

    def icon_usage_post_commands(
        self,
        style_manifest_path: Path,
        app_dir: Path,
        report_path: Path,
    ) -> List[List[str]]:
        """Build the icon-usage gate post-command for a Coder/DevCoder stage.

        Runs after the codex command succeeds; a non-zero exit (usage below the
        default 0.9 threshold) fails the stage script and triggers a retry, so
        the resumed session must wire the still-unused icons reported in
        ``report_path``. The gate can be tuned/disabled via the
        SOFTWAREGYM_ICON_* env vars read inside ``validate_icon_usage``.

        Invoked by absolute script path (not ``-m``) because the resume branch of
        a Coder/DevCoder stage ``cd``s into the work dir before running the
        post-commands, where the ``agent`` package is not importable. The script
        is stdlib-only, so a direct path invocation needs no PYTHONPATH.
        """
        script_path = self.repo_root / "agent" / "icons" / "validate_icon_usage.py"
        return [
            [
                sys.executable,
                str(script_path.resolve()),
                "--style-manifest",
                str(Path(style_manifest_path).resolve()),
                "--app-dir",
                str(Path(app_dir).resolve()),
                "--report",
                str(Path(report_path).resolve()),
            ]
        ]

    def icon_slot_projection_post_commands(
        self,
        superset_manifest_path: Path,
        icon_slots_path: Path,
        style_id: str,
        projected_manifest_path: Path,
        report_path: Path,
    ) -> List[List[str]]:
        """Validate SubStyleDesigner slots and project the style manifest.

        The absolute script path remains importable after a resumed Codex
        command changes cwd into the SubStyleDesigner work directory.
        """
        script_path = self.repo_root / "agent" / "icons" / "project_style_manifest.py"
        return [
            [
                sys.executable,
                str(script_path.resolve()),
                "--superset-manifest",
                str(Path(superset_manifest_path).resolve()),
                "--icon-slots",
                str(Path(icon_slots_path).resolve()),
                "--style-id",
                style_id,
                "--projected-manifest",
                str(Path(projected_manifest_path).resolve()),
                "--report",
                str(Path(report_path).resolve()),
            ]
        ]

    def avalonia_blueprint_post_commands(
        self,
        blueprint_dir: Path,
        report_path: Path,
    ) -> List[List[str]]:
        """Run the platform-owned Avalonia dialect and compiler gate."""
        script_path = (
            self.repo_root / "agent" / "loop_controller" / "avalonia_blueprint.py"
        )
        return [
            [
                sys.executable,
                str(script_path.resolve()),
                "--blueprint-dir",
                str(Path(blueprint_dir).resolve()),
                "--report",
                str(Path(report_path).resolve()),
                "--compile-probe",
                "required",
            ]
        ]

    def render_agent(
        self,
        agent_name: str,
        work_dir: Path,
        variables: Mapping[str, Any],
        template_key: str = "template",
        resume_session_file: Optional[str] = None,
        post_commands: Optional[Iterable[Iterable[str]]] = None,
    ) -> CodexCommand:
        prompt_file = PROMPT_DIR / PROMPT_FILES[agent_name]
        skill_file = SKILL_DIR / SKILL_DIRS[agent_name] / "SKILL.md"
        prompt_config = load_json(prompt_file)
        skill_text = read_text(skill_file)
        normalized_vars = {key: self.path_to_string(value) for key, value in variables.items()}
        prompt_text = format_prompt(prompt_config, skill_text, normalized_vars, template_key)
        prompt_path = work_dir / "prompt.rendered.md"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        command = self.build_codex_command(
            agent_name,
            work_dir,
            prompt_path,
            resume_session_file=resume_session_file,
            post_commands=post_commands,
        )
        (work_dir / "codex-command.sh").write_text(command.shell_script(), encoding="utf-8")
        if self.execute:
            self.execute_command(command)
        return command

    def _dynamo_sticky_model(self, work_dir: Path) -> str:
        """Pin each style's codex session to one channel alias from
        SOFTWAREGYM_CODEX_CHANNEL_ALIASES so server-side prompt-cache affinity is
        preserved while load spreads across channels.

        Opt-in via SOFTWAREGYM_CODEX_CHANNEL_ALIASES (comma-separated alias list, e.g.
        "model-a,model-b"). When unset, falls back to the default MODEL — a no-op for
        single-endpoint setups. When set, EVERY codex session is pinned
        to one channel (by its ``style*`` segment if present, else by its work_dir),
        because codex carries a responses ``previous_response_id`` across turns and the
        backend binds that item to the resource that created it — a load-balanced session
        that hops channels gets "created under a different resource" 400s.
        """
        default_model = self.runtime.model()
        raw = os.environ.get("SOFTWAREGYM_CODEX_CHANNEL_ALIASES", "")
        aliases = [a.strip() for a in raw.split(",") if a.strip()]
        if not aliases:
            return default_model
        # Match an actual style id, not an unrelated ancestor such as
        # ``style_design_ablation_runs``.  The old prefix match pinned every
        # ablation job below that directory to the same channel alias.
        style = next(
            (part for part in reversed(work_dir.parts) if re.fullmatch(r"style_\d+", part)),
            None,
        )
        # Per-style channel override: manually pin a style on an unstable channel to a stable one.
        # Format: SOFTWAREGYM_CODEX_CHANNEL_OVERRIDES="style_02=<model-alias>,style_07=<model-alias>"
        if style:
            overrides_raw = os.environ.get("SOFTWAREGYM_CODEX_CHANNEL_OVERRIDES", "")
            for item in overrides_raw.split(","):
                item = item.strip()
                if "=" not in item:
                    continue
                key, _, val = item.partition("=")
                if key.strip() == style and val.strip():
                    return val.strip()
        # Sticky key: prefer the style segment (so all stages of one style share a
        # channel), else the full work_dir path so design-stage sessions (Function/
        # Icon/Style — no style_* segment) are ALSO pinned to a single channel.
        sticky_key = style or str(work_dir)
        digest = hashlib.sha1(sticky_key.encode("utf-8")).hexdigest()
        return aliases[int(digest, 16) % len(aliases)]

    def build_codex_command(
        self,
        agent_name: str,
        work_dir: Path,
        prompt_path: Path,
        resume_session_file: Optional[str] = None,
        post_commands: Optional[Iterable[Iterable[str]]] = None,
    ) -> CodexCommand:
        base_url = self.runtime.base_url().rstrip("/")
        openai_base_url = os.environ.get("OPENAI_BASE_URL", f"{base_url}/v1")
        provider_id = (os.environ.get("MODEL_PROVIDER") or "").strip()
        if not provider_id:
            raise RuntimeError("MODEL_PROVIDER is not set; configure the Codex model provider in the repo-root .env")
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY") or ""
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY/API_KEY is not set; configure the Codex API key in the repo-root .env")
        model_name = self._dynamo_sticky_model(work_dir)
        env = {
            "API_KEY": os.environ.get("API_KEY", ""),
            "OPENAI_API_KEY": api_key,
            "BASE_URL": base_url,
            "OPENAI_BASE_URL": openai_base_url,
            "MODEL": model_name,
            # By default codex summarizes API failures at ERROR level as "stream disconnected ...",
            # swallowing the underlying HTTP status/error body (e.g. 429 rate_limit_exceeded).
            # Enable codex tracing (RUST_LOG) to emit internals to stderr; the runner already
            # captures stderr into codex.log, so no log-pipeline change is needed to see the
            # concrete API errors and retry pacing in codex.log.
            # Default codex_core=debug is enough to see API errors and retry pacing; override
            # with SOFTWAREGYM_CODEX_RUST_LOG=codex_core=trace for raw SSE/HTTP details.
            "RUST_LOG": os.environ.get("SOFTWAREGYM_CODEX_RUST_LOG", "codex_core=debug"),
        }
        web_search = self.runtime.web_search(agent_name)
        prompt_text = prompt_path.read_text(encoding="utf-8")
        # DevCoder edits the sibling coder/app directory, but codex workspace-write only treats
        # --cd (= the devcoder work dir) and /tmp as writable roots by default, making coder/app
        # read-only so fixes cannot be persisted (fresh sometimes relaxes this, resume always
        # keeps it read-only). Explicitly add coder/ and this work_dir as writable roots so
        # DevCoder can actually modify the evaluated coder/app in either branch while still
        # being able to write its own review/feature_matrix.
        extra_config: List[str] = []
        if agent_name == "DevCoder":
            coder_dir = work_dir.parent / "coder"
            writable_roots = f"[\"{coder_dir}\", \"{work_dir}\"]"
            extra_config = [
                "--config",
                f"sandbox_workspace_write.writable_roots={writable_roots}",
            ]
        # Non-OpenAI-native models (e.g. Claude) have no built-in codex metadata and fall back
        # to conservative defaults. If .env specifies a context window / max output, forward
        # them explicitly to avoid truncation and warnings.
        _ctx_win = os.environ.get("MODEL_CONTEXT_WINDOW", "").strip()
        if _ctx_win:
            extra_config += ["--config", f"model_context_window={_ctx_win}"]
        _max_out = os.environ.get("MODEL_MAX_OUTPUT_TOKENS", "").strip()
        if _max_out:
            extra_config += ["--config", f"model_max_output_tokens={_max_out}"]
        command = [
            "codex",
            "exec",
            "--model",
            model_name,
            "--config",
            f"model_provider=\"{provider_id}\"",
            "--config",
            f"model_providers.{provider_id}.name=\"{provider_id}\"",
            "--config",
            f"model_providers.{provider_id}.base_url=\"{openai_base_url}\"",
            "--config",
            f"model_providers.{provider_id}.wire_api=\"responses\"",
            "--config",
            f"model_providers.{provider_id}.env_key=\"OPENAI_API_KEY\"",
            "--config",
            f"web_search=\"{web_search}\"",
            *extra_config,
            "--sandbox",
            self.runtime.sandbox(),
            "--cd",
            str(work_dir),
            "--skip-git-repo-check",
            prompt_text,
        ]
        resume_rest: Optional[List[str]] = None
        if resume_session_file:
            # Resume branch: strip the "codex exec" prefix and parameters not supported by
            # `codex exec resume`. The resume subcommand has no `--cd` (the shell script cds
            # first) and no `-s/--sandbox`; pass the sandbox policy via `--config sandbox_mode="..."`.
            rest = command[2:]
            resume_rest = []
            skip_next = False
            for token in rest:
                if skip_next:
                    skip_next = False
                    continue
                if token == "--cd":
                    skip_next = True
                    continue
                if token == "--sandbox":
                    skip_next = True
                    resume_rest.extend(
                        ["--config", f"sandbox_mode=\"{self.runtime.sandbox()}\""]
                    )
                    continue
                resume_rest.append(token)
        normalized_post = tuple(tuple(str(token) for token in cmd) for cmd in (post_commands or ()))
        return CodexCommand(
            agent_name=agent_name,
            work_dir=work_dir,
            prompt_path=prompt_path,
            log_path=work_dir / "codex.log",
            command=command,
            env=env,
            resume_session_file=resume_session_file,
            resume_rest=resume_rest,
            post_commands=normalized_post,
        )

    def execute_command(self, command: CodexCommand) -> None:
        env = os.environ.copy()
        env.update(command.env)
        with command.log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command.command, cwd=command.work_dir, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    @staticmethod
    def path_to_string(value: Any) -> str:
        if isinstance(value, Path):
            path = value if value.is_absolute() else REPO_ROOT / value
            # Codex runs with --cd set to each stage directory. Repo-relative
            # prompt paths would therefore be resolved below that directory
            # (for example, 01_function_design/output/.../requirement.md).
            # Keep the stable /home-facing path instead of resolving symlinks
            # to /data so sandbox writable roots and prompt paths stay aligned.
            return str(path.absolute())
        return str(value)


def parse_style_allowlist(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="software-gym LoopController")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scaffold = subparsers.add_parser("scaffold", help="Create a pipeline run directory and rendered Codex prompts.")
    scaffold.add_argument("--software-type", required=True)
    scaffold.add_argument(
        "--software-description",
        default="",
        help="Optional product positioning/use-case description passed to FunctionDesigner.",
    )
    scaffold.add_argument("--style-count", type=int, default=5)
    scaffold.add_argument("--max-rounds", type=int, default=3)
    scaffold.add_argument("--run-id", default=None)
    scaffold.add_argument("--style-allowlist", default=None, help="Comma separated style ids, e.g. style_01,style_03.")
    scaffold.add_argument(
        "--execute",
        action="store_true",
        help="Debug only: run Codex while scaffolding. Use agent.loop_controller.runner for the ordered full pipeline.",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    ensure_softwaregym_for_runner()
    activate_softwaregym_env_vars()
    load_env_file(REPO_ROOT / ".env")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "scaffold":
        if args.execute and args.max_rounds != 1:
            parser.error("scaffold --execute is only allowed with --max-rounds 1 for prompt debugging; use agent.loop_controller.runner for the full loop.")
        controller = LoopController(execute=args.execute)
        run_dir = controller.scaffold_run(
            software_type=args.software_type,
            style_count=args.style_count,
            max_rounds=args.max_rounds,
            software_description=args.software_description,
            run_id=args.run_id,
            style_allowlist=parse_style_allowlist(args.style_allowlist),
        )
        print(run_dir)
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
