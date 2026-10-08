"""Token / model / time usage statistics for the software synthesis pipeline (pipeline one).

Design notes:
- After each codex phase (FunctionDesigner / StyleDesigner / SubStyleDesigner / Coder /
  DevCoder / EvalTaskGenerateAgent) runs, it leaves a ``codex.log`` in its working directory,
  with a final line ``tokens used`` + number. That number = ``input - cached_input + output``
  (cost-relevant "effective tokens"), and **is cumulative within the same session when resumed**.
- Therefore, for fix rounds (Coder/DevCoder v2, v3 … that actually resumed the previous session)
  compute ``this_round_effective = this_round_cumulative - previous_round_cumulative`` to avoid
  double-counting v1 into v2/v3.
- Whether it actually resumed: codex-command.sh prints ``[resume] continuing codex session``
  when resume succeeds; use this marker to avoid mis-classifying a fresh fallback as resumed.
- GUIEval's GUI-agent tokens are already written to ``loop/vX/eval/cost.json`` by
  ``agent/eval/cost.py``; aggregate them directly here.

This module is called by the runner in real time (``reconstruct_codex_record``, with duration
appended by the runner), and can also **rebuild purely from disk logs** (``build_cost_summary``),
so it can generate summaries for historically completed runs. All parsing is tolerant of missing
or malformed data and never raises errors that would break the main flow.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

_EFFECTIVE_RE = re.compile(r"tokens used\s*[:\-]?\s*([\d,]+)", re.IGNORECASE)
_MODEL_RE = re.compile(r"^model:\s*(\S+)", re.MULTILINE)
_PROVIDER_RE = re.compile(r"^provider:\s*(\S+)", re.MULTILINE)
_SESSION_RE = re.compile(r"session id:\s*([0-9a-fA-F-]{32,36})")
_EFFORT_RE = re.compile(r"reasoning effort:\s*(\S+)", re.IGNORECASE)
_BASE_URL_RE = re.compile(r'base_url:\s*Some\("([^"]+)"\)')
_RESUMED_MARK = "[resume] continuing codex session"

# Real context window for gpt-5.5 (from codex rollout's token_count.info.model_context_window).
GPT55_CONTEXT_WINDOW = 258400

# Channel alias suffix -> channel identifier. After codex connects to multi-channel proxies,
# --model carries a suffix (gpt-5.5-dc02 etc.); normalize it back to a canonical model name
# plus a separate channel field.
_ALIAS_SUFFIX = (("-dc02", "dc02"), ("-dc03", "dc03"), ("-dc01", "dc01"), ("-gw", "gw"))


def _channel_from_base_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    if "sjtu-dc-02" in url:
        return "dc02"
    if "sjtu-dc-03" in url:
        return "dc03"
    if "sjtu-dc-01" in url:
        return "dc01"
    if "ai-gateway-internal" in url:
        return "gw"
    return None


def normalize_model_channel(model: Optional[str], base_url: Optional[str] = None) -> tuple:
    """Normalize a codex model name to (canonical_model_name, channel).

    - ``gpt-5.5-dc02`` -> (``gpt-5.5``, ``dc02``) (channel-sticky alias).
    - Without a suffix, infer the channel from base_url (known when directly connecting to
      Azure/gateway; unknown when going through a shared proxy pool).
    """
    if not model:
        return (None, _channel_from_base_url(base_url))
    channel = None
    for suffix, ch in _ALIAS_SUFFIX:
        if model.endswith(suffix):
            model = model[: -len(suffix)]
            channel = ch
            break
    if channel is None:
        channel = _channel_from_base_url(base_url)
    # Gateway model id is ``sjtu/gpt-5.5``, the same model as Azure's ``gpt-5.5``; normalize to
    # the latter so by_model totals are not split by different ids for the same model (channel
    # differences are preserved in the channel field).
    if model.startswith("sjtu/"):
        model = model.split("/", 1)[1]
    return (model, channel)


# Classification rule for codex.log paths relative to run_dir.
_CODEX_ROLE_DIRS = {"coder": "Coder", "devcoder": "DevCoder"}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def parse_effective_tokens(text: str) -> Optional[int]:
    """Parse the ``tokens used`` number from codex output (takes the last occurrence). None if absent."""
    matches = _EFFECTIVE_RE.findall(text or "")
    if not matches:
        return None
    try:
        return int(matches[-1].replace(",", ""))
    except ValueError:
        return None


def parse_codex_meta(text: str) -> Dict[str, Any]:
    """Parse model / provider / session id / reasoning effort from the head of codex.log,
    and whether the previous session was actually resumed."""
    model = _MODEL_RE.search(text or "")
    provider = _PROVIDER_RE.search(text or "")
    session = _SESSION_RE.search(text or "")
    effort = _EFFORT_RE.search(text or "")
    base_url = _BASE_URL_RE.search(text or "")
    return {
        "model": model.group(1) if model else None,
        "provider": provider.group(1) if provider else None,
        "session_id": session.group(1) if session else None,
        "reasoning_effort": effort.group(1) if effort else None,
        "base_url": base_url.group(1) if base_url else None,
        "resumed": (_RESUMED_MARK in (text or "")),
    }


def classify_codex_log(log_path: Path, run_dir: Path) -> Optional[Dict[str, Any]]:
    """Determine which phase a codex.log belongs to based on its path relative to run_dir.
    Returns None if unrecognized."""
    try:
        rel = log_path.resolve().relative_to(run_dir.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if parts[-1] != "codex.log":
        return None
    head = parts[0] if parts else ""
    if head == "01_function_design":
        return {"stage": "FunctionDesigner", "style_id": None, "version": None, "role": None}
    if head == "02_icon_design":
        return {"stage": "IconDesigner", "style_id": None, "version": None, "role": None}
    if head == "02_style_design":
        return {"stage": "StyleDesigner", "style_id": None, "version": None, "role": None}
    if head == "03_substyle_design" and len(parts) >= 3:
        return {"stage": "SubStyleDesigner", "style_id": parts[1], "version": None, "role": None}
    if head == "software_variants" and len(parts) >= 3:
        style_id = parts[1]
        # SubIconDesigner's codex.log has icon-validation post_commands output appended after codex
        # finishes, but that segment contains no "tokens used", so parse_effective_tokens taking the
        # last match still returns codex's true value.
        if len(parts) >= 4 and parts[2] == "04_icons":
            return {"stage": "SubIconDesigner", "style_id": style_id, "version": None, "role": None}
        if len(parts) >= 4 and parts[2] == "05_eval_tasks":
            return {"stage": "EvalTaskGenerateAgent", "style_id": style_id, "version": None, "role": None}
        if len(parts) >= 6 and parts[2] == "loop":
            version = parts[3]
            role = parts[4]
            stage = _CODEX_ROLE_DIRS.get(role)
            if stage:
                return {"stage": stage, "style_id": style_id, "version": version, "role": role}
    return None


def _prev_role_log(log_path: Path, version: str, role: str) -> Optional[Path]:
    """For Coder/DevCoder vN (N>1), return the previous round's codex.log path for the same role."""
    m = re.match(r"v(\d+)$", version or "")
    if not m:
        return None
    n = int(m.group(1))
    if n <= 1:
        return None
    # log_path = .../loop/vN/<role>/codex.log
    loop_dir = log_path.parents[2]
    prev = loop_dir / f"v{n - 1}" / role / "codex.log"
    return prev if prev.exists() else None


def find_rollout(session_id: Optional[str]) -> Optional[Path]:
    """Locate the rollout file under ~/.codex/sessions by session id (best-effort)."""
    if not session_id:
        return None
    base = Path.home() / ".codex" / "sessions"
    if not base.exists():
        return None
    matches = list(base.glob(f"**/rollout-*-{session_id}.jsonl"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def rollout_cumulative_usage(rollout_path: Optional[Path]) -> Optional[Dict[str, int]]:
    """Read the last token_count event's total_token_usage (cumulative session breakdown) from the rollout file."""
    if not rollout_path or not rollout_path.exists():
        return None
    last: Optional[Dict[str, int]] = None
    try:
        with rollout_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "token_count" not in line or "total_token_usage" not in line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                info = (record.get("payload") or {}).get("info") or {}
                usage = info.get("total_token_usage")
                if isinstance(usage, dict):
                    last = usage
    except OSError:
        return None
    if not last:
        return None
    return {
        "input_tokens": int(last.get("input_tokens") or 0),
        "cached_input_tokens": int(last.get("cached_input_tokens") or 0),
        "output_tokens": int(last.get("output_tokens") or 0),
        "reasoning_output_tokens": int(last.get("reasoning_output_tokens") or 0),
        "total_tokens": int(last.get("total_tokens") or 0),
    }


def window_rollout_usage(
    session_id: Optional[str],
    prev_cumulative: int,
    this_cumulative: Optional[int],
) -> Optional[Dict[str, int]]:
    """In the shared rollout, cut out **this phase's** turns and breakdown usage by the
    cumulative effective-token window ``(prev_cumulative, this_cumulative]``.

    - ``turns`` = number of token_count events in the window with real model output
      (last.output_tokens>0), i.e. the actual number of model requests in this phase.
    - Breakdown = incremental sum of ``last_token_usage`` within the window (also accurate for
      post-hoc reconstruction because cumulative effective tokens are monotonic and the boundaries
      are fixed, unaffected by later turns in the same session).

    Returns None when ``this_cumulative`` is None (cannot locate the boundary).
    """
    rollout = find_rollout(session_id)
    if rollout is None or not rollout.exists() or this_cumulative is None:
        return None
    _FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens",
               "reasoning_output_tokens", "total_tokens")
    # codex's total_token_usage is cumulative (not the sum of last increments: each round resends
    # the full context, so last.input contains the full context and summing last increments would
    # double-count context). Therefore this phase's breakdown = cumulative amount at this round's
    # boundary minus cumulative amount at the previous round's boundary.
    prev_totals = {f: 0 for f in _FIELDS}
    this_totals = {f: 0 for f in _FIELDS}
    turns = 0
    try:
        with rollout.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "token_count" not in line or "total_token_usage" not in line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                info = (record.get("payload") or {}).get("info") or {}
                total = info.get("total_token_usage") or {}
                last = info.get("last_token_usage") or {}
                eff_cum = (
                    int(total.get("input_tokens") or 0)
                    - int(total.get("cached_input_tokens") or 0)
                    + int(total.get("output_tokens") or 0)
                )
                snap = {f: int(total.get(f) or 0) for f in _FIELDS}
                if prev_cumulative > 0 and eff_cum <= prev_cumulative:
                    prev_totals = snap
                if eff_cum <= this_cumulative:
                    this_totals = snap
                if prev_cumulative < eff_cum <= this_cumulative and int(last.get("output_tokens") or 0) > 0:
                    turns += 1
    except OSError:
        return None
    agg = {f: max(this_totals[f] - prev_totals[f], 0) for f in _FIELDS}
    agg["turns"] = turns
    return agg


def reconstruct_codex_record(log_path: Path, run_dir: Path, with_breakdown: bool = True) -> Optional[Dict[str, Any]]:
    """Rebuild a phase usage record from a single codex.log (excluding duration; duration is added
    separately by the runner).

    The ``tokens`` field is **this phase's** effective tokens: for truly resumed fix rounds it is
    differenced against the previous round.
    """
    info = classify_codex_log(log_path, run_dir)
    if info is None:
        return None
    text = _read_text(log_path)
    meta = parse_codex_meta(text)
    cumulative = parse_effective_tokens(text)

    stage_dir = log_path.parent
    # Collect all attempts for this phase (including 429/failure retries that were discarded).
    # codex.log is a copy of the last attempt.
    attempts: List[Dict[str, Any]] = []
    for ap in sorted(stage_dir.glob("codex.attempt_*.log")):
        atext = _read_text(ap)
        attempts.append({
            "path": str(ap),
            "effective": parse_effective_tokens(atext),
            "resumed": (_RESUMED_MARK in atext),
        })
    # Identify the "final/successful" attempt. codex.log is a copy of the adopted attempt (not
    # necessarily the last one -- the runner adopts the successful attempt). Match the attempt by
    # codex.log's effective token value; if there is no match (codex.log has no "tokens used" line,
    # e.g. the last attempt immediately returns 429), fall back to the last attempt that has a token
    # line as the final representative, so this phase is not mis-recorded as 0. That attempt is not
    # counted in retry_overhead (avoid double counting).
    final_idx: Optional[int] = None
    if cumulative is not None and attempts:
        for i in range(len(attempts) - 1, -1, -1):
            if attempts[i]["effective"] == cumulative:
                final_idx = i
                break
    if final_idx is None:
        for i in range(len(attempts) - 1, -1, -1):
            if attempts[i]["effective"] is not None:
                final_idx = i
                break
        if cumulative is None and final_idx is not None:
            cumulative = attempts[final_idx]["effective"]

    resumed = bool(meta.get("resumed"))
    # Previous version (same role) session cumulative -- used for cross-version diffing on resume,
    # and for deduplicating failed attempts.
    prev_version_cum = 0
    prev_found: Optional[int] = None
    if info.get("role"):
        prev_log = _prev_role_log(log_path, info.get("version") or "", info["role"])
        if prev_log is not None:
            prev_found = parse_effective_tokens(_read_text(prev_log))
            if prev_found is not None:
                prev_version_cum = prev_found

    stage_tokens = cumulative
    delta_unavailable = False
    if resumed and cumulative is not None and info.get("role"):
        if prev_found is not None:
            stage_tokens = max(0, cumulative - prev_version_cum)
        else:
            delta_unavailable = True

    # tokens_with_retries: also count effective tokens actually burned by failed/discarded attempts
    # (real bill). The successful attempt uses the already-differenced stage_tokens; each other
    # attempt counts its "new" burn (fresh retry = full; if the attempt also resumed the previous
    # version, subtract the previous version cumulative to avoid duplication).
    overhead = 0
    for i, a in enumerate(attempts):
        if i == final_idx:
            continue
        eff = a["effective"]
        if eff is None:
            continue
        if a["resumed"] and prev_version_cum > 0:
            overhead += max(0, eff - prev_version_cum)
        else:
            overhead += eff
    tokens_clean = stage_tokens
    tokens_billed = (tokens_clean + overhead) if tokens_clean is not None else None

    record: Dict[str, Any] = {
        "stage": info["stage"],
        "kind": "codex",
        "style_id": info["style_id"],
        "version": info["version"],
        "model": meta.get("model"),
        "provider": meta.get("provider"),
        "reasoning_effort": meta.get("reasoning_effort"),
        "session_id": meta.get("session_id"),
        "resumed": resumed,
        "tokens": tokens_clean,
        "tokens_without_retries": tokens_clean,
        "tokens_with_retries": tokens_billed,
        "retry_overhead": overhead,
        "attempt_count": len(attempts) or 1,
        "session_cumulative_tokens": cumulative,
        "delta_unavailable": delta_unavailable,
        "tokens_missing": cumulative is None,
        "log_path": str(log_path),
    }
    # Breakdown (input/cached/output/reasoning): only reliable for non-resumed phases where the
    # entire session equals this phase.
    if with_breakdown and not resumed:
        breakdown = rollout_cumulative_usage(find_rollout(meta.get("session_id")))
        if breakdown is not None:
            breakdown["scope"] = "stage"
            record["token_breakdown"] = breakdown
    return record


def _read_eval_record(cost_path: Path, run_dir: Path) -> Optional[Dict[str, Any]]:
    """Rebuild a GUIEval phase usage record from loop/vX/eval/cost.json."""
    try:
        rel = cost_path.resolve().relative_to(run_dir.resolve())
    except ValueError:
        return None
    parts = rel.parts
    # software_variants/<style>/loop/<vN>/eval/cost.json
    if len(parts) < 6 or parts[0] != "software_variants" or parts[2] != "loop" or parts[4] != "eval":
        return None
    style_id = parts[1]
    version = parts[3]
    try:
        cost = json.loads(cost_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    gui = (cost.get("components") or {}).get("gui_agent") or {}
    billed = int(
        cost["effective_tokens_with_retries"]
        if "effective_tokens_with_retries" in cost
        else cost.get("total_tokens", 0)
    )
    clean = int(
        cost["effective_tokens"]
        if "effective_tokens" in cost
        else cost.get("total_tokens_without_retries", cost.get("total_tokens", 0))
    )
    record: Dict[str, Any] = {
        "stage": "GUIEval",
        "kind": "gui_eval",
        "style_id": style_id,
        "version": version,
        "model": cost.get("model"),
        "provider": None,
        # Main metric tokens = clean figure (without retries, needed for a normal run), aligned with codex;
        # tokens_with_retries = real bill including model retries (GUIEval's cost.json total_tokens).
        "tokens": clean,
        "tokens_without_retries": clean,
        "tokens_with_retries": billed,
        "retry_overhead": max(0, billed - clean),
        "turns": int(gui.get("model_calls") or 0) or None,
        "token_breakdown": {
            "input_tokens": int(gui.get("input_tokens") or 0),
            "cached_input_tokens": int(gui.get("cached_input_tokens") or 0),
            "output_tokens": int(gui.get("output_tokens") or 0),
            "reasoning_output_tokens": int(gui.get("reasoning_output_tokens") or 0),
            "total_tokens": int(gui.get("raw_total_tokens", gui.get("total_tokens", 0)) or 0),
            "scope": "stage",
        },
        "cost_by_model": cost.get("by_model") or {},
        "log_path": str(cost_path),
    }
    return record


def _overlay_duration(record: Dict[str, Any], stage_dir: Path) -> None:
    """If a usage.json written by the runner exists in the same directory, overlay
    duration/timestamp/retry count into the record."""
    usage_path = stage_dir / "usage.json"
    if not usage_path.exists():
        return
    try:
        extra = json.loads(usage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    for key in ("duration_seconds", "started_at", "ended_at", "attempts"):
        if extra.get(key) is not None:
            record[key] = extra[key]


def _empty_bucket() -> Dict[str, Any]:
    return {"tokens": 0, "duration_seconds": 0.0, "count": 0, "models": []}


def _accumulate(bucket: Dict[str, Any], record: Dict[str, Any]) -> None:
    bucket["tokens"] += int(record.get("tokens") or 0)
    dur = record.get("duration_seconds")
    if isinstance(dur, (int, float)):
        bucket["duration_seconds"] = round(bucket["duration_seconds"] + float(dur), 1)
    bucket["count"] += 1
    # Use the normalized real model name (historical GUIEval may contain both executor and judge),
    # not the channel-suffixed raw name from codex.log, so stage-table model columns stay consistent
    # with by_model.
    names = list((record.get("by_model") or {}).keys()) or [record.get("model_canonical") or record.get("model")]
    for model in names:
        if model and model not in bucket["models"]:
            bucket["models"].append(model)


def _wall_clock_seconds(run_dir: Path) -> Optional[float]:
    events_path = run_dir / "runner-events.jsonl"
    if not events_path.exists():
        return None
    first: Optional[datetime] = None
    last: Optional[datetime] = None
    try:
        with events_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stamp = event.get("time")
                if not stamp:
                    continue
                try:
                    moment = datetime.fromisoformat(stamp)
                except ValueError:
                    continue
                if first is None:
                    first = moment
                last = moment
    except OSError:
        return None
    if first is None or last is None:
        return None
    return round((last - first).total_seconds(), 1)


# Display order of stages in each software table.
_STAGE_ORDER = [
    "FunctionDesigner", "IconDesigner", "StyleDesigner",
    "SubStyleDesigner", "SubIconDesigner", "EvalTaskGenerateAgent",
    "Coder", "DevCoder", "GUIEval",
]
_RUN_SHARED_STAGES = ("FunctionDesigner", "IconDesigner", "StyleDesigner")


def enrich_record(record: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Add unified-protocol fields to a phase record: canonical model name + channel + turns + by_model.

    - codex: read base_url from codex.log to normalize channel; use the rollout to count this phase's
      turns within the cumulative window, and (for resumed fix rounds) recalculate this round's
      breakdown by window to avoid cross-phase leakage from shared-session growth.
    - GUIEval: channel is the shared proxy pool (per-request routing, cannot be attributed to a single
      channel), turns = model_calls.
    """
    if record.get("kind") == "gui_eval":
        # Current seed GUIEval uses the same Seed model for executor + result-only judge independent
        # requests. Prefer eval cost.json's by_model (with clean and with-retries figures), normalized
        # per model.
        eval_bm = record.get("cost_by_model") or {}
        by_model: Dict[str, Any] = {}
        for raw_name, mv in eval_bm.items():
            cname, _ch = normalize_model_channel(raw_name)
            cname = cname or raw_name or "unknown"
            roles = mv.get("roles") or []
            channel = "gw" if "executor" in roles else "pool"  # executor goes directly to gateway; judge through shared proxy pool
            billed = int(
                mv["tokens_with_retries"]
                if "tokens_with_retries" in mv
                else mv.get("total_tokens", 0)
            )
            clean = int(
                mv["effective_tokens"]
                if "effective_tokens" in mv
                else mv.get("total_tokens_without_retries", billed)
            )
            entry = by_model.setdefault(cname, {
                "channel": channel, "effective_tokens": 0, "tokens_with_retries": 0,
                "retry_overhead": 0, "input_tokens": 0, "cached_input_tokens": 0,
                "output_tokens": 0, "reasoning_output_tokens": 0,
                "raw_total_tokens": 0, "turns": 0, "roles": [],
            })
            entry["effective_tokens"] += clean
            entry["tokens_with_retries"] += billed
            entry["retry_overhead"] += max(0, billed - clean)
            entry["input_tokens"] += int(mv.get("input_tokens") or 0)
            entry["cached_input_tokens"] += int(mv.get("cached_input_tokens") or 0)
            entry["output_tokens"] += int(mv.get("output_tokens") or 0)
            entry["reasoning_output_tokens"] += int(mv.get("reasoning_output_tokens") or 0)
            entry["raw_total_tokens"] += int(mv.get("raw_total_tokens", mv.get("total_tokens", 0)) or 0)
            entry["turns"] += int(mv.get("model_calls") or 0)
            for r in roles:
                if r not in entry["roles"]:
                    entry["roles"].append(r)
        if not by_model:  # Old eval cost.json has no by_model: fall back to single model
            model = record.get("model") or "unknown"
            cname, _ = normalize_model_channel(model)
            cname = cname or model
            bd = record.get("token_breakdown") or {}
            by_model = {cname: {
                "channel": "pool",
                "effective_tokens": int(record.get("tokens") or 0),
                "tokens_with_retries": int(record.get("tokens_with_retries") or record.get("tokens") or 0),
                "retry_overhead": int(record.get("retry_overhead") or 0),
                "input_tokens": int(bd.get("input_tokens") or 0),
                "cached_input_tokens": int(bd.get("cached_input_tokens") or 0),
                "output_tokens": int(bd.get("output_tokens") or 0),
                "reasoning_output_tokens": int(bd.get("reasoning_output_tokens") or 0),
                "raw_total_tokens": int(bd.get("total_tokens") or 0),
                "turns": record.get("turns"),
            }}
        # Display canonical model name = the one with the most tokens (usually executor)
        record["model_canonical"] = (
            max(by_model, key=lambda m: by_model[m]["effective_tokens"]) if by_model else (record.get("model") or "unknown")
        )
        record["channel"] = "pool"
        record["by_model"] = by_model
        return record

    log_path = Path(record.get("log_path") or "")
    base_url = None
    if log_path.exists():
        m = _BASE_URL_RE.search(_read_text(log_path))
        base_url = m.group(1) if m else None
    model, channel = normalize_model_channel(record.get("model"), base_url)
    record["model_canonical"] = model
    record["channel"] = channel

    this_cum = record.get("session_cumulative_tokens")
    prev_cum = 0
    role = log_path.parent.name if log_path.name == "codex.log" else None
    if record.get("resumed") and role in _CODEX_ROLE_DIRS:
        prev_log = _prev_role_log(log_path, record.get("version") or "", role)
        if prev_log is not None:
            pc = parse_effective_tokens(_read_text(prev_log))
            if pc is not None:
                prev_cum = pc
    win = window_rollout_usage(record.get("session_id"), prev_cum, this_cum)
    if win is not None:
        record["turns"] = win["turns"]
        record["token_breakdown"] = {
            "input_tokens": win["input_tokens"],
            "cached_input_tokens": win["cached_input_tokens"],
            "output_tokens": win["output_tokens"],
            "reasoning_output_tokens": win["reasoning_output_tokens"],
            "total_tokens": win["total_tokens"],
            "scope": "stage",
        }
    record.setdefault("turns", None)
    bd = record.get("token_breakdown") or {}
    record["by_model"] = {
        (model or "unknown"): {
            "channel": channel,
            "effective_tokens": int(record.get("tokens") or 0),
            "tokens_with_retries": int(record.get("tokens_with_retries") or record.get("tokens") or 0),
            "retry_overhead": int(record.get("retry_overhead") or 0),
            "input_tokens": int(bd.get("input_tokens") or 0),
            "cached_input_tokens": int(bd.get("cached_input_tokens") or 0),
            "output_tokens": int(bd.get("output_tokens") or 0),
            "reasoning_output_tokens": int(bd.get("reasoning_output_tokens") or 0),
            "raw_total_tokens": int(bd.get("total_tokens") or 0),
            "turns": record.get("turns"),
        }
    }
    return record


def _write_stage_cost_json(stage_dir: Path, record: Dict[str, Any]) -> None:
    """Write a single-stage unified-protocol cost.json into the codex phase directory
    (GUIEval already has its own cost.json, do not overwrite)."""
    if record.get("kind") != "codex":
        return
    by_model = record.get("by_model") or {}
    payload = {
        "schema_version": 2,
        "stage": record.get("stage"),
        "kind": record.get("kind"),
        "style_id": record.get("style_id"),
        "version": record.get("version"),
        "status": "completed",
        "time": {
            "started_at": record.get("started_at"),
            "ended_at": record.get("ended_at"),
            "duration_seconds": record.get("duration_seconds"),
            "attempts": record.get("attempts"),
        },
        "turns": record.get("turns"),
        "attempt_count": record.get("attempt_count"),
        "by_model": by_model,
        "totals": {
            "effective_tokens": int(record.get("tokens") or 0),
            "effective_tokens_with_retries": int(record.get("tokens_with_retries") or record.get("tokens") or 0),
            "retry_overhead": int(record.get("retry_overhead") or 0),
            "raw_total_tokens": sum(int(v.get("raw_total_tokens") or 0) for v in by_model.values()),
            "turns": record.get("turns"),
        },
        "source": "codex.log+rollout",
    }
    try:
        (stage_dir / "cost.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError:
        pass


def _table_sort_key(record: Dict[str, Any]) -> tuple:
    version = record.get("version") or ""
    vnum = int(version[1:]) if version.startswith("v") and version[1:].isdigit() else 0
    try:
        sidx = _STAGE_ORDER.index(record.get("stage"))
    except ValueError:
        sidx = 99
    return (vnum, sidx)


def build_software_table(
    style_id: str, records: List[Dict[str, Any]], software_type: str = ""
) -> Dict[str, Any]:
    """Generate a phase-level usage table for a single software (style): rows = phases
    (including v1/v2/v3), columns = time / tokens per model / turns.

    Includes the style's own phases plus run-level shared design phases
    (FunctionDesigner / StyleDesigner etc., marked ``scope=run_shared``)."""
    relevant = [
        r for r in records
        if r.get("style_id") == style_id
        or (r.get("style_id") is None and r.get("stage") in _RUN_SHARED_STAGES)
    ]
    relevant.sort(key=_table_sort_key)
    rows: List[Dict[str, Any]] = []
    totals_by_model: Dict[str, int] = {}
    total_eff = 0
    total_billed = 0
    total_overhead = 0
    total_dur = 0.0
    for r in relevant:
        tokens_by_model = {
            m: int(v.get("effective_tokens") or 0) for m, v in (r.get("by_model") or {}).items()
        }
        eff = int(r.get("tokens") or 0)
        billed = int(r.get("tokens_with_retries") or r.get("tokens") or 0)
        overhead = int(r.get("retry_overhead") or 0)
        dur = r.get("duration_seconds")
        rows.append({
            "stage": r.get("stage"),
            "version": r.get("version"),
            "scope": "run_shared" if r.get("style_id") is None else "style",
            "channel": r.get("channel"),
            "duration_seconds": dur,
            "turns": r.get("turns"),
            "attempt_count": r.get("attempt_count"),
            "tokens_by_model": tokens_by_model,
            "effective_tokens": eff,
            "tokens_with_retries": billed,
            "retry_overhead": overhead,
        })
        for m, t in tokens_by_model.items():
            totals_by_model[m] = totals_by_model.get(m, 0) + t
        total_eff += eff
        total_billed += billed
        total_overhead += overhead
        if isinstance(dur, (int, float)):
            total_dur = round(total_dur + float(dur), 1)
    return {
        "schema_version": 2,
        "style_id": style_id,
        "software_type": software_type,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "columns": ["stage", "version", "duration_seconds", "tokens_by_model", "turns",
                    "tokens_with_retries", "retry_overhead"],
        "rows": rows,
        "totals": {
            "effective_tokens": total_eff,
            "effective_tokens_with_retries": total_billed,
            "retry_overhead": total_overhead,
            "duration_seconds": total_dur,
            "by_model": totals_by_model,
        },
    }


def _style_codex_logs(run_dir: Path, style_id: str) -> List[Path]:
    """Collect codex.logs related to a software (style): run-level shared design phases plus
    style-specific phases.

    Path rules correspond one-to-one with classify_codex_log; reconstructing by run_dir naturally
    implements "resume accumulates, restart (new run_dir) starts from zero"."""
    paths: List[Path] = []
    for shared in ("01_function_design", "02_icon_design", "02_style_design"):
        p = run_dir / shared / "codex.log"
        if p.exists():
            paths.append(p)
    p = run_dir / "03_substyle_design" / style_id / "codex.log"
    if p.exists():
        paths.append(p)
    vdir = run_dir / "software_variants" / style_id
    for sub in ("04_icons", "05_eval_tasks"):
        p = vdir / sub / "codex.log"
        if p.exists():
            paths.append(p)
    paths.extend(sorted(vdir.glob("loop/v*/coder/codex.log")))
    paths.extend(sorted(vdir.glob("loop/v*/devcoder/codex.log")))
    return paths


def _software_type_of(run_dir: Path) -> str:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            return str(json.loads(manifest_path.read_text(encoding="utf-8")).get("software_type") or "")
        except (OSError, json.JSONDecodeError):
            return ""
    return ""


def build_software_table_for_style(
    run_dir: Path, style_id: str, write: bool = True
) -> Dict[str, Any]:
    """Rebuild a phase-level usage table for a **single** software (style), and optionally write
    cost_by_stage.json + cost_by_stage.md.

    For the runner to call immediately after each software synthesis completes; only scans logs
    related to that style, without affecting other parallel styles."""
    run_dir = run_dir.resolve()
    records: List[Dict[str, Any]] = []
    for log_path in _style_codex_logs(run_dir, style_id):
        record = reconstruct_codex_record(log_path, run_dir)
        if record is None:
            continue
        _overlay_duration(record, log_path.parent)
        enrich_record(record, run_dir)
        records.append(record)
    vdir = run_dir / "software_variants" / style_id
    for cost_path in sorted(vdir.glob("loop/v*/eval/cost.json")):
        record = _read_eval_record(cost_path, run_dir)
        if record is None:
            continue
        _overlay_duration(record, cost_path.parent)
        enrich_record(record, run_dir)
        records.append(record)

    table = build_software_table(style_id, records, _software_type_of(run_dir))
    if write:
        try:
            vdir.mkdir(parents=True, exist_ok=True)
            (vdir / "cost_by_stage.json").write_text(
                json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except OSError:
            pass
    return table


def build_cost_summary(run_dir: Path, write: bool = True) -> Dict[str, Any]:
    """Scan the entire run directory, rebuild the pipeline-one usage summary, and write it to
    ``cost_summary.json``.

    Reconstructed purely from disk logs (codex.log + eval/cost.json), so it can generate summaries
    for historical runs; if the runner wrote ``usage.json``, its duration info is overlaid."""
    run_dir = run_dir.resolve()
    records: List[Dict[str, Any]] = []

    for log_path in sorted(run_dir.glob("**/codex.log")):
        record = reconstruct_codex_record(log_path, run_dir)
        if record is None:
            continue
        _overlay_duration(record, log_path.parent)
        enrich_record(record, run_dir)
        if write:
            _write_stage_cost_json(log_path.parent, record)
        records.append(record)

    for cost_path in sorted(run_dir.glob("software_variants/*/loop/*/eval/cost.json")):
        record = _read_eval_record(cost_path, run_dir)
        if record is None:
            continue
        _overlay_duration(record, cost_path.parent)
        enrich_record(record, run_dir)
        records.append(record)

    by_stage_type: Dict[str, Dict[str, Any]] = {}
    by_style: Dict[str, Dict[str, Any]] = {}
    by_model: Dict[str, Dict[str, Any]] = {}
    codex_tokens = 0
    gui_tokens = 0
    gui_tokens_wo = 0  # GUIEval without-retries figure (clean cumulative)
    tokens_with_retries = 0
    retry_overhead = 0
    total_duration = 0.0
    models: List[str] = []

    for record in records:
        stage = record["stage"]
        by_stage_type.setdefault(stage, _empty_bucket())
        _accumulate(by_stage_type[stage], record)

        # Separate stats by model kind: accumulate per model from this record's by_model. Current
        # GUIEval is usually single-model; historical dual-model seed records may still contain both
        # executor and judge. Model names are normalized (gpt-5.5-dc02 -> gpt-5.5) so the same model
        # is not split by channel aliases.
        kind = record.get("kind")
        rec_by_model = record.get("by_model") or {}
        primary = record.get("model_canonical") or record.get("model") or "unknown"
        if rec_by_model:
            primary = max(rec_by_model, key=lambda m: int(rec_by_model[m].get("effective_tokens") or 0))

        def _mbucket(name: str) -> Dict[str, Any]:
            return by_model.setdefault(
                name,
                {"tokens": 0, "codex_tokens": 0, "gui_eval_tokens": 0, "tokens_with_retries": 0,
                 "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
                 "duration_seconds": 0.0, "count": 0, "stage_types": []},
            )

        for m, mv in (rec_by_model or {primary: {"effective_tokens": int(record.get("tokens") or 0)}}).items():
            mb = _mbucket(m)
            t = int(mv.get("effective_tokens") or 0)
            mb["tokens"] += t
            mb["tokens_with_retries"] += int(mv.get("tokens_with_retries") or t)
            mb["input_tokens"] += int(mv.get("input_tokens") or 0)
            mb["cached_input_tokens"] += int(mv.get("cached_input_tokens") or 0)
            mb["output_tokens"] += int(mv.get("output_tokens") or 0)
            if kind == "gui_eval":
                mb["gui_eval_tokens"] += t
            else:
                mb["codex_tokens"] += t
            if stage not in mb["stage_types"]:
                mb["stage_types"].append(stage)
        # Duration and phase count are attributed to the primary model only once, to avoid
        # double-counting for multi-model records.
        pb = _mbucket(primary)
        rec_dur = record.get("duration_seconds")
        if isinstance(rec_dur, (int, float)):
            pb["duration_seconds"] = round(pb["duration_seconds"] + float(rec_dur), 1)
        pb["count"] += 1

        style_id = record.get("style_id") or "_run_level"
        style_bucket = by_style.setdefault(style_id, {"tokens": 0, "duration_seconds": 0.0, "count": 0, "by_stage": {}})
        style_bucket["tokens"] += int(record.get("tokens") or 0)
        dur = record.get("duration_seconds")
        if isinstance(dur, (int, float)):
            style_bucket["duration_seconds"] = round(style_bucket["duration_seconds"] + float(dur), 1)
        style_bucket["count"] += 1
        stage_key = f"{stage}/{record['version']}" if record.get("version") else stage
        style_bucket["by_stage"].setdefault(stage_key, _empty_bucket())
        _accumulate(style_bucket["by_stage"][stage_key], record)

        if record.get("kind") == "gui_eval":
            gui_tokens += int(record.get("tokens") or 0)
            gui_tokens_wo += int(record.get("tokens_without_retries", record.get("tokens")) or 0)
        else:
            codex_tokens += int(record.get("tokens") or 0)
        tokens_with_retries += int(record.get("tokens_with_retries") or record.get("tokens") or 0)
        retry_overhead += int(record.get("retry_overhead") or 0)
        if isinstance(dur, (int, float)):
            total_duration = round(total_duration + float(dur), 1)
        model = record.get("model_canonical") or record.get("model")
        if model and model not in models:
            models.append(model)

    wall_clock = _wall_clock_seconds(run_dir)
    manifest_path = run_dir / "manifest.json"
    software_type = ""
    if manifest_path.exists():
        try:
            software_type = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("software_type") or "")
        except (OSError, json.JSONDecodeError):
            software_type = ""

    # Level 2: generate a phase-level usage table for each software (style) and write it to
    # software_variants/<style>/cost_by_stage.json
    by_software: Dict[str, Any] = {}
    style_ids = sorted({r.get("style_id") for r in records if r.get("style_id")})
    for sid in style_ids:
        table = build_software_table(sid, records, software_type)
        by_software[sid] = table
        if write:
            try:
                (run_dir / "software_variants" / sid / "cost_by_stage.json").write_text(
                    json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            except OSError:
                pass

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "software_type": software_type,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": models,
        "totals": {
            "tokens": codex_tokens + gui_tokens,
            "tokens_without_retries": codex_tokens + gui_tokens_wo,
            "codex_tokens": codex_tokens,
            "gui_eval_tokens": gui_tokens,
            "gui_eval_tokens_without_retries": gui_tokens_wo,
            "tokens_with_retries": tokens_with_retries,
            "retry_overhead": retry_overhead,
            "stage_duration_seconds_sum": total_duration,
            "wall_clock_seconds": wall_clock,
            "stage_count": len(records),
        },
        "by_stage_type": by_stage_type,
        "by_style": by_style,
        "by_model": by_model,
        "by_software": by_software,
        "stages": records,
        "notes": [
            "tokens/tokens_without_retries = clean figure: effective tokens needed for a normal run without retries/failures (primary metric).",
            "tokens_with_retries = real bill figure: includes attempts discarded due to 429/failure; retry_overhead = difference from the final retained attempt's effective tokens.",
            "Codex and GUIEval both use input - cached_input + output as the cost-relevant effective token.",
            "For fix rounds (Coder/DevCoder v2+) that resume continuation, tokens are calculated as the difference from the previous round to avoid double counting.",
            "GUIEval's Input/Cached-in/Output come from model API usage; old traces that did not collect cache keep Cached-in at 0.",
            "stage_duration_seconds_sum is the sum of phase durations (will exceed wall_clock when run in parallel); wall_clock is the time difference between first and last runner events.",
            "by_model stats per model: current seed GUIEval's GUI operations go to executor, RESULT_JSON pure result requests go to judge; by default both use the same Seed model and are merged into one model item, roles remain separate. Duration/phase count are attributed to the primary model (most tokens) only once to avoid duplication.",
            "turns = number of model requests in this phase: codex counts token_count events with last.output>0 in the rollout cumulative window (prev_cumulative, this_cumulative]; GUIEval = model_calls.",
            "channel = actual channel this phase went through (dc02/gw/dc01): codex inferred from alias suffix or base_url; GUIEval goes through the shared proxy pool so recorded as pool.",
            "by_software = phase-level usage table for each software (style) (rows = phases including v1/v2/v3, columns = time / tokens per model / turns), also written to software_variants/<style>/cost_by_stage.json.",
            "Each codex phase directory also writes a unified-protocol cost.json (schema_version=2); GUIEval keeps its original cost.json.",
        ],
    }

    if write:
        out_path = run_dir / "cost_summary.json"
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Backfill run-level cost.json: scaffold only wrote {"total_tokens":0,"components":{}} as a
        # placeholder and never updated it. Write the aggregated totals back here, keeping the
        # original schema (total_tokens + components), so the top-level cost.json matches
        # cost_summary.json and is no longer always 0.
        try:
            (run_dir / "cost.json").write_text(
                json.dumps(
                    {
                        "total_tokens": codex_tokens + gui_tokens,
                        "total_tokens_with_retries": tokens_with_retries,
                        "retry_overhead": retry_overhead,
                        "components": {
                            "codex": {"total_tokens": codex_tokens},
                            "gui_eval": {"total_tokens": gui_tokens},
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
    return summary


def _format_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild/view token/time usage summary for the software synthesis pipeline.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--print", action="store_true", help="Also print a brief table to the terminal.")
    parser.add_argument("--software", type=str, default=None,
                        help="Print phase-level usage table for a software (style), e.g. style_03; pass 'all' to print all.")
    args = parser.parse_args(argv)

    summary = build_cost_summary(args.run_dir, write=True)
    out_path = Path(args.run_dir).resolve() / "cost_summary.json"
    print(f"wrote {out_path}")
    if args.print:
        totals = summary["totals"]
        print("\n== Totals ==")
        print(f"  Effective tokens total (no retries): {_format_int(totals['tokens'])} "
              f"(codex {_format_int(totals['codex_tokens'])} + GUIEval {_format_int(totals['gui_eval_tokens'])})")
        print(f"  Bill with retries: {_format_int(totals.get('tokens_with_retries'))} "
              f"(retry overhead {_format_int(totals.get('retry_overhead'))})")
        print(f"  Sum of phase durations: {totals['stage_duration_seconds_sum']}s; wall_clock: {totals['wall_clock_seconds']}s")
        print(f"  Models: {', '.join(summary['models']) or '(unknown)'})")
        print("\n== By Stage Type ==")
        for stage, bucket in summary["by_stage_type"].items():
            print(f"  {stage:20s} tokens={_format_int(bucket['tokens']):>14s} "
                  f"duration={bucket['duration_seconds']:>8.1f}s count={bucket['count']} "
                  f"models={','.join(bucket['models']) or '-'}")
        print("\n== By Model ==")
        for model_name, bucket in summary["by_model"].items():
            print(f"  {model_name:24s} tokens={_format_int(bucket['tokens']):>14s} "
                  f"(codex {_format_int(bucket['codex_tokens'])} + GUIEval {_format_int(bucket['gui_eval_tokens'])}) "
                  f"duration={bucket['duration_seconds']:>8.1f}s count={bucket['count']} "
                  f"stages={','.join(bucket['stage_types']) or '-'}")

    if args.software:
        tables = summary.get("by_software") or {}
        targets = sorted(tables) if args.software == "all" else [args.software]
        for sid in targets:
            table = tables.get(sid)
            if not table:
                print(f"\n(no usage table for {sid})")
                continue
            print(f"\n== Software {sid} ({table.get('software_type') or '?'}) phase-level usage ==")
            print(f"  {'stage':24s} {'time(s)':>9s} {'turns':>6s} {'+retry':>9s}  tokens_by_model (no retries)")
            for row in table["rows"]:
                stg = row["stage"] + (f"/{row['version']}" if row.get("version") else "")
                ch = f"[{row['channel']}]" if row.get("channel") else ""
                dur = row.get("duration_seconds")
                dur_s = f"{dur:.1f}" if isinstance(dur, (int, float)) else "-"
                turns = row.get("turns")
                turns_s = str(turns) if turns is not None else "-"
                ov = int(row.get("retry_overhead") or 0)
                ov_s = f"+{_format_int(ov)}" if ov else "-"
                tbm = ", ".join(f"{m} {_format_int(t)}" for m, t in row["tokens_by_model"].items()) or "-"
                print(f"  {stg:24s} {dur_s:>9s} {turns_s:>6s} {ov_s:>9s}  {tbm} {ch}")
            tot = table["totals"]
            tbm = ", ".join(f"{m} {_format_int(t)}" for m, t in tot["by_model"].items()) or "-"
            ov_s = f"+{_format_int(tot.get('retry_overhead'))}" if tot.get("retry_overhead") else "-"
            print(f"  {'TOTAL':24s} {tot['duration_seconds']:>9.1f} {'':>6s} {ov_s:>9s}  {tbm}")
            print(f"  {'(bill with retries)':24s} {'':>9s} {'':>6s} {'':>9s}  "
                  f"{_format_int(tot.get('effective_tokens_with_retries'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
