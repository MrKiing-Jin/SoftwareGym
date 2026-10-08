from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable


def extract_usage(value: Any) -> Dict[str, int]:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    patterns = {
        "input_tokens": [r"input_tokens[\"']?\s*[:=]\s*(\d+)", r"prompt_tokens[\"']?\s*[:=]\s*(\d+)"],
        "output_tokens": [r"output_tokens[\"']?\s*[:=]\s*(\d+)", r"completion_tokens[\"']?\s*[:=]\s*(\d+)"],
        "total_tokens": [r"total_tokens[\"']?\s*[:=]\s*(\d+)"],
    }
    for name, regexes in patterns.items():
        for regex in regexes:
            for match in re.finditer(regex, text):
                if name == "input_tokens":
                    input_tokens += int(match.group(1))
                elif name == "output_tokens":
                    output_tokens += int(match.group(1))
                else:
                    total_tokens += int(match.group(1))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


def aggregate_trace_usage(trace_files: Iterable[Path]) -> Dict[str, int]:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for path in trace_files:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        usage = extract_usage(value)
        for key in total:
            total[key] += usage[key]
    return total


def _usage_parts(usage: Dict[str, Any]) -> tuple:
    pin = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    cached = min(pin, max(0, int(usage.get("cached_input_tokens", 0) or 0)))
    pout = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    reasoning = min(pout, max(0, int(usage.get("reasoning_output_tokens", 0) or 0)))
    return pin, cached, pout, reasoning


# Events that incur real billing: normal execution step (step_predicted) + pure judge
# finalization request after exhausting the step budget (final_result_requested).
# Seed's executor/judge writes into the same event's usage_breakdown.
_USAGE_EVENTS = ("step_predicted", "final_result_requested")


def usage_from_steps_jsonl(path: Path) -> tuple:
    """Sum every real model call from a single ``steps.jsonl`` event stream.

    Returns ``(total, by_role)``:
    - ``total``: total token sum of all model calls in this file (real billing basis).
    - ``by_role``: split by ``executor`` / ``judge``, taken from
      ``predict_info.usage_breakdown``; current seed uses an independent judge, gpt54
      calls go to executor. When no breakdown exists, the whole amount goes to executor.

    Covers retry attempts, intermediate dropped calls, and final scoring after step
    budget exhaustion---they are all actually billed. Only one usage is taken per step;
    never scan the whole JSON with regex (would duplicate/double count).
    """
    total = _empty_usage()
    by_role = {"executor": _empty_usage(), "judge": _empty_usage()}
    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return total, by_role
    with handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") not in _USAGE_EVENTS:
                continue
            pi = event.get("predict_info") or {}
            usage = pi.get("model_usage")
            if not isinstance(usage, dict):
                continue
            pin, cached, pout, reasoning = _usage_parts(usage)
            ptot = int(usage.get("total_tokens", 0) or 0) or (pin + pout)
            total["input_tokens"] += pin
            total["cached_input_tokens"] += cached
            total["output_tokens"] += pout
            total["reasoning_output_tokens"] += reasoning
            total["total_tokens"] += ptot
            total["effective_tokens"] += max(0, pin - cached) + pout
            total["model_calls"] += max(1, int(usage.get("model_calls", 0) or 0))
            breakdown = pi.get("usage_breakdown")
            if isinstance(breakdown, dict) and (breakdown.get("executor") or breakdown.get("judge")):
                for role in ("executor", "judge"):
                    ru = breakdown.get(role)
                    if not isinstance(ru, dict):
                        continue
                    rin, rcached, rout, rreasoning = _usage_parts(ru)
                    if rin or rout:
                        by_role[role]["input_tokens"] += rin
                        by_role[role]["cached_input_tokens"] += rcached
                        by_role[role]["output_tokens"] += rout
                        by_role[role]["reasoning_output_tokens"] += rreasoning
                        by_role[role]["total_tokens"] += rin + rout
                        by_role[role]["effective_tokens"] += max(0, rin - rcached) + rout
                        by_role[role]["model_calls"] += max(1, int(ru.get("model_calls", 0) or 0))
            else:
                by_role["executor"]["input_tokens"] += pin
                by_role["executor"]["cached_input_tokens"] += cached
                by_role["executor"]["output_tokens"] += pout
                by_role["executor"]["reasoning_output_tokens"] += reasoning
                by_role["executor"]["total_tokens"] += ptot
                by_role["executor"]["effective_tokens"] += max(0, pin - cached) + pout
                by_role["executor"]["model_calls"] += max(
                    1,
                    int(usage.get("model_calls", 0) or 0),
                )
    return total, by_role


def models_from_steps_jsonl(path: Path) -> Dict[str, Counter]:
    """Read the actual executor/judge model names recorded with billed events."""
    found: Dict[str, Counter] = {
        "executor": Counter(),
        "judge": Counter(),
    }
    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return found
    with handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") not in _USAGE_EVENTS:
                continue
            predict_info = event.get("predict_info") or {}
            models = predict_info.get("models")
            if not isinstance(models, dict):
                continue
            breakdown = predict_info.get("usage_breakdown") or {}
            for role in ("executor", "judge"):
                name = str(models.get(role) or "").strip()
                role_usage = breakdown.get(role) if isinstance(breakdown, dict) else None
                calls = (
                    max(1, int(role_usage.get("model_calls", 0) or 0))
                    if isinstance(role_usage, dict)
                    and (
                        role_usage.get("prompt_tokens")
                        or role_usage.get("input_tokens")
                        or role_usage.get("completion_tokens")
                        or role_usage.get("output_tokens")
                    )
                    else 0
                )
                if name and calls:
                    found[role][name] += calls
    return found


def _attempt_index(steps_jsonl_path: Path) -> int:
    """Parse attempt index from ``traces/case_XXX/attempt_YY/steps.jsonl``; treat as 1 if absent."""
    name = steps_jsonl_path.parent.name  # attempt_YY
    m = re.search(r"attempt[_-]?(\d+)", name)
    return int(m.group(1)) if m else 1


def _case_key(steps_jsonl_path: Path) -> str:
    """Case identifier: take the parent of the attempt directory (case_XXX); fall back to the file's parent directory if missing."""
    p = steps_jsonl_path.parent
    if re.search(r"attempt[_-]?\d+", p.name):
        return p.parent.name
    return p.name


def _empty_usage() -> Dict[str, int]:
    return {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 0,
        "effective_tokens": 0,
        "model_calls": 0,
    }


def _add(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for key in dst:
        dst[key] += int(src.get(key, 0) or 0)


def write_cost(eval_dir: Path, model: str = "unknown") -> Path:
    """Aggregate GUIEval token costs and write ``cost.json``.

    Two metrics are computed separately (based on the full ``steps.jsonl`` event stream):
    - ``with_retries``: all model calls across all attempts = tokens actually billed.
    - ``without_retries``: for each case only the final attempt = tokens a normal,
      single successful run should spend (excluding infra retry overhead).

    ``total_tokens`` uses the effective basis aligned with Codex:
    ``input - cached_input + output``; the original provider total is stored separately as
    ``raw_total_tokens``. Both metrics keep with-retries and without-retries versions.
    Unsuffixed Input/Cached/Output/turns refer to the final retained attempt; the
    corresponding real full values use ``*_with_retries``.
    When no jsonl exists (very old run), fall back to old ``steps.json`` regex
    aggregation, in which case the two metrics are identical.
    """
    jsonl_files = sorted(eval_dir.glob("traces/**/steps.jsonl"))
    # Accumulate per-role with/without retry totals for the by_model breakdown.
    wr_by_role = {"executor": _empty_usage(), "judge": _empty_usage()}  # with retries (real bill)
    wor_by_role = {"executor": _empty_usage(), "judge": _empty_usage()}  # without retries (clean run)
    if jsonl_files:
        with_retries = _empty_usage()
        model_votes = {"executor": Counter(), "judge": Counter()}
        # For each case pick the highest attempt index as the "final retained trace".
        best_per_case: Dict[str, tuple] = {}  # case -> (attempt_idx, total, by_role)
        for path in jsonl_files:
            part, part_role = usage_from_steps_jsonl(path)
            _add(with_retries, part)
            for role in wr_by_role:
                _add(wr_by_role[role], part_role[role])
            discovered_models = models_from_steps_jsonl(path)
            for role in model_votes:
                model_votes[role].update(discovered_models[role])
            case = _case_key(path)
            idx = _attempt_index(path)
            if case not in best_per_case or idx > best_per_case[case][0]:
                best_per_case[case] = (idx, part, part_role)
        without_retries = _empty_usage()
        for _idx, part, part_role in best_per_case.values():
            _add(without_retries, part)
            for role in wor_by_role:
                _add(wor_by_role[role], part_role[role])
        source = "steps.jsonl"
    else:
        model_votes = {"executor": Counter(), "judge": Counter()}
        usage = aggregate_trace_usage(eval_dir.glob("traces/**/steps.json"))
        usage = {
            **usage,
            "cached_input_tokens": 0,
            "reasoning_output_tokens": 0,
            "effective_tokens": int(usage.get("total_tokens", 0) or 0),
            "model_calls": 0,
        }
        with_retries = dict(usage)
        without_retries = dict(usage)
        _add(wr_by_role["executor"], usage)  # old steps.json has no role info; attribute everything to executor
        _add(wor_by_role["executor"], usage)
        source = "steps.json"

    retry_overhead = {k: with_retries[k] - without_retries[k] for k in with_retries}

    # Split by actual model: by default seed makes executor/judge reuse the same Seed model
    # and naturally merges; only when SEED_JUDGE_USE_EXEC_CONFIG is explicitly disabled are
    # independent judge model items split out.
    impl = (os.environ.get("APPEVAL_AGENT_IMPL") or "gpt54").strip().lower()
    if impl in {"seed", "doubao", "seed_gui"}:
        exec_model_name = os.environ.get("SEED_EXEC_MODEL") or model or "unknown"
        judge_uses_exec = (
            os.environ.get("SEED_JUDGE_USE_EXEC_CONFIG", "1").strip().lower()
            not in {"0", "false", "no"}
        )
        judge_model_name = (
            exec_model_name
            if judge_uses_exec
            else os.environ.get("SEED_JUDGE_MODEL") or model or "judge"
        )
    else:
        exec_model_name = model or "unknown"
        judge_model_name = model or "unknown"
    if model_votes["executor"]:
        exec_model_name = model_votes["executor"].most_common(1)[0][0]
    if model_votes["judge"]:
        judge_model_name = model_votes["judge"].most_common(1)[0][0]
    role_to_model = {"executor": exec_model_name, "judge": judge_model_name}

    by_model: Dict[str, Any] = {}
    for role in ("executor", "judge"):
        billed = wr_by_role[role]
        clean = wor_by_role[role]
        if billed["input_tokens"] == 0 and billed["output_tokens"] == 0 and billed["model_calls"] == 0:
            continue
        name = role_to_model[role]
        entry = by_model.setdefault(
            name,
            {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_output_tokens": 0,
                "input_tokens_with_retries": 0,
                "cached_input_tokens_with_retries": 0,
                "output_tokens_with_retries": 0,
                "reasoning_output_tokens_with_retries": 0,
                "raw_total_tokens": 0,
                "raw_total_tokens_with_retries": 0,
                "raw_total_tokens_without_retries": 0,
                "effective_tokens": 0,
                "effective_tokens_without_retries": 0,
                "tokens_with_retries": 0,
                "total_tokens": 0,
                "total_tokens_without_retries": 0,
                "retry_overhead": 0,
                "model_calls": 0,
                "model_calls_with_retries": 0,
                "roles": [],
            },
        )
        entry["input_tokens"] += clean["input_tokens"]
        entry["cached_input_tokens"] += clean["cached_input_tokens"]
        entry["output_tokens"] += clean["output_tokens"]
        entry["reasoning_output_tokens"] += clean["reasoning_output_tokens"]
        entry["input_tokens_with_retries"] += billed["input_tokens"]
        entry["cached_input_tokens_with_retries"] += billed["cached_input_tokens"]
        entry["output_tokens_with_retries"] += billed["output_tokens"]
        entry["reasoning_output_tokens_with_retries"] += billed["reasoning_output_tokens"]
        entry["raw_total_tokens"] += clean["total_tokens"]
        entry["raw_total_tokens_with_retries"] += billed["total_tokens"]
        entry["raw_total_tokens_without_retries"] += clean["total_tokens"]
        entry["effective_tokens"] += clean["effective_tokens"]
        entry["effective_tokens_without_retries"] += clean["effective_tokens"]
        entry["tokens_with_retries"] += billed["effective_tokens"]
        # Keep the historical names as the cost-facing effective-token metric.
        entry["total_tokens"] += billed["effective_tokens"]
        entry["total_tokens_without_retries"] += clean["effective_tokens"]
        entry["retry_overhead"] += max(
            0,
            billed["effective_tokens"] - clean["effective_tokens"],
        )
        entry["model_calls"] += clean["model_calls"]
        entry["model_calls_with_retries"] += billed["model_calls"]
        if role not in entry["roles"]:
            entry["roles"].append(role)
    if not by_model:  # Fallback: keep at least one entry when no role data exists.
        by_model[model or "unknown"] = {
            "input_tokens": without_retries["input_tokens"],
            "cached_input_tokens": without_retries["cached_input_tokens"],
            "output_tokens": without_retries["output_tokens"],
            "reasoning_output_tokens": without_retries["reasoning_output_tokens"],
            "input_tokens_with_retries": with_retries["input_tokens"],
            "cached_input_tokens_with_retries": with_retries["cached_input_tokens"],
            "output_tokens_with_retries": with_retries["output_tokens"],
            "reasoning_output_tokens_with_retries": with_retries["reasoning_output_tokens"],
            "raw_total_tokens": without_retries["total_tokens"],
            "raw_total_tokens_with_retries": with_retries["total_tokens"],
            "raw_total_tokens_without_retries": without_retries["total_tokens"],
            "effective_tokens": without_retries["effective_tokens"],
            "effective_tokens_without_retries": without_retries["effective_tokens"],
            "tokens_with_retries": with_retries["effective_tokens"],
            "total_tokens": with_retries["effective_tokens"],
            "total_tokens_without_retries": without_retries["effective_tokens"],
            "retry_overhead": max(
                0,
                with_retries["effective_tokens"] - without_retries["effective_tokens"],
            ),
            "model_calls": without_retries["model_calls"],
            "model_calls_with_retries": with_retries["model_calls"],
            "roles": ["executor"],
        }

    by_role = {"with_retries": wr_by_role, "without_retries": wor_by_role}
    primary_model = (
        max(by_model, key=lambda name: int(by_model[name].get("tokens_with_retries", 0) or 0))
        if by_model
        else (model or "unknown")
    )
    gui = {
        "model": primary_model,
        "input_tokens": without_retries["input_tokens"],
        "cached_input_tokens": without_retries["cached_input_tokens"],
        "output_tokens": without_retries["output_tokens"],
        "reasoning_output_tokens": without_retries["reasoning_output_tokens"],
        "input_tokens_with_retries": with_retries["input_tokens"],
        "cached_input_tokens_with_retries": with_retries["cached_input_tokens"],
        "output_tokens_with_retries": with_retries["output_tokens"],
        "reasoning_output_tokens_with_retries": with_retries["reasoning_output_tokens"],
        "raw_total_tokens": without_retries["total_tokens"],
        "raw_total_tokens_with_retries": with_retries["total_tokens"],
        "raw_total_tokens_without_retries": without_retries["total_tokens"],
        "effective_tokens": without_retries["effective_tokens"],
        "effective_tokens_without_retries": without_retries["effective_tokens"],
        "effective_tokens_with_retries": with_retries["effective_tokens"],
        "total_tokens": with_retries["effective_tokens"],
        "model_calls": without_retries["model_calls"],
        "model_calls_with_retries": with_retries["model_calls"],
        "with_retries": with_retries,
        "without_retries": without_retries,
        "retry_overhead": retry_overhead,
        "by_role": by_role,
        "by_model": by_model,
    }
    payload = {
        "schema_version": 2,
        "stage": "GUIEval",
        "kind": "gui_eval",
        "status": "completed",
        "model": primary_model,
        "models": sorted(by_model.keys()),
        "usage_source": source,
        "input_tokens": without_retries["input_tokens"],
        "cached_input_tokens": without_retries["cached_input_tokens"],
        "output_tokens": without_retries["output_tokens"],
        "reasoning_output_tokens": without_retries["reasoning_output_tokens"],
        "input_tokens_with_retries": with_retries["input_tokens"],
        "cached_input_tokens_with_retries": with_retries["cached_input_tokens"],
        "output_tokens_with_retries": with_retries["output_tokens"],
        "reasoning_output_tokens_with_retries": with_retries["reasoning_output_tokens"],
        "raw_total_tokens": without_retries["total_tokens"],
        "raw_total_tokens_with_retries": with_retries["total_tokens"],
        "raw_total_tokens_without_retries": without_retries["total_tokens"],
        "effective_tokens": without_retries["effective_tokens"],
        "effective_tokens_without_retries": without_retries["effective_tokens"],
        "effective_tokens_with_retries": with_retries["effective_tokens"],
        "total_tokens": with_retries["effective_tokens"],
        "total_tokens_without_retries": without_retries["effective_tokens"],
        "retry_overhead": max(
            0,
            with_retries["effective_tokens"] - without_retries["effective_tokens"],
        ),
        "by_model": by_model,
        "by_role": by_role,
        "components": {"gui_agent": gui},
        "turns": without_retries["model_calls"],
        "turns_with_retries": with_retries["model_calls"],
        "totals": {
            "effective_tokens": without_retries["effective_tokens"],
            "effective_tokens_with_retries": with_retries["effective_tokens"],
            "retry_overhead": max(
                0,
                with_retries["effective_tokens"] - without_retries["effective_tokens"],
            ),
            "raw_total_tokens": without_retries["total_tokens"],
            "raw_total_tokens_with_retries": with_retries["total_tokens"],
            "turns": without_retries["model_calls"],
            "turns_with_retries": with_retries["model_calls"],
        },
        "source": "steps.jsonl" if jsonl_files else "steps.json",
    }
    path = eval_dir / "cost.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
