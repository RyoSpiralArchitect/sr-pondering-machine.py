#!/usr/bin/env python3
"""Build an HTML analysis report for a Claude/Gemini artifact sweep."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
MONO_FONT_FAMILY = ["SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "monospace"]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLOR_FAMILIES = {
    "blue": {"xlight": "#EAF1FE", "light": "#CEDFFE", "base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"},
    "gold": {"xlight": "#FFF4C2", "light": "#FFEA8F", "base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"},
    "orange": {"xlight": "#FFEDDE", "light": "#FFBDA1", "base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
    "olive": {"xlight": "#D8ECBD", "light": "#BEEB96", "base": "#A3D576", "mid": "#71B436", "dark": "#386411"},
    "pink": {"xlight": "#FCDAD6", "light": "#F5BACC", "base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"},
}

CONTRACT_ORDER = [
    "none",
    "final_closure",
    "log_closure",
    "log_final_closure",
    "log_skeleton_closure",
    "log_skeleton_final_closure",
]

LOG_PHASE_ROUTE_ORDER = [
    "inherit_1024_no_rescue",
    "low_1024_no_rescue",
    "low_log1024_final_low2048",
    "low_log2048_final_low2048",
    "inherit_2048_no_rescue",
    "low_2048_rescue",
]


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.monospace": MONO_FONT_FAMILY,
            "patch.linewidth": 1.0,
        },
    )


def add_chart_header(fig: Any, ax: Any, title: str, subtitle: str) -> None:
    if not title or not subtitle:
        raise ValueError("Every shipped chart needs a non-empty title and subtitle.")
    ax.set_title("")
    fig.subplots_adjust(top=0.82)
    left = ax.get_position().x0
    fig.text(left, 0.97, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.91, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    sns.despine(ax=ax)


def save_fig(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def safe_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(parsed):
        return parsed
    return None


def safe_int(value: Any) -> Optional[int]:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def infer_scaffold_meta(item: Dict[str, Any]) -> tuple[str, Optional[int]]:
    name = str(item.get("name") or "").strip()
    cfg = item.get("cfg_overrides") if isinstance(item.get("cfg_overrides"), dict) else {}
    extras = item.get("extras") if isinstance(item.get("extras"), dict) else {}
    scaffold = extras.get("scaffold") if isinstance(extras.get("scaffold"), dict) else {}

    condition = str(cfg.get("scaffold_condition") or scaffold.get("condition") or "").strip()
    dose = safe_int(cfg.get("scaffold_token_target"))
    if dose is None:
        dose = safe_int(scaffold.get("target_tokens"))

    match = re.match(r"^(?P<condition>[A-Za-z0-9_-]+)_dose_(?P<dose>\d+)(?:_[A-Za-z0-9_-]+)?$", name)
    if match:
        condition = condition or match.group("condition")
        dose = dose if dose is not None else int(match.group("dose"))

    if not condition and name in {"assoc", "random", "facts", "isomorphic"}:
        condition = name
    return condition, dose


def infer_output_contract(item: Dict[str, Any]) -> str:
    name = str(item.get("name") or "").strip()
    cfg = item.get("cfg_overrides") if isinstance(item.get("cfg_overrides"), dict) else {}
    extras = item.get("extras") if isinstance(item.get("extras"), dict) else {}
    contract = str(cfg.get("output_contract") or extras.get("output_contract") or "").strip()
    if not contract:
        for candidate in sorted(CONTRACT_ORDER, key=len, reverse=True):
            suffix = f"_{candidate}"
            if name.endswith(suffix):
                contract = candidate
                break
    return contract or "none"


def infer_log_phase_route(item: Dict[str, Any]) -> str:
    name = str(item.get("name") or "").strip()
    cfg = item.get("cfg_overrides") if isinstance(item.get("cfg_overrides"), dict) else {}
    for route in LOG_PHASE_ROUTE_ORDER:
        if name.endswith(f"_{route}"):
            return route

    effort = str(cfg.get("log_phase_reasoning_effort") or "inherit").strip() or "inherit"
    budget = safe_int(cfg.get("log_phase_max_new_tokens"))
    budget_label = str(budget) if budget and budget > 0 else "inherit"
    rescue = bool(cfg.get("log_phase_rescue"))
    final_effort = str(cfg.get("final_phase_reasoning_effort") or "inherit").strip() or "inherit"
    final_budget = safe_int(cfg.get("final_phase_max_new_tokens"))
    final_budget_label = str(final_budget) if final_budget and final_budget > 0 else "inherit"
    final_is_default = final_effort == "inherit" and final_budget_label == "inherit"
    if effort == "inherit" and budget_label == "inherit" and not rescue and final_is_default:
        return ""
    log_part = f"{effort}_{budget_label}_{'rescue' if rescue else 'no_rescue'}"
    if final_is_default:
        return log_part
    return f"{log_part}__final_{final_effort}_{final_budget_label}"


def _nonempty_lines(text: Any) -> List[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def marker_line_present(text: Any, marker: str) -> int:
    return int(any(line == marker for line in _nonempty_lines(text)))


def marker_is_last_line(text: Any, marker: str) -> int:
    lines = _nonempty_lines(text)
    return int(bool(lines) and lines[-1] == marker)


def log_skeleton_prefix_count(text: Any) -> int:
    lines = _nonempty_lines(text)
    prefixes = ("X1|", "X2|", "X3|", "X4|")
    return sum(1 for prefix in prefixes if any(line.startswith(prefix) for line in lines))


def log_skeleton_complete(text: Any) -> int:
    lines = _nonempty_lines(text)
    if len(lines) != 5 or marker_is_last_line(text, "END_LOG") != 1:
        return 0
    return int(all(lines[ix].startswith(f"X{ix + 1}|") for ix in range(4)))


def prompt_leakage_flag(text: Any) -> int:
    lowered = str(text or "").lower()
    needles = [
        "<ponder_log>",
        "</ponder_log>",
        "actual question:",
        "additional output contract",
        "additional log contract",
        "本題の質問",
        "追加の出力契約",
        "追加のログ契約",
        "出力は回答本文のみ",
    ]
    return int(any(needle.lower() in lowered for needle in needles))


def record_logs(item: Dict[str, Any]) -> List[str]:
    records = item.get("records")
    if not isinstance(records, list):
        return []
    logs: List[str] = []
    for record in records:
        if isinstance(record, dict):
            log = str(record.get("ponder_log") or "").strip()
            if log:
                logs.append(log)
    return logs


def finish_reason(meta: Any) -> str:
    if not isinstance(meta, dict):
        return ""
    return str(meta.get("finish_reason") or meta.get("stop_reason") or "").strip().lower()


def count_record_finish_reason(item: Dict[str, Any], reason: str) -> int:
    records = item.get("records")
    if not isinstance(records, list):
        return 0
    count = 0
    for record in records:
        if isinstance(record, dict) and finish_reason(record.get("api_generation")) == reason:
            count += 1
    return count


def count_record_log_phase_flag(item: Dict[str, Any], key: str) -> int:
    records = item.get("records")
    if not isinstance(records, list):
        return 0
    count = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        log_phase = record.get("log_phase") if isinstance(record.get("log_phase"), dict) else {}
        if bool(log_phase.get(key)):
            count += 1
    return count


def palette_for(values: Iterable[Any]) -> Dict[str, str]:
    color_values = [
        COLOR_FAMILIES["blue"]["base"],
        COLOR_FAMILIES["gold"]["base"],
        COLOR_FAMILIES["orange"]["base"],
        COLOR_FAMILIES["olive"]["base"],
        COLOR_FAMILIES["pink"]["base"],
        COLOR_FAMILIES["blue"]["mid"],
        COLOR_FAMILIES["gold"]["mid"],
        COLOR_FAMILIES["orange"]["mid"],
        COLOR_FAMILIES["olive"]["mid"],
    ]
    labels = unique_texts(values)
    return {label: color_values[ix % len(color_values)] for ix, label in enumerate(labels)}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_rows(sweep_dir: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    manifest = read_json(sweep_dir / "manifest.json")
    run_rows: List[Dict[str, Any]] = []
    item_rows: List[Dict[str, Any]] = []

    for run in manifest.get("runs", []):
        if not isinstance(run, dict):
            continue
        raw_result_path = Path(str(run.get("json_out") or ""))
        result_candidates = [sweep_dir / raw_result_path.name]
        if raw_result_path.is_absolute():
            result_candidates.append(raw_result_path)
        else:
            result_candidates.extend([Path.cwd() / raw_result_path, sweep_dir.parents[0] / raw_result_path])
        result_path = next((p for p in result_candidates if p.exists()), result_candidates[0])
        result = read_json(result_path)
        provider = str(run.get("provider") or result.get("provider") or "").strip()
        query_id = str(run.get("query_id") or "").strip()
        baseline_chars = None
        for item in result.get("items", []):
            if isinstance(item, dict) and item.get("kind") == "baseline":
                metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
                baseline_chars = metrics.get("answer_chars")
                break

        run_rows.append(
            {
                "provider": provider,
                "query_id": query_id,
                "model": result.get("model"),
                "ok": bool(run.get("ok")),
                "elapsed_s": safe_float(run.get("elapsed_s")),
                "item_count": len(result.get("items") or []),
                "baseline_answer_chars": baseline_chars,
                "query": str(run.get("query") or result.get("query") or ""),
                "json_out": str(result_path),
                "trace_out": str(sweep_dir / Path(str(run.get("trace_out") or "")).name),
                "matrix_report_out": str(sweep_dir / Path(str(run.get("matrix_report_out") or "")).name),
                "trace_report_out": str(sweep_dir / Path(str(run.get("trace_report_out") or "")).name),
                "stdout": str(sweep_dir / Path(str(run.get("stdout") or "")).name),
                "stderr": str(sweep_dir / Path(str(run.get("stderr") or "")).name),
            }
        )

        for item in result.get("items", []):
            if not isinstance(item, dict) or item.get("kind") != "ponder":
                continue
            metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
            comp = item.get("comparison") if isinstance(item.get("comparison"), dict) else {}
            semantic = comp.get("semantic") if isinstance(comp.get("semantic"), dict) else {}
            stance = comp.get("stance") if isinstance(comp.get("stance"), dict) else {}
            spatial = comp.get("spatial_metaphor") if isinstance(comp.get("spatial_metaphor"), dict) else {}
            spatial_logs = spatial.get("logs") if isinstance(spatial.get("logs"), dict) else {}
            budget = comp.get("token_budget") if isinstance(comp.get("token_budget"), dict) else {}
            delta = budget.get("delta") if isinstance(budget.get("delta"), dict) else {}
            all_usage = budget.get("all_api_usage") if isinstance(budget.get("all_api_usage"), dict) else {}
            extras = item.get("extras") if isinstance(item.get("extras"), dict) else {}
            warnings = extras.get("api_warnings") if isinstance(extras.get("api_warnings"), list) else []
            scaffold_condition, scaffold_dose = infer_scaffold_meta(item)
            output_contract = infer_output_contract(item)
            cfg_overrides = item.get("cfg_overrides") if isinstance(item.get("cfg_overrides"), dict) else {}
            log_phase_route = infer_log_phase_route(item)
            log_phase_reasoning_effort = str(cfg_overrides.get("log_phase_reasoning_effort") or "inherit").strip() or "inherit"
            log_phase_max_new_tokens = safe_int(cfg_overrides.get("log_phase_max_new_tokens")) or 0
            log_phase_rescue_enabled = int(bool(cfg_overrides.get("log_phase_rescue")))
            log_phase_rescue_used_count = count_record_log_phase_flag(item, "rescue_used")
            final_phase_reasoning_effort = str(cfg_overrides.get("final_phase_reasoning_effort") or "inherit").strip() or "inherit"
            final_phase_max_new_tokens = safe_int(cfg_overrides.get("final_phase_max_new_tokens")) or 0
            answer = str(item.get("answer") or "")
            record_log_values = record_logs(item)
            log_marker_count = sum(marker_line_present(log, "END_LOG") for log in record_log_values)
            log_skeleton_prefixes = sum(log_skeleton_prefix_count(log) for log in record_log_values)
            log_skeleton_complete_count = sum(log_skeleton_complete(log) for log in record_log_values)
            final_meta = extras.get("api_final_generation") if isinstance(extras.get("api_final_generation"), dict) else {}
            final_reason = finish_reason(final_meta)
            item_rows.append(
                {
                    "provider": provider,
                    "query_id": query_id,
                    "model": result.get("model"),
                    "condition": str(item.get("name") or ""),
                    "scaffold_condition": scaffold_condition,
                    "scaffold_dose": scaffold_dose,
                    "output_contract": output_contract,
                    "log_phase_route": log_phase_route,
                    "log_phase_reasoning_effort": log_phase_reasoning_effort,
                    "log_phase_max_new_tokens": log_phase_max_new_tokens,
                    "log_phase_rescue_enabled": log_phase_rescue_enabled,
                    "log_phase_rescue_used_count": log_phase_rescue_used_count,
                    "log_phase_rescue_used_rate": (log_phase_rescue_used_count / len(record_log_values)) if record_log_values else None,
                    "final_phase_reasoning_effort": final_phase_reasoning_effort,
                    "final_phase_max_new_tokens": final_phase_max_new_tokens,
                    "answer_chars": safe_float(metrics.get("answer_chars")),
                    "final_marker_line_present": marker_line_present(answer, "END_ANSWER"),
                    "final_marker_is_last_line": marker_is_last_line(answer, "END_ANSWER"),
                    "prompt_leakage_flag": prompt_leakage_flag(answer),
                    "ponder_log_count": len(record_log_values),
                    "ponder_log_marker_count": log_marker_count,
                    "ponder_log_marker_rate": (log_marker_count / len(record_log_values)) if record_log_values else None,
                    "ponder_log_skeleton_prefix_count": log_skeleton_prefixes,
                    "ponder_log_skeleton_prefix_rate": (log_skeleton_prefixes / (4 * len(record_log_values))) if record_log_values else None,
                    "ponder_log_skeleton_complete_count": log_skeleton_complete_count,
                    "ponder_log_skeleton_complete_rate": (log_skeleton_complete_count / len(record_log_values)) if record_log_values else None,
                    "final_finish_reason": final_reason,
                    "final_finish_is_length": int(final_reason == "length") if final_reason else None,
                    "ponder_finish_length_count": count_record_finish_reason(item, "length"),
                    "ponder_finish_stop_count": count_record_finish_reason(item, "stop"),
                    "elapsed_s": safe_float(metrics.get("elapsed_s")),
                    "diff_ratio": safe_float(comp.get("diff_ratio")),
                    "answer_cosine": safe_float(semantic.get("answer_cosine")),
                    "query_alignment_delta": safe_float(semantic.get("query_alignment_delta")),
                    "stance_shift": safe_float(stance.get("shift_score")),
                    "spatial_log_density": safe_float(spatial_logs.get("density_per_1k_chars")),
                    "api_total_delta": safe_float(delta.get("api_total_tokens")),
                    "api_prompt_delta": safe_float(delta.get("api_prompt_tokens")),
                    "api_completion_delta": safe_float(delta.get("api_completion_tokens")),
                    "external_scaffold_tokens": safe_float(delta.get("external_scaffold_tokens_est")),
                    "all_api_calls": safe_float(all_usage.get("calls")),
                    "all_api_total_tokens": safe_float(all_usage.get("total_tokens")),
                    "warnings_count": len(warnings),
                }
            )

    return run_rows, item_rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(series: Iterable[Any]) -> float:
    vals = [float(x) for x in series if safe_float(x) is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def build_charts(run_df: pd.DataFrame, item_df: pd.DataFrame, chart_dir: Path) -> Dict[str, str]:
    use_chart_theme()
    charts: Dict[str, str] = {}
    provider_palette = palette_for(run_df.get("provider", pd.Series(dtype=str)).tolist())

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sns.barplot(data=run_df, x="provider", y="elapsed_s", hue="provider", palette=provider_palette, legend=False, ax=ax, edgecolor=TOKENS["ink"])
    sns.stripplot(data=run_df, x="provider", y="elapsed_s", ax=ax, color=COLOR_FAMILIES["orange"]["dark"], size=5, jitter=0.08)
    for patch in ax.patches:
        patch.set_edgecolor(TOKENS["ink"])
    ax.set_xlabel("")
    ax.set_ylabel("Seconds per lab-matrix run")
    add_chart_header(fig, ax, "Runtime by provider", "Selected query runs per provider; bars show mean elapsed seconds and dots show individual runs.")
    charts["runtime"] = str(chart_dir / "runtime_by_provider.png")
    save_fig(fig, Path(charts["runtime"]))

    condition_field = "condition"
    if "scaffold_condition" in item_df.columns:
        non_empty_conditions = item_df["scaffold_condition"].fillna("").astype(str).str.len() > 0
        if bool(non_empty_conditions.any()):
            condition_field = "scaffold_condition"
    preferred_order = ["assoc", "random", "facts", "isomorphic"]
    seen_conditions = unique_texts(item_df.get(condition_field, pd.Series(dtype=str)).tolist())
    condition_order = [x for x in preferred_order if x in seen_conditions] + [x for x in seen_conditions if x not in preferred_order]
    plot_df = item_df[item_df.get(condition_field, pd.Series(dtype=str)).fillna("").astype(str).str.len() > 0].copy()
    plot_df["condition_group"] = plot_df[condition_field].astype(str)

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    sns.barplot(
        data=plot_df,
        x="condition_group",
        y="query_alignment_delta",
        hue="provider",
        order=condition_order,
        palette=provider_palette,
        errorbar=None,
        ax=ax,
        edgecolor=TOKENS["ink"],
    )
    ax.axhline(0, color=TOKENS["ink"], linewidth=1.0, linestyle=":")
    ax.set_xlabel("")
    ax.set_ylabel("Mean query-alignment delta")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, borderaxespad=0)
    add_chart_header(fig, ax, "Alignment movement by scaffold condition", "Hashed char-ngram cosine delta vs baseline; read alongside answer length and the output atlas.")
    charts["alignment"] = str(chart_dir / "alignment_by_condition.png")
    save_fig(fig, Path(charts["alignment"]))

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    sns.barplot(
        data=plot_df,
        x="condition_group",
        y="api_total_delta",
        hue="provider",
        order=condition_order,
        palette=provider_palette,
        errorbar=None,
        ax=ax,
        edgecolor=TOKENS["ink"],
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("")
    ax.set_ylabel("Mean additional API total tokens, log scale")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, borderaxespad=0)
    add_chart_header(fig, ax, "Token overhead rises for generated scaffold conditions", "Average additional API total tokens vs baseline; Claude CLI includes its prompt/cache wrapper.")
    charts["token_overhead"] = str(chart_dir / "token_overhead_by_condition.png")
    save_fig(fig, Path(charts["token_overhead"]))

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    sns.barplot(
        data=plot_df,
        x="condition_group",
        y="answer_chars",
        hue="provider",
        order=condition_order,
        palette=provider_palette,
        errorbar=None,
        ax=ax,
        edgecolor=TOKENS["ink"],
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean pondered answer characters")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, borderaxespad=0)
    add_chart_header(fig, ax, "Answer length changes the comparison surface", "Length differences can dominate downstream metrics, especially for hash alignment and stance heuristics.")
    charts["answer_length"] = str(chart_dir / "answer_length_by_condition.png")
    save_fig(fig, Path(charts["answer_length"]))
    return charts


def char_ngrams(text: str, min_n: int = 2, max_n: int = 5) -> List[str]:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not normalized:
        return []
    grams: List[str] = []
    for n in range(min_n, max_n + 1):
        if len(normalized) < n:
            continue
        grams.extend(normalized[i : i + n] for i in range(0, len(normalized) - n + 1))
    return grams or [normalized]


def stable_bucket(text: str, dim: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def hashed_text_vector(text: str, dim: int = 768) -> np.ndarray:
    vec = np.zeros(dim, dtype=float)
    for gram in char_ngrams(text):
        vec[stable_bucket(gram, dim)] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec


def cosine_vectors(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size == 0 or b.size == 0:
        return None
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 0 or b_norm <= 0:
        return None
    return float(np.dot(a, b) / (a_norm * b_norm))


def cosine_texts(a: str, b: str) -> Optional[float]:
    return cosine_vectors(hashed_text_vector(a), hashed_text_vector(b))


def collect_answer_docs(run_df: pd.DataFrame) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for _, row in run_df.sort_values(["provider", "query_id"]).iterrows():
        result_path = Path(str(row["json_out"]))
        if not result_path.exists():
            continue
        result = read_json(result_path)
        for item in result.get("items", []):
            if not isinstance(item, dict):
                continue
            answer = str(item.get("answer") or "").strip()
            if not answer:
                continue
            name = str(item.get("name") or "")
            kind = str(item.get("kind") or "")
            output_contract = infer_output_contract(item)
            docs.append(
                {
                    "provider": str(row["provider"]),
                    "query_id": str(row["query_id"]),
                    "model": str(result.get("model") or row.get("model") or ""),
                    "condition": "baseline" if kind == "baseline" else name,
                    "output_contract": output_contract,
                    "kind": kind,
                    "answer_chars": len(answer),
                    "answer": answer,
                }
            )
    return docs


def build_answer_pca(run_df: pd.DataFrame, analysis_dir: Path, chart_dir: Path) -> tuple[Optional[str], Dict[str, Any]]:
    docs = collect_answer_docs(run_df)
    pca_csv = analysis_dir / "answer_pca.csv"
    if len(docs) < 3:
        write_csv(pca_csv, docs)
        return None, {"doc_count": len(docs), "explained_variance_ratio": []}

    dim = 768
    counts = np.zeros((len(docs), dim), dtype=float)
    doc_freq = np.zeros(dim, dtype=float)
    for row_ix, doc in enumerate(docs):
        seen = set()
        for gram in char_ngrams(str(doc.get("answer") or "")):
            bucket = stable_bucket(gram, dim)
            counts[row_ix, bucket] += 1.0
            seen.add(bucket)
        for bucket in seen:
            doc_freq[bucket] += 1.0

    tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    idf = np.log((len(docs) + 1.0) / (doc_freq + 1.0)) + 1.0
    matrix = tf * idf
    matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :2] * s[:2]
    variances = (s**2) / max(1, len(docs) - 1)
    total_variance = float(variances.sum()) if variances.size else 0.0
    explained = (variances[:2] / total_variance).tolist() if total_variance else [0.0, 0.0]

    pca_rows: List[Dict[str, Any]] = []
    for doc, score in zip(docs, scores):
        row = dict(doc)
        row["pc1"] = float(score[0])
        row["pc2"] = float(score[1])
        row["answer"] = str(row["answer"])[:1200]
        pca_rows.append(row)
    write_csv(pca_csv, pca_rows)

    pca_df = pd.DataFrame(pca_rows)
    provider_palette = {"gemini": COLOR_FAMILIES["blue"]["base"], "claude": COLOR_FAMILIES["gold"]["base"]}
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    sns.scatterplot(
        data=pca_df,
        x="pc1",
        y="pc2",
        hue="provider",
        style="condition",
        palette=provider_palette,
        s=82,
        edgecolor=TOKENS["ink"],
        linewidth=0.7,
        ax=ax,
    )
    ax.axhline(0, color=TOKENS["axis"], linewidth=0.9)
    ax.axvline(0, color=TOKENS["axis"], linewidth=0.9)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    add_chart_header(
        fig,
        ax,
        "Generated answer PCA",
        "Char n-gram TF-IDF projected to two dimensions; color encodes provider and marker style encodes condition.",
    )
    chart_path = chart_dir / "answer_pca_by_provider_condition.png"
    save_fig(fig, chart_path)

    centroid_distance = None
    centroids = pca_df.groupby("provider")[["pc1", "pc2"]].mean()
    if {"claude", "gemini"}.issubset(set(centroids.index)):
        delta = centroids.loc["claude"] - centroids.loc["gemini"]
        centroid_distance = float(np.linalg.norm(delta.to_numpy(dtype=float)))
    return str(chart_path), {
        "doc_count": len(docs),
        "feature_dim": dim,
        "explained_variance_ratio": [float(x) for x in explained],
        "provider_centroid_distance": centroid_distance,
        "csv": str(pca_csv),
    }


def classify_attractor(step_drift: Any, origin_drift: Any, recurrence: Any, query_relevance: Any) -> str:
    step = safe_float(step_drift)
    origin = safe_float(origin_drift)
    recur = safe_float(recurrence)
    relevance = safe_float(query_relevance)
    if relevance is not None and relevance < 0.04 and origin is not None and origin > 0.55:
        return "off_track"
    if recur is not None and recur >= 0.88 and step is not None and step <= 0.12:
        return "loop_or_convergence"
    if origin is not None and origin >= 0.45 and (recur is None or recur < 0.72):
        return "divergence"
    if recur is not None and recur >= 0.78 and origin is not None and origin >= 0.25:
        return "recurring_shift"
    if origin is not None and origin <= 0.18 and step is not None and step <= 0.18:
        return "convergence"
    return "mixed"


def build_attractor_rows(run_df: pd.DataFrame, item_df: pd.DataFrame) -> List[Dict[str, Any]]:
    metrics_by_item: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for _, row in item_df.iterrows():
        metrics_by_item[(str(row["provider"]), str(row["query_id"]), str(row["condition"]))] = dict(row)

    rows: List[Dict[str, Any]] = []
    for _, run in run_df.sort_values(["provider", "query_id"]).iterrows():
        provider = str(run["provider"])
        query_id = str(run["query_id"])
        query_text = str(run.get("query") or "")
        result_path = Path(str(run.get("json_out") or ""))
        if not result_path.exists():
            continue
        result = read_json(result_path)
        trace_summary = load_trace_summary(Path(str(run.get("trace_out") or "")))
        items = [it for it in result.get("items", []) if isinstance(it, dict)]
        baseline = next((it for it in items if str(it.get("kind") or "") == "baseline"), None)
        baseline_answer = str((baseline or {}).get("answer") or "")
        baseline_vec = hashed_text_vector(baseline_answer)
        query_vec = hashed_text_vector(query_text)

        previous: List[tuple[str, np.ndarray]] = []
        if baseline_answer.strip():
            previous.append(("baseline", baseline_vec))

        for item in items:
            if str(item.get("kind") or "") != "ponder":
                continue
            name = str(item.get("name") or "")
            answer = str(item.get("answer") or "")
            answer_vec = hashed_text_vector(answer)
            metric_row = metrics_by_item.get((provider, query_id, name), {})
            scaffold_condition, scaffold_dose = infer_scaffold_meta(item)
            output_contract = str(metric_row.get("output_contract") or infer_output_contract(item) or "none")
            trace = trace_summary.get(name, {})
            scaffold_text = "\n\n".join(
                unique_texts(list(trace.get("scaffold_previews") or []) + record_texts(item))
            )

            origin_similarity = cosine_vectors(answer_vec, baseline_vec)
            origin_drift = None if origin_similarity is None else 1.0 - origin_similarity
            query_relevance = cosine_vectors(answer_vec, query_vec)
            scaffold_leakage = cosine_texts(answer, scaffold_text) if scaffold_text.strip() else None

            prior_name = ""
            step_drift = None
            recurrence = None
            if previous:
                prior_name, prior_vec = previous[-1]
                step_similarity = cosine_vectors(answer_vec, prior_vec)
                step_drift = None if step_similarity is None else 1.0 - step_similarity
                recurrence_values = [cosine_vectors(answer_vec, vec) for _, vec in previous]
                recurrence_values = [x for x in recurrence_values if x is not None]
                recurrence = max(recurrence_values) if recurrence_values else None

            alignment_delta = safe_float(metric_row.get("query_alignment_delta"))
            coil_index = None
            if recurrence is not None and query_relevance is not None and alignment_delta is not None:
                coil_index = max(0.0, recurrence) * max(0.0, query_relevance) * max(0.0, alignment_delta)

            rows.append(
                {
                    "provider": provider,
                    "query_id": query_id,
                    "model": str(result.get("model") or run.get("model") or ""),
                    "condition": name,
                    "scaffold_condition": scaffold_condition,
                    "scaffold_dose": scaffold_dose,
                    "output_contract": output_contract,
                    "prior_item": prior_name,
                    "step_drift": step_drift,
                    "origin_drift": origin_drift,
                    "recurrence": recurrence,
                    "query_relevance": query_relevance,
                    "scaffold_leakage": scaffold_leakage,
                    "coil_index": coil_index,
                    "attractor_class": classify_attractor(step_drift, origin_drift, recurrence, query_relevance),
                    "query_alignment_delta": alignment_delta,
                    "answer_chars": safe_float(metric_row.get("answer_chars")),
                }
            )
            previous.append((name, answer_vec))
    return rows


def build_dose_response_rows(run_df: pd.DataFrame, item_df: pd.DataFrame, attractor_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    attractor_by_item = {
        (str(row.get("provider")), str(row.get("query_id")), str(row.get("condition"))): row for row in attractor_rows
    }
    rows: List[Dict[str, Any]] = []
    if item_df.empty or "scaffold_dose" not in item_df.columns:
        return rows

    for _, item in item_df.iterrows():
        dose = safe_int(item.get("scaffold_dose"))
        condition = str(item.get("scaffold_condition") or "").strip()
        if dose is None or dose <= 0 or not condition:
            continue
        output_contract = str(item.get("output_contract") or "none").strip() or "none"
        key = (str(item.get("provider")), str(item.get("query_id")), str(item.get("condition")))
        attractor = attractor_by_item.get(key, {})
        rows.append(
            {
                "provider": str(item.get("provider")),
                "query_id": str(item.get("query_id")),
                "model": str(item.get("model") or ""),
                "condition": str(item.get("condition") or ""),
                "scaffold_condition": condition,
                "scaffold_dose": dose,
                "output_contract": output_contract,
                "answer_chars": safe_float(item.get("answer_chars")),
                "query_alignment_delta": safe_float(item.get("query_alignment_delta")),
                "diff_ratio": safe_float(item.get("diff_ratio")),
                "stance_shift": safe_float(item.get("stance_shift")),
                "spatial_log_density": safe_float(item.get("spatial_log_density")),
                "api_total_delta": safe_float(item.get("api_total_delta")),
                "origin_drift": safe_float(attractor.get("origin_drift")),
                "recurrence": safe_float(attractor.get("recurrence")),
                "query_relevance": safe_float(attractor.get("query_relevance")),
                "scaffold_leakage": safe_float(attractor.get("scaffold_leakage")),
                "coil_index": safe_float(attractor.get("coil_index")),
                "attractor_class": str(attractor.get("attractor_class") or ""),
                "is_baseline_reference": False,
            }
        )

    present = {(row["provider"], row["query_id"], row["scaffold_condition"], row["output_contract"]) for row in rows}
    run_lookup = {(str(row["provider"]), str(row["query_id"])): dict(row) for _, row in run_df.iterrows()}
    for provider, query_id, condition, output_contract in sorted(present):
        run = run_lookup.get((provider, query_id), {})
        rows.append(
            {
                "provider": provider,
                "query_id": query_id,
                "model": str(run.get("model") or ""),
                "condition": "baseline",
                "scaffold_condition": condition,
                "scaffold_dose": 0,
                "output_contract": output_contract,
                "answer_chars": safe_float(run.get("baseline_answer_chars")),
                "query_alignment_delta": 0.0,
                "diff_ratio": 0.0,
                "stance_shift": 0.0,
                "spatial_log_density": None,
                "api_total_delta": 0.0,
                "origin_drift": 0.0,
                "recurrence": 1.0,
                "query_relevance": None,
                "scaffold_leakage": None,
                "coil_index": 0.0,
                "attractor_class": "baseline_reference",
                "is_baseline_reference": True,
            }
        )
    return sorted(
        rows,
        key=lambda r: (
            str(r["provider"]),
            str(r["query_id"]),
            str(r["scaffold_condition"]),
            str(r.get("output_contract") or "none"),
            int(r["scaffold_dose"]),
        ),
    )


def build_dose_charts(dose_df: pd.DataFrame, chart_dir: Path) -> Dict[str, str]:
    charts: Dict[str, str] = {}
    if dose_df.empty or "scaffold_dose" not in dose_df.columns:
        return charts
    plot_df = dose_df[dose_df["scaffold_dose"].notna()].copy()
    if plot_df.empty or float(plot_df["scaffold_dose"].max()) <= 0:
        return charts

    condition_order = unique_texts(plot_df["scaffold_condition"].tolist())
    condition_palette = palette_for(condition_order)
    style_field = "provider" if len(unique_texts(plot_df["provider"].tolist())) > 1 else None
    if style_field is None and "output_contract" in plot_df.columns:
        contracts = unique_texts(plot_df["output_contract"].tolist())
        if len(contracts) > 1:
            style_field = "output_contract"

    def line_chart(metric: str, ylabel: str, title: str, subtitle: str, filename: str) -> None:
        if metric not in plot_df.columns or plot_df[metric].dropna().empty:
            return
        fig, ax = plt.subplots(figsize=(9.6, 4.8))
        sns.lineplot(
            data=plot_df,
            x="scaffold_dose",
            y=metric,
            hue="scaffold_condition",
            style=style_field,
            hue_order=condition_order,
            palette=condition_palette,
            marker="o",
            estimator="mean",
            errorbar=None,
            ax=ax,
        )
        if metric == "query_alignment_delta":
            ax.axhline(0, color=TOKENS["ink"], linewidth=1.0, linestyle=":")
        ax.set_xlabel("Injected scaffold target tokens")
        ax.set_ylabel(ylabel)
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
        add_chart_header(fig, ax, title, subtitle)
        charts[metric] = str(chart_dir / filename)
        save_fig(fig, Path(charts[metric]))

    line_chart(
        "query_alignment_delta",
        "Mean query-alignment delta",
        "Scaffold dose changes query alignment nonlinearly",
        "Dose 0 repeats the shared baseline; mid-dose peaks are the first sign of a treatment-like effect.",
        "dose_query_alignment_delta.png",
    )
    line_chart(
        "origin_drift",
        "Origin drift from baseline",
        "Higher scaffold doses pull answers farther from the baseline surface",
        "Origin drift is 1 - cosine(answer, baseline) over hashed character n-gram vectors.",
        "dose_origin_drift.png",
    )
    line_chart(
        "scaffold_leakage",
        "Answer-to-scaffold similarity",
        "Scaffold leakage proxy rises when the injected path dominates the final answer",
        "This uses scaffold previews and retained ponder records as a surface proxy, not hidden-state evidence.",
        "dose_scaffold_leakage.png",
    )
    line_chart(
        "coil_index",
        "Coil index proxy",
        "Coil index highlights recurrent, query-relevant alignment gain",
        "Computed as recurrence x query relevance x positive alignment delta, so zero values are meaningful.",
        "dose_coil_index.png",
    )
    return charts


def build_closure_contract_rows(item_df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if item_df.empty or "output_contract" not in item_df.columns:
        return rows
    for _, item in item_df.iterrows():
        contract = str(item.get("output_contract") or "none").strip() or "none"
        rows.append(
            {
                "provider": str(item.get("provider")),
                "query_id": str(item.get("query_id")),
                "model": str(item.get("model") or ""),
                "condition": str(item.get("condition") or ""),
                "scaffold_condition": str(item.get("scaffold_condition") or ""),
                "scaffold_dose": safe_int(item.get("scaffold_dose")),
                "output_contract": contract,
                "log_phase_route": str(item.get("log_phase_route") or ""),
                "log_phase_reasoning_effort": str(item.get("log_phase_reasoning_effort") or "inherit"),
                "log_phase_max_new_tokens": safe_int(item.get("log_phase_max_new_tokens")) or 0,
                "log_phase_rescue_enabled": safe_float(item.get("log_phase_rescue_enabled")),
                "log_phase_rescue_used_rate": safe_float(item.get("log_phase_rescue_used_rate")),
                "final_phase_reasoning_effort": str(item.get("final_phase_reasoning_effort") or "inherit"),
                "final_phase_max_new_tokens": safe_int(item.get("final_phase_max_new_tokens")) or 0,
                "answer_chars": safe_float(item.get("answer_chars")),
                "final_marker_line_present": safe_float(item.get("final_marker_line_present")),
                "final_marker_is_last_line": safe_float(item.get("final_marker_is_last_line")),
                "ponder_log_marker_rate": safe_float(item.get("ponder_log_marker_rate")),
                "ponder_log_skeleton_prefix_rate": safe_float(item.get("ponder_log_skeleton_prefix_rate")),
                "ponder_log_skeleton_complete_rate": safe_float(item.get("ponder_log_skeleton_complete_rate")),
                "final_finish_is_length": safe_float(item.get("final_finish_is_length")),
                "ponder_finish_length_count": safe_float(item.get("ponder_finish_length_count")),
                "prompt_leakage_flag": safe_float(item.get("prompt_leakage_flag")),
                "final_finish_reason": str(item.get("final_finish_reason") or ""),
            }
        )
    return rows


def build_closure_charts(closure_df: pd.DataFrame, chart_dir: Path) -> Dict[str, str]:
    charts: Dict[str, str] = {}
    if closure_df.empty or "output_contract" not in closure_df.columns:
        return charts
    contracts = unique_texts(closure_df["output_contract"].tolist())
    if not contracts or set(contracts) == {"none"}:
        return charts

    order = [x for x in CONTRACT_ORDER if x in contracts] + [x for x in contracts if x not in CONTRACT_ORDER]
    palette = palette_for(order)

    def bar_chart(metric: str, ylabel: str, title: str, subtitle: str, filename: str) -> None:
        if metric not in closure_df.columns or closure_df[metric].dropna().empty:
            return
        fig, ax = plt.subplots(figsize=(9.6, 4.8))
        sns.barplot(
            data=closure_df,
            x="output_contract",
            y=metric,
            hue="output_contract",
            order=order,
            palette=palette,
            legend=False,
            errorbar=None,
            ax=ax,
            edgecolor=TOKENS["ink"],
        )
        sns.stripplot(
            data=closure_df,
            x="output_contract",
            y=metric,
            order=order,
            ax=ax,
            color=COLOR_FAMILIES["orange"]["dark"],
            size=4,
            jitter=0.08,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        add_chart_header(fig, ax, title, subtitle)
        charts[metric] = str(chart_dir / filename)
        save_fig(fig, Path(charts[metric]))

    bar_chart(
        "final_finish_is_length",
        "Final answer length-stop rate",
        "Closure contracts expose whether Gemini still stops by length",
        "Bars show mean 0/1 final-answer length finish; dots are individual scaffold rows.",
        "closure_final_finish_length_rate.png",
    )
    bar_chart(
        "final_marker_is_last_line",
        "END_ANSWER as final line",
        "Final closure success is visible in the answer text",
        "Rows without a final-answer contract are expected to stay near zero.",
        "closure_final_marker_rate.png",
    )
    bar_chart(
        "ponder_log_marker_rate",
        "Ponder logs with END_LOG",
        "Log closure success is measured before scaffold synthesis",
        "Rows without a log contract are expected to stay near zero.",
        "closure_log_marker_rate.png",
    )
    bar_chart(
        "ponder_log_skeleton_complete_rate",
        "Ponder logs matching X1-X4 skeleton",
        "Skeleton closure tests whether a non-semantic frame survives generation",
        "A pass requires exactly X1| through X4| followed by END_LOG.",
        "closure_log_skeleton_rate.png",
    )
    bar_chart(
        "answer_chars",
        "Final answer characters",
        "Closure contracts also control the output-length surface",
        "Use this with marker and finish-rate charts to separate shorter-but-complete from truncated.",
        "closure_answer_chars.png",
    )
    return charts


def build_log_phase_route_charts(closure_df: pd.DataFrame, chart_dir: Path) -> Dict[str, str]:
    charts: Dict[str, str] = {}
    if closure_df.empty or "log_phase_route" not in closure_df.columns:
        return charts
    plot_df = closure_df[closure_df["log_phase_route"].fillna("").astype(str).str.len() > 0].copy()
    if plot_df.empty:
        return charts
    routes = unique_texts(plot_df["log_phase_route"].tolist())
    if len(routes) <= 1:
        return charts
    order = [x for x in LOG_PHASE_ROUTE_ORDER if x in routes] + [x for x in routes if x not in LOG_PHASE_ROUTE_ORDER]
    palette = palette_for(order)

    def bar_chart(metric: str, ylabel: str, title: str, subtitle: str, filename: str) -> None:
        if metric not in plot_df.columns or plot_df[metric].dropna().empty:
            return
        fig, ax = plt.subplots(figsize=(9.8, 4.8))
        sns.barplot(
            data=plot_df,
            x="log_phase_route",
            y=metric,
            hue="log_phase_route",
            order=order,
            palette=palette,
            legend=False,
            errorbar=None,
            ax=ax,
            edgecolor=TOKENS["ink"],
        )
        sns.stripplot(
            data=plot_df,
            x="log_phase_route",
            y=metric,
            order=order,
            ax=ax,
            color=COLOR_FAMILIES["orange"]["dark"],
            size=4,
            jitter=0.08,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=18)
        add_chart_header(fig, ax, title, subtitle)
        chart_key = f"log_phase_route_{metric}"
        charts[chart_key] = str(chart_dir / filename)
        save_fig(fig, Path(charts[chart_key]))

    bar_chart(
        "ponder_log_marker_rate",
        "Ponder logs with END_LOG",
        "Log-only routes change visible closure before the final answer",
        "Bars show mean END_LOG presence; dots are scaffold rows under the same log contract.",
        "log_phase_route_marker_rate.png",
    )
    bar_chart(
        "ponder_log_skeleton_complete_rate",
        "Ponder logs matching X1-X4 skeleton",
        "Larger log budget plus rescue targets skeleton completion",
        "A pass requires exactly X1| through X4| followed by END_LOG.",
        "log_phase_route_skeleton_rate.png",
    )
    bar_chart(
        "ponder_finish_length_count",
        "Length finishes per item",
        "Log route changes whether Gemini still runs out of visible budget",
        "Lower is better for this diagnostic because each item has one ponder stage in the focused sweep.",
        "log_phase_route_length_finishes.png",
    )
    bar_chart(
        "log_phase_rescue_used_rate",
        "Rescue used rate",
        "Rescue is visible only when the first log generation misses the contract",
        "Rows without rescue enabled should remain at zero by construction.",
        "log_phase_route_rescue_used_rate.png",
    )
    return charts


def summarize(
    run_df: pd.DataFrame,
    item_df: pd.DataFrame,
    dose_df: Optional[pd.DataFrame] = None,
    closure_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    by_provider: Dict[str, Any] = {}
    for provider, group in item_df.groupby("provider"):
        by_provider[provider] = {
            "ponder_items": int(len(group)),
            "answer_chars_mean": mean(group["answer_chars"]),
            "diff_ratio_mean": mean(group["diff_ratio"]),
            "query_alignment_delta_mean": mean(group["query_alignment_delta"]),
            "stance_shift_mean": mean(group["stance_shift"]),
            "spatial_log_density_mean": mean(group["spatial_log_density"]),
            "api_total_delta_mean": mean(group["api_total_delta"]),
            "external_scaffold_tokens_mean": mean(group["external_scaffold_tokens"]),
            "all_api_total_tokens_mean": mean(group["all_api_total_tokens"]),
        }
    run_summary = {
        provider: {
            "runs": int(len(group)),
            "elapsed_s_mean": mean(group["elapsed_s"]),
            "elapsed_s_min": float(group["elapsed_s"].min()),
            "elapsed_s_max": float(group["elapsed_s"].max()),
        }
        for provider, group in run_df.groupby("provider")
    }
    condition_summary = (
        item_df.groupby(["provider", "condition"], as_index=False)
        .agg(
            answer_chars_mean=("answer_chars", "mean"),
            query_alignment_delta_mean=("query_alignment_delta", "mean"),
            api_total_delta_mean=("api_total_delta", "mean"),
            stance_shift_mean=("stance_shift", "mean"),
            diff_ratio_mean=("diff_ratio", "mean"),
        )
        .to_dict(orient="records")
    )
    dose_summary: Dict[str, Any] = {}
    if dose_df is not None and not dose_df.empty:
        non_baseline = dose_df[dose_df.get("is_baseline_reference", pd.Series(dtype=bool)) != True].copy()
        if not non_baseline.empty:
            best_alignment = non_baseline.sort_values("query_alignment_delta", ascending=False).head(1).to_dict(orient="records")
            highest_drift = non_baseline.sort_values("origin_drift", ascending=False).head(1).to_dict(orient="records")
            dose_summary = {
                "row_count": int(len(dose_df)),
                "max_dose": safe_float(dose_df["scaffold_dose"].max()),
                "conditions": unique_texts(dose_df["scaffold_condition"].tolist()),
                "best_alignment_row": best_alignment[0] if best_alignment else {},
                "highest_origin_drift_row": highest_drift[0] if highest_drift else {},
                "by_condition": (
                    non_baseline.groupby("scaffold_condition", as_index=False)
                    .agg(
                        query_alignment_delta_mean=("query_alignment_delta", "mean"),
                        origin_drift_mean=("origin_drift", "mean"),
                        recurrence_mean=("recurrence", "mean"),
                        scaffold_leakage_mean=("scaffold_leakage", "mean"),
                        coil_index_mean=("coil_index", "mean"),
                    )
                    .to_dict(orient="records")
                ),
            }
    closure_summary: Dict[str, Any] = {}
    if closure_df is not None and not closure_df.empty:
        contracts = unique_texts(closure_df.get("output_contract", pd.Series(dtype=str)).tolist())
        if contracts and set(contracts) != {"none"}:
            by_contract = (
                closure_df.groupby("output_contract", as_index=False)
                .agg(
                    rows=("condition", "count"),
                    answer_chars_mean=("answer_chars", "mean"),
                    final_finish_length_rate=("final_finish_is_length", "mean"),
                    final_marker_last_rate=("final_marker_is_last_line", "mean"),
                    log_marker_rate=("ponder_log_marker_rate", "mean"),
                    log_skeleton_complete_rate=("ponder_log_skeleton_complete_rate", "mean"),
                    prompt_leakage_rate=("prompt_leakage_flag", "mean"),
                )
                .to_dict(orient="records")
            )
            closure_summary = {
                "row_count": int(len(closure_df)),
                "contracts": contracts,
                "by_contract": by_contract,
            }
        route_df = closure_df[closure_df.get("log_phase_route", pd.Series(dtype=str)).fillna("").astype(str).str.len() > 0].copy()
        if not route_df.empty:
            closure_summary["log_phase_routes"] = unique_texts(route_df["log_phase_route"].tolist())
            closure_summary["by_log_phase_route"] = (
                route_df.groupby("log_phase_route", as_index=False)
                .agg(
                    rows=("condition", "count"),
                    log_marker_rate=("ponder_log_marker_rate", "mean"),
                    log_skeleton_complete_rate=("ponder_log_skeleton_complete_rate", "mean"),
                    ponder_finish_length_mean=("ponder_finish_length_count", "mean"),
                    final_finish_length_rate=("final_finish_is_length", "mean"),
                    rescue_used_rate=("log_phase_rescue_used_rate", "mean"),
                    answer_chars_mean=("answer_chars", "mean"),
                )
                .to_dict(orient="records")
            )
    return {
        "run_count": int(len(run_df)),
        "ponder_item_count": int(len(item_df)),
        "by_provider": by_provider,
        "run_summary": run_summary,
        "condition_summary": condition_summary,
        "dose_summary": dose_summary,
        "closure_summary": closure_summary,
    }


def fmt_num(value: Any, digits: int = 1) -> str:
    f = safe_float(value)
    if f is None or math.isnan(f):
        return "n/a"
    return f"{f:,.{digits}f}"


def img_tag(path: str, alt: str, report_dir: Path) -> str:
    rel = Path(path).resolve().relative_to(report_dir.resolve())
    return f'<img src="{html.escape(str(rel))}" alt="{html.escape(alt)}" loading="lazy">'


def unique_texts(values: Iterable[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def load_trace_summary(trace_path: Path) -> Dict[str, Dict[str, Any]]:
    by_item: Dict[str, Dict[str, Any]] = {}
    if not trace_path.exists():
        return by_item
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = str(event.get("pack_item") or event.get("item") or "").strip()
        if not item:
            continue
        slot = by_item.setdefault(
            item,
            {
                "keywords": [],
                "ponder_previews": [],
                "scaffold_previews": [],
                "answer_preview": "",
                "baseline_preview": "",
                "api_finish_reasons": [],
            },
        )
        event_name = str(event.get("event") or "")
        api_meta = event.get("api_meta") if isinstance(event.get("api_meta"), dict) else {}
        finish_reason = api_meta.get("finish_reason")
        if finish_reason:
            slot["api_finish_reasons"].append(str(finish_reason))
        if event_name == "seed_keywords":
            keywords = event.get("keywords")
            if isinstance(keywords, list):
                slot["keywords"].extend(str(k) for k in keywords)
        elif event_name == "ponder_stage":
            slot["ponder_previews"].append(str(event.get("text_preview") or ""))
        elif event_name == "scaffold_conditioned":
            slot["scaffold_previews"].append(str(event.get("text_preview") or ""))
        elif event_name == "answer_done":
            slot["answer_preview"] = str(event.get("answer_preview") or "")
        elif event_name == "baseline_answer":
            slot["baseline_preview"] = str(event.get("answer_preview") or "")

    for slot in by_item.values():
        slot["keywords"] = unique_texts(slot.get("keywords", []))
        slot["ponder_previews"] = unique_texts(slot.get("ponder_previews", []))
        slot["scaffold_previews"] = unique_texts(slot.get("scaffold_previews", []))
        slot["api_finish_reasons"] = unique_texts(slot.get("api_finish_reasons", []))
    return by_item


def rel_link(path_value: Any, report_dir: Path, label: str) -> str:
    text = str(path_value or "").strip()
    if not text or text == "nan":
        return html.escape(label)
    return f'<a href="{html.escape(os.path.relpath(text, str(report_dir)))}">{html.escape(label)}</a>'


def render_pre(text: Any, *, empty: str = "(empty output)") -> str:
    body = str(text or "").strip()
    if not body:
        body = empty
    body = "\n".join(line.rstrip() for line in body.splitlines())
    return f'<pre class="output-text">{html.escape(body)}</pre>'


def render_trace_detail(label: str, values: Iterable[Any]) -> str:
    rendered = []
    for ix, value in enumerate(unique_texts(values), start=1):
        rendered.append(f"<h5>{html.escape(label)} {ix}</h5>{render_pre(value, empty='(no preview captured)')}")
    return "".join(rendered)


def record_texts(item: Dict[str, Any]) -> List[str]:
    records = item.get("records")
    if not isinstance(records, list):
        return []
    texts: List[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        pieces = []
        q = str(record.get("ponder_question") or "").strip()
        log = str(record.get("ponder_log") or "").strip()
        if q:
            pieces.append(f"ponder_q: {q}")
        if log:
            pieces.append(f"ponder_log:\n{log}")
        if pieces:
            texts.append("\n\n".join(pieces))
    return texts


def build_output_atlas_html(report_dir: Path, run_df: pd.DataFrame, item_df: pd.DataFrame) -> str:
    metrics: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for _, row in item_df.iterrows():
        metrics[(str(row["provider"]), str(row["query_id"]), str(row["condition"]))] = dict(row)

    parts: List[str] = []
    for query_id, query_group in run_df.sort_values(["query_id", "provider"]).groupby("query_id", sort=True):
        query_text = str(query_group.iloc[0].get("query") or "").strip()
        parts.append(
            '<div class="atlas-query">'
            f"<h3>{html.escape(str(query_id))}</h3>"
            f"<p class=\"query-text\">{html.escape(query_text)}</p>"
        )
        for _, row in query_group.sort_values("provider").iterrows():
            provider = str(row["provider"])
            result_path = Path(str(row["json_out"]))
            trace_summary = load_trace_summary(Path(str(row.get("trace_out") or "")))
            result = read_json(result_path)
            items = [it for it in result.get("items", []) if isinstance(it, dict)]
            model = str(result.get("model") or row.get("model") or "")
            link_line = " / ".join(
                [
                    rel_link(row.get("json_out"), report_dir, "json"),
                    rel_link(row.get("matrix_report_out"), report_dir, "matrix"),
                    rel_link(row.get("trace_report_out"), report_dir, "trace"),
                    rel_link(row.get("stdout"), report_dir, "stdout"),
                    rel_link(row.get("stderr"), report_dir, "stderr"),
                ]
            )
            parts.append(
                '<details class="output-run">'
                f"<summary><strong>{html.escape(provider)}</strong> "
                f"<span>{html.escape(model)}</span> "
                f"<em>{fmt_num(row.get('elapsed_s'), 1)}s</em></summary>"
                f'<p class="artifact-links">{link_line}</p>'
            )
            for item in items:
                name = str(item.get("name") or "")
                kind = str(item.get("kind") or "")
                trace = trace_summary.get(name, {})
                answer = item.get("answer") or ""
                condition_metrics = metrics.get((provider, str(query_id), name), {})
                if kind == "baseline":
                    meta = f"baseline | answer chars {len(str(answer))}"
                else:
                    dose_label = ""
                    if condition_metrics.get("scaffold_dose") is not None:
                        dose_label = (
                            f" | scaffold {html.escape(str(condition_metrics.get('scaffold_condition') or ''))}"
                            f"/{fmt_num(condition_metrics.get('scaffold_dose'), 0)}"
                        )
                    contract_label = ""
                    if condition_metrics.get("output_contract") and str(condition_metrics.get("output_contract")) != "none":
                        contract_label = f" | contract {html.escape(str(condition_metrics.get('output_contract')))}"
                    route_label = ""
                    if condition_metrics.get("log_phase_route"):
                        route_label = (
                            f" | log route {html.escape(str(condition_metrics.get('log_phase_route')))}"
                            f" rescue {fmt_num(condition_metrics.get('log_phase_rescue_used_rate'), 2)}"
                        )
                    meta = (
                        f"{html.escape(str(item.get('control') or name))} | "
                        f"answer chars {fmt_num(condition_metrics.get('answer_chars'), 0)} | "
                        f"diff ratio {fmt_num(condition_metrics.get('diff_ratio'), 3)} | "
                        f"alignment delta {fmt_num(condition_metrics.get('query_alignment_delta'), 3)} | "
                        f"api total delta {fmt_num(condition_metrics.get('api_total_delta'), 0)}"
                        f"{dose_label}"
                        f"{contract_label}"
                        f"{route_label}"
                    )
                finish_reasons = ", ".join(trace.get("api_finish_reasons") or [])
                keyword_html = ", ".join(html.escape(k) for k in trace.get("keywords", [])) or "(none captured)"
                parts.append(
                    '<details class="output-item">'
                    f"<summary><strong>{html.escape(name)}</strong> <span>{meta}</span></summary>"
                    f'<p class="finish-reasons">finish reasons: {html.escape(finish_reasons or "n/a")}</p>'
                    f'<p class="keyword-list">keywords: {keyword_html}</p>'
                )
                record_values = record_texts(item)
                if record_values:
                    parts.append(render_trace_detail("Full ponder record", record_values))
                else:
                    parts.append(render_trace_detail("Ponder preview", trace.get("ponder_previews", [])))
                parts.append(render_trace_detail("Scaffold preview", trace.get("scaffold_previews", [])))
                preview = trace.get("baseline_preview") or trace.get("answer_preview")
                if preview:
                    parts.append(f"<h5>Trace answer preview</h5>{render_pre(preview)}")
                parts.append(f"<h5>Final answer from result JSON</h5>{render_pre(answer)}")
                parts.append("</details>")
            parts.append("</details>")
        parts.append("</div>")

    return "".join(parts)


def build_report_html(
    report_dir: Path,
    charts: Dict[str, str],
    summary: Dict[str, Any],
    run_df: pd.DataFrame,
    output_atlas_html: str,
) -> str:
    gemini = summary["by_provider"].get("gemini", {})
    claude = summary["by_provider"].get("claude", {})
    gemini_run = summary["run_summary"].get("gemini", {})
    claude_run = summary["run_summary"].get("claude", {})
    providers = unique_texts(run_df.get("provider", pd.Series(dtype=str)).tolist())
    elapsed_ratio = safe_float(claude_run.get("elapsed_s_mean")) or 0
    if safe_float(gemini_run.get("elapsed_s_mean")):
        elapsed_ratio = elapsed_ratio / float(gemini_run["elapsed_s_mean"])
    model_names = ", ".join(unique_texts(run_df.get("model", pd.Series(dtype=str)).tolist()))
    if {"claude", "gemini"}.issubset(set(providers)):
        provider_runtime_bullet = (
            f"<strong>Claude was much slower but produced fuller answers.</strong> Mean runtime was "
            f"{fmt_num(claude_run.get('elapsed_s_mean'), 1)}s for Claude versus "
            f"{fmt_num(gemini_run.get('elapsed_s_mean'), 1)}s for Gemini, roughly "
            f"{fmt_num(elapsed_ratio, 1)}x slower; Claude answers averaged "
            f"{fmt_num(claude.get('answer_chars_mean'), 0)} chars versus Gemini's "
            f"{fmt_num(gemini.get('answer_chars_mean'), 0)}."
        )
        alignment_bullet = (
            f"<strong>Alignment lift is a triage signal, not a verdict.</strong> Gemini's mean "
            f"query-alignment delta was {fmt_num(gemini.get('query_alignment_delta_mean'), 3)} "
            f"versus Claude's {fmt_num(claude.get('query_alignment_delta_mean'), 3)}; read that "
            f"alongside answer length, PCA position, and the raw output atlas."
        )
    elif providers:
        provider = providers[0]
        provider_stats = summary["by_provider"].get(provider, {})
        provider_run = summary["run_summary"].get(provider, {})
        provider_runtime_bullet = (
            f"<strong>{html.escape(provider)} ran as the focused comparison lane.</strong> Mean runtime was "
            f"{fmt_num(provider_run.get('elapsed_s_mean'), 1)}s and pondered answers averaged "
            f"{fmt_num(provider_stats.get('answer_chars_mean'), 0)} chars."
        )
        alignment_bullet = (
            f"<strong>Alignment lift is a triage signal, not a verdict.</strong> Mean query-alignment "
            f"delta for {html.escape(provider)} was {fmt_num(provider_stats.get('query_alignment_delta_mean'), 3)}; "
            f"read that alongside answer length, PCA position, and the raw output atlas."
        )
    else:
        provider_runtime_bullet = "<strong>No provider rows were detected.</strong> Inspect the manifest and raw run logs."
        alignment_bullet = "<strong>Alignment lift is unavailable.</strong> No pondered item rows were detected."

    dose_summary = summary.get("dose_summary") if isinstance(summary.get("dose_summary"), dict) else {}
    closure_summary = summary.get("closure_summary") if isinstance(summary.get("closure_summary"), dict) else {}
    log_phase_rows = closure_summary.get("by_log_phase_route") if isinstance(closure_summary.get("by_log_phase_route"), list) else []
    profile_name = str(summary.get("profile") or "").strip()
    dose_conditions = dose_summary.get("conditions") if isinstance(dose_summary.get("conditions"), list) else []
    has_evo_scaffolds = "evo" in profile_name or any(str(x).startswith("evo_") for x in dose_conditions)
    if dose_summary:
        best = dose_summary.get("best_alignment_row") if isinstance(dose_summary.get("best_alignment_row"), dict) else {}
        if has_evo_scaffolds:
            dose_bullet = (
                f"<li><strong>Evolutionary scaffold conditions are now isolated under the route lock.</strong> "
                f"The strongest alignment row was <code>{html.escape(str(best.get('scaffold_condition') or 'n/a'))}</code> "
                f"at dose {fmt_num(best.get('scaffold_dose'), 0)} with alignment delta "
                f"{fmt_num(best.get('query_alignment_delta'), 3)}; compare it against origin drift, leakage, and the raw text.</li>"
            )
        else:
            dose_bullet = (
                f"<li><strong>Dose ladder metrics are now part of the report.</strong> The strongest alignment row was "
                f"<code>{html.escape(str(best.get('scaffold_condition') or 'n/a'))}</code> at dose "
                f"{fmt_num(best.get('scaffold_dose'), 0)} with alignment delta "
                f"{fmt_num(best.get('query_alignment_delta'), 3)}; verify it against origin drift, leakage, and the raw text.</li>"
            )
    else:
        dose_bullet = (
            "<li><strong>Generated scaffold conditions are the expensive branch.</strong> "
            "<code>facts</code> and <code>isomorphic</code> raise token overhead most strongly; "
            "read token cost as operational overhead rather than a pure cognition claim.</li>"
        )
    if closure_summary:
        closure_rows = closure_summary.get("by_contract") if isinstance(closure_summary.get("by_contract"), list) else []
        best_marker = sorted(
            closure_rows,
            key=lambda row: (
                safe_float(row.get("final_marker_last_rate")) or 0.0,
                -(safe_float(row.get("final_finish_length_rate")) or 0.0),
            ),
            reverse=True,
        )
        best_contract = best_marker[0] if best_marker else {}
        closure_bullet = (
            f"<li><strong>Output closure is now a first-class experimental axis.</strong> "
            f"Best final marker rate in this run was <code>{html.escape(str(best_contract.get('output_contract') or 'n/a'))}</code> "
            f"at {fmt_num((safe_float(best_contract.get('final_marker_last_rate')) or 0.0) * 100, 0)}%; "
            "read it with the finish-reason chart and raw output atlas.</li>"
        )
    else:
        closure_bullet = ""
    log_phase_bullet = ""
    if log_phase_rows:
        def _route_budget(row: Dict[str, Any]) -> int:
            route = str(row.get("log_phase_route") or "")
            match = re.search(r"_(\d+)_(?:rescue|no_rescue)", route)
            if match:
                return int(match.group(1))
            if "_inherit_" in route:
                return 1024
            return 999999

        def _route_rescue_penalty(row: Dict[str, Any]) -> int:
            route = str(row.get("log_phase_route") or "")
            return int("_rescue" in route and "_no_rescue" not in route)

        best_route_rows = sorted(
            log_phase_rows,
            key=lambda row: (
                -(safe_float(row.get("log_skeleton_complete_rate")) or 0.0),
                -(safe_float(row.get("log_marker_rate")) or 0.0),
                safe_float(row.get("ponder_finish_length_mean")) or 0.0,
                safe_float(row.get("final_finish_length_rate")) or 0.0,
                _route_budget(row),
                _route_rescue_penalty(row),
            ),
        )
        best_route = best_route_rows[0] if best_route_rows else {}
        if has_evo_scaffolds:
            log_phase_bullet = (
                f"<li><strong>The closure-protecting route is held fixed.</strong> "
                f"The route lock is <code>{html.escape(str(best_route.get('log_phase_route') or 'n/a'))}</code>, "
                f"with skeleton completion {fmt_num((safe_float(best_route.get('log_skeleton_complete_rate')) or 0.0) * 100, 0)}% "
                f"and final length rate {fmt_num((safe_float(best_route.get('final_finish_length_rate')) or 0.0) * 100, 0)}%; "
                "read scaffold effects against this closure baseline.</li>"
            )
        else:
            log_phase_bullet = (
                f"<li><strong>Log-phase routing is now isolated from final-answer routing.</strong> "
                f"Best skeleton-complete route was <code>{html.escape(str(best_route.get('log_phase_route') or 'n/a'))}</code> "
                f"at {fmt_num((safe_float(best_route.get('log_skeleton_complete_rate')) or 0.0) * 100, 0)}%, "
                f"with final length rate {fmt_num((safe_float(best_route.get('final_finish_length_rate')) or 0.0) * 100, 0)}%; "
                "compare it with rescue-used rate, final finish rate, and raw ponder logs.</li>"
            )

    metric_cards = [
        (str(summary["run_count"]), "completed provider/query runs"),
        (str(summary["ponder_item_count"]), "pondered item rows"),
    ]
    for provider in providers[:2]:
        provider_run = summary["run_summary"].get(provider, {})
        metric_cards.append((f"{fmt_num(provider_run.get('elapsed_s_mean'), 1)}s", f"{provider} mean run time"))
    if len(metric_cards) < 4 and dose_summary:
        metric_cards.append((fmt_num(dose_summary.get("max_dose"), 0), "max scaffold dose"))
    if len(metric_cards) < 4 and closure_summary:
        metric_cards.append((str(len(closure_summary.get("contracts") or [])), "output contracts"))
    if len(metric_cards) < 4:
        metric_cards.append((str(len(providers)), "provider lanes"))
    metric_grid_html = "".join(
        f'<div class="metric"><strong>{html.escape(value)}</strong><span>{html.escape(label)}</span></div>'
        for value, label in metric_cards[:4]
    )

    artifact_rows = []
    for _, row in run_df.sort_values(["provider", "query_id"]).iterrows():
        matrix_href = os.path.relpath(str(row["matrix_report_out"]), str(report_dir))
        trace_href = os.path.relpath(str(row["trace_report_out"]), str(report_dir))
        artifact_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['provider']))}</td>"
            f"<td>{html.escape(str(row['query_id']))}</td>"
            f"<td>{fmt_num(row['elapsed_s'], 1)}</td>"
            f"<td><a href=\"{html.escape(matrix_href)}\">matrix</a> / "
            f"<a href=\"{html.escape(trace_href)}\">trace</a></td>"
            "</tr>"
        )

    pca = summary.get("pca") if isinstance(summary.get("pca"), dict) else {}
    pca_section = ""
    if charts.get("answer_pca"):
        pca_section = f"""
    <section data-contract-section="key-findings">
      <h2>Generated Content PCA</h2>
      <p>The PCA view treats each final answer as a document, hashes character n-gram TF-IDF features, and projects them into two dimensions. This is not semantic truth, but it is useful for spotting whether provider, prompt, or scaffold condition dominates the generated surface.</p>
      <figure>
        {img_tag(charts["answer_pca"], "Generated answer PCA by provider and condition", report_dir)}
        <figcaption>PC1 explains {fmt_num((pca.get("explained_variance_ratio") or [0, 0])[0] * 100, 1)}% and PC2 explains {fmt_num((pca.get("explained_variance_ratio") or [0, 0])[1] * 100, 1)}% of the hashed TF-IDF variance across {int(pca.get("doc_count") or 0)} generated answers.</figcaption>
      </figure>
      <p>Provider centroid distance in PCA space: <strong>{fmt_num(pca.get("provider_centroid_distance"), 3)}</strong>. Use <a href="{html.escape(os.path.relpath(str(pca.get("csv") or ""), str(report_dir)))}">answer_pca.csv</a> for row-level inspection.</p>
    </section>
"""

    dose_section = ""
    if dose_summary and charts.get("query_alignment_delta"):
        dose_figures = []
        for key, alt, caption in [
            (
                "query_alignment_delta",
                "Scaffold dose query alignment",
                "Positive values mean the pondered answer aligned more with the query surface than the baseline did.",
            ),
            (
                "origin_drift",
                "Scaffold dose origin drift",
                "Higher values mean the final answer moved farther from the baseline answer surface.",
            ),
            (
                "scaffold_leakage",
                "Scaffold dose leakage proxy",
                "Leakage is a proxy from scaffold previews and retained records, not hidden-state access.",
            ),
            (
                "coil_index",
                "Scaffold dose coil index",
                "Coil index combines recurrence, query relevance, and positive alignment gain.",
            ),
        ]:
            if charts.get(key):
                dose_figures.append(
                    f"<figure>{img_tag(charts[key], alt, report_dir)}<figcaption>{html.escape(caption)}</figcaption></figure>"
                )
        dose_section = f"""
    <section data-contract-section="key-findings">
      <h2>Scaffold Dose Response and Attractor Proxies</h2>
      <p><strong>Dose 0 is the shared baseline reference.</strong> The ladder then varies injected scaffold target tokens by condition, making it easier to see weak-injection, useful-intervention, and takeover-like regimes separately.</p>
      {''.join(dose_figures)}
      <p>Use <a href="dose_response_metrics.csv">dose_response_metrics.csv</a> for dose rows and <a href="attractor_metrics.csv">attractor_metrics.csv</a> for step drift, origin drift, recurrence, leakage proxy, and coil-index rows.</p>
    </section>
"""

    closure_section = ""
    if closure_summary:
        closure_figures = []
        for key, alt, caption in [
            (
                "final_finish_is_length",
                "Final answer length finish rate by output contract",
                "Lower is better when it means the answer closed normally rather than exhausting completion tokens.",
            ),
            (
                "final_marker_is_last_line",
                "Final marker pass rate by output contract",
                "A pass means END_ANSWER was the final non-empty line of the generated answer.",
            ),
            (
                "ponder_log_marker_rate",
                "Ponder log marker rate by output contract",
                "A pass means the saved ponder log included END_LOG as a standalone line.",
            ),
            (
                "ponder_log_skeleton_complete_rate",
                "Ponder log skeleton pass rate by output contract",
                "A pass means the saved ponder log matched X1| through X4| plus END_LOG exactly.",
            ),
            (
                "answer_chars",
                "Answer length by output contract",
                "Shorter output is useful only when the marker and raw answer show it completed rather than collapsed.",
            ),
        ]:
            if charts.get(key):
                closure_figures.append(
                    f"<figure>{img_tag(charts[key], alt, report_dir)}<figcaption>{html.escape(caption)}</figcaption></figure>"
                )
        closure_section = f"""
    <section data-contract-section="key-findings">
      <h2>Gemini Output Closure Contract</h2>
      <p><strong>The closure contract splits the problem into log closure and final-answer closure.</strong> This helps distinguish a scaffold that fails while producing the ponder log from one that fails when landing the final answer.</p>
      {''.join(closure_figures)}
      <p>Use <a href="closure_contract_metrics.csv">closure_contract_metrics.csv</a> for marker pass rates, final finish reasons, prompt-leak flags, and row-level answer lengths.</p>
    </section>
"""

    log_phase_section = ""
    if log_phase_rows:
        route_figures = []
        for key, alt, caption in [
            (
                "log_phase_route_ponder_log_marker_rate",
                "Log phase route marker rate",
                "END_LOG presence indicates whether the visible log reached a closure marker.",
            ),
            (
                "log_phase_route_ponder_log_skeleton_complete_rate",
                "Log phase route skeleton completion",
                "Skeleton completion is the stricter X1-X4 plus END_LOG contract.",
            ),
            (
                "log_phase_route_ponder_finish_length_count",
                "Log phase route length finishes",
                "Length finishes show when the first log pass still runs out of visible output budget.",
            ),
            (
                "log_phase_route_log_phase_rescue_used_rate",
                "Log phase rescue usage",
                "Rescue usage should rise only where the initial log pass misses the contract.",
            ),
        ]:
            if charts.get(key):
                route_figures.append(
                    f"<figure>{img_tag(charts[key], alt, report_dir)}<figcaption>{html.escape(caption)}</figcaption></figure>"
                )
        log_phase_section = f"""
    <section data-contract-section="key-findings">
      <h2>Log-Phase Dedicated Routes</h2>
      <p><strong>This isolates the ponder-log call from the final-answer call.</strong> The route axis can vary log-stage reasoning effort, log-stage visible-token budget, log rescue, and final-answer routing knobs when the matrix includes them.</p>
      {''.join(route_figures)}
      <p>Use <a href="closure_contract_metrics.csv">closure_contract_metrics.csv</a> to compare route-level marker pass rates, log/final length finishes, and rescue-used rates.</p>
    </section>
"""

    if has_evo_scaffolds:
        title = "Gemini Route-Locked Evolutionary Scaffold Mini Sweep"
    elif closure_summary:
        title = "Gemini Log-Phase Route Sweep" if log_phase_rows else "Gemini Closure Contract Sweep"
    elif dose_summary:
        title = "Scaffold Dose Ladder Sweep"
    else:
        title = "Claude/Gemini Pondering Machine Sweep"
    runtime_section_title = (
        "Runtime and answer length frame the route lock"
        if has_evo_scaffolds
        else "Runtime and answer length frame the closure contract"
        if closure_summary
        else "Runtime and answer length frame the dose ladder"
        if dose_summary
        else "Runtime and answer length split provider behavior"
    )
    runtime_note = (
        "This focused run locks the log/final route and uses one provider lane, so runtime is mainly a planning handle for larger scaffold-evolution sweeps."
        if has_evo_scaffolds
        else "This focused run uses one provider lane, so runtime is mainly a planning handle for larger closure sweeps."
        if closure_summary
        else "This focused run uses one provider lane, so runtime is mainly a planning handle for larger dose ladders."
        if len(providers) == 1
        else "Gemini and Claude are both usable for artifact sweeps, but their latency/cost surfaces differ enough that high-volume sampling and qualitative inspection should be planned separately."
    )
    token_caption = (
        "Log scale. Generated scaffold conditions add operational cost; use this as a planning metric, not a pure model-internals comparison."
        if len(providers) == 1
        else "Log scale. Claude values include Claude Code CLI prompt/cache wrapper overhead, so they are operationally real for this environment but not a pure model API comparison."
    )
    caveat_text = (
        "This is an exploratory Gemini route-locked evolutionary scaffold mini sweep, not a stable benchmark. The route lock is a control surface, marker checks only prove visible completion, finish reasons are provider-reported API metadata, semantic alignment and PCA use hashed character n-grams rather than embedding models, and the raw output atlas remains the main qualitative evidence lane."
        if has_evo_scaffolds
        else "This is an exploratory Gemini log-phase route sweep, not a stable benchmark. Marker checks only prove visible completion, rescue replaces the saved ponder log only when it improves the visible log contract, finish reasons are provider-reported API metadata, semantic alignment and PCA use hashed character n-grams rather than embedding models, and the raw output atlas remains the main qualitative evidence lane."
        if log_phase_rows
        else "This is an exploratory Gemini closure-contract sweep, not a stable benchmark. Marker checks only prove visible completion, finish reasons are provider-reported API metadata, semantic alignment and PCA use hashed character n-grams rather than embedding models, and the raw output atlas remains the main qualitative evidence lane."
        if closure_summary
        else "This is an exploratory dose-ladder artifact sweep, not a stable benchmark. The curated bundle uses one prompt and one provider lane, semantic alignment and PCA use hashed character n-grams rather than embedding models, and attractor metrics are text-surface proxies rather than hidden-state measurements. The raw output atlas is the main qualitative evidence lane."
        if dose_summary
        else "This is an exploratory artifact sweep, not a stable benchmark. Semantic alignment used hashed character n-grams rather than an embedding model, provider routing differs by backend, and older pack JSON files may only retain ponder/scaffold previews in the trace while newer pack outputs retain full records."
    )
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --surface: #FCFCFD;
      --panel: #FFFFFF;
      --ink: #1F2430;
      --muted: #6F768A;
      --line: #D7DBE7;
      --blue: #5477C4;
      --gold: #B8A037;
      --orange: #CC6F47;
      --olive: #71B436;
    }}
    body {{ margin: 0; background: var(--surface); color: var(--ink); font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 42px 22px 72px; }}
    header, section {{ margin-bottom: 34px; }}
    h1, h2, h3 {{ line-height: 1.18; margin: 0 0 12px; letter-spacing: 0; }}
    h1 {{ font-size: 34px; }}
    h2 {{ font-size: 23px; }}
    p, li {{ line-height: 1.7; }}
    a {{ color: var(--blue); }}
    .summary {{ border: 1px solid var(--line); background: var(--panel); border-left: 5px solid var(--blue); padding: 20px 22px; }}
    .summary li + li {{ margin-top: 10px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid var(--line); background: var(--panel); padding: 14px 16px; }}
    .metric strong {{ display: block; font-size: 22px; }}
    .metric span {{ color: var(--muted); font-size: 13px; }}
    figure {{ margin: 20px 0 28px; }}
    figure img {{ display: block; width: 100%; height: auto; border: 1px solid var(--line); background: var(--panel); }}
    figcaption {{ color: var(--muted); font-size: 14px; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); }}
    th, td {{ text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--line); font-size: 14px; }}
    th {{ color: var(--muted); font-weight: 600; }}
    details {{ border: 1px solid var(--line); background: var(--panel); margin: 12px 0; }}
    summary {{ cursor: pointer; padding: 12px 14px; }}
    summary span, summary em {{ color: var(--muted); font-style: normal; margin-left: 8px; }}
    .atlas-query {{ border-top: 1px solid var(--line); padding-top: 18px; margin-top: 22px; }}
    .query-text {{ font-weight: 600; }}
    .output-run {{ border-left: 5px solid var(--gold); }}
    .output-item {{ margin: 10px 14px 16px; border-left: 4px solid var(--olive); }}
    .artifact-links, .keyword-list, .finish-reasons {{ color: var(--muted); font-size: 14px; margin: 0 14px 10px; }}
    .output-item h5 {{ margin: 12px 14px 6px; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0; }}
    .output-text {{ white-space: pre-wrap; overflow-wrap: anywhere; margin: 0 14px 14px; padding: 12px; border: 1px solid var(--line); background: #F8F9FC; color: var(--ink); font-size: 13px; line-height: 1.55; }}
    .callout {{ border: 1px solid var(--line); border-left: 5px solid var(--orange); background: var(--panel); padding: 16px 18px; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    @media (max-width: 760px) {{ .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} h1 {{ font-size: 28px; }} }}
  </style>
</head>
<body>
  <main data-report-audience="product stakeholders">
    <header data-contract-section="title">
      <h1>{html.escape(title)}</h1>
      <p>Generated {html.escape(generated)} from the local sweep artifacts.</p>
    </header>

    <section class="summary" data-contract-section="executive-summary">
      <h2>Executive Summary</h2>
      <ul>
        <li><strong>{summary["run_count"]} lab-matrix runs completed cleanly.</strong> The sweep produced {summary["ponder_item_count"]} pondered comparison rows across {html.escape(model_names or "the selected models")}, plus per-run JSON, trace, and matrix artifacts.</li>
        <li>{provider_runtime_bullet}</li>
        <li>{alignment_bullet}</li>
        {closure_bullet}
        {log_phase_bullet}
        {dose_bullet}
      </ul>
    </section>

    <section data-contract-section="key-findings">
      <h2>{html.escape(runtime_section_title)}</h2>
      <div class="metric-grid">
        {metric_grid_html}
      </div>
      <p><strong>Runtime is part of the experimental surface.</strong> {html.escape(runtime_note)}</p>
      <figure>
        {img_tag(charts["runtime"], "Runtime by provider", report_dir)}
        <figcaption>Each dot is one query-level lab matrix. The bar is the provider mean.</figcaption>
      </figure>
      <p><strong>The output-length gap shapes the first read.</strong> Stance, metaphor-density, and hash-alignment heuristics become easier to interpret when answer lengths are comparable; when they are not, the raw output atlas and PCA view matter more than a single scalar.</p>
      <figure>
        {img_tag(charts["answer_length"], "Answer length by condition", report_dir)}
        <figcaption>Mean answer characters for pondered items. Treat large length gaps as a caveat for any text-surface metric.</figcaption>
      </figure>
    </section>

    <section data-contract-section="key-findings">
      <h2>Scaffold condition changes the metric surface, but not all changes are meaningful yet</h2>
      <p><strong>Alignment deltas moved by condition.</strong> The chart shows where each scaffold condition pulled answers closer to or farther from the original query surface. This should guide the next experiment design rather than be read as a final claim about model cognition.</p>
      <figure>
        {img_tag(charts["alignment"], "Alignment movement by scaffold condition", report_dir)}
        <figcaption>Hashed char-ngram cosine against the original query. Positive means the pondered answer aligned more with the query than the baseline answer did.</figcaption>
      </figure>
      <p><strong>Cost rises where the machine asks the model to synthesize a new scaffold.</strong> The plain associative and random controls are cheaper; <code>facts</code> and <code>isomorphic</code> require extra generation and larger final contexts.</p>
      <figure>
        {img_tag(charts["token_overhead"], "Token overhead by condition", report_dir)}
        <figcaption>{html.escape(token_caption)}</figcaption>
      </figure>
    </section>

	{closure_section}

	{log_phase_section}

	{dose_section}

{pca_section}

    <section data-contract-section="recommended-next-steps">
      <h2>Recommended Next Steps</h2>
      <ol>
        <li><strong>Use closure markers as a diagnostic, not a quality score.</strong> They show whether Gemini landed the output; they do not prove the answer is better.</li>
        <li><strong>Normalize answer-length targets before making provider claims.</strong> Require a minimum answer length, or stratify metrics by length band.</li>
        <li><strong>Use the output atlas as the qualitative inspection lane.</strong> The charts show surface movement; the actual generated text shows whether the movement is meaningful.</li>
      </ol>
    </section>

    <section id="actual-output-atlas">
      <h2>Actual Output Atlas</h2>
      <p>Each provider/query run below includes the full final answers stored in the result JSON, plus captured keywords, trace answer previews, and scaffold previews. Runs created after this report builder update can also include full ponder records inside each pack item.</p>
      {output_atlas_html}
    </section>

    <section data-contract-section="further-questions">
      <h2>Further Questions</h2>
      <p>The next high-value question is whether the same scaffold condition stays separated after output length and thinking effort are controlled. A second question is whether the PCA provider split is mostly language/style, model-family behavior, or the scaffold machinery itself.</p>
      <table>
        <thead><tr><th>Provider</th><th>Query</th><th>Elapsed seconds</th><th>Artifacts</th></tr></thead>
        <tbody>
          {''.join(artifact_rows)}
        </tbody>
      </table>
    </section>

    <section data-contract-section="caveats-and-assumptions">
      <h2>Caveats and Assumptions</h2>
      <div class="callout">
        <p>{html.escape(caveat_text)}</p>
      </div>
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    analysis_dir = sweep_dir / "analysis"
    chart_dir = analysis_dir / "charts"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(sweep_dir / "manifest.json")
    run_rows, item_rows = extract_rows(sweep_dir)
    write_csv(analysis_dir / "run_metrics.csv", run_rows)
    write_csv(analysis_dir / "item_metrics.csv", item_rows)
    run_df = pd.DataFrame(run_rows)
    item_df = pd.DataFrame(item_rows)
    condition_df = item_df.groupby(["provider", "condition"], as_index=False).agg(
        answer_chars_mean=("answer_chars", "mean"),
        query_alignment_delta_mean=("query_alignment_delta", "mean"),
        api_total_delta_mean=("api_total_delta", "mean"),
        stance_shift_mean=("stance_shift", "mean"),
        diff_ratio_mean=("diff_ratio", "mean"),
    )
    condition_df.to_csv(analysis_dir / "condition_summary.csv", index=False)

    attractor_rows = build_attractor_rows(run_df, item_df)
    write_csv(analysis_dir / "attractor_metrics.csv", attractor_rows)
    dose_rows = build_dose_response_rows(run_df, item_df, attractor_rows)
    write_csv(analysis_dir / "dose_response_metrics.csv", dose_rows)
    dose_df = pd.DataFrame(dose_rows)
    closure_rows = build_closure_contract_rows(item_df)
    write_csv(analysis_dir / "closure_contract_metrics.csv", closure_rows)
    closure_df = pd.DataFrame(closure_rows)

    charts = build_charts(run_df, item_df, chart_dir)
    charts.update(build_dose_charts(dose_df, chart_dir))
    charts.update(build_closure_charts(closure_df, chart_dir))
    charts.update(build_log_phase_route_charts(closure_df, chart_dir))
    pca_chart, pca_summary = build_answer_pca(run_df, analysis_dir, chart_dir)
    if pca_chart:
        charts["answer_pca"] = pca_chart
    summary = summarize(run_df, item_df, dose_df, closure_df)
    summary["profile"] = manifest.get("profile")
    summary["profile_config"] = manifest.get("profile_config") if isinstance(manifest.get("profile_config"), dict) else {}
    summary["pca"] = pca_summary
    summary["charts"] = charts
    (analysis_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    output_atlas_html = build_output_atlas_html(analysis_dir, run_df, item_df)
    html_text = build_report_html(analysis_dir, charts, summary, run_df, output_atlas_html)
    (analysis_dir / "report.html").write_text(html_text, encoding="utf-8")
    print(json.dumps({"report": str(analysis_dir / "report.html"), "charts": charts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
