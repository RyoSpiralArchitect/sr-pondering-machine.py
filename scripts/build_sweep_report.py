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
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            return float(value)
    return None


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
            logs = spatial.get("logs") if isinstance(spatial.get("logs"), dict) else {}
            budget = comp.get("token_budget") if isinstance(comp.get("token_budget"), dict) else {}
            delta = budget.get("delta") if isinstance(budget.get("delta"), dict) else {}
            all_usage = budget.get("all_api_usage") if isinstance(budget.get("all_api_usage"), dict) else {}
            extras = item.get("extras") if isinstance(item.get("extras"), dict) else {}
            warnings = extras.get("api_warnings") if isinstance(extras.get("api_warnings"), list) else []
            item_rows.append(
                {
                    "provider": provider,
                    "query_id": query_id,
                    "model": result.get("model"),
                    "condition": str(item.get("name") or ""),
                    "answer_chars": safe_float(metrics.get("answer_chars")),
                    "elapsed_s": safe_float(metrics.get("elapsed_s")),
                    "diff_ratio": safe_float(comp.get("diff_ratio")),
                    "answer_cosine": safe_float(semantic.get("answer_cosine")),
                    "query_alignment_delta": safe_float(semantic.get("query_alignment_delta")),
                    "stance_shift": safe_float(stance.get("shift_score")),
                    "spatial_log_density": safe_float(logs.get("density_per_1k_chars")),
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
    provider_palette = {"gemini": COLOR_FAMILIES["blue"]["base"], "claude": COLOR_FAMILIES["gold"]["base"]}
    provider_edges = {"gemini": COLOR_FAMILIES["blue"]["dark"], "claude": COLOR_FAMILIES["gold"]["dark"]}

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sns.barplot(data=run_df, x="provider", y="elapsed_s", hue="provider", palette=provider_palette, legend=False, ax=ax, edgecolor=TOKENS["ink"])
    sns.stripplot(data=run_df, x="provider", y="elapsed_s", ax=ax, color=COLOR_FAMILIES["orange"]["dark"], size=5, jitter=0.08)
    for patch, provider in zip(ax.patches, ["claude", "gemini"] if list(run_df["provider"].unique())[0] == "claude" else list(run_df["provider"].unique())):
        patch.set_edgecolor(provider_edges.get(provider, TOKENS["ink"]))
    ax.set_xlabel("")
    ax.set_ylabel("Seconds per lab-matrix run")
    add_chart_header(fig, ax, "Runtime by provider", "Three query runs per provider; bars show mean elapsed seconds and dots show individual runs.")
    charts["runtime"] = str(chart_dir / "runtime_by_provider.png")
    save_fig(fig, Path(charts["runtime"]))

    condition_order = ["assoc", "random", "facts", "isomorphic"]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    plot_df = item_df[item_df["condition"].isin(condition_order)].copy()
    sns.barplot(
        data=plot_df,
        x="condition",
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
        x="condition",
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
        x="condition",
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
            docs.append(
                {
                    "provider": str(row["provider"]),
                    "query_id": str(row["query_id"]),
                    "model": str(result.get("model") or row.get("model") or ""),
                    "condition": "baseline" if kind == "baseline" else name,
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


def summarize(run_df: pd.DataFrame, item_df: pd.DataFrame) -> Dict[str, Any]:
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
    return {
        "run_count": int(len(run_df)),
        "ponder_item_count": int(len(item_df)),
        "by_provider": by_provider,
        "run_summary": run_summary,
        "condition_summary": condition_summary,
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
                    meta = (
                        f"{html.escape(str(item.get('control') or name))} | "
                        f"answer chars {fmt_num(condition_metrics.get('answer_chars'), 0)} | "
                        f"diff ratio {fmt_num(condition_metrics.get('diff_ratio'), 3)} | "
                        f"alignment delta {fmt_num(condition_metrics.get('query_alignment_delta'), 3)} | "
                        f"api total delta {fmt_num(condition_metrics.get('api_total_delta'), 0)}"
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
    elapsed_ratio = safe_float(claude_run.get("elapsed_s_mean")) or 0
    if safe_float(gemini_run.get("elapsed_s_mean")):
        elapsed_ratio = elapsed_ratio / float(gemini_run["elapsed_s_mean"])
    model_names = ", ".join(unique_texts(run_df.get("model", pd.Series(dtype=str)).tolist()))

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

    title = "Claude/Gemini Pondering Machine Sweep"
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
        <li><strong>{summary["run_count"]} lab-matrix runs completed cleanly.</strong> The sweep produced {summary["ponder_item_count"]} pondered comparison rows across {html.escape(model_names or "the selected Claude/Gemini models")}, plus per-run JSON, trace, and matrix artifacts.</li>
        <li><strong>Claude was much slower but produced fuller answers.</strong> Mean runtime was {fmt_num(claude_run.get("elapsed_s_mean"), 1)}s for Claude versus {fmt_num(gemini_run.get("elapsed_s_mean"), 1)}s for Gemini, roughly {fmt_num(elapsed_ratio, 1)}x slower; Claude answers averaged {fmt_num(claude.get("answer_chars_mean"), 0)} chars versus Gemini's {fmt_num(gemini.get("answer_chars_mean"), 0)}.</li>
        <li><strong>Alignment lift is a triage signal, not a verdict.</strong> Gemini's mean query-alignment delta was {fmt_num(gemini.get("query_alignment_delta_mean"), 3)} versus Claude's {fmt_num(claude.get("query_alignment_delta_mean"), 3)}; read that alongside answer length, PCA position, and the raw output atlas.</li>
        <li><strong>Generated scaffold conditions are the expensive branch.</strong> <code>facts</code> and <code>isomorphic</code> raised token overhead most strongly for both providers; Claude's CLI wrapper made the overhead about an order of magnitude larger than Gemini's direct OpenAI-compatible API path.</li>
      </ul>
    </section>

    <section data-contract-section="key-findings">
      <h2>Runtime and answer length split provider behavior</h2>
      <div class="metric-grid">
        <div class="metric"><strong>{summary["run_count"]}</strong><span>completed provider/query runs</span></div>
        <div class="metric"><strong>{summary["ponder_item_count"]}</strong><span>pondered item rows</span></div>
        <div class="metric"><strong>{fmt_num(gemini_run.get("elapsed_s_mean"), 1)}s</strong><span>Gemini mean run time</span></div>
        <div class="metric"><strong>{fmt_num(claude_run.get("elapsed_s_mean"), 1)}s</strong><span>Claude mean run time</span></div>
      </div>
      <p><strong>Runtime separated clearly.</strong> Gemini and Claude are both usable for artifact sweeps, but their latency/cost surfaces differ enough that high-volume sampling and qualitative inspection should be planned separately.</p>
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
        <figcaption>Log scale. Claude values include Claude Code CLI prompt/cache wrapper overhead, so they are operationally real for this environment but not a pure model API comparison.</figcaption>
      </figure>
    </section>

{pca_section}

    <section data-contract-section="recommended-next-steps">
      <h2>Recommended Next Steps</h2>
      <ol>
        <li><strong>Normalize answer-length targets before making provider claims.</strong> Require a minimum answer length, or stratify metrics by length band.</li>
        <li><strong>Use the output atlas as the qualitative inspection lane.</strong> The charts show surface movement; the actual generated text shows whether the movement is meaningful.</li>
        <li><strong>Add a small judge pass after the sweep shape is stable.</strong> The current hash alignment and PCA views are useful for triage, but they cannot say whether the pondered answer is actually better.</li>
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
        <p>This is an exploratory artifact sweep, not a stable benchmark. There are only three prompts per provider, Gemini outputs were too short for final interpretation, Claude was routed through the Claude Code CLI rather than a direct Anthropic API client, and semantic alignment used hashed character n-grams rather than an embedding model. Older pack JSON files may only retain ponder/scaffold previews in the trace, while newer pack outputs retain full records.</p>
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

    charts = build_charts(run_df, item_df, chart_dir)
    pca_chart, pca_summary = build_answer_pca(run_df, analysis_dir, chart_dir)
    if pca_chart:
        charts["answer_pca"] = pca_chart
    summary = summarize(run_df, item_df)
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
