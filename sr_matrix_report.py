#!/usr/bin/env python3
"""
sr_matrix_report.py

Dependency-free HTML summary for sr_pondering_machine pack / lab-matrix result JSON.

Examples:
  python3 sr_matrix_report.py --results ./artifacts/scaffold_abcd.json
  python3 sr_matrix_report.py --results ./artifacts/scaffold_abcd.json --out ./artifacts/scaffold_abcd.matrix.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _esc(x: Any) -> str:
    return html.escape(str(x if x is not None else ""))


def _fmt_float(x: Any, digits: int = 4) -> str:
    if isinstance(x, (int, float)):
        try:
            return f"{float(x):.{int(digits)}f}"
        except Exception:
            return "?"
    return "?"


def _preview(text: Any, *, limit: int = 420) -> str:
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"


def load_results(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"[sr_matrix_report] ERROR: failed to read {str(path)!r}: {e}")
    if not isinstance(data, dict):
        raise SystemExit(f"[sr_matrix_report] ERROR: results JSON must be an object, got {type(data).__name__}")
    return data


def _row_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
    extras = item.get("extras") if isinstance(item.get("extras"), dict) else {}
    comparison = item.get("comparison") if isinstance(item.get("comparison"), dict) else {}
    semantic = comparison.get("semantic") if isinstance(comparison.get("semantic"), dict) else {}
    judge = comparison.get("judge") if isinstance(comparison.get("judge"), dict) else {}
    stance = comparison.get("stance") if isinstance(comparison.get("stance"), dict) else {}
    spatial = comparison.get("spatial_metaphor") if isinstance(comparison.get("spatial_metaphor"), dict) else {}
    budget = comparison.get("token_budget") if isinstance(comparison.get("token_budget"), dict) else {}
    budget_delta = budget.get("delta") if isinstance(budget.get("delta"), dict) else {}
    scaffold = extras.get("scaffold") if isinstance(extras.get("scaffold"), dict) else {}
    api_warnings = extras.get("api_warnings") if isinstance(extras.get("api_warnings"), list) else []

    scaffold_condition = str(scaffold.get("condition") or (item.get("cfg_overrides") or {}).get("scaffold_condition") or "")
    scaffold_target = scaffold.get("target_tokens")
    if scaffold_target is None:
        scaffold_target = (item.get("cfg_overrides") or {}).get("scaffold_token_target")

    baseline_dom = str(stance.get("dominant_baseline") or "")
    ponder_dom = str(stance.get("dominant_ponder") or "")
    logs = spatial.get("logs") if isinstance(spatial.get("logs"), dict) else {}

    return {
        "name": str(item.get("name") or ""),
        "kind": str(item.get("kind") or ""),
        "control": str(item.get("control") or ""),
        "query": str(item.get("query") or ""),
        "answer": str(item.get("answer") or ""),
        "answer_chars": metrics.get("answer_chars"),
        "records": metrics.get("records"),
        "elapsed_s": metrics.get("elapsed_s"),
        "warnings_count": len(api_warnings),
        "warnings": [str(x) for x in api_warnings],
        "scaffold_condition": scaffold_condition,
        "scaffold_target": scaffold_target,
        "comparison": comparison,
        "diff_ratio": comparison.get("diff_ratio"),
        "answer_changed": comparison.get("answer_changed"),
        "semantic_method": str(semantic.get("method") or ""),
        "answer_cosine": semantic.get("answer_cosine"),
        "query_alignment_delta": semantic.get("query_alignment_delta"),
        "judge_method": str(judge.get("method") or ""),
        "judge_winner": str(judge.get("winner") or ""),
        "judge_confidence": judge.get("confidence"),
        "judge_score_delta": judge.get("score_delta"),
        "judge_resolution_delta": judge.get("resolution_delta"),
        "stance_base": baseline_dom,
        "stance_ponder": ponder_dom,
        "stance_shift": stance.get("shift_score"),
        "spatial_log_density": logs.get("density_per_1k_chars"),
        "budget_method": str(budget.get("method") or ""),
        "budget_scaffold_tokens": budget_delta.get("external_scaffold_tokens_est"),
        "budget_reasoning_delta": budget_delta.get("api_reasoning_tokens"),
        "budget_completion_delta": budget_delta.get("api_completion_tokens"),
        "budget_total_delta": budget_delta.get("api_total_tokens"),
    }


HEATMAP_METRICS: List[Dict[str, Any]] = [
    {"key": "judge_score_delta", "label": "Judge score Δ", "direction": "max", "digits": 4},
    {"key": "judge_resolution_delta", "label": "Judge resolution Δ", "direction": "max", "digits": 4},
    {"key": "query_alignment_delta", "label": "Query align Δ", "direction": "max", "digits": 4},
    {"key": "stance_shift", "label": "Stance shift", "direction": "max", "digits": 4},
    {"key": "budget_reasoning_delta", "label": "Reasoning token Δ", "direction": "min", "digits": 0},
    {"key": "budget_total_delta", "label": "Total token Δ", "direction": "min", "digits": 0},
]


def _analyze_heatmaps(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = [row for row in rows if str(row.get("kind") or "") != "baseline"]
    if len(candidates) < 2:
        return []

    heatmaps: List[Dict[str, Any]] = []
    for spec in HEATMAP_METRICS:
        key = str(spec.get("key") or "")
        direction = str(spec.get("direction") or "max")
        digits = int(spec.get("digits") or 4)
        item_values: List[Optional[float]] = []
        usable = 0
        for row in candidates:
            val = row.get(key)
            if isinstance(val, (int, float)):
                item_values.append(float(val))
                usable += 1
            else:
                item_values.append(None)
        if usable < 2:
            continue

        cells: List[List[Dict[str, Any]]] = []
        row_wins: List[int] = []
        for ix, aval in enumerate(item_values):
            row_cells: List[Dict[str, Any]] = []
            wins = 0
            for jx, bval in enumerate(item_values):
                if ix == jx:
                    row_cells.append({"display": "—", "class": "heat-diag"})
                    continue
                if aval is None or bval is None:
                    row_cells.append({"display": "?", "class": "heat-missing"})
                    continue
                if direction == "min":
                    signed_delta = bval - aval
                else:
                    signed_delta = aval - bval
                if signed_delta > 0:
                    cell_class = "heat-win"
                    wins += 1
                elif signed_delta < 0:
                    cell_class = "heat-loss"
                else:
                    cell_class = "heat-tie"
                row_cells.append(
                    {
                        "display": _fmt_float(signed_delta, digits),
                        "class": cell_class,
                        "delta": signed_delta,
                    }
                )
            row_wins.append(wins)
            cells.append(row_cells)

        items = []
        for row, val, wins in zip(candidates, item_values, row_wins):
            items.append(
                {
                    "name": str(row.get("name") or ""),
                    "scaffold_condition": str(row.get("scaffold_condition") or ""),
                    "value": val,
                    "value_display": _fmt_float(val, digits) if val is not None else "?",
                    "wins": wins,
                }
            )
        heatmaps.append(
            {
                "key": key,
                "label": str(spec.get("label") or key),
                "direction": direction,
                "digits": digits,
                "items": items,
                "cells": cells,
            }
        )
    return heatmaps


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    kind = str(results.get("kind") or "").strip()
    if kind not in ("pack", "lab_matrix"):
        raise SystemExit(f"[sr_matrix_report] ERROR: expected kind=pack|lab_matrix, got {kind!r}")
    items_raw = results.get("items")
    if not isinstance(items_raw, list):
        raise SystemExit("[sr_matrix_report] ERROR: results JSON is missing an items array")

    rows: List[Dict[str, Any]] = []
    for item in items_raw:
        if isinstance(item, dict):
            rows.append(_row_from_item(item))

    baseline_name = ""
    for row in rows:
        if row.get("kind") == "baseline":
            baseline_name = str(row.get("name") or "")
            break

    max_reasoning_delta: Optional[float] = None
    strongest_shift: Optional[float] = None
    best_alignment_delta: Optional[float] = None
    for row in rows:
        val = row.get("budget_reasoning_delta")
        if isinstance(val, (int, float)):
            max_reasoning_delta = float(val) if max_reasoning_delta is None else max(max_reasoning_delta, float(val))
        val = row.get("stance_shift")
        if isinstance(val, (int, float)):
            strongest_shift = float(val) if strongest_shift is None else max(strongest_shift, float(val))
        val = row.get("query_alignment_delta")
        if isinstance(val, (int, float)):
            best_alignment_delta = float(val) if best_alignment_delta is None else max(best_alignment_delta, float(val))

    return {
        "kind": kind,
        "pack": str(results.get("pack") or ""),
        "lab_matrix": str(results.get("lab_matrix") or ""),
        "provider": str(results.get("provider") or ""),
        "model": str(results.get("model") or ""),
        "query": str(results.get("query") or ""),
        "base_cfg": results.get("base_cfg") if isinstance(results.get("base_cfg"), dict) else {},
        "items": rows,
        "heatmaps": _analyze_heatmaps(rows),
        "summary": {
            "baseline_name": baseline_name,
            "item_count": len(rows),
            "ponder_count": sum(1 for r in rows if r.get("kind") == "ponder"),
            "warning_count": sum(int(r.get("warnings_count") or 0) for r in rows),
            "max_reasoning_delta": max_reasoning_delta,
            "strongest_shift": strongest_shift,
            "best_alignment_delta": best_alignment_delta,
        },
    }


def render_html(report: Dict[str, Any]) -> str:
    items = report.get("items") if isinstance(report.get("items"), list) else []
    heatmaps = report.get("heatmaps") if isinstance(report.get("heatmaps"), list) else []
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}

    parts: List[str] = []
    parts.append("<!doctype html>")
    parts.append("<html><head><meta charset='utf-8'/>")
    parts.append("<title>sr_ponder matrix report</title>")
    parts.append(
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;color:#111;background:#fafafa;}"
        "h1,h2,h3{margin:0 0 12px 0;} .muted{color:#666;} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:16px 0;}"
        ".card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:14px;box-shadow:0 1px 2px rgba(0,0,0,0.04);}"
        ".toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:12px 0 14px 0;padding:10px 12px;background:#fff;border:1px solid #ddd;border-radius:10px;}"
        ".toolbar label{font-size:13px;color:#333;display:flex;gap:6px;align-items:center;}"
        ".toolbar select,.toolbar button{font:inherit;padding:6px 10px;border:1px solid #ccc;border-radius:8px;background:#fff;color:#111;}"
        ".toolbar button{cursor:pointer;}"
        ".heatmap-wrap{margin:12px 0 20px 0;}"
        ".heatmap-cell{min-width:74px;text-align:center;font-variant-numeric:tabular-nums;}"
        ".heat-win{background:#e8f7ee;color:#0a5a2e;}"
        ".heat-loss{background:#fdecec;color:#8b1d1d;}"
        ".heat-tie,.heat-diag{background:#f3f3f3;color:#666;}"
        ".heat-missing{background:#fff8e6;color:#8a6d1f;}"
        "table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #ddd;} th,td{padding:8px 10px;border-bottom:1px solid #eee;vertical-align:top;text-align:left;font-size:13px;}"
        "th{background:#f3f3f3;position:sticky;top:0;} code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}"
        "pre{white-space:pre-wrap;background:#fbfbfb;border:1px solid #eee;padding:12px;border-radius:8px;}"
        ".pill{display:inline-block;padding:2px 8px;border-radius:999px;background:#eef4ff;color:#245; margin-right:6px;font-size:12px;}"
        "details{margin:12px 0;} .num{text-align:right;} .ok{color:#0a6;} .warn{color:#a60;}"
        "</style>"
    )
    parts.append("</head><body>")
    parts.append("<h1>sr_ponder matrix report</h1>")
    subtitle = report.get("lab_matrix") or report.get("pack") or report.get("kind")
    parts.append(f"<div class='muted'>{_esc(subtitle)} · {_esc(report.get('provider'))} / {_esc(report.get('model'))}</div>")
    parts.append(f"<p><strong>Query:</strong> {_esc(report.get('query'))}</p>")

    parts.append("<div class='grid'>")
    cards = [
        ("Items", summary.get("item_count")),
        ("Ponder items", summary.get("ponder_count")),
        ("Warnings", summary.get("warning_count")),
        ("Baseline", summary.get("baseline_name") or "—"),
        ("Max reasoning Δ", _fmt_float(summary.get("max_reasoning_delta"), 0) if summary.get("max_reasoning_delta") is not None else "—"),
        ("Max stance shift", _fmt_float(summary.get("strongest_shift"), 4) if summary.get("strongest_shift") is not None else "—"),
        ("Best align Δ", _fmt_float(summary.get("best_alignment_delta"), 4) if summary.get("best_alignment_delta") is not None else "—"),
    ]
    for title, value in cards:
        parts.append(f"<div class='card'><div class='muted'>{_esc(title)}</div><div><strong>{_esc(value)}</strong></div></div>")
    parts.append("</div>")

    if heatmaps:
        parts.append("<h2>Pairwise heatmap</h2>")
        parts.append(
            "<div class='toolbar'>"
            "<strong>A/B/C/D win-loss heatmap</strong>"
            "<label>Metric"
            "<select id='heatmap-metric-key'>"
        )
        for ix, heatmap in enumerate(heatmaps):
            selected = " selected" if ix == 0 else ""
            parts.append(f"<option value='{_esc(heatmap.get('key'))}'{selected}>{_esc(heatmap.get('label'))}</option>")
        parts.append(
            "</select>"
            "</label>"
            "<span class='muted'>Green means the row beats the column on the selected metric. Cost metrics use lower-is-better.</span>"
            "</div>"
        )
        for ix, heatmap in enumerate(heatmaps):
            display = "" if ix == 0 else " style='display:none'"
            direction_label = "higher wins" if heatmap.get("direction") == "max" else "lower wins"
            parts.append(
                f"<div class='heatmap-wrap' data-heatmap-key='{_esc(heatmap.get('key'))}'{display}>"
                f"<div class='muted'>{_esc(heatmap.get('label'))} · {_esc(direction_label)}</div>"
            )
            parts.append("<table><thead><tr><th>Item</th>")
            for item in heatmap.get("items") or []:
                parts.append(f"<th>{_esc(item.get('name'))}</th>")
            parts.append("</tr></thead><tbody>")
            heatmap_items = heatmap.get("items") or []
            cells = heatmap.get("cells") or []
            for item, row_cells in zip(heatmap_items, cells):
                row_meta = f"{item.get('value_display')} · {int(item.get('wins') or 0)}W"
                parts.append("<tr>")
                parts.append(
                    f"<td><strong>{_esc(item.get('name'))}</strong>"
                    f"<br/><span class='muted'>{_esc(row_meta)}</span></td>"
                )
                for cell in row_cells:
                    parts.append(
                        f"<td class='heatmap-cell {_esc(cell.get('class'))}'>{_esc(cell.get('display'))}</td>"
                    )
                parts.append("</tr>")
            parts.append("</tbody></table></div>")

    parts.append("<h2>Comparison table</h2>")
    parts.append(
        "<div class='toolbar'>"
        "<strong>Sort by baseline-relative metric</strong>"
        "<label>Metric"
        "<select id='matrix-sort-key'>"
        "<option value='judge_score_delta'>Judge score Δ</option>"
        "<option value='judge_resolution_delta'>Judge resolution Δ</option>"
        "<option value='query_alignment_delta'>Query align Δ</option>"
        "<option value='stance_shift'>Stance shift</option>"
        "<option value='budget_reasoning_delta'>Reasoning token Δ</option>"
        "<option value='budget_completion_delta'>Completion token Δ</option>"
        "<option value='budget_total_delta'>Total token Δ</option>"
        "<option value='answer_cosine'>Answer cosine</option>"
        "<option value='diff_ratio'>Diff ratio</option>"
        "<option value='answer_chars'>Answer chars</option>"
        "<option value='elapsed_s'>Elapsed seconds</option>"
        "<option value='name'>Name</option>"
        "</select>"
        "</label>"
        "<label>Direction"
        "<select id='matrix-sort-dir'>"
        "<option value='desc'>Descending</option>"
        "<option value='asc'>Ascending</option>"
        "</select>"
        "</label>"
        "<button type='button' id='matrix-sort-apply'>Apply</button>"
        "<span class='muted'>Baseline rows stay pinned to the top.</span>"
        "</div>"
    )
    parts.append("<table><thead><tr>")
    headers = [
        "Item",
        "Kind",
        "Scaffold",
        "Chars",
        "Elapsed",
        "Semantic",
        "Judge",
        "Stance",
        "Budget",
        "Warnings",
    ]
    for h in headers:
        parts.append(f"<th>{_esc(h)}</th>")
    parts.append("</tr></thead><tbody id='matrix-report-body'>")
    for row in items:
        scaffold = row.get("scaffold_condition") or "—"
        target = row.get("scaffold_target")
        scaffold_s = scaffold if target in (None, "", 0) else f"{scaffold}/{int(target)}"
        semantic_bits = []
        if row.get("answer_cosine") is not None:
            semantic_bits.append(f"cos={_fmt_float(row.get('answer_cosine'), 4)}")
        if row.get("query_alignment_delta") is not None:
            semantic_bits.append(f"Δalign={_fmt_float(row.get('query_alignment_delta'), 4)}")
        judge_bits = []
        if row.get("judge_winner"):
            judge_bits.append(f"winner={row.get('judge_winner')}")
        if row.get("judge_score_delta") is not None:
            judge_bits.append(f"scoreΔ={_fmt_float(row.get('judge_score_delta'), 4)}")
        if row.get("judge_resolution_delta") is not None:
            judge_bits.append(f"resΔ={_fmt_float(row.get('judge_resolution_delta'), 4)}")
        if row.get("judge_confidence") is not None:
            judge_bits.append(f"conf={_fmt_float(row.get('judge_confidence'), 3)}")
        stance_bits = []
        if row.get("stance_base") or row.get("stance_ponder"):
            stance_bits.append(f"{row.get('stance_base') or '—'}→{row.get('stance_ponder') or '—'}")
        if row.get("stance_shift") is not None:
            stance_bits.append(f"shift={_fmt_float(row.get('stance_shift'), 4)}")
        budget_bits = []
        if row.get("budget_scaffold_tokens") is not None:
            budget_bits.append(f"scaf={int(row.get('budget_scaffold_tokens') or 0)}")
        if row.get("budget_reasoning_delta") is not None:
            budget_bits.append(f"reasonΔ={int(row.get('budget_reasoning_delta') or 0)}")
        if row.get("budget_completion_delta") is not None:
            budget_bits.append(f"compΔ={int(row.get('budget_completion_delta') or 0)}")
        warn_count = int(row.get("warnings_count") or 0)
        warn_cls = "warn" if warn_count else "ok"
        data_attrs = {
            "name": str(row.get("name") or ""),
            "kind": str(row.get("kind") or ""),
            "answer_chars": row.get("answer_chars"),
            "elapsed_s": row.get("elapsed_s"),
            "diff_ratio": row.get("diff_ratio"),
            "answer_cosine": row.get("answer_cosine"),
            "query_alignment_delta": row.get("query_alignment_delta"),
            "judge_score_delta": row.get("judge_score_delta"),
            "judge_resolution_delta": row.get("judge_resolution_delta"),
            "judge_confidence": row.get("judge_confidence"),
            "stance_shift": row.get("stance_shift"),
            "budget_scaffold_tokens": row.get("budget_scaffold_tokens"),
            "budget_reasoning_delta": row.get("budget_reasoning_delta"),
            "budget_completion_delta": row.get("budget_completion_delta"),
            "budget_total_delta": row.get("budget_total_delta"),
        }
        attr_s = " ".join(f"data-{k.replace('_', '-')}='{_esc(v)}'" for k, v in data_attrs.items())
        parts.append(f"<tr {attr_s}>")
        parts.append(f"<td><strong>{_esc(row.get('name'))}</strong><br/><span class='muted'>{_esc(row.get('control') or '')}</span></td>")
        parts.append(f"<td>{_esc(row.get('kind'))}</td>")
        parts.append(f"<td>{_esc(scaffold_s)}</td>")
        parts.append(f"<td class='num'>{_esc(row.get('answer_chars'))}</td>")
        parts.append(f"<td class='num'>{_esc(_fmt_float(row.get('elapsed_s'), 2) if row.get('elapsed_s') is not None else '—')}</td>")
        parts.append(f"<td>{_esc(' | '.join(semantic_bits) if semantic_bits else '—')}</td>")
        parts.append(f"<td>{_esc(' | '.join(judge_bits) if judge_bits else '—')}</td>")
        parts.append(f"<td>{_esc(' | '.join(stance_bits) if stance_bits else '—')}</td>")
        parts.append(f"<td>{_esc(' | '.join(budget_bits) if budget_bits else '—')}</td>")
        parts.append(f"<td class='{warn_cls}'>{warn_count}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")

    parts.append("<h2>Item details</h2>")
    for row in items:
        parts.append("<details>")
        parts.append(f"<summary><strong>{_esc(row.get('name'))}</strong> · {_esc(row.get('kind'))}</summary>")
        parts.append("<div class='card'>")
        if row.get("comparison"):
            comp = row["comparison"]
            bits = []
            if comp.get("diff_ratio") is not None:
                bits.append(f"diff={_fmt_float(comp.get('diff_ratio'), 4)}")
            if comp.get("answer_changed") is not None:
                bits.append(f"changed={comp.get('answer_changed')}")
            if bits:
                parts.append(f"<div class='muted'>{_esc(' | '.join(bits))}</div>")
        if row.get("warnings"):
            parts.append("<h3>warnings</h3>")
            parts.append("<ul>")
            for w in row.get("warnings") or []:
                parts.append(f"<li>{_esc(w)}</li>")
            parts.append("</ul>")
        parts.append("<h3>answer</h3>")
        parts.append(f"<pre>{_esc(_preview(row.get('answer'), limit=1200))}</pre>")
        parts.append("</div>")
        parts.append("</details>")

    parts.append(
        "<script>"
        "(function(){"
        "const body=document.getElementById('matrix-report-body');"
        "const heatSel=document.getElementById('heatmap-metric-key');"
        "const keySel=document.getElementById('matrix-sort-key');"
        "const dirSel=document.getElementById('matrix-sort-dir');"
        "const applyBtn=document.getElementById('matrix-sort-apply');"
        "const heatmaps=Array.from(document.querySelectorAll('[data-heatmap-key]'));"
        "if(heatSel&&heatmaps.length){"
        "function syncHeatmap(){"
        "const wanted=heatSel.value||'';"
        "for(const node of heatmaps){"
        "node.style.display=(node.getAttribute('data-heatmap-key')===wanted)?'':'none';"
        "}"
        "}"
        "heatSel.addEventListener('change',syncHeatmap);"
        "syncHeatmap();"
        "}"
        "if(!body||!keySel||!dirSel||!applyBtn){return;}"
        "function metricValue(row,key){"
        "const attr='data-'+String(key||'').replace(/_/g,'-');"
        "const raw=row.getAttribute(attr)||'';"
        "if(key==='name'){return raw.toLowerCase();}"
        "const num=Number(raw);"
        "return Number.isFinite(num)?num:null;"
        "}"
        "function compareRows(a,b,key,dir){"
        "const av=metricValue(a,key);"
        "const bv=metricValue(b,key);"
        "if(typeof av==='string'||typeof bv==='string'){"
        "const as=String(av||'');"
        "const bs=String(bv||'');"
        "return dir==='asc'?as.localeCompare(bs):bs.localeCompare(as);"
        "}"
        "if(av===null&&bv===null){return 0;}"
        "if(av===null){return 1;}"
        "if(bv===null){return -1;}"
        "return dir==='asc'?(av-bv):(bv-av);"
        "}"
        "function sortRows(){"
        "const key=keySel.value||'query_alignment_delta';"
        "const dir=dirSel.value||'desc';"
        "const rows=Array.from(body.querySelectorAll('tr'));"
        "const pinned=rows.filter((row)=>row.getAttribute('data-kind')==='baseline');"
        "const sortable=rows.filter((row)=>row.getAttribute('data-kind')!=='baseline');"
        "sortable.sort((a,b)=>{"
        "const main=compareRows(a,b,key,dir);"
        "if(main!==0){return main;}"
        "return compareRows(a,b,'name','asc');"
        "});"
        "for(const row of pinned.concat(sortable)){body.appendChild(row);}"
        "}"
        "applyBtn.addEventListener('click',sortRows);"
        "keySel.addEventListener('change',sortRows);"
        "dirSel.addEventListener('change',sortRows);"
        "sortRows();"
        "})();"
        "</script>"
    )
    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render an HTML report for sr_pondering_machine pack/lab-matrix results")
    ap.add_argument("--results", required=True, help="Pack / lab-matrix result JSON path")
    ap.add_argument("--out", default="", help="Optional HTML output path")
    args = ap.parse_args()

    path = Path(str(args.results))
    report = analyze_results(load_results(path))
    html_text = render_html(report)
    out = str(args.out or "").strip()
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html_text, encoding="utf-8")
        print(f"[sr_matrix_report] wrote {out_path}")
        return

    print(html_text)


if __name__ == "__main__":
    main()
