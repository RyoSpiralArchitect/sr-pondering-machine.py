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

    parts.append("<h2>Comparison table</h2>")
    parts.append("<table><thead><tr>")
    headers = [
        "Item",
        "Kind",
        "Scaffold",
        "Chars",
        "Elapsed",
        "Semantic",
        "Stance",
        "Budget",
        "Warnings",
    ]
    for h in headers:
        parts.append(f"<th>{_esc(h)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in items:
        scaffold = row.get("scaffold_condition") or "—"
        target = row.get("scaffold_target")
        scaffold_s = scaffold if target in (None, "", 0) else f"{scaffold}/{int(target)}"
        semantic_bits = []
        if row.get("answer_cosine") is not None:
            semantic_bits.append(f"cos={_fmt_float(row.get('answer_cosine'), 4)}")
        if row.get("query_alignment_delta") is not None:
            semantic_bits.append(f"Δalign={_fmt_float(row.get('query_alignment_delta'), 4)}")
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
        parts.append("<tr>")
        parts.append(f"<td><strong>{_esc(row.get('name'))}</strong><br/><span class='muted'>{_esc(row.get('control') or '')}</span></td>")
        parts.append(f"<td>{_esc(row.get('kind'))}</td>")
        parts.append(f"<td>{_esc(scaffold_s)}</td>")
        parts.append(f"<td class='num'>{_esc(row.get('answer_chars'))}</td>")
        parts.append(f"<td class='num'>{_esc(_fmt_float(row.get('elapsed_s'), 2) if row.get('elapsed_s') is not None else '—')}</td>")
        parts.append(f"<td>{_esc(' | '.join(semantic_bits) if semantic_bits else '—')}</td>")
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
