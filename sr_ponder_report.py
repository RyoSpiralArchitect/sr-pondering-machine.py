#!/usr/bin/env python3
"""
sr_ponder_report.py

Tiny, dependency-free analyzer for sr_pondering_machine JSONL memory logs.

Examples:
  python3 sr_ponder_report.py --memory ./ponder_logs.jsonl
  python3 sr_ponder_report.py --memory ./ponder_logs.jsonl --out ./ponder_report.html
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _parse_ts(ts: str) -> Optional[dt.datetime]:
    if not ts:
        return None
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None


def _iter_jsonl(path: Path, *, max_records: int = 0) -> Iterable[Dict[str, Any]]:
    n = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if max_records and n >= max_records:
                    return
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
                    n += 1
    except FileNotFoundError:
        return


def _bar(n: int, max_n: int, *, width: int = 24) -> str:
    if max_n <= 0:
        return ""
    filled = int(round(width * (n / max_n)))
    filled = max(0, min(width, filled))
    return "█" * filled + " " * (width - filled)


def _fmt_pct(n: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{(100.0 * n / total):.1f}%"


def _format_counter(counter: Counter[str], *, title: str, total: int, max_rows: int) -> str:
    if not counter:
        return f"{title}\n(no data)\n"
    items = counter.most_common(max_rows)
    max_n = max(n for _, n in items) if items else 1
    lines = [title]
    for k, n in items:
        lines.append(f"- {k:>16}  {n:>6}  {_fmt_pct(n, total):>7}  {_bar(n, max_n)}")
    return "\n".join(lines) + "\n"


def _bucket_counts(values: Sequence[float], edges: Sequence[float]) -> List[int]:
    """
    edges: ascending, len>=2. Buckets are [e[i], e[i+1]) except the last which is [.., +inf).
    """
    if not values:
        return [0] * max(0, len(edges) - 1)
    out = [0] * max(0, len(edges) - 1)
    for v in values:
        placed = False
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                out[i] += 1
                placed = True
                break
        if not placed and out:
            out[-1] += 1
    return out


def _rank_bins() -> Tuple[List[int], List[str]]:
    edges = [0, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096, 1_000_000_000]
    labels: List[str] = []
    for i in range(len(edges) - 2):
        labels.append(f"{edges[i]}–{edges[i + 1] - 1}")
    labels.append(f"{edges[-2]}+")
    return edges, labels


def _prob_bins() -> Tuple[List[float], List[str]]:
    edges = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    labels = [
        "0–1e-6",
        "1e-6–1e-5",
        "1e-5–1e-4",
        "1e-4–1e-3",
        "1e-3–1e-2",
        "1e-2–1e-1",
        "1e-1–1",
    ]
    return edges, labels


def _describe(vals: Sequence[float]) -> Dict[str, Any]:
    if not vals:
        return {"count": 0}
    v = list(vals)
    v.sort()
    out: Dict[str, Any] = {
        "count": len(v),
        "min": v[0],
        "max": v[-1],
        "mean": float(statistics.fmean(v)),
        "median": float(statistics.median(v)),
    }
    if len(v) >= 10:
        out["p90"] = float(v[int(0.9 * (len(v) - 1))])
        out["p99"] = float(v[int(0.99 * (len(v) - 1))])
    return out


def analyze_memory(path: Path, *, max_records: int, top_keywords: int) -> Dict[str, Any]:
    mode = Counter()
    band = Counter()
    lang = Counter()
    kw_source = Counter()
    keywords = Counter()
    days = Counter()
    run_ids = set()

    token_ranks: List[float] = []
    token_probs: List[float] = []

    ts_min: Optional[dt.datetime] = None
    ts_max: Optional[dt.datetime] = None

    n_records = 0
    for r in _iter_jsonl(path, max_records=max_records):
        n_records += 1

        if rid := r.get("run_id"):
            run_ids.add(str(rid))

        ts = _parse_ts(str(r.get("ts", "")))
        if ts is not None:
            if ts_min is None or ts < ts_min:
                ts_min = ts
            if ts_max is None or ts > ts_max:
                ts_max = ts
            days[ts.date().isoformat()] += 1

        mode[str(r.get("ponder_mode", "unknown"))] += 1
        lang[str(r.get("prompt_lang", "unknown"))] += 1
        kw_source[str(r.get("keywords_source", "unknown"))] += 1

        band_label = r.get("band_label")
        if not band_label and isinstance(r.get("band"), dict):
            b = r["band"]
            if "start_rank" in b and "end_rank" in b:
                band_label = f"{b['start_rank']}:{b['end_rank']}"
        band[str(band_label or "unknown")] += 1

        kws = r.get("keywords")
        if isinstance(kws, list):
            for k in kws:
                if isinstance(k, str):
                    kk = k.strip()
                    if kk:
                        keywords[kk] += 1

        sel = r.get("selected_tokens")
        if isinstance(sel, list):
            for t in sel:
                if not isinstance(t, dict):
                    continue
                rk = t.get("rank")
                pr = t.get("prob")
                try:
                    if rk is not None:
                        token_ranks.append(float(rk))
                except Exception:
                    pass
                try:
                    if pr is not None:
                        token_probs.append(float(pr))
                except Exception:
                    pass

    token_rank_desc = _describe(token_ranks)
    token_prob_desc = _describe(token_probs)

    rank_edges, rank_labels = _rank_bins()
    prob_edges, prob_labels = _prob_bins()

    rank_hist = _bucket_counts(token_ranks, [float(x) for x in rank_edges])
    prob_hist = _bucket_counts(token_probs, prob_edges)

    return {
        "path": str(path),
        "records": n_records,
        "unique_runs": len(run_ids),
        "ts_min": ts_min.isoformat() if ts_min else None,
        "ts_max": ts_max.isoformat() if ts_max else None,
        "mode": mode,
        "band": band,
        "lang": lang,
        "keywords_source": kw_source,
        "top_keywords": keywords.most_common(top_keywords),
        "days": days,
        "token_rank": token_rank_desc,
        "token_prob": token_prob_desc,
        "rank_hist": {"labels": rank_labels, "counts": rank_hist},
        "prob_hist": {"labels": prob_labels, "counts": prob_hist},
    }


def render_text_report(stats: Dict[str, Any], *, max_rows: int) -> str:
    total = int(stats.get("records", 0))
    lines: List[str] = []
    lines.append(f"Memory: {stats.get('path')}")
    lines.append(f"Records: {total} (unique_runs={stats.get('unique_runs')})")
    if stats.get("ts_min") or stats.get("ts_max"):
        lines.append(f"Time: {stats.get('ts_min')} .. {stats.get('ts_max')}")
    lines.append("")

    lines.append(_format_counter(stats.get("mode", Counter()), title="Ponder mode", total=total, max_rows=max_rows))
    lines.append(_format_counter(stats.get("band", Counter()), title="Band label", total=total, max_rows=max_rows))
    lines.append(_format_counter(stats.get("keywords_source", Counter()), title="Keyword source", total=total, max_rows=max_rows))
    lines.append(_format_counter(stats.get("lang", Counter()), title="Prompt lang", total=total, max_rows=max_rows))

    tr = stats.get("token_rank", {}) or {}
    tp = stats.get("token_prob", {}) or {}
    lines.append("Selected token stats")
    lines.append(f"- rank: {tr}")
    lines.append(f"- prob: {tp}")
    lines.append("")

    lines.append("Top keywords")
    for k, n in stats.get("top_keywords", []):
        lines.append(f"- {k}: {n}")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_html_bar_table(title: str, counter: Counter[str], *, total: int, max_rows: int) -> str:
    if not counter:
        return f"<h2>{html.escape(title)}</h2><p>(no data)</p>"
    items = counter.most_common(max_rows)
    max_n = max(n for _, n in items) if items else 1
    rows = []
    for k, n in items:
        w = int(round(360 * (n / max_n))) if max_n else 0
        rows.append(
            "<tr>"
            f"<td class='k'>{html.escape(str(k))}</td>"
            f"<td class='n'>{n}</td>"
            f"<td class='p'>{html.escape(_fmt_pct(n, total))}</td>"
            f"<td class='barcell'><div class='bar' style='width:{w}px'></div></td>"
            "</tr>"
        )
    return (
        f"<h2>{html.escape(title)}</h2>"
        "<table class='tbl'>"
        "<thead><tr><th>key</th><th>count</th><th>%</th><th></th></tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_html_hist(title: str, labels: Sequence[str], counts: Sequence[int]) -> str:
    if not labels or not counts:
        return f"<h2>{html.escape(title)}</h2><p>(no data)</p>"
    max_n = max(counts) if counts else 1
    bars = []
    for lab, n in zip(labels, counts):
        w = int(round(360 * (n / max_n))) if max_n else 0
        bars.append(
            "<tr>"
            f"<td class='k'>{html.escape(str(lab))}</td>"
            f"<td class='n'>{n}</td>"
            f"<td class='barcell'><div class='bar' style='width:{w}px'></div></td>"
            "</tr>"
        )
    return (
        f"<h2>{html.escape(title)}</h2>"
        "<table class='tbl'>"
        "<thead><tr><th>bucket</th><th>count</th><th></th></tr></thead>"
        "<tbody>"
        + "".join(bars)
        + "</tbody></table>"
    )


def render_html_report(stats: Dict[str, Any], *, max_rows: int) -> str:
    total = int(stats.get("records", 0))
    top_kws = stats.get("top_keywords", []) or []

    tr = html.escape(json.dumps(stats.get("token_rank", {}), ensure_ascii=False))
    tp = html.escape(json.dumps(stats.get("token_prob", {}), ensure_ascii=False))

    rank_hist = stats.get("rank_hist", {}) or {}
    prob_hist = stats.get("prob_hist", {}) or {}

    kw_rows = []
    for k, n in top_kws:
        kw_rows.append(f"<tr><td class='k'>{html.escape(str(k))}</td><td class='n'>{n}</td></tr>")

    return f"""<!doctype html>
<meta charset="utf-8" />
<title>sr_ponder report</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; }}
  .meta {{ color: #444; margin-bottom: 12px; }}
  .tbl {{ border-collapse: collapse; width: 100%; max-width: 980px; }}
  .tbl th, .tbl td {{ border-bottom: 1px solid #eee; padding: 6px 8px; font-size: 14px; }}
  .tbl th {{ text-align: left; color: #222; }}
  .k {{ white-space: nowrap; }}
  .n {{ text-align: right; width: 90px; }}
  .p {{ text-align: right; width: 70px; color: #444; }}
  .barcell {{ width: 400px; }}
  .bar {{ height: 12px; background: linear-gradient(90deg, #222, #666); border-radius: 999px; }}
  code {{ background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }}
  h1 {{ margin: 0 0 8px 0; }}
  h2 {{ margin: 20px 0 8px 0; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
  @media (min-width: 980px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
  .box {{ border: 1px solid #eee; border-radius: 12px; padding: 12px 14px; }}
</style>

<h1>sr_ponder report</h1>
<div class="meta">
  <div>Memory: <code>{html.escape(str(stats.get("path", "")))}</code></div>
  <div>Records: <code>{total}</code> (unique_runs=<code>{html.escape(str(stats.get("unique_runs")))}</code>)</div>
  <div>Time: <code>{html.escape(str(stats.get("ts_min")))}</code> .. <code>{html.escape(str(stats.get("ts_max")))}</code></div>
</div>

<div class="grid">
  <div class="box">{_render_html_bar_table("Ponder mode", stats.get("mode", Counter()), total=total, max_rows=max_rows)}</div>
  <div class="box">{_render_html_bar_table("Band label", stats.get("band", Counter()), total=total, max_rows=max_rows)}</div>
  <div class="box">{_render_html_bar_table("Keyword source", stats.get("keywords_source", Counter()), total=total, max_rows=max_rows)}</div>
  <div class="box">{_render_html_bar_table("Prompt lang", stats.get("lang", Counter()), total=total, max_rows=max_rows)}</div>
</div>

<div class="grid">
  <div class="box">{_render_html_hist("Selected token rank histogram", rank_hist.get("labels", []), rank_hist.get("counts", []))}</div>
  <div class="box">{_render_html_hist("Selected token prob histogram", prob_hist.get("labels", []), prob_hist.get("counts", []))}</div>
</div>

<h2>Selected token stats</h2>
<ul>
  <li>rank: <code>{tr}</code></li>
  <li>prob: <code>{tp}</code></li>
</ul>

<h2>Top keywords</h2>
<table class="tbl" style="max-width: 640px;">
  <thead><tr><th>keyword</th><th class="n">count</th></tr></thead>
  <tbody>{''.join(kw_rows)}</tbody>
</table>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", required=True, help="Path to ponder_logs.jsonl")
    ap.add_argument("--out", default="", help="Optional output file (e.g., report.html)")
    ap.add_argument("--format", choices=["auto", "text", "html"], default="auto")
    ap.add_argument("--max_records", type=int, default=0, help="Analyze only first N records (0=all)")
    ap.add_argument("--max_rows", type=int, default=20, help="Max rows per table")
    ap.add_argument("--top_keywords", type=int, default=40)
    args = ap.parse_args()

    path = Path(args.memory)
    stats = analyze_memory(path, max_records=max(0, int(args.max_records)), top_keywords=max(0, int(args.top_keywords)))

    fmt = args.format
    out = (args.out or "").strip()
    if fmt == "auto":
        if out.lower().endswith(".html"):
            fmt = "html"
        else:
            fmt = "text"

    if fmt == "html":
        html_text = render_html_report(stats, max_rows=max(1, int(args.max_rows)))
        if out:
            Path(out).write_text(html_text, encoding="utf-8")
            print(out)
        else:
            print(html_text)
        return

    # text
    text = render_text_report(stats, max_rows=max(1, int(args.max_rows)))
    if out:
        Path(out).write_text(text, encoding="utf-8")
        print(out)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()

