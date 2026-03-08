#!/usr/bin/env python3
"""
sr_trace_report.py

Tiny, dependency-free analyzer for sr_pondering_machine trace JSONL files.

Examples:
  python3 sr_trace_report.py --trace ./run.trace.jsonl
  python3 sr_trace_report.py --trace ./run.trace.jsonl --out ./trace_report.html
  python3 sr_trace_report.py --trace ./run.trace.jsonl --session_id abc123
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def _esc(x: Any) -> str:
    return html.escape(str(x if x is not None else ""))

def _slug(s: str) -> str:
    out = []
    for ch in str(s or ""):
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    t = "".join(out).strip("_")
    return t or "unknown"


def _as_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        try:
            return float(x)
        except Exception:
            return None
    return None


def _fmt_float(x: Any, digits: int = 6) -> str:
    if isinstance(x, (int, float)):
        try:
            return f"{float(x):.{int(digits)}f}"
        except Exception:
            return "?"
    return "?"


def _token_label(x: Any) -> str:
    if not isinstance(x, dict):
        return ""
    tok = str(x.get("token") or "").strip()
    tid = x.get("token_id")
    if tok and tid is not None:
        return f"{tok} (id={tid})"
    if tok:
        return tok
    if tid is not None:
        return f"id={tid}"
    return ""


def analyze_trace(path: Path, *, max_records: int, session_id: str) -> Dict[str, Any]:
    sessions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ev in _iter_jsonl(path, max_records=max_records):
        sid = str(ev.get("session_id") or "unknown")
        if session_id and sid != session_id:
            continue
        sessions[sid].append(ev)

    out: Dict[str, Any] = {"path": str(path), "sessions": []}
    for sid, events in sorted(sessions.items(), key=lambda kv: kv[0]):
        # Sort by timestamp when available, keep stable order otherwise.
        events2: List[Tuple[Optional[dt.datetime], int, Dict[str, Any]]] = []
        for i, ev in enumerate(events):
            ts = _parse_ts(str(ev.get("ts") or ""))
            events2.append((ts, i, ev))
        events2.sort(key=lambda x: (x[0] is None, x[0] or dt.datetime.min.replace(tzinfo=dt.timezone.utc), x[1]))
        ordered = [ev for _, _, ev in events2]

        counts = Counter(str(ev.get("event") or "unknown") for ev in ordered)
        ts_vals = [_parse_ts(str(ev.get("ts") or "")) for ev in ordered]
        ts_vals2 = [t for t in ts_vals if t is not None]
        ts_min = min(ts_vals2) if ts_vals2 else None
        ts_max = max(ts_vals2) if ts_vals2 else None

        start_ts: Optional[dt.datetime] = None
        end_ts: Optional[dt.datetime] = None
        for ev in ordered:
            nm = str(ev.get("event") or "")
            if nm == "session_start" and start_ts is None:
                start_ts = _parse_ts(str(ev.get("ts") or "")) or start_ts
            if nm == "session_end":
                end_ts = _parse_ts(str(ev.get("ts") or "")) or end_ts
        if start_ts is None:
            start_ts = ts_min
        if end_ts is None:
            end_ts = ts_max
        duration_s: Optional[float] = None
        if start_ts is not None and end_ts is not None:
            try:
                duration_s = float((end_ts - start_ts).total_seconds())
            except Exception:
                duration_s = None

        pack_id = ""
        pack_items = 0
        pack_skips = 0
        per_item_elapsed: Dict[str, float] = {}
        for ev in ordered:
            name = str(ev.get("event") or "")
            if name == "pack_start":
                pack_id = str(ev.get("pack") or "") or pack_id
                try:
                    pack_items = int(ev.get("items") or pack_items)
                except Exception:
                    pass
            if name == "pack_item_skip":
                pack_skips += 1
            if name == "pack_item_end":
                it = str(ev.get("item") or "")
                try:
                    per_item_elapsed[it] = float(ev.get("elapsed_s"))
                except Exception:
                    continue

        pack_total_elapsed: Optional[float] = None
        if per_item_elapsed:
            try:
                pack_total_elapsed = float(sum(float(v) for v in per_item_elapsed.values()))
            except Exception:
                pack_total_elapsed = None

        probe_final: Optional[Dict[str, Any]] = None
        probe_stages: List[Dict[str, Any]] = []
        comparison_summary: Optional[Dict[str, Any]] = None
        for ev in ordered:
            name = str(ev.get("event") or "")
            if name == "probe_compare":
                probe_final = {
                    "status": str(ev.get("status") or "ok"),
                    "reason": str(ev.get("reason") or ""),
                    "top_n": int(ev.get("top_n") or 0),
                    "js_divergence": _as_float(ev.get("js_divergence")),
                    "js_divergence_mode": str(ev.get("js_divergence_mode") or ""),
                    "overlap_count": int(ev.get("overlap_count") or 0),
                    "jaccard": _as_float(ev.get("jaccard")),
                    "mover_count": int(ev.get("mover_count") or 0),
                    "entered_count": int(ev.get("entered_count") or 0),
                    "exited_count": int(ev.get("exited_count") or 0),
                    "top1_before": _token_label(ev.get("top1_before")),
                    "top1_after": _token_label(ev.get("top1_after")),
                    "movers": ev.get("movers") if isinstance(ev.get("movers"), list) else [],
                    "entered": ev.get("entered") if isinstance(ev.get("entered"), list) else [],
                    "exited": ev.get("exited") if isinstance(ev.get("exited"), list) else [],
                }
            elif name == "probe_compare_stage":
                probe_stages.append(
                    {
                        "status": str(ev.get("status") or "ok"),
                        "reason": str(ev.get("reason") or ""),
                        "source": str(ev.get("source") or ""),
                        "band_label": str(ev.get("band_label") or ""),
                        "band_ponder_ix": ev.get("band_ponder_ix"),
                        "hop_ix": ev.get("hop_ix"),
                        "stage_ix": ev.get("stage_ix"),
                        "ponder_ix": ev.get("ponder_ix"),
                        "ponder_mode": str(ev.get("ponder_mode") or ""),
                        "memory_chars": int(ev.get("memory_chars") or 0),
                        "prompt_chars": int(ev.get("prompt_chars") or 0),
                        "top_n": int(ev.get("top_n") or 0),
                        "js_divergence": _as_float(ev.get("js_divergence")),
                        "prev_js_divergence": _as_float(ev.get("prev_js_divergence")),
                        "js_divergence_mode": str(ev.get("js_divergence_mode") or ""),
                        "overlap_count": int(ev.get("overlap_count") or 0),
                        "jaccard": _as_float(ev.get("jaccard")),
                        "mover_count": int(ev.get("mover_count") or 0),
                        "entered_count": int(ev.get("entered_count") or 0),
                        "exited_count": int(ev.get("exited_count") or 0),
                        "top1_before": _token_label(ev.get("top1_before")),
                        "top1_after": _token_label(ev.get("top1_after")),
                    }
                )
            elif name == "run_comparison":
                comp = ev.get("comparison")
                if isinstance(comp, dict):
                    comparison_summary = comp

        probe_summary: Optional[Dict[str, Any]] = None
        if probe_final or probe_stages:
            stage_js = [x for x in (_as_float(ev.get("js_divergence")) for ev in probe_stages) if x is not None]
            prev_stage_js = [x for x in (_as_float(ev.get("prev_js_divergence")) for ev in probe_stages) if x is not None]
            probe_summary = {
                "final": probe_final,
                "stages": probe_stages,
                "stage_count": int(len(probe_stages)),
                "stage_max_js": max(stage_js) if stage_js else None,
                "stage_last_js": stage_js[-1] if stage_js else None,
                "stage_max_prev_js": max(prev_stage_js) if prev_stage_js else None,
            }

        out["sessions"].append(
            {
                "session_id": sid,
                "ts_min": ts_min.isoformat() if ts_min else "",
                "ts_max": ts_max.isoformat() if ts_max else "",
                "duration_s": duration_s,
                "counts": dict(counts),
                "pack": {
                    "id": pack_id,
                    "items": pack_items,
                    "skips": pack_skips,
                    "total_elapsed_s": pack_total_elapsed,
                    "per_item_elapsed_s": per_item_elapsed,
                }
                if pack_id or ("pack_start" in counts)
                else None,
                "comparison": comparison_summary,
                "probe_compare": probe_summary,
                "events": ordered,
            }
        )
    return out


def render_html(report: Dict[str, Any]) -> str:
    sessions = report.get("sessions") or []
    parts: List[str] = []
    parts.append("<!doctype html>")
    parts.append("<html><head><meta charset='utf-8'/>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'/>")
    parts.append("<title>sr_ponder trace report</title>")
    parts.append(
        "<style>"
        "body{font-family:ui-monospace,Consolas,Menlo,monospace;margin:16px;}"
        ".muted{color:#666}"
        ".row{display:flex;flex-wrap:wrap;gap:10px;align-items:center}"
        ".filters{margin:10px 0;padding:8px 10px;border:1px solid #eee;border-radius:10px;background:#fafafa}"
        ".filters label{margin-right:10px;white-space:nowrap}"
        ".card{margin:12px 0;padding:10px 12px;border:1px solid #ddd;border-radius:10px;background:#fcfcfc}"
        "table{border-collapse:collapse;width:100%;margin:12px 0;}"
        "th,td{border:1px solid #ddd;padding:6px 8px;vertical-align:top;}"
        "th{background:#f6f6f6;text-align:left;position:sticky;top:0;z-index:2}"
        ".pill{display:inline-block;padding:2px 8px;border-radius:999px;background:#eee;margin-right:6px;}"
        ".ev-session_start{background:#eef7ff}"
        ".ev-session_end{background:#f0fff0}"
        ".ev-pack_item_skip{background:#fff8e1}"
        "</style>"
    )
    parts.append("</head><body>")
    parts.append(f"<h1>sr_ponder trace report</h1>")
    parts.append(f"<div class='muted'>trace: <code>{_esc(report.get('path'))}</code></div>")

    if sessions:
        parts.append("<h2>sessions</h2>")
        parts.append("<ul>")
        for sess in sessions:
            sid = sess.get("session_id")
            dur = sess.get("duration_s")
            dur_s = f"{float(dur):.2f}s" if isinstance(dur, (int, float)) else ""
            parts.append(
                f"<li><a href='#{_esc(_slug(str(sid)))}'><code>{_esc(sid)}</code></a>"
                + (f" <span class='muted'>({dur_s})</span>" if dur_s else "")
                + "</li>"
            )
        parts.append("</ul>")

    for sess in sessions:
        sid = sess.get("session_id")
        sid_slug = _slug(str(sid))
        parts.append(f"<h2 id='{_esc(sid_slug)}'>session <code>{_esc(sid)}</code></h2>")
        ts_min = sess.get("ts_min") or ""
        ts_max = sess.get("ts_max") or ""
        dur = sess.get("duration_s")
        dur_s = f"{float(dur):.2f}s" if isinstance(dur, (int, float)) else ""
        if ts_min or ts_max or dur_s:
            bits: List[str] = []
            if ts_min or ts_max:
                bits.append(f"range: {_esc(ts_min)} → {_esc(ts_max)}")
            if dur_s:
                bits.append(f"duration: {dur_s}")
            parts.append(f"<div class='muted'>{' | '.join(bits)}</div>")

        counts = sess.get("counts") or {}
        if counts:
            pills = " ".join(
                f"<span class='pill'>{_esc(k)}: {_esc(v)}</span>"
                for k, v in sorted(counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))
            )
            parts.append(f"<div class='row'>{pills}</div>")

            # Per-session event filter.
            parts.append(f"<div class='filters' data-session='{_esc(sid_slug)}'>")
            parts.append("<div class='muted'>filter events:</div>")
            for k, _v in sorted(counts.items(), key=lambda kv: kv[0]):
                k2 = str(k)
                ks = _slug(k2)
                parts.append(
                    "<label>"
                    f"<input type='checkbox' checked data-session='{_esc(sid_slug)}' data-event='{_esc(ks)}'/>"
                    f" {_esc(k2)}"
                    "</label>"
                )
            parts.append("</div>")

        pack = sess.get("pack")
        if isinstance(pack, dict):
            parts.append("<h3>pack</h3>")
            parts.append("<ul>")
            parts.append(f"<li>id: <code>{_esc(pack.get('id'))}</code></li>")
            tel = pack.get("total_elapsed_s")
            tel_s = f"{float(tel):.2f}s" if isinstance(tel, (int, float)) else ""
            parts.append(
                f"<li>items: {_esc(pack.get('items'))} (skips: {_esc(pack.get('skips'))})"
                + (f" | sum elapsed: {tel_s}" if tel_s else "")
                + "</li>"
            )
            per = pack.get("per_item_elapsed_s") or {}
            if isinstance(per, dict) and per:
                parts.append("<li>per-item elapsed_s:</li>")
                parts.append("<ul>")
                for k, v in sorted(per.items(), key=lambda kv: (-float(kv[1]), kv[0])):
                    parts.append(f"<li><code>{_esc(k)}</code>: {_esc(v)}</li>")
                parts.append("</ul>")
            parts.append("</ul>")

        comparison = sess.get("comparison")
        if isinstance(comparison, dict):
            parts.append("<h3>comparison</h3>")
            parts.append("<div class='card'>")
            diff_ratio = comparison.get("diff_ratio")
            diff_ratio_s = f"{float(diff_ratio):.4f}" if isinstance(diff_ratio, (int, float)) else "?"
            parts.append(
                "<div>"
                f"answers_changed={_esc(comparison.get('answer_changed'))}"
                f" | chars={_esc(comparison.get('baseline_chars'))}→{_esc(comparison.get('ponder_chars'))}"
                f" | diff={_esc(diff_ratio_s)}"
                "</div>"
            )
            semantic = comparison.get("semantic")
            if isinstance(semantic, dict) and str(semantic.get("status") or "") == "ok":
                answer_cos_s = _fmt_float(semantic.get("answer_cosine"), 6)
                query_base_s = _fmt_float(semantic.get("query_baseline_cosine"), 6)
                query_ponder_s = _fmt_float(semantic.get("query_ponder_cosine"), 6)
                align_delta_s = _fmt_float(semantic.get("query_alignment_delta"), 6)
                parts.append(
                    "<div>"
                    f"semantic[{_esc(semantic.get('method') or '')}]"
                    f" answer_cos={_esc(answer_cos_s)}"
                    f" | query_base={_esc(query_base_s)}"
                    f" | query_ponder={_esc(query_ponder_s)}"
                    f" | align_delta={_esc(align_delta_s)}"
                    "</div>"
                )
            stance = comparison.get("stance")
            if isinstance(stance, dict) and str(stance.get("status") or "") == "ok":
                shift_score = stance.get("shift_score")
                shift_score_s = f"{float(shift_score):.6f}" if isinstance(shift_score, (int, float)) else "?"
                parts.append(
                    "<div>"
                    f"stance[{_esc(stance.get('method') or '')}]"
                    f" { _esc(stance.get('dominant_baseline') or '') } → { _esc(stance.get('dominant_ponder') or '') }"
                    f" | shift={_esc(shift_score_s)}"
                    f" | gain={_esc(stance.get('top_gain') or '')}"
                    f" | drop={_esc(stance.get('top_drop') or '')}"
                    "</div>"
                )
            spatial = comparison.get("spatial_metaphor")
            if isinstance(spatial, dict) and str(spatial.get("status") or "") == "ok":
                baseline = spatial.get("baseline") if isinstance(spatial.get("baseline"), dict) else {}
                ponder = spatial.get("ponder") if isinstance(spatial.get("ponder"), dict) else {}
                questions = spatial.get("questions") if isinstance(spatial.get("questions"), dict) else {}
                logs = spatial.get("logs") if isinstance(spatial.get("logs"), dict) else {}
                baseline_density_s = _fmt_float(baseline.get("density_per_1k_chars"), 3)
                ponder_density_s = _fmt_float(ponder.get("density_per_1k_chars"), 3)
                log_density_s = _fmt_float(logs.get("density_per_1k_chars"), 3)
                question_density_s = _fmt_float(questions.get("density_per_1k_chars"), 3)
                parts.append(
                    "<div>"
                    f"spatial[{_esc(spatial.get('method') or '')}]"
                    f" ans={_esc(baseline_density_s)}→{_esc(ponder_density_s)}"
                    f" | logs={_esc(log_density_s)}"
                    f" | questions={_esc(question_density_s)}"
                    f" | groups={_esc(baseline.get('dominant_group') or 'none')}→{_esc(ponder.get('dominant_group') or 'none')}"
                    f" log={_esc(logs.get('dominant_group') or 'none')}"
                    "</div>"
                )
            parts.append("</div>")

        probe = sess.get("probe_compare")
        if isinstance(probe, dict):
            parts.append("<h3>probe compare</h3>")
            parts.append("<div class='card'>")
            final = probe.get("final")
            if isinstance(final, dict):
                js = final.get("js_divergence")
                js_s = f"{float(js):.6f}" if isinstance(js, (int, float)) else "?"
                jac = final.get("jaccard")
                jac_s = f"{float(jac):.3f}" if isinstance(jac, (int, float)) else "?"
                parts.append(
                    "<div>"
                    f"final: js={_esc(js_s)}"
                    f" | mode={_esc(final.get('js_divergence_mode') or '')}"
                    f" | overlap={_esc(final.get('overlap_count'))}"
                    f" | jaccard={_esc(jac_s)}"
                    f" | movers={_esc(final.get('mover_count'))}"
                    "</div>"
                )
                t1b = str(final.get("top1_before") or "").strip()
                t1a = str(final.get("top1_after") or "").strip()
                if t1b or t1a:
                    parts.append(f"<div>top1: <code>{_esc(t1b)}</code> → <code>{_esc(t1a)}</code></div>")
                if str(final.get("status") or "ok") != "ok":
                    parts.append(f"<div class='muted'>status: {_esc(final.get('status'))} {_esc(final.get('reason') or '')}</div>")

            stage_count = int(probe.get("stage_count") or 0)
            if stage_count > 0:
                stage_max_js = probe.get("stage_max_js")
                stage_last_js = probe.get("stage_last_js")
                stage_max_prev_js = probe.get("stage_max_prev_js")
                stage_max_js_s = f"{float(stage_max_js):.6f}" if isinstance(stage_max_js, (int, float)) else "?"
                stage_last_js_s = f"{float(stage_last_js):.6f}" if isinstance(stage_last_js, (int, float)) else "?"
                stage_max_prev_js_s = f"{float(stage_max_prev_js):.6f}" if isinstance(stage_max_prev_js, (int, float)) else "?"
                parts.append(
                    "<div>"
                    f"stages: {_esc(stage_count)}"
                    f" | max js(base): {_esc(stage_max_js_s)}"
                    f" | last js(base): {_esc(stage_last_js_s)}"
                    f" | max js(prev): {_esc(stage_max_prev_js_s)}"
                    "</div>"
                )
            parts.append("</div>")

            stages = probe.get("stages") or []
            if isinstance(stages, list) and stages:
                parts.append("<table>")
                parts.append(
                    "<thead><tr>"
                    "<th>band</th><th>hop</th><th>stage</th><th>mode</th><th>js(base)</th><th>js(prev)</th>"
                    "<th>top1</th><th>movers</th><th>chars</th><th>status</th>"
                    "</tr></thead>"
                )
                parts.append("<tbody>")
                for ev in stages:
                    if not isinstance(ev, dict):
                        continue
                    js = ev.get("js_divergence")
                    js_s = f"{float(js):.6f}" if isinstance(js, (int, float)) else ""
                    pjs = ev.get("prev_js_divergence")
                    pjs_s = f"{float(pjs):.6f}" if isinstance(pjs, (int, float)) else ""
                    top1b = str(ev.get("top1_before") or "").strip()
                    top1a = str(ev.get("top1_after") or "").strip()
                    top1_s = f"{top1b} → {top1a}".strip(" →")
                    chars_s = f"mem={int(ev.get('memory_chars') or 0)} prompt={int(ev.get('prompt_chars') or 0)}"
                    status_s = str(ev.get("status") or "ok")
                    if str(ev.get("reason") or "").strip():
                        status_s += f" | {str(ev.get('reason') or '').strip()}"
                    parts.append(
                        "<tr>"
                        f"<td><code>{_esc(ev.get('band_label') or '')}</code></td>"
                        f"<td><code>{_esc(ev.get('hop_ix'))}</code></td>"
                        f"<td><code>{_esc(ev.get('stage_ix'))}</code></td>"
                        f"<td><code>{_esc(ev.get('ponder_mode') or '')}</code></td>"
                        f"<td><code>{_esc(js_s)}</code></td>"
                        f"<td><code>{_esc(pjs_s)}</code></td>"
                        f"<td><code>{_esc(top1_s)}</code></td>"
                        f"<td><code>{_esc(ev.get('mover_count'))}</code></td>"
                        f"<td><code>{_esc(chars_s)}</code></td>"
                        f"<td><code>{_esc(status_s)}</code></td>"
                        "</tr>"
                    )
                parts.append("</tbody></table>")

        parts.append("<h3>events</h3>")
        parts.append("<table>")
        parts.append("<thead><tr><th>ts</th><th>event</th><th>item</th><th>elapsed_s</th><th>details</th></tr></thead>")
        parts.append("<tbody>")
        for ev in sess.get("events") or []:
            if not isinstance(ev, dict):
                continue
            ts = ev.get("ts")
            name = ev.get("event")
            name_s = str(name or "")
            name_slug = _slug(name_s)
            item = ev.get("item") or ev.get("pack_item") or ""
            elapsed = ev.get("elapsed_s") or ""
            # Keep details compact; drop large fields.
            ev2 = dict(ev)
            for k in ("ts", "session_id", "event"):
                ev2.pop(k, None)
            details = json.dumps(ev2, ensure_ascii=False, sort_keys=True)
            if len(details) > 1600:
                details = details[:1597] + "..."
            parts.append(
                f"<tr data-session='{_esc(sid_slug)}' data-event='{_esc(name_slug)}' class='ev-{_esc(name_slug)}'>"
                f"<td><code>{_esc(ts)}</code></td>"
                f"<td><code>{_esc(name_s)}</code></td>"
                f"<td><code>{_esc(item)}</code></td>"
                f"<td><code>{_esc(elapsed)}</code></td>"
                f"<td><code>{_esc(details)}</code></td>"
                "</tr>"
            )
        parts.append("</tbody></table>")

    parts.append(
        "<script>"
        "document.addEventListener('change', (ev) => {"
        "  const t = ev.target;"
        "  if (!t || !t.matches('input[data-session][data-event]')) return;"
        "  const sid = t.getAttribute('data-session');"
        "  const en = t.getAttribute('data-event');"
        "  const rows = document.querySelectorAll(`tr[data-session=\"${sid}\"][data-event=\"${en}\"]`);"
        "  for (const r of rows) { r.style.display = t.checked ? '' : 'none'; }"
        "});"
        "</script>"
    )
    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="Trace JSONL file (from --trace_out)")
    ap.add_argument("--out", default="", help="Optional HTML output path")
    ap.add_argument("--max_records", type=int, default=0, help="Max JSONL records to read (0=all)")
    ap.add_argument("--session_id", default="", help="Filter to one session_id")
    args = ap.parse_args()

    trace_path = Path(str(args.trace))
    report = analyze_trace(trace_path, max_records=int(args.max_records or 0), session_id=str(args.session_id or ""))

    out_s = str(args.out or "").strip()
    if out_s:
        out_path = Path(out_s)
        out_path.write_text(render_html(report), encoding="utf-8")
        print(f"[sr_trace_report] wrote {out_path}")
        return

    # Plain text fallback: list sessions + counts.
    sessions = report.get("sessions") or []
    print(f"trace: {report.get('path')} sessions={len(sessions)}")
    for s in sessions:
        sid = s.get("session_id")
        rng = f"{s.get('ts_min') or ''} .. {s.get('ts_max') or ''}".strip()
        counts = s.get("counts") or {}
        top = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:8])
        probe = s.get("probe_compare") or {}
        probe_s = ""
        if isinstance(probe, dict):
            final = probe.get("final") or {}
            js = final.get("js_divergence") if isinstance(final, dict) else None
            stage_count = probe.get("stage_count")
            if isinstance(js, (int, float)) or isinstance(stage_count, int):
                js_s = f"{float(js):.4f}" if isinstance(js, (int, float)) else "?"
                sc_s = str(int(stage_count)) if isinstance(stage_count, int) else "0"
                probe_s = f"  probe(js={js_s}, stages={sc_s})"
        comp = s.get("comparison") or {}
        comp_s = ""
        if isinstance(comp, dict):
            stance = comp.get("stance") or {}
            spatial = comp.get("spatial_metaphor") or {}
            stance_bits: List[str] = []
            if isinstance(stance, dict) and str(stance.get("status") or "") == "ok":
                stance_bits.append(f"stance={stance.get('dominant_baseline') or ''}->{stance.get('dominant_ponder') or ''}")
            if isinstance(spatial, dict) and str(spatial.get("status") or "") == "ok":
                log_density = ((spatial.get("logs") or {}).get("density_per_1k_chars") if isinstance(spatial.get("logs"), dict) else None)
                if isinstance(log_density, (int, float)):
                    stance_bits.append(f"spatial_log={float(log_density):.3f}")
            if stance_bits:
                comp_s = "  " + " ".join(stance_bits)
        print(f"- {sid}  {rng}  {top}{probe_s}{comp_s}")


if __name__ == "__main__":
    main()
