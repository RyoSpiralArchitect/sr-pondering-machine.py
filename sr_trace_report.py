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

        out["sessions"].append(
            {
                "session_id": sid,
                "ts_min": ts_min.isoformat() if ts_min else "",
                "ts_max": ts_max.isoformat() if ts_max else "",
                "counts": dict(counts),
                "pack": {
                    "id": pack_id,
                    "items": pack_items,
                    "skips": pack_skips,
                    "per_item_elapsed_s": per_item_elapsed,
                }
                if pack_id or ("pack_start" in counts)
                else None,
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
        "table{border-collapse:collapse;width:100%;margin:12px 0;}"
        "th,td{border:1px solid #ddd;padding:6px 8px;vertical-align:top;}"
        "th{background:#f6f6f6;text-align:left;}"
        ".pill{display:inline-block;padding:2px 8px;border-radius:999px;background:#eee;margin-right:6px;}"
        "</style>"
    )
    parts.append("</head><body>")
    parts.append(f"<h1>sr_ponder trace report</h1>")
    parts.append(f"<div class='muted'>trace: <code>{_esc(report.get('path'))}</code></div>")

    for sess in sessions:
        sid = sess.get("session_id")
        parts.append(f"<h2>session <code>{_esc(sid)}</code></h2>")
        ts_min = sess.get("ts_min") or ""
        ts_max = sess.get("ts_max") or ""
        if ts_min or ts_max:
            parts.append(f"<div class='muted'>range: { _esc(ts_min) } → { _esc(ts_max) }</div>")

        counts = sess.get("counts") or {}
        if counts:
            pills = " ".join(
                f"<span class='pill'>{_esc(k)}: {_esc(v)}</span>"
                for k, v in sorted(counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))
            )
            parts.append(f"<div>{pills}</div>")

        pack = sess.get("pack")
        if isinstance(pack, dict):
            parts.append("<h3>pack</h3>")
            parts.append("<ul>")
            parts.append(f"<li>id: <code>{_esc(pack.get('id'))}</code></li>")
            parts.append(f"<li>items: {_esc(pack.get('items'))} (skips: {_esc(pack.get('skips'))})</li>")
            per = pack.get("per_item_elapsed_s") or {}
            if isinstance(per, dict) and per:
                parts.append("<li>per-item elapsed_s:</li>")
                parts.append("<ul>")
                for k, v in sorted(per.items(), key=lambda kv: (-float(kv[1]), kv[0])):
                    parts.append(f"<li><code>{_esc(k)}</code>: {_esc(v)}</li>")
                parts.append("</ul>")
            parts.append("</ul>")

        parts.append("<h3>events</h3>")
        parts.append("<table>")
        parts.append("<thead><tr><th>ts</th><th>event</th><th>item</th><th>elapsed_s</th><th>details</th></tr></thead>")
        parts.append("<tbody>")
        for ev in sess.get("events") or []:
            if not isinstance(ev, dict):
                continue
            ts = ev.get("ts")
            name = ev.get("event")
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
                "<tr>"
                f"<td><code>{_esc(ts)}</code></td>"
                f"<td><code>{_esc(name)}</code></td>"
                f"<td><code>{_esc(item)}</code></td>"
                f"<td><code>{_esc(elapsed)}</code></td>"
                f"<td><code>{_esc(details)}</code></td>"
                "</tr>"
            )
        parts.append("</tbody></table>")

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
        print(f"- {sid}  {rng}  {top}")


if __name__ == "__main__":
    main()
