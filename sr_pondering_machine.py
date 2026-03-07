#!/usr/bin/env python3
"""sr_pondering_machine.py

MPS-first + Gemma-turn-format aware variant of the minimal “pondering machine”.

This version fixes 2 common failure modes you just hit on Apple Silicon:

1) MPS device mismatch
   - Ensures *all* model inputs live on the same device as the input-embedding weights.

2) Gemma prompt formatting / endless transcript loops
   - Gemma instruction-tuned models expect the special turn tokens:
       <start_of_turn>user ... <end_of_turn>\n<start_of_turn>model
     and typically stop at <end_of_turn>.
   - If the tokenizer does not provide a chat template (or provides one that’s not ideal),
     we still format prompts in the Gemma-native way whenever the tokens exist.
   - We also stop generation at the first <end_of_turn> token if available.

Usage:
  python3 sr_pondering_machine.py \
    --model ./model/gemma-3-270m-it \
    --query "量子もつれを高校生にも分かるように説明して" \
    --mode both \
    --memory ./ponder_logs.jsonl

Notes:
  - If you’re using the *base* (non-IT) Gemma variant, don’t expect good instruction-following.
    Prefer the “-it” models.
  - If you hit missing MPS ops:
      export PYTORCH_ENABLE_MPS_FALLBACK=1
"""

from __future__ import annotations

import argparse
from collections import deque
import dataclasses
import datetime as dt
import difflib
import hashlib
import json
import math
import os
import random
import re
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Optional heavy deps (allow --backend openai_compat to work without torch/transformers installed).
try:  # pragma: no cover - environment dependent
    import torch  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore

try:  # pragma: no cover - environment dependent
    import transformers  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    transformers = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

from urllib import error as urlerror
from urllib import request as urlrequest


# -----------------------------
# Utilities
# -----------------------------


def _inference_mode() -> Any:
    """Decorator/context wrapper for torch.inference_mode() that becomes a no-op when torch is missing.

    This keeps the module importable for API-only usage (openai_compat backend).
    """

    if torch is None:
        def _noop(fn: Any) -> Any:
            return fn

        return _noop
    return torch.inference_mode()


def _expand_path_str(s: str) -> str:
    """Expand env vars and ~ in filesystem paths. Keeps '-' intact for stream-style destinations."""

    t = str(s or "").strip()
    if not t:
        return ""
    if t == "-":
        return t
    try:
        t = os.path.expandvars(t)
    except Exception:
        pass
    try:
        return str(Path(t).expanduser())
    except Exception:
        return t

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_iso() -> str:
    # timezone-aware UTC timestamp (avoids datetime.utcnow() deprecation)
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def now_slug() -> str:
    # Filesystem-friendly UTC timestamp (Windows-safe).
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%SZ")


_JA_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")  # hiragana/katakana/CJK


def is_japanese(text: str) -> bool:
    return _JA_RE.search(text or "") is not None


def resolve_prompt_lang(prompt_lang: str, query: str) -> str:
    if prompt_lang in ("en", "ja"):
        return prompt_lang
    return "ja" if is_japanese(query) else "en"


def sha256_short(s: str, n: int = 12) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:n]


def stable_hash_mod(s: str, mod: int) -> int:
    mm = max(1, int(mod))
    digest = hashlib.sha256((s or "").encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % mm


def make_run_id(seed: int, query: str) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")
    nonce = secrets.token_hex(4)
    return sha256_short(f"{stamp}|pid={os.getpid()}|nonce={nonce}|seed={seed}|query={query}", n=16)


def safe_mkdir(p: Path) -> None:
    # For "trace.jsonl" / "run.json", `p.parent` becomes "." (current directory),
    # which is safe to mkdir with exist_ok=True.
    p.parent.mkdir(parents=True, exist_ok=True)


def _jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if dataclasses.is_dataclass(x):
        return _jsonable(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, set):
        return [_jsonable(v) for v in sorted(list(x), key=lambda z: str(z))]
    return str(x)


def write_json(path: Path, payload: Any) -> None:
    safe_mkdir(path)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_json_dest(dest: str, payload: Any, *, stream: Optional[Any] = None) -> Optional[Path]:
    """Write a JSON payload to a destination path.

    - dest == "-" writes to the provided stream (default stdout).
    - otherwise writes to a file (creating parent dirs via safe_mkdir).
    """

    d = _expand_path_str(dest)
    if not d:
        return None
    if d == "-":
        out = stream if stream is not None else sys.stdout
        try:
            out.write(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n")
        except Exception:
            pass
        return None
    path = Path(d)
    write_json(path, payload)
    return path


def write_text_dest(dest: str, text: str) -> Optional[Path]:
    """Write a UTF-8 text payload to a destination path.

    - dest == "-" is intentionally treated as disabled (return None) because
      this CLI prints normal output to stdout; mixing HTML with stdout is easy to
      footgun. Use a file path.
    - otherwise writes to a file (creating parent dirs via safe_mkdir).
    """

    d = _expand_path_str(dest)
    if not d:
        return None
    if d == "-":
        return None
    path = Path(d)
    safe_mkdir(path)
    path.write_text(str(text), encoding="utf-8")
    return path


def maybe_write_trace_report(
    *,
    trace: Optional[Any],
    dest: str,
    session_id: str,
    max_records: int = 0,
    session_filter: str = "",
) -> Optional[Path]:
    """Optionally write an HTML trace report for the current session."""

    d = _expand_path_str(str(dest or ""))
    if not d or d == "-":
        return None
    if trace is None:
        return None
    tp_s = str(getattr(trace, "path", "") or "").strip()
    if (not tp_s) or tp_s == "-":
        return None
    tp = Path(tp_s)
    if not tp.exists():
        return None
    try:
        import sr_trace_report as tr
    except Exception:
        return None
    try:
        sid = str(session_filter or session_id or "").strip()
        report = tr.analyze_trace(tp, max_records=int(max_records or 0), session_id=sid)
        html_s = tr.render_html(report)
    except Exception:
        return None
    return write_text_dest(d, html_s)


_REDACTED = "***REDACTED***"
_SENSITIVE_HEADER_KEYS = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "x-auth-token",
    "x-authorization",
}


_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify_filename(s: str, *, max_len: int = 64) -> str:
    t = _FILENAME_SAFE_RE.sub("-", str(s or "").strip())
    t = re.sub(r"-{2,}", "-", t).strip("-")
    if not t:
        return "run"
    return t[: max(8, int(max_len))]


def _redact_header_line(line: str) -> str:
    s = str(line or "").strip()
    if not s:
        return s
    if ":" not in s:
        return _REDACTED if "bearer " in s.lower() else s
    k, v = s.split(":", 1)
    k2 = k.strip()
    v2 = v.strip()
    if k2.lower() in _SENSITIVE_HEADER_KEYS:
        return f"{k2}: {_REDACTED}"
    if "bearer " in v2.lower():
        return f"{k2}: {_REDACTED}"
    return s


def sanitize_cfg_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d or {})
    if out.get("api_key"):
        out["api_key"] = _REDACTED
    hdrs = out.get("api_headers")
    if isinstance(hdrs, list):
        out["api_headers"] = [_redact_header_line(str(x)) for x in hdrs]
    return out


_CONTROL_VARIANTS = ("none", "no_inject", "random_log", "random_keywords", "lens_only")


def load_pack_file(path: Path) -> Tuple[str, List[Tuple[str, Dict[str, Any]]]]:
    try:
        raw = path.read_text(encoding="utf-8-sig")
        data = json.loads(raw)
    except Exception as e:
        raise SystemExit(f"[sr_ponder] ERROR: failed to read --pack_file {str(path)!r}: {e}")

    if not isinstance(data, dict):
        raise SystemExit(
            f"[sr_ponder] ERROR: --pack_file must be a JSON object (dict), got {type(data).__name__}"
        )

    name = str(data.get("name") or "").strip() or path.stem

    base_cfg = data.get("base_cfg") or {}
    if base_cfg and not isinstance(base_cfg, dict):
        raise SystemExit(
            f"[sr_ponder] ERROR: --pack_file base_cfg must be an object (dict), got {type(base_cfg).__name__}"
        )

    items_raw = data.get("items")
    if not isinstance(items_raw, list) or not items_raw:
        raise SystemExit("[sr_ponder] ERROR: --pack_file must include a non-empty 'items' array")

    allowed_controls = set(_CONTROL_VARIANTS)
    items: List[Tuple[str, Dict[str, Any]]] = []
    for ix, it in enumerate(items_raw):
        if not isinstance(it, dict):
            raise SystemExit(
                f"[sr_ponder] ERROR: --pack_file items[{ix}] must be an object (dict), got {type(it).__name__}"
            )
        item_name = str(it.get("name") or "").strip()
        if not item_name:
            raise SystemExit(f"[sr_ponder] ERROR: --pack_file items[{ix}].name is required")

        kind = str(it.get("kind") or "ponder").strip().lower()
        if kind not in ("baseline", "ponder"):
            raise SystemExit(
                f"[sr_ponder] ERROR: --pack_file items[{ix}].kind must be 'baseline'|'ponder', got {kind!r}"
            )

        control = str(it.get("control") or "none").strip()
        if kind == "ponder" and control not in allowed_controls:
            raise SystemExit(
                f"[sr_ponder] ERROR: --pack_file items[{ix}].control must be one of {sorted(list(allowed_controls))}, got {control!r}"
            )

        cfg = it.get("cfg") or {}
        if cfg and not isinstance(cfg, dict):
            raise SystemExit(
                f"[sr_ponder] ERROR: --pack_file items[{ix}].cfg must be an object (dict), got {type(cfg).__name__}"
            )

        cfg_merged: Dict[str, Any] = dict(base_cfg or {})
        cfg_merged.update(dict(cfg or {}))

        spec: Dict[str, Any] = {"kind": kind}
        if kind == "ponder":
            spec["control"] = control
        if cfg_merged:
            spec["cfg"] = cfg_merged
        qv = it.get("query")
        if qv is not None:
            spec["query"] = str(qv)
        items.append((item_name, spec))

    return name, items


class TraceWriter:
    def __init__(self, path: Path, *, session_id: str, preview_chars: int = 0) -> None:
        self.path = path
        self.session_id = str(session_id)
        self.preview_chars = max(0, int(preview_chars))
        safe_mkdir(path)

    def preview(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        n = int(self.preview_chars)
        if n <= 0:
            return None
        t = str(text).strip()
        if len(t) <= n:
            return t
        return t[:n].rstrip() + "…"

    def event(self, name: str, /, **fields: Any) -> None:
        rec: Dict[str, Any] = {"ts": now_iso(), "session_id": self.session_id, "event": str(name)}
        rec.update(fields)
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_jsonable(rec), ensure_ascii=False) + "\n")
        except Exception:
            pass


class StreamTraceWriter:
    """TraceWriter that writes JSONL events to a stream (e.g. stderr) instead of a file."""

    def __init__(self, stream: Any, *, session_id: str, preview_chars: int = 0, label: str = "-") -> None:
        self.path = Path(str(label or "-"))
        self.session_id = str(session_id)
        self.preview_chars = max(0, int(preview_chars))
        self._stream = stream

    def preview(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        n = int(self.preview_chars)
        if n <= 0:
            return None
        t = str(text).strip()
        if len(t) <= n:
            return t
        return t[:n].rstrip() + "…"

    def event(self, name: str, /, **fields: Any) -> None:
        rec: Dict[str, Any] = {"ts": now_iso(), "session_id": self.session_id, "event": str(name)}
        rec.update(fields)
        try:
            self._stream.write(json.dumps(_jsonable(rec), ensure_ascii=False) + "\n")
            try:
                self._stream.flush()
            except Exception:
                pass
        except Exception:
            pass


def compute_run_metrics(
    *,
    query: str,
    baseline_answer: Optional[str],
    ponder_answer: Optional[str],
    records: Sequence[Dict[str, Any]],
    extras: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metrics["query_chars"] = len(query or "")
    if baseline_answer is not None:
        metrics["baseline_chars"] = len(baseline_answer or "")
    if ponder_answer is not None:
        metrics["ponder_chars"] = len(ponder_answer or "")
    if baseline_answer and ponder_answer:
        try:
            ratio = difflib.SequenceMatcher(None, baseline_answer, ponder_answer).ratio()
            metrics["baseline_ponder_diff"] = float(1.0 - float(ratio))
        except Exception:
            pass

    metrics["records"] = int(len(list(records)))
    uniq: set[str] = set()
    max_hop = 0
    for r in records or []:
        hop_ix = r.get("hop_ix")
        if isinstance(hop_ix, int):
            max_hop = max(max_hop, hop_ix)
        kws = r.get("keywords")
        if isinstance(kws, list):
            for k in kws:
                kk = str(k or "").strip()
                if kk:
                    uniq.add(kk)
    metrics["unique_keywords"] = int(len(uniq))
    metrics["max_hop_ix"] = int(max_hop) if records else 0

    if isinstance(extras, dict):
        ms = extras.get("memory_selected")
        if isinstance(ms, list):
            metrics["memory_selected"] = int(len(ms))
        pc = extras.get("probe_compare")
        if isinstance(pc, dict):
            js = pc.get("js_divergence")
            if isinstance(js, (int, float)):
                metrics["probe_js_divergence"] = float(js)
            overlap = pc.get("overlap_count")
            if isinstance(overlap, int):
                metrics["probe_overlap_count"] = int(overlap)
            movers = pc.get("mover_count")
            if isinstance(movers, int):
                metrics["probe_mover_count"] = int(movers)
            if isinstance(pc.get("top1_changed"), bool):
                metrics["probe_top1_changed"] = bool(pc.get("top1_changed"))
        pcs = extras.get("probe_compare_stages")
        if isinstance(pcs, list):
            stage_js: List[float] = []
            for item in pcs:
                if not isinstance(item, dict):
                    continue
                comp = item.get("compare_from_base")
                if not isinstance(comp, dict):
                    continue
                js2 = comp.get("js_divergence")
                if isinstance(js2, (int, float)):
                    stage_js.append(float(js2))
            metrics["probe_stage_count"] = int(len(stage_js))
            if stage_js:
                metrics["probe_stage_max_js"] = float(max(stage_js))
                metrics["probe_stage_last_js"] = float(stage_js[-1])
    return metrics


def build_run_comparison(
    *,
    query: str,
    baseline_answer: Optional[str],
    ponder_answer: Optional[str],
    records: Sequence[Dict[str, Any]],
    extras: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics = compute_run_metrics(
        query=query,
        baseline_answer=baseline_answer,
        ponder_answer=ponder_answer,
        records=records,
        extras=extras,
    )
    out: Dict[str, Any] = {}
    if baseline_answer is not None:
        out["baseline_chars"] = len(baseline_answer or "")
    if ponder_answer is not None:
        out["ponder_chars"] = len(ponder_answer or "")
    if baseline_answer is not None and ponder_answer is not None:
        out["answer_changed"] = bool((baseline_answer or "") != (ponder_answer or ""))
        out["char_delta"] = int(len(ponder_answer or "") - len(baseline_answer or ""))
        diff_ratio = metrics.get("baseline_ponder_diff")
        if isinstance(diff_ratio, (int, float)):
            out["diff_ratio"] = float(diff_ratio)

    for key in (
        "records",
        "unique_keywords",
        "max_hop_ix",
        "memory_selected",
        "probe_js_divergence",
        "probe_overlap_count",
        "probe_mover_count",
        "probe_top1_changed",
        "probe_stage_count",
        "probe_stage_max_js",
        "probe_stage_last_js",
    ):
        val = metrics.get(key)
        if val is not None:
            out[key] = val

    if isinstance(extras, dict):
        api_warnings = extras.get("api_warnings")
        if isinstance(api_warnings, list):
            out["api_warnings_count"] = int(len(api_warnings))
    return out


def _fmt_metric(value: Any, *, digits: int = 3) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _print_run_comparison(comp: Dict[str, Any], *, mode: str) -> None:
    if not comp or mode == "none":
        return
    if mode == "json":
        print("\n=== COMPARISON ===\n")
        print(json.dumps(comp, ensure_ascii=False, indent=2))
        return

    print("\n=== COMPARISON ===\n")
    if "baseline_chars" in comp and "ponder_chars" in comp:
        diff_ratio = comp.get("diff_ratio")
        diff_s = _fmt_metric(diff_ratio, digits=4) if isinstance(diff_ratio, (int, float)) else "?"
        print(
            f"answers_changed={_fmt_metric(comp.get('answer_changed'))} "
            f"chars={int(comp.get('baseline_chars', 0))}->{int(comp.get('ponder_chars', 0))} "
            f"delta={int(comp.get('char_delta', 0)):+d} diff_ratio={diff_s}"
        )

    detail_parts: List[str] = []
    for key in ("records", "unique_keywords", "max_hop_ix", "memory_selected"):
        if key in comp:
            detail_parts.append(f"{key}={_fmt_metric(comp.get(key))}")
    if detail_parts:
        print(" ".join(detail_parts))

    probe_parts: List[str] = []
    for key in ("probe_js_divergence", "probe_overlap_count", "probe_mover_count", "probe_top1_changed"):
        if key in comp:
            probe_parts.append(f"{key}={_fmt_metric(comp.get(key), digits=6)}")
    for key in ("probe_stage_count", "probe_stage_max_js", "probe_stage_last_js"):
        if key in comp:
            probe_parts.append(f"{key}={_fmt_metric(comp.get(key), digits=6)}")
    if probe_parts:
        print(" ".join(probe_parts))

    if "api_warnings_count" in comp:
        print(f"api_warnings={_fmt_metric(comp.get('api_warnings_count'))}")


def _ponder_record_label(record: Dict[str, Any]) -> str:
    parts: List[str] = []
    band = str(record.get("band_label") or "").strip()
    if band:
        parts.append(f"band={band}")
    hop_ix = record.get("hop_ix")
    if isinstance(hop_ix, int):
        parts.append(f"hop={hop_ix}")
    stage_ix = record.get("pipeline_stage_ix")
    if isinstance(stage_ix, int):
        parts.append(f"stage={stage_ix}")
    mode = str(record.get("ponder_mode") or "").strip()
    if mode:
        parts.append(f"mode={mode}")
    return " ".join(parts) if parts else "ponder"


def _empty_ponder_log_reason(record: Dict[str, Any]) -> str:
    meta = record.get("api_generation")
    if not isinstance(meta, dict):
        return "[empty ponder log]"
    parts: List[str] = []
    finish = str(meta.get("finish_reason") or "").strip()
    if finish:
        parts.append(f"finish_reason={finish}")
    completion_tokens = meta.get("completion_tokens")
    if isinstance(completion_tokens, int):
        parts.append(f"completion_tokens={completion_tokens}")
    reasoning_tokens = meta.get("reasoning_tokens")
    if isinstance(reasoning_tokens, int):
        parts.append(f"reasoning_tokens={reasoning_tokens}")
    refusal = str(meta.get("refusal") or "").strip()
    if refusal:
        parts.append(f"refusal={refusal}")
    if not parts:
        return "[empty ponder log]"
    return "[empty ponder log] " + " ".join(parts)


def _print_ponder_logs(records: Sequence[Dict[str, Any]], *, mode: str) -> None:
    if mode == "none":
        return
    items = [x for x in records if isinstance(x, dict)]
    if not items:
        return
    limit = len(items) if mode == "full" else min(len(items), 2)
    print("\n=== PONDER LOGS ===\n")
    for ix, record in enumerate(items[:limit], start=1):
        print(f"--- {ix}. {_ponder_record_label(record)} ---")
        kws = record.get("keywords")
        if isinstance(kws, list) and kws:
            kws_s = ", ".join(str(x).strip() for x in kws if str(x).strip())
            if kws_s:
                print(f"keywords: {kws_s}")
        question = str(record.get("ponder_question") or "").strip()
        if question:
            print(f"question: {question}")
        log = str(record.get("ponder_log") or "").strip()
        if log:
            if mode == "auto" and len(log) > 1600:
                print(log[:1600].rstrip() + "…")
                print("[truncated; use --print_ponder full for full logs]")
            else:
                print(log)
        else:
            print(_empty_ponder_log_reason(record))
        if ix < limit:
            print("")
    if mode == "auto" and len(items) > limit:
        print(f"\n[info] showing {limit}/{len(items)} ponder logs (use --print_ponder full to expand)")


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    safe_mkdir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def tail_jsonl(path: Path, n: int) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in deque(f, maxlen=n):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


_MODEL_PATHLIKE_EXTS = (".json", ".safetensors", ".bin", ".gguf", ".pt", ".pth")


def _looks_like_local_path(s: str) -> bool:
    if not s:
        return False
    if s.startswith(("~", "./", "../")):
        return True
    if os.path.isabs(s):
        return True
    if re.match(r"^[A-Za-z]:[\\/]", s or ""):
        return True
    if any(s.endswith(ext) for ext in _MODEL_PATHLIKE_EXTS):
        return True
    if "/" in s or "\\" in s:
        first = re.split(r"[\\/]+", s, maxsplit=1)[0]
        if first and Path(first).exists():
            return True
    return False


def _normalize_model_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


_PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    "hf": {"backend": "hf"},
    "openai": {
        "backend": "openai_compat",
        "api_base_url": "https://api.openai.com/v1",
        "api_chat_path": "/chat/completions",
        "api_key_env": "OPENAI_API_KEY",
    },
    "mistral": {
        "backend": "openai_compat",
        "api_base_url": "https://api.mistral.ai/v1",
        "api_chat_path": "/chat/completions",
        "api_key_env": "MISTRAL_API_KEY",
    },
    "groq": {
        "backend": "openai_compat",
        "api_base_url": "https://api.groq.com/openai/v1",
        "api_chat_path": "/chat/completions",
        "api_key_env": "GROQ_API_KEY",
    },
    "openrouter": {
        "backend": "openai_compat",
        "api_base_url": "https://openrouter.ai/api/v1",
        "api_chat_path": "/chat/completions",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "deepseek": {
        "backend": "openai_compat",
        "api_base_url": "https://api.deepseek.com",
        "api_chat_path": "/chat/completions",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "custom": {"backend": "openai_compat"},
}


def infer_provider_name(*, backend: str, base_url: str) -> str:
    backend_s = str(backend or "hf").strip().lower()
    if backend_s == "hf":
        return "hf"
    s = (base_url or "").strip().lower()
    if "api.openai.com" in s:
        return "openai"
    if "api.mistral.ai" in s:
        return "mistral"
    if "api.groq.com" in s:
        return "groq"
    if "openrouter.ai" in s:
        return "openrouter"
    if "api.deepseek.com" in s:
        return "deepseek"
    return "custom"


def apply_provider_defaults_inplace(args: argparse.Namespace, *, explicit_dests: Sequence[str]) -> str:
    explicit = set(str(x) for x in explicit_dests)
    requested = str(getattr(args, "provider", "auto") or "auto").strip().lower()
    if requested in ("", "auto"):
        resolved = infer_provider_name(
            backend=str(getattr(args, "backend", "hf") or "hf"),
            base_url=str(getattr(args, "api_base_url", "") or ""),
        )
        args.provider = resolved
        return resolved

    preset = _PROVIDER_PRESETS.get(requested)
    if not isinstance(preset, dict):
        raise SystemExit(f"[sr_ponder] ERROR: unknown provider preset: {requested!r}")

    backend = str(preset.get("backend", getattr(args, "backend", "hf")) or "hf")
    args.backend = backend
    for key, value in preset.items():
        if key == "backend":
            continue
        if key not in explicit:
            setattr(args, key, value)
    args.provider = requested
    return requested


def _suggest_model_dirs(missing_path: Path, *, limit: int = 4) -> List[Path]:
    parent = missing_path.parent
    if not parent.is_dir():
        return []

    target = _normalize_model_name(missing_path.name)
    candidates: List[Tuple[float, Path]] = []
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        child_norm = _normalize_model_name(child.name)
        if not child_norm:
            continue
        if target and child_norm == target:
            score = 1.0
        else:
            score = difflib.SequenceMatcher(a=target, b=child_norm).ratio() if target else 0.0
        candidates.append((score, child))

    candidates.sort(key=lambda t: t[0], reverse=True)
    out: List[Path] = []
    for score, child in candidates:
        if len(out) >= limit:
            break
        if score < 0.45:
            continue
        out.append(child)
    return out


def resolve_model_ref(model_ref: str) -> str:
    """Resolve a model ref intended as a *local directory*.

    If the user passes something that looks like a local path but it doesn't exist,
    fail fast with a clearer message than the HF Hub validation error.

    If it doesn't look like a local path, return as-is (allows cached HF IDs).
    """

    expanded = os.path.expandvars(os.path.expanduser(model_ref or ""))
    p = Path(expanded)

    if p.exists():
        if not p.is_dir():
            raise SystemExit(f"[sr_ponder] ERROR: --model must be a directory, got: {p}")
        if not (p / "config.json").exists():
            snap_root = p / "snapshots"
            if snap_root.is_dir():
                candidates = [d for d in snap_root.iterdir() if d.is_dir() and (d / "config.json").exists()]
                if candidates:
                    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
                    return str(candidates[0].resolve())
            raise SystemExit(
                f"[sr_ponder] ERROR: not a HF model dir (missing config.json): {p}\n"
                "[sr_ponder] Tip: if this is a Hugging Face cache root, pass the snapshots/<hash> directory "
                "or the cache root itself and the newest snapshot will be auto-resolved."
            )
        return str(p.resolve())

    if _looks_like_local_path(model_ref) or _looks_like_local_path(expanded):
        suggestions = _suggest_model_dirs(p)
        lines = [f"[sr_ponder] ERROR: model path not found: {p}"]
        if suggestions:
            lines.append("[sr_ponder] Did you mean:")
            lines.extend([f"  - {s}" for s in suggestions])
        lines.append("[sr_ponder] Tip: `--model` expects a local Hugging Face model directory (offline).")
        raise SystemExit("\n".join(lines))

    return model_ref


def configure_transformers_allocator_warmup(mode: str) -> None:
    """Configure/patch Transformers `caching_allocator_warmup`.

    Transformers 5.x added `caching_allocator_warmup()` which pre-allocates a single fp16 buffer roughly the
    size of the model on accelerator devices. On Apple Silicon (MPS) this can fail with:

        RuntimeError: Invalid buffer size: XX GiB

    because Metal has stricter single-buffer size limits.

    Modes:
      - on: keep Transformers default behavior
      - off: disable warmup (safer, a bit slower to load)
      - auto: run warmup only for cuda/xpu, skip for mps/others
    """

    mode = (mode or "auto").strip().lower()
    if mode not in ("auto", "on", "off"):
        raise ValueError(f"allocator_warmup must be auto|on|off, got: {mode!r}")

    try:
        import transformers.modeling_utils as mu  # type: ignore
    except Exception:
        return

    if not hasattr(mu, "caching_allocator_warmup"):
        return

    # Store mode on the module so the patched function can read it.
    setattr(mu, "_sr_ponder_allocator_warmup_mode", mode)

    if getattr(mu, "_sr_ponder_allocator_warmup_patched", False):
        return

    orig = getattr(mu, "caching_allocator_warmup")
    setattr(mu, "_sr_ponder_allocator_warmup_orig", orig)

    def _patched_caching_allocator_warmup(model: Any, expanded_device_map: Dict[str, Any], hf_quantizer: Any) -> None:
        m = getattr(mu, "_sr_ponder_allocator_warmup_mode", "auto")
        if m == "off":
            return

        if m == "auto":
            # Warmup is designed/optimized for cuda/xpu; skip for MPS (and any other non-cuda/xpu accelerator types).
            saw_accelerator = False
            for dev in (expanded_device_map or {}).values():
                if dev in ("disk", "cpu", "meta"):
                    continue
                try:
                    tdev = torch.device(dev)
                except Exception:
                    continue
                saw_accelerator = True
                if tdev.type not in ("cuda", "xpu"):
                    return
            if not saw_accelerator:
                return

        return getattr(mu, "_sr_ponder_allocator_warmup_orig")(model, expanded_device_map, hf_quantizer)

    setattr(mu, "caching_allocator_warmup", _patched_caching_allocator_warmup)
    setattr(mu, "_sr_ponder_allocator_warmup_patched", True)


def resolve_device(device: str) -> str:
    if device != "auto":
        try:
            return str(torch.device(device))
        except Exception as e:
            raise ValueError(f"Invalid device: {device!r}") from e
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype != "auto":
        dt_obj = getattr(torch, dtype, None)
        if not isinstance(dt_obj, torch.dtype):
            raise ValueError(f"Invalid dtype: {dtype!r}")
        return dt_obj
    if device == "mps" or str(device).startswith("cuda"):
        return torch.float16
    return torch.float32


# -----------------------------
# Token selection
# -----------------------------


@dataclasses.dataclass
class RejectedTokenConfig:
    top_k: int = 80
    strategy: str = "outside_topk"  # within_topk|outside_topk
    exclude_top: int = 8
    band_width: int = 256
    n_keywords: int = 6


_TOKEN_CLEAN_RE = re.compile(r"\s+")


def clean_token_text(t: str) -> str:
    t = t.replace("▁", " ").replace("Ġ", " ")
    t = _TOKEN_CLEAN_RE.sub(" ", t).strip()
    return t


def choose_rejected_token_ids(logits_1d: torch.Tensor, cfg: RejectedTokenConfig) -> List[int]:
    return choose_rejected_token_ids_with_rng(logits_1d, cfg)


def choose_rejected_token_ids_with_rng(
    logits_1d: torch.Tensor,
    cfg: RejectedTokenConfig,
    *,
    rng: Optional[random.Random] = None,
    sorted_ids: Optional[torch.Tensor] = None,
) -> List[int]:
    if logits_1d.dim() != 1:
        raise ValueError(f"logits_1d must be 1D, got shape={tuple(logits_1d.shape)}")

    if sorted_ids is None:
        sorted_ids = torch.argsort(logits_1d, descending=True)

    if cfg.strategy == "within_topk":
        start = max(0, min(cfg.exclude_top, cfg.top_k))
        end = max(start, cfg.top_k)
    elif cfg.strategy == "outside_topk":
        start = cfg.top_k
        end = cfg.top_k + cfg.band_width
    else:
        raise ValueError(f"Unknown strategy: {cfg.strategy!r}")

    end = min(end, sorted_ids.numel())
    if start >= end:
        start = max(0, end - 1)

    candidates = sorted_ids[start:end].tolist()
    if rng is None:
        random.shuffle(candidates)
    else:
        rng.shuffle(candidates)
    return candidates[: cfg.n_keywords]


def decode_keyword_tokens(tokenizer, token_ids: Sequence[int], *, max_len: int = 18) -> List[str]:
    special = set(getattr(tokenizer, "all_special_ids", []))
    keywords: List[str] = []
    for tid in token_ids:
        if tid in special:
            continue
        s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        s = clean_token_text(s)
        if not s:
            continue
        if len(s) > max_len:
            continue
        if "<" in s and ">" in s:
            continue
        if not s.isprintable():
            continue
        # avoid pure punctuation / ultra-short noise
        if len(s) == 1 and not ("ぁ" <= s <= "ん" or "ァ" <= s <= "ヶ" or "一" <= s <= "龯" or s.isalnum()):
            continue
        keywords.append(s)

    seen = set()
    out: List[str] = []
    for k in keywords:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def token_ids_from_keywords_text(
    hf: "LocalHFModel",
    keywords: Sequence[str],
    *,
    special_ids: Sequence[int],
    max_total: int = 96,
) -> List[int]:
    special = set(int(x) for x in (special_ids or []))
    out: List[int] = []
    seen = set()
    for kw in keywords or []:
        s = str(kw or "").strip()
        if not s:
            continue
        try:
            enc = hf.tokenizer(s, add_special_tokens=False)
            ids = enc.get("input_ids") if isinstance(enc, dict) else None
        except Exception:
            ids = None
        if not isinstance(ids, list):
            continue
        for tid in ids:
            try:
                tid_i = int(tid)
            except Exception:
                continue
            if tid_i in special:
                continue
            if tid_i in seen:
                continue
            seen.add(tid_i)
            out.append(tid_i)
            if len(out) >= int(max_total):
                return out
    return out


# -----------------------------
# Ponder question + prompts
# -----------------------------


_PONDER_TEMPLATES_EN = [
    "Using {a} and {b} and the other words, what do you freely associate without any specific purpose?",
    "With {a} and {b} as seeds, write down whatever comes to mind, even if it feels irrelevant.",
    "Let {a} and {b} collide. What images, questions, or fragments appear?",
    "Treat {a} and {b} as a dream prompt. What does the dream do?",
]

_PONDER_TEMPLATES_JA = [
    "「{a}」と「{b}」と他の言葉を種にして、目的なく自由連想してみて。",
    "「{a}」と「{b}」をぶつけたときに浮かぶイメージ・疑問・断片をそのまま書いて。",
    "「{a}」と「{b}」を夢の合図だと思って。夢はどう展開する？",
    "「{a}」と「{b}」から、関係なさそうでもいいので連想を走らせて。",
]


def make_unrelated_question(keywords: Sequence[str], *, lang: str, rng: random.Random) -> str:
    if len(keywords) >= 2:
        a, b = keywords[0], keywords[1]
    elif len(keywords) == 1:
        a, b = keywords[0], "Emptiness"
    else:
        a, b = " Unknown", "Not-defined"
    templates = _PONDER_TEMPLATES_JA if lang == "ja" else _PONDER_TEMPLATES_EN
    tmpl = rng.choice(templates)
    return tmpl.format(a=a, b=b)


def build_memory_block(records: Sequence[Dict[str, Any]], *, max_chars_per_log: int = 600) -> str:
    chunks: List[str] = []
    for r in records:
        ts = r.get("ts", "?")
        meta_bits: List[str] = []
        if r.get("run_id"):
            meta_bits.append(f"run:{r['run_id']}")
        if r.get("band_label"):
            meta_bits.append(f"band:{r['band_label']}")
        if r.get("hop_ix") is not None:
            meta_bits.append(f"hop:{r['hop_ix']}")
        if r.get("ponder_ix") is not None:
            meta_bits.append(f"ix:{r['ponder_ix']}")
        if r.get("ponder_mode"):
            meta_bits.append(f"mode:{r['ponder_mode']}")
        meta = (" " + " ".join(meta_bits)) if meta_bits else ""
        kw = r.get("keywords", [])
        kw_s = ", ".join(kw) if isinstance(kw, list) else str(kw)
        pq = r.get("ponder_question", "")
        plog = (r.get("ponder_log", "") or "")[:max_chars_per_log].rstrip()
        chunks.append(
            f"[{ts}]{meta} keywords: {kw_s}\n"
            f"ponder_q: {pq}\n"
            f"ponder_log:\n{plog}\n"
        )
    return "\n".join(chunks).strip()


def default_system_text(lang: str) -> str:
    # Gemma IT does not support a dedicated system role; keep this inside the user turn.
    if lang == "ja":
        return "寄り道して熟考（ponder）したあと、最後に質問へ答えてください。"
    return "After pondering the question, you provide an answer."


def _answer_style_guidance(style: str, *, lang: str) -> str:
    s = (style or "plain").strip().lower()
    if s in ("", "plain", "default"):
        return ""
    if lang == "ja":
        if s == "surreal":
            return (
                "文体: シュールで比喩・象徴を多めに（少しメタでもよい）。ただし意味を崩しすぎず、最後に1段落で平易に要点をまとめる。"
            )
        if s == "metaphor":
            return "文体: 比喩・アナロジー中心。ただし最後に1〜2文で平易な要点まとめを添える。"
        if s == "meta":
            return "文体: メタ認知的に（問いの前提/フレーミングにも短く触れる）。その上で端的に回答する。"
    else:
        if s == "surreal":
            return (
                "Style: surreal, metaphor-heavy, slightly self-referential. Keep an anchor: end with a short plain-language summary paragraph."
            )
        if s == "metaphor":
            return "Style: metaphor/analogy-driven, but end with 1-2 plain sentences summarizing the answer."
        if s == "meta":
            return "Style: briefly comment on the framing/assumptions of the question, then answer directly."
    raise ValueError(f"Unknown answer_style: {style!r}")


def build_prompt_for_answer(query: str, memory_block: Optional[str], *, lang: str, style: str = "plain") -> str:
    style_note = _answer_style_guidance(style, lang=lang).strip()
    prefix = f"{default_system_text(lang)}\n\n"
    if style_note:
        prefix += style_note + "\n\n"
    if memory_block:
        if lang == "ja":
            return (
                prefix
                + "以下は最近生成された「本題と直接関係しない Ponder Log」です。"
                + "ただし、隠れた前提や別の切り口に気づく助けになるなら軽く参照してもよい。\n"
                + "<ponder_log>\n"
                + f"{memory_block}\n"
                + "</ponder_log>\n\n"
                + "本題の質問:\n"
                + f"{query}\n\n"
                + "出力は回答本文のみ（見出しや引用は不要）。\n"
            )
        return (
            prefix
            + "The following is a recently generated Ponder Log Not Directly Related to the Main Topic."
            + "But you may use it casually if it helps you notice hidden assumptions or alternative framings.\n"
            + "<ponder_log>\n"
            + f"{memory_block}\n"
            + "</ponder_log>\n\n"
            + "Actual Question:\n"
            + f"{query}\n\n"
            + "Write only the answer in your output (headings and quotes are not needed).\n"
        )
    if lang == "ja":
        return (
            prefix
            + "本題の質問:\n"
            + f"{query}\n\n"
            + "出力は回答本文のみ（見出しや引用は不要）。\n"
        )
    return (
        prefix
        + "Actual Question:\n"
        + f"{query}\n\n"
        + "Write only the answer in your output (headings and quotes are not needed).\n"
    )


def build_prompt_for_pondering(ponder_q: str, *, mode: str, lang: str, context: Optional[str] = None) -> str:
    context = (context or "").strip()
    if lang == "ja":
        if mode == "assoc":
            body = (
                "次の問いに対して、結論を出さずに短い ponder log（寄り道ログ）を書いてください。\n"
                "条件:\n"
                "- 実用的な助言や最終結論は書かない\n"
                "- 連想・比喩・反例・仮定の揺らぎを混ぜる\n"
                "- 10〜15行。各行は - で始める\n\n"
                f"問い: {ponder_q}\n"
            )
        elif mode == "assumption":
            body = (
                "次の問いの背後にありそうな「隠れた前提」を列挙し、前提の置き換えも添えてください（結論は出さない）。\n"
                "条件:\n"
                "- 各行は1つの前提（または置き換え）\n"
                "- 10〜15行。各行は - で始める\n\n"
                f"問い: {ponder_q}\n"
            )
        elif mode == "counterexample":
            body = (
                "次の問いに対して、素朴な理解を崩す反例・境界事例・例外だけを集めてください（結論は出さない）。\n"
                "条件:\n"
                "- 10〜15行。各行は - で始める\n\n"
                f"問い: {ponder_q}\n"
            )
        elif mode == "questions_only":
            body = (
                "次の問いから派生する「問い」だけを出力してください（答えや断定は禁止）。\n"
                "条件:\n"
                "- 10〜15行。各行は - で始める\n\n"
                f"問い: {ponder_q}\n"
            )
        elif mode == "metaphor":
            body = (
                "次の問いを素材に、比喩・アナロジー・イメージの断片だけで ponder log を書いてください（結論は出さない）。\n"
                "条件:\n"
                "- 10〜15行。各行は - で始める\n\n"
                f"問い: {ponder_q}\n"
            )
        else:
            raise ValueError(f"Unknown ponder_mode: {mode!r}")
        if context:
            return (
                "参考（直前のログ。結論ではない）：\n"
                "<prev_ponder>\n"
                f"{context}\n"
                "</prev_ponder>\n\n"
                + body
            )
        return body

    # English
    if mode == "assoc":
        body = (
            "Create a brief ponder log for the following question without drawing any conclusions.\n"
            "Conditions:\n"
            "- Do not provide practical advice or reach a final conclusion.\n"
            "- Mix assumptions, counterexamples, and analogies.\n"
            "- 10 to 15 lines. Each line begins with - .\n\n"
            f"Question: {ponder_q}\n"
        )
    elif mode == "assumption":
        body = (
            "List the hidden assumptions behind the following question, and suggest alternative assumptions.\n"
            "Conditions:\n"
            "- No conclusion.\n"
            "- 10 to 15 lines. Each line begins with - .\n\n"
            f"Question: {ponder_q}\n"
        )
    elif mode == "counterexample":
        body = (
            "Generate counterexamples, edge cases, and exceptions related to the following question.\n"
            "Conditions:\n"
            "- No conclusion.\n"
            "- 10 to 15 lines. Each line begins with - .\n\n"
            f"Question: {ponder_q}\n"
        )
    elif mode == "questions_only":
        body = (
            "Output only questions derived from the following question.\n"
            "Conditions:\n"
            "- No answers or assertions.\n"
            "- 10 to 15 lines. Each line begins with - .\n\n"
            f"Question: {ponder_q}\n"
        )
    elif mode == "metaphor":
        body = (
            "Write a ponder log using only metaphors, analogies, and image fragments inspired by the following question.\n"
            "Conditions:\n"
            "- No conclusion.\n"
            "- 10 to 15 lines. Each line begins with - .\n\n"
            f"Question: {ponder_q}\n"
        )
    else:
        raise ValueError(f"Unknown ponder_mode: {mode!r}")

    if context:
        return (
            "Context from the previous stage (not a conclusion):\n"
            "<prev_ponder>\n"
            f"{context}\n"
            "</prev_ponder>\n\n"
            + body
        )
    return body


_PUNCT_ONLY_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)


def extract_json_array(text: str) -> List[str]:
    """
    Tries hard to parse a JSON array of strings from a model output.
    Falls back to newline/comma splitting. Returns a de-duplicated list (preserving order).
    """
    if not text:
        return []

    start = text.find("[")
    end = text.rfind("]")
    candidates: List[str] = []

    if 0 <= start < end:
        snippet = text[start : end + 1].strip()
        try:
            arr = json.loads(snippet)
            if isinstance(arr, list):
                for x in arr:
                    if isinstance(x, str):
                        candidates.append(x.strip())
        except Exception:
            pass

    if not candidates:
        rough = re.split(r"[\n,]+", text)
        candidates = [x.strip().strip('"').strip("'") for x in rough if x.strip()]

    out: List[str] = []
    seen = set()
    for w in candidates:
        w2 = re.sub(r"\s+", " ", (w or "")).strip()
        w2 = re.sub(r"^[-*•\d\.\)\s]+", "", w2).strip()
        w2 = w2.strip(" \t\r\n\"'`，,。．.：:;；()（）[]【】")
        if not w2:
            continue
        if len(w2) > 32:
            continue
        if _PUNCT_ONLY_RE.match(w2):
            continue
        if "<" in w2 and ">" in w2:
            continue
        if w2 in seen:
            continue
        seen.add(w2)
        out.append(w2)
    return out


def build_prompt_for_keyword_refine(query: str, seed_keywords: Sequence[str], *, n: int, lang: str) -> str:
    kws = ", ".join(seed_keywords) if seed_keywords else ""
    if lang == "ja":
        return (
            "あなたはキーワード抽出器です。\n"
            "入力のキーワード断片（token由来）をヒントに、面白くて少し逸脱したキーワードを生成してください。\n"
            "制約:\n"
            f"- 出力は JSON の文字列配列のみ（要素数はちょうど {n}）\n"
            "- 説明文は禁止\n"
            "- 記号だけ/助詞だけ/role名は禁止\n"
            "- 1〜3語程度の短い語句\n\n"
            f"元の質問: {query}\n"
            f"断片: {kws}\n"
        )
    return (
        "You are a keyword extractor.\n"
        "Given token-fragment seed words, generate interesting, slightly off-axis keywords.\n"
        "Constraints:\n"
        f"- Output ONLY a JSON array of strings (exactly {n} items)\n"
        "- No explanations\n"
        "- Avoid pure punctuation, stopwords, and role labels\n"
        "- Short words/phrases (1 to 3 words)\n\n"
        f"Original question: {query}\n"
        f"Fragments: {kws}\n"
    )


def build_prompt_for_hop_keyword_extract(
    query: str,
    prev_keywords: Sequence[str],
    ponder_log: str,
    *,
    n: int,
    lang: str,
) -> str:
    prev = ", ".join([str(x).strip() for x in (prev_keywords or []) if str(x).strip()][:24])
    plog = (ponder_log or "").strip()
    if len(plog) > 1400:
        plog = plog[:1400] + "\n…"
    if lang == "ja":
        return (
            "あなたは pondering machine の「次ホップ用キーワード抽出器」です。\n"
            "直前の ponder log から、さらに思考を逸脱させるための新しい種キーワードを提案してください。\n"
            "制約:\n"
            f"- 出力は JSON の文字列配列のみ（要素数はちょうど {n}）\n"
            "- 説明文は禁止\n"
            "- 直前キーワードの言い換え/重複は禁止\n"
            "- 本題の単語をそのまま繰り返すのは避ける\n"
            "- 1〜3語程度の短い語句（抽象語より具体語を優先）\n\n"
            f"本題: {query}\n"
            f"直前キーワード: {prev}\n\n"
            "<ponder_log>\n"
            f"{plog}\n"
            "</ponder_log>\n"
        )
    return (
        "You generate next-hop seed keywords for a pondering machine.\n"
        "From the previous ponder log, propose NEW seed keywords that push the thought into adjacent, surprising regions.\n"
        "Constraints:\n"
        f"- Output ONLY a JSON array of strings (exactly {n} items)\n"
        "- No explanations\n"
        "- Avoid paraphrases/duplicates of previous keywords\n"
        "- Avoid simply repeating obvious words from the original question\n"
        "- Short phrases (1 to 3 words), prefer concrete words over abstractions\n\n"
        f"Original question: {query}\n"
        f"Previous keywords: {prev}\n\n"
        "<ponder_log>\n"
        f"{plog}\n"
        "</ponder_log>\n"
    )


_EN_HOP_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")
_JA_HOP_WORD_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]{2,}")


def _ban_norm_set(query: str, prev_keywords: Sequence[str], *, lang: str) -> set:
    ban = set()
    for x in (prev_keywords or []):
        ban.add(_norm_for_sim(str(x)))
    # also ban obvious query words (heuristic)
    q = str(query or "")
    if lang == "ja":
        toks = _JA_HOP_WORD_RE.findall(q)
    else:
        toks = _EN_HOP_WORD_RE.findall(q)
    for t in toks[:200]:
        ban.add(_norm_for_sim(t))
    return ban


def extract_hop_keywords_heuristic(
    *,
    query: str,
    prev_keywords: Sequence[str],
    ponder_log: str,
    n: int,
    lang: str,
    rng: random.Random,
) -> List[str]:
    if n <= 0:
        return []
    s = re.sub(r"<[^>]+>", " ", str(ponder_log or ""))
    s = re.sub(r"[\r\n\t]+", " ", s)
    if lang == "ja":
        toks = _JA_HOP_WORD_RE.findall(s)
    else:
        toks = _EN_HOP_WORD_RE.findall(s)

    ban = _ban_norm_set(query, prev_keywords, lang=lang)
    counts: Dict[str, int] = {}
    raw_map: Dict[str, str] = {}
    for t in toks:
        t2 = clean_token_text(t)
        nt = _norm_for_sim(t2)
        if not nt:
            continue
        if nt in ban:
            continue
        if len(nt) < 3:
            continue
        if len(nt) > 24:
            continue
        counts[nt] = int(counts.get(nt, 0)) + 1
        raw_map.setdefault(nt, t2)

    if not counts:
        return []

    scored = sorted(counts.items(), key=lambda kv: (kv[1] * len(kv[0]), kv[1], len(kv[0])), reverse=True)
    pool = [raw_map.get(k, k) for k, _ in scored[: max(24, n * 8)]]
    pool = [p for p in pool if p and _norm_for_sim(p) not in ban]
    if len(pool) <= n:
        return pool[:n]
    return rng.sample(pool, k=n)


def extract_hop_keywords_with_model(
    hf: Any,
    *,
    query: str,
    prev_keywords: Sequence[str],
    ponder_log: str,
    n: int,
    lang: str,
    seed: int,
) -> List[str]:
    if n <= 0:
        return []
    prompt = hf._apply_chat(
        build_prompt_for_hop_keyword_extract(query, prev_keywords, ponder_log, n=n, lang=lang),
        system_text=None,
    )
    out = hf.generate_text(
        prompt,
        max_new_tokens=180,
        temperature=0.5,
        top_p=0.9,
        top_k=0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        seed=int(seed),
    )
    kws = extract_json_array(out)
    ban = _ban_norm_set(query, prev_keywords, lang=lang)
    out2: List[str] = []
    seen = set()
    for k in kws:
        k2 = clean_token_text(str(k or "")).strip()
        nk = _norm_for_sim(k2)
        if not nk:
            continue
        if nk in ban:
            continue
        if nk in seen:
            continue
        if len(k2) > 32:
            continue
        seen.add(nk)
        out2.append(k2)
        if len(out2) >= n:
            break
    return out2[:n]


def build_prompt_for_api_seed_keywords(
    query: str,
    *,
    n: int,
    lang: str,
    band_label: str,
    objective: str,
) -> str:
    bl = (band_label or "single").strip().lower()
    obj = (objective or "random_band").strip().lower()

    if bl == "near":
        style_en = "near = plausible/on-topic, but avoid the most obvious words"
        style_ja = "near = もっともらしく本題寄り。ただし一番ベタな語は避ける"
    elif bl == "mid":
        style_en = "mid = slightly off-axis 'rejected-token' vibe (useful drift)"
        style_ja = "mid = ほどよく逸脱（rejected-tokenっぽいドリフト）"
    elif bl == "far":
        style_en = "far = distant/metaphorical, surprising but still usable as seeds"
        style_ja = "far = かなり遠い/比喩的。意外だが種として使える"
    else:
        style_en = f"band = {bl} (treat as a style label)"
        style_ja = f"band = {bl}（スタイルラベルとして扱う）"

    if obj == "random_vocab":
        obj_en = "Objective: random words/phrases (not necessarily related)."
        obj_ja = "目的: ランダムな語句（関連は不要）。"
    elif obj == "unstable":
        obj_en = "Objective: ambiguous keywords likely to change under paraphrasing."
        obj_ja = "目的: 言い換えで揺れそうな曖昧キーワード。"
    elif obj == "dissonance":
        obj_en = "Objective: dissonant/off-axis keywords (conceptual distance)."
        obj_ja = "目的: 不協和/逸脱キーワード（概念距離）。"
    else:
        obj_en = "Objective: slightly off-axis seed keywords (no answers)."
        obj_ja = "目的: 少し逸脱した種キーワード（答えは出さない）。"

    if lang == "ja":
        return (
            "あなたは pondering machine 用のキーワード抽出器です。\n"
            f"{obj_ja}\n"
            f"スタイル: {style_ja}\n"
            "制約:\n"
            f"- 出力は JSON の文字列配列のみ（要素数はちょうど {n}）\n"
            "- 説明文は禁止\n"
            "- 記号だけ/助詞だけ/role名は禁止\n"
            "- 1〜3語程度の短い語句\n\n"
            f"質問: {query}\n"
        )
    return (
        "You are a keyword extractor for a pondering machine.\n"
        f"{obj_en}\n"
        f"Style: {style_en}\n"
        "Constraints:\n"
        f"- Output ONLY a JSON array of strings (exactly {n} items)\n"
        "- No explanations\n"
        "- Avoid pure punctuation, stopwords, and role labels\n"
        "- Short words/phrases (1 to 3 words)\n\n"
        f"Question: {query}\n"
    )


def generate_seed_keywords_self(
    hf: Any,
    *,
    query: str,
    n_keywords: int,
    lang: str,
    band_label: str,
    objective: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> List[str]:
    prompt = hf._apply_chat(
        build_prompt_for_api_seed_keywords(query, n=n_keywords, lang=lang, band_label=band_label, objective=objective),
        system_text=None,
    )
    out = hf.generate_text(
        prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=0.9,
        top_k=0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        seed=int(seed),
    )
    kws = extract_json_array(out)
    return kws[: max(0, int(n_keywords))]


def refine_keywords_with_model(
    hf: Any,
    *,
    query: str,
    seed_keywords: Sequence[str],
    n_keywords: int,
    lang: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> List[str]:
    if not seed_keywords:
        return []
    prompt = hf._apply_chat(
        build_prompt_for_keyword_refine(query, seed_keywords, n=n_keywords, lang=lang),
        system_text=None,
    )
    out = hf.generate_text(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=float(temperature),
        top_p=0.9,
        top_k=0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        seed=seed,
    )
    refined = extract_json_array(out)
    return refined[:n_keywords]


def build_prompt_for_prompt_jitter(query: str, *, n: int, lang: str) -> str:
    if lang == "ja":
        return (
            "あなたは言い換え生成器です。\n"
            "次の質問の意味を保ったまま、言い換えを生成してください。\n"
            "制約:\n"
            f"- 出力は JSON の文字列配列のみ（要素数はちょうど {n}）\n"
            "- 回答は禁止（質問文だけ）\n"
            "- できるだけ表現を変えるが、意図は変えない\n\n"
            f"質問: {query}\n"
        )
    return (
        "You are a paraphrase generator.\n"
        "Generate paraphrases of the following question while preserving meaning.\n"
        "Constraints:\n"
        f"- Output ONLY a JSON array of strings (exactly {n} items)\n"
        "- Do NOT answer the question (questions only)\n"
        "- Change wording as much as possible without changing intent\n\n"
        f"Question: {query}\n"
    )


def generate_prompt_jitters(
    hf: "LocalHFModel",
    *,
    query: str,
    n: int,
    lang: str,
    include_original: bool,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> List[str]:
    if n <= 0:
        return [query]
    prompt = hf._apply_chat(build_prompt_for_prompt_jitter(query, n=n, lang=lang), system_text=None)
    out = hf.generate_text(
        prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=0.9,
        top_k=0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        seed=int(seed),
    )
    arr = extract_json_array(out)
    out_list: List[str] = []
    seen = set()
    for s in arr:
        s2 = (s or "").strip()
        if not s2:
            continue
        if s2 in seen:
            continue
        seen.add(s2)
        out_list.append(s2)

    if include_original:
        q = (query or "").strip()
        if q and q not in seen:
            out_list.insert(0, q)

    if not out_list:
        return [query]
    return out_list


def _token_prob(logits_1d: torch.Tensor, token_id: int, log_z: torch.Tensor) -> float:
    return float(torch.exp(logits_1d[int(token_id)] - log_z).item())


def _token_text(tokenizer, token_id: int) -> str:
    try:
        s = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    except Exception:
        s = str(token_id)
    s = clean_token_text(s)
    return s


def _l2_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if v.dim() <= 1:
        denom = torch.linalg.norm(v) + eps
        return v / denom
    denom = torch.linalg.norm(v, dim=-1, keepdim=True) + eps
    return v / denom


def mean_embedding_for_token_ids(hf: "LocalHFModel", token_ids: Sequence[int]) -> Optional[torch.Tensor]:
    ids = [int(x) for x in token_ids if x is not None]
    if not ids:
        return None
    try:
        emb = hf.model.get_input_embeddings()
        weight = emb.weight
        idx = torch.tensor(ids, device=weight.device, dtype=torch.long)
        rows = weight.index_select(0, idx).detach()
        vec = rows.float().mean(dim=0)
        return vec.cpu()
    except Exception:
        return None


def prompt_embedding_tail(
    hf: "LocalHFModel",
    prompt: str,
    *,
    tail_k: int = 64,
) -> Optional[torch.Tensor]:
    try:
        encoded = hf.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            return None
        ids = input_ids[0].tolist()
    except Exception:
        return None

    special = set(getattr(hf.tokenizer, "all_special_ids", []) or [])
    for x in (getattr(hf, "start_of_turn_id", None), getattr(hf, "end_of_turn_id", None)):
        if x is not None:
            special.add(int(x))

    tail: List[int] = []
    for tid in reversed(ids):
        tid_i = int(tid)
        if tid_i in special:
            continue
        tail.append(tid_i)
        if len(tail) >= int(tail_k):
            break
    tail.reverse()
    if not tail:
        return None
    return mean_embedding_for_token_ids(hf, tail)


def dissonance_scores(
    hf: "LocalHFModel",
    *,
    query_vec: torch.Tensor,
    candidate_ids: Sequence[int],
) -> Dict[int, float]:
    q = query_vec.detach().float().cpu()
    q = _l2_normalize(q)
    scores: Dict[int, float] = {}

    # NOTE: mean_embedding_for_token_ids currently returns a mean vector; for candidates we need per-token vectors.
    # We'll compute per-token vectors here efficiently.
    try:
        emb = hf.model.get_input_embeddings()
        weight = emb.weight
        idx = torch.tensor([int(x) for x in candidate_ids], device=weight.device, dtype=torch.long)
        rows = weight.index_select(0, idx).detach().float().cpu()
    except Exception:
        return scores

    rows = _l2_normalize(rows, eps=1e-8)
    sims = (rows @ q).tolist()
    for tid, sim in zip(candidate_ids, sims):
        # dissonance: 1 - cosine similarity (range ~[0,2])
        scores[int(tid)] = float(1.0 - float(sim))
    return scores


def unstable_scores_from_jitters(
    candidate_ids: Sequence[int],
    jitter_logits: Sequence[torch.Tensor],
) -> Dict[int, float]:
    if not candidate_ids or len(jitter_logits) < 2:
        return {}
    idx = torch.tensor([int(x) for x in candidate_ids], device=jitter_logits[0].device, dtype=torch.long)
    vals = torch.stack([lg.index_select(0, idx) for lg in jitter_logits], dim=0)
    std = torch.std(vals, dim=0, unbiased=False).tolist()
    return {int(tid): float(s) for tid, s in zip(candidate_ids, std)}


def sample_token_ids(
    *,
    rng: random.Random,
    pool: Sequence[int],
    n: int,
) -> List[int]:
    if n <= 0:
        return []
    if len(pool) <= n:
        return [int(x) for x in pool[:n]]
    return [int(x) for x in rng.sample(list(pool), k=int(n))]


def select_token_ids_by_objective(
    *,
    objective: str,
    rng: random.Random,
    vocab_size: int,
    candidate_pool: Sequence[int],
    n_keywords: int,
    special_ids: Sequence[int],
    dissonance: Optional[Dict[int, float]] = None,
    dissonance_target: float = 0.9,
    dissonance_width: float = 0.6,
    unstable: Optional[Dict[int, float]] = None,
    select_top: int = 128,
) -> List[int]:
    n = int(n_keywords)
    if n <= 0:
        return []

    special = set(int(x) for x in (special_ids or []))

    if objective == "random_vocab":
        out: List[int] = []
        seen = set()
        tries = 0
        while len(out) < n and tries < n * 50:
            tries += 1
            tid = int(rng.randrange(0, int(vocab_size)))
            if tid in special:
                continue
            if tid in seen:
                continue
            seen.add(tid)
            out.append(tid)
        return out

    if objective == "unstable" and unstable:
        items = [(tid, unstable.get(int(tid), 0.0)) for tid in candidate_pool if int(tid) not in special]
        items.sort(key=lambda x: x[1], reverse=True)
        top = [tid for tid, _ in items[: max(1, int(select_top))]]
        return sample_token_ids(rng=rng, pool=top or candidate_pool, n=n)

    if objective == "dissonance" and dissonance:
        lo = float(dissonance_target) - float(dissonance_width) / 2.0
        hi = float(dissonance_target) + float(dissonance_width) / 2.0
        filtered = [
            int(tid)
            for tid in candidate_pool
            if int(tid) not in special and lo <= float(dissonance.get(int(tid), -1e9)) <= hi
        ]
        if filtered:
            # Prefer closer-to-target candidates, then sample for variety.
            filtered.sort(key=lambda tid: abs(float(dissonance.get(int(tid), 0.0)) - float(dissonance_target)))
            top = filtered[: max(1, int(select_top))]
            return sample_token_ids(rng=rng, pool=top, n=n)
        return sample_token_ids(rng=rng, pool=[tid for tid in candidate_pool if int(tid) not in special], n=n)

    # default: random within band
    return sample_token_ids(rng=rng, pool=[tid for tid in candidate_pool if int(tid) not in special], n=n)


def _norm_for_sim(s: str) -> str:
    s2 = clean_token_text(str(s or ""))
    s2 = re.sub(r"\s+", " ", s2).strip().lower()
    return s2


def _sim_ratio(a: str, b: str) -> float:
    aa = _norm_for_sim(a)
    bb = _norm_for_sim(b)
    if not aa or not bb:
        return 0.0
    if aa == bb:
        return 1.0
    return float(difflib.SequenceMatcher(a=aa, b=bb).ratio())


def pick_diverse_strings(
    strings: Sequence[str],
    *,
    rng: random.Random,
    n: int,
    threshold: float,
    preserve_order: bool,
) -> List[str]:
    if n <= 0:
        return []
    items = [str(s) for s in (strings or []) if isinstance(s, str) and str(s).strip()]
    if not preserve_order:
        items = list(items)
        rng.shuffle(items)
    out: List[str] = []
    out_norm: List[str] = []
    for s in items:
        if len(out) >= n:
            break
        ns = _norm_for_sim(s)
        if not ns:
            continue
        if any(float(difflib.SequenceMatcher(a=ns, b=t).ratio()) >= float(threshold) for t in out_norm):
            continue
        out.append(s)
        out_norm.append(ns)
    if len(out) < n:
        rest = [s for s in items if s not in set(out)]
        rng.shuffle(rest)
        out.extend(rest[: max(0, n - len(out))])
    return out[:n]


def pick_diverse_token_ids_by_text(
    hf: "LocalHFModel",
    *,
    rng: random.Random,
    candidate_ids: Sequence[int],
    n: int,
    threshold: float,
    preserve_order: bool,
) -> List[int]:
    if n <= 0:
        return []
    items: List[Tuple[int, str]] = []
    for tid in candidate_ids or []:
        try:
            tid_i = int(tid)
        except Exception:
            continue
        txt = _token_text(hf.tokenizer, tid_i)
        nt = _norm_for_sim(txt)
        if not nt:
            continue
        items.append((tid_i, nt))

    if not preserve_order:
        rng.shuffle(items)

    out: List[int] = []
    out_norm: List[str] = []
    for tid_i, nt in items:
        if len(out) >= n:
            break
        if any(float(difflib.SequenceMatcher(a=nt, b=t).ratio()) >= float(threshold) for t in out_norm):
            continue
        out.append(tid_i)
        out_norm.append(nt)

    if len(out) < n:
        rest = [tid_i for tid_i, _ in items if tid_i not in set(out)]
        out.extend(sample_token_ids(rng=rng, pool=rest, n=max(0, n - len(out))))
    return out[:n]


def pick_diverse_token_ids_by_embedding(
    hf: "LocalHFModel",
    *,
    rng: random.Random,
    candidate_ids: Sequence[int],
    n: int,
) -> List[int]:
    if n <= 0:
        return []
    ids: List[int] = []
    for tid in candidate_ids or []:
        try:
            ids.append(int(tid))
        except Exception:
            continue
    if not ids:
        return []
    if len(ids) <= n:
        return ids[:n]

    try:
        emb = hf.model.get_input_embeddings()
        weight = emb.weight
        idx = torch.tensor(ids, device=weight.device, dtype=torch.long)
        rows = weight.index_select(0, idx).detach().float().cpu()
    except Exception:
        return sample_token_ids(rng=rng, pool=ids, n=n)

    rows = _l2_normalize(rows, eps=1e-8)
    m = int(rows.shape[0])
    if m <= n:
        return ids[:n]

    start = int(rng.randrange(0, max(1, min(m, 8))))
    selected = [start]
    min_dist = (1.0 - (rows @ rows[start])).to(torch.float32)
    min_dist[start] = -1.0

    for _ in range(1, int(n)):
        next_i = int(torch.argmax(min_dist).item())
        if float(min_dist[next_i].item()) < 0.0:
            break
        selected.append(next_i)
        new_dist = (1.0 - (rows @ rows[next_i])).to(torch.float32)
        min_dist = torch.minimum(min_dist, new_dist)
        min_dist[next_i] = -1.0

    out_ids = [ids[i] for i in selected]
    if len(out_ids) < n:
        rest = [tid for tid in ids if tid not in set(out_ids)]
        out_ids.extend(sample_token_ids(rng=rng, pool=rest, n=max(0, n - len(out_ids))))
    return out_ids[:n]


def select_keyword_token_ids_hf(
    hf: "LocalHFModel",
    *,
    objective: str,
    rng: random.Random,
    vocab_size: int,
    candidate_pool: Sequence[int],
    n_keywords: int,
    special_ids: Sequence[int],
    dissonance: Optional[Dict[int, float]] = None,
    dissonance_target: float = 0.9,
    dissonance_width: float = 0.6,
    unstable: Optional[Dict[int, float]] = None,
    select_top: int = 128,
    diversity: str = "off",  # off|lex|embed
    diversity_threshold: float = 0.82,
) -> List[int]:
    n = int(n_keywords)
    if n <= 0:
        return []

    if (diversity or "off").strip() == "off":
        return select_token_ids_by_objective(
            objective=objective,
            rng=rng,
            vocab_size=vocab_size,
            candidate_pool=candidate_pool,
            n_keywords=n_keywords,
            special_ids=special_ids,
            dissonance=dissonance,
            dissonance_target=dissonance_target,
            dissonance_width=dissonance_width,
            unstable=unstable,
            select_top=select_top,
        )

    special = set(int(x) for x in (special_ids or []))
    base_pool = [int(t) for t in (candidate_pool or []) if int(t) not in special]
    if not base_pool and objective != "random_vocab":
        return []

    k_top = max(1, int(select_top))
    want = max(k_top, n * 16)

    # Objective: narrow down candidates, then pick diverse samples.
    cand = base_pool
    if objective == "random_vocab":
        cand = []
        seen = set()
        tries = 0
        while len(cand) < want and tries < want * 80:
            tries += 1
            tid = int(rng.randrange(0, int(vocab_size)))
            if tid in special or tid in seen:
                continue
            seen.add(tid)
            cand.append(tid)
    elif objective == "unstable" and unstable:
        items = [(tid, float(unstable.get(int(tid), 0.0))) for tid in cand]
        items.sort(key=lambda x: x[1], reverse=True)
        cand = [tid for tid, _ in items[:want]]
    elif objective == "dissonance" and dissonance:
        lo = float(dissonance_target) - float(dissonance_width) / 2.0
        hi = float(dissonance_target) + float(dissonance_width) / 2.0
        filtered = [tid for tid in cand if lo <= float(dissonance.get(int(tid), -1e9)) <= hi]
        if filtered:
            filtered.sort(key=lambda tid: abs(float(dissonance.get(int(tid), 0.0)) - float(dissonance_target)))
            cand = filtered[:want]
        else:
            items = [(tid, float(dissonance.get(int(tid), -1e9))) for tid in cand]
            items.sort(key=lambda x: x[1], reverse=True)
            cand = [tid for tid, _ in items[:want]]
    else:
        cand = cand[:want]

    cand = list(dict.fromkeys([int(t) for t in cand if int(t) not in special]))
    if not cand:
        return []

    div = (diversity or "off").strip()
    if div == "embed":
        picked = pick_diverse_token_ids_by_embedding(hf, rng=rng, candidate_ids=cand, n=n)
        if picked:
            return picked[:n]

    preserve = objective in ("dissonance", "unstable")
    return pick_diverse_token_ids_by_text(
        hf,
        rng=rng,
        candidate_ids=cand,
        n=n,
        threshold=float(diversity_threshold),
        preserve_order=preserve,
    )


def api_lex_dissonance_scores(query: str, tokens: Sequence[str], *, lang: str) -> Dict[str, float]:
    q = str(query or "")
    if lang == "ja":
        q_words = _JA_HOP_WORD_RE.findall(q)
    else:
        q_words = _EN_HOP_WORD_RE.findall(q)
    q_words = [w for w in q_words if w]
    q_words = q_words[:128]

    out: Dict[str, float] = {}
    for t in tokens or []:
        if not isinstance(t, str):
            continue
        tt = clean_token_text(t).strip()
        if not tt:
            continue
        if q_words:
            max_sim = 0.0
            for qw in q_words:
                max_sim = max(max_sim, _sim_ratio(tt, qw))
            out[tt] = float(1.0 - max_sim)
        else:
            out[tt] = float(1.0 - _sim_ratio(tt, q))
    return out


def interactive_pick_token_ids(
    *,
    prompt: str,
    rng: random.Random,
    candidates: Sequence[Dict[str, Any]],
    n_keywords: int,
) -> Optional[List[int]]:
    if not sys.stdin.isatty():
        return None
    if not candidates:
        return None

    n = int(n_keywords)
    print(f"\n=== INTERACTIVE PICK: {prompt} ===\n")
    for i, c in enumerate(candidates, start=1):
        tok = c.get("token", "")
        tid = c.get("token_id", "?")
        rk = c.get("rank", "?")
        p = c.get("prob", None)
        extra_bits: List[str] = []
        if "dissonance" in c:
            extra_bits.append(f"d={float(c['dissonance']):.3f}")
        if "unstable" in c:
            extra_bits.append(f"u={float(c['unstable']):.3f}")
        extra = (" " + " ".join(extra_bits)) if extra_bits else ""
        p_s = f"{p:.4f}" if isinstance(p, float) else "?"
        print(f"{i:>2}. {tok!r} id={tid} rank={rk} p={p_s}{extra}")

    try:
        raw = input(f"\nPick {n} indices (space/comma). Enter=auto: ").strip()
    except EOFError:
        return None

    if not raw:
        return None

    parts = re.split(r"[\s,]+", raw)
    picked: List[int] = []
    for p in parts:
        if not p:
            continue
        try:
            ix = int(p)
        except ValueError:
            continue
        if ix < 1 or ix > len(candidates):
            continue
        tid = candidates[ix - 1].get("token_id")
        if tid is None:
            continue
        tid_i = int(tid)
        if tid_i in picked:
            continue
        picked.append(tid_i)
        if len(picked) >= n:
            break

    if not picked:
        return None
    if len(picked) < n:
        rest = [int(c.get("token_id")) for c in candidates if c.get("token_id") is not None and int(c["token_id"]) not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: (n - len(picked))])
    return picked[:n]


def interactive_pick_keywords(
    *,
    prompt: str,
    rng: random.Random,
    candidates: Sequence[Dict[str, Any]],
    n_keywords: int,
) -> Optional[List[str]]:
    if not sys.stdin.isatty():
        return None
    if not candidates:
        return None

    n = int(n_keywords)
    print(f"\n=== INTERACTIVE PICK (keywords): {prompt} ===\n")
    for i, c in enumerate(candidates, start=1):
        tok = str(c.get("token", "") or "")
        rk = c.get("rank", "?")
        p = c.get("prob", None)
        p_s = f"{p:.4f}" if isinstance(p, float) else "?"
        print(f"{i:>2}. {tok!r} rank={rk} p={p_s}")

    try:
        raw = input(f"\nPick {n} indices (space/comma). Enter=auto: ").strip()
    except EOFError:
        return None

    if not raw:
        return None

    parts = re.split(r"[\s,]+", raw)
    picked: List[str] = []
    for p in parts:
        if not p:
            continue
        try:
            ix = int(p)
        except ValueError:
            continue
        if ix < 1 or ix > len(candidates):
            continue
        tok = candidates[ix - 1].get("token")
        if not isinstance(tok, str):
            continue
        tok2 = tok.strip()
        if not tok2:
            continue
        if tok2 in picked:
            continue
        picked.append(tok2)
        if len(picked) >= n:
            break

    if not picked:
        return None
    if len(picked) < n:
        rest = [str(c.get("token", "")).strip() for c in candidates if str(c.get("token", "")).strip() and str(c.get("token", "")).strip() not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: (n - len(picked))])
    return picked[:n]


def parse_ponder_pipeline(pipeline: str, *, fallback_mode: str) -> List[str]:
    allowed = {"assoc", "assumption", "counterexample", "questions_only", "metaphor"}
    s = (pipeline or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in re.split(r"[,\s]+", s) if p.strip()]
    out: List[str] = []
    for p in parts:
        if p not in allowed:
            raise ValueError(f"Unknown ponder_mode in pipeline: {p!r}")
        out.append(p)
    return out or []


def explicit_dests_from_argv(ap: argparse.ArgumentParser, argv: Sequence[str]) -> set[str]:
    opt_to_dest: Dict[str, str] = {}
    for a in ap._actions:
        for opt in getattr(a, "option_strings", []) or []:
            opt_to_dest[str(opt)] = str(getattr(a, "dest", ""))

    out: set[str] = set()
    for tok in list(argv or [])[1:]:
        t = str(tok)
        if not t.startswith("-"):
            continue
        key = t.split("=", 1)[0] if "=" in t else t
        dest = opt_to_dest.get(key)
        if dest:
            out.add(dest)
    return out


def apply_preset_inplace(args: argparse.Namespace, *, explicit_dests: Optional[set[str]] = None) -> None:
    preset = str(getattr(args, "preset", "none") or "none").strip().lower()
    if preset in ("", "none"):
        return
    if preset != "surreal":
        raise ValueError(f"Unknown preset: {preset!r}")

    # Apply curated surreal defaults, but only when the user didn't already opt into something else.
    # Prefer options that can still be overridden by explicit CLI flags.
    explicit_dests = explicit_dests or set()

    def ok(dest: str) -> bool:
        return str(dest) not in explicit_dests

    if ok("ponder_pipeline") and str(getattr(args, "ponder_pipeline", "") or "").strip() == "":
        args.ponder_pipeline = "metaphor,metaphor"
    if (
        ok("band_profile")
        and ok("band")
        and str(getattr(args, "band_profile", "single") or "single").strip() == "single"
        and not list(getattr(args, "band", []) or [])
    ):
        args.band_profile = "spectrum3"
    if ok("ponder_hops") and int(getattr(args, "ponder_hops", 1) or 1) <= 1:
        args.ponder_hops = 3
    if ok("keyword_objective") and str(getattr(args, "keyword_objective", "random_band") or "random_band").strip() == "random_band":
        args.keyword_objective = "dissonance"
    if ok("keyword_diversity") and str(getattr(args, "keyword_diversity", "off") or "off").strip() == "off":
        args.keyword_diversity = "embed"
    if ok("memory_policy") and str(getattr(args, "memory_policy", "tail") or "tail").strip() == "tail":
        args.memory_policy = "current_only"
    if ok("memory_remix") and str(getattr(args, "memory_remix", "off") or "off").strip() == "off":
        args.memory_remix = "dream"
    if ok("prompt_jitter") and int(getattr(args, "prompt_jitter", 0) or 0) <= 0:
        args.prompt_jitter = 2
    if ok("answer_style") and str(getattr(args, "answer_style", "plain") or "plain").strip() in ("plain", "default"):
        args.answer_style = "surreal"


def _cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    aa = a.detach().float().cpu()
    bb = b.detach().float().cpu()
    denom = (torch.linalg.norm(aa) * torch.linalg.norm(bb)).item() + eps
    if denom <= 0:
        return 0.0
    return float(torch.dot(aa, bb).item() / denom)


_SIM_TEXT_CLEAN_RE = re.compile(r"[^\w\u3040-\u30ff\u3400-\u9fff]+", re.UNICODE)


def _normalize_sim_text(text: str) -> str:
    s = (text or "").lower()
    s = _SIM_TEXT_CLEAN_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _fnv1a_32(s: str) -> int:
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _hashed_char_ngrams(text: str, *, n: int = 3, dim: int = 4096, max_chars: int = 4000) -> Dict[int, float]:
    s = _normalize_sim_text(text)
    if max_chars > 0 and len(s) > max_chars:
        s = s[-int(max_chars) :]
    if not s:
        return {}
    d: Dict[int, float] = {}
    nn = max(1, int(n))
    if len(s) <= nn:
        ix = _fnv1a_32(s) % int(dim)
        d[ix] = d.get(ix, 0.0) + 1.0
        return d
    for i in range(0, len(s) - nn + 1):
        gram = s[i : i + nn]
        ix = _fnv1a_32(gram) % int(dim)
        d[ix] = d.get(ix, 0.0) + 1.0
    return d


def _add_hashed_char_ngrams_multi(
    dst: Dict[int, float],
    text: str,
    *,
    ns: Sequence[int],
    dim: int,
    max_chars: int,
    scale: float,
    n_weights: Dict[int, float],
) -> None:
    s = _normalize_sim_text(text)
    if max_chars > 0 and len(s) > int(max_chars):
        s = s[-int(max_chars) :]
    if not s:
        return

    base_dim = max(1, int(dim))
    for n_i, n in enumerate(ns):
        nn = max(1, int(n))
        w = float(scale) * float(n_weights.get(nn, 1.0))
        off = n_i * base_dim
        if len(s) <= nn:
            ix = (_fnv1a_32(s) % base_dim) + off
            dst[ix] = dst.get(ix, 0.0) + w
            continue
        for i in range(0, len(s) - nn + 1):
            gram = s[i : i + nn]
            ix = (_fnv1a_32(gram) % base_dim) + off
            dst[ix] = dst.get(ix, 0.0) + w


def _sparse_dot(a: Dict[int, float], b: Dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for k, av in a.items():
        bv = b.get(k)
        if bv:
            dot += float(av) * float(bv)
    return float(dot)


def _sparse_norm(a: Dict[int, float]) -> float:
    if not a:
        return 0.0
    return float(math.sqrt(sum(float(v) * float(v) for v in a.values())))


def _cosine_with_norm(a: Dict[int, float], na: float, b: Dict[int, float], nb: float, eps: float = 1e-12) -> float:
    denom = float(na) * float(nb) + float(eps)
    if denom <= 0:
        return 0.0
    return float(_sparse_dot(a, b) / denom)


def _tfidf_vec(tf: Dict[int, float], idf: Dict[int, float]) -> Tuple[Dict[int, float], float]:
    """Return (tfidf_vec, norm)."""
    vec: Dict[int, float] = {}
    norm2 = 0.0
    for k, f in tf.items():
        idf_w = float(idf.get(k, 0.0))
        if idf_w <= 0:
            continue
        w = math.log1p(float(f)) * idf_w
        if w == 0.0:
            continue
        vec[k] = w
        norm2 += w * w
    return vec, float(math.sqrt(norm2))


def _memory_record_text_for_sim(r: Dict[str, Any]) -> str:
    kws = r.get("keywords")
    if not kws:
        kws = r.get("keywords_raw")
    if isinstance(kws, list):
        kw_s = " ".join(str(x) for x in kws if x is not None)
    else:
        kw_s = str(kws or "")
    pq = str(r.get("ponder_question", "") or "")
    plog = str(r.get("ponder_log", "") or "")
    if len(plog) > 1600:
        plog = plog[:1600]
    return f"{kw_s}\n{pq}\n{plog}".strip()


def _current_text_for_sim(query: str, current_records: Sequence[Dict[str, Any]]) -> str:
    # Prefer the raw query; add a small keyword summary to stabilize similarity across languages.
    bits: List[str] = []
    q = (query or "").strip()
    if q:
        bits.append(q)

    seen = set()
    kws_out: List[str] = []
    for r in current_records or []:
        kws = r.get("keywords") or r.get("keywords_raw") or []
        if not isinstance(kws, list):
            continue
        for k in kws:
            k2 = str(k or "").strip()
            if not k2:
                continue
            if k2 in seen:
                continue
            seen.add(k2)
            kws_out.append(k2)
            if len(kws_out) >= 32:
                break
        if len(kws_out) >= 32:
            break
    if kws_out:
        bits.append("keywords: " + " ".join(kws_out))

    if not bits and current_records:
        # Final fallback: use some of the freshly generated logs.
        for r in current_records[:2]:
            bits.append(_memory_record_text_for_sim(dict(r)))

    return "\n".join(bits).strip()


def select_memory_records_fuzzy(
    *,
    memory_path: Path,
    current_records: Sequence[Dict[str, Any]],
    query: str,
    memory_policy: str,
    memory_retrieve: str,
    n_memory: int,
    pool_size: int,
    mix_ratio: float,
    exclude_run_id: Optional[str],
    sim_ngram_n: int = 3,
    sim_dim: int = 4096,
    sim_max_chars: int = 4000,
) -> List[Dict[str, Any]]:
    """Approximate memory retrieval using hashed character n-gram retrieval (IDF-weighted).

    This is a provider-agnostic fallback (works for API backends and when embedding retrieval isn't available).
    """

    if memory_policy == "off":
        return []
    if memory_policy == "current_only":
        take = max(0, int(n_memory))
        if take <= 0:
            return []
        cur = list(current_records)
        return cur[-take:] if len(cur) > take else cur
    if memory_policy != "tail":
        raise ValueError(f"Unknown memory_policy: {memory_policy!r}")

    pool = tail_jsonl(memory_path, max(0, int(pool_size)))
    if not pool:
        return []
    if exclude_run_id:
        pool = [r for r in pool if str(r.get("run_id", "")) != str(exclude_run_id)]
    if not pool:
        return []

    take = max(0, int(n_memory))
    if take <= 0:
        return []

    if memory_retrieve == "tail":
        return pool[-take:]

    # Multi-gram hashed TF -> IDF-weighted cosine similarity.
    # We treat `sim_dim` as "dim per n-gram", and reserve a separate slice per n to reduce collisions.
    center_n = max(2, int(sim_ngram_n))
    ns = [n for n in (center_n - 1, center_n, center_n + 1) if n >= 2]
    ns = list(dict.fromkeys(ns))  # stable unique
    n_weights: Dict[int, float] = {}
    for n in ns:
        if n <= 2:
            n_weights[n] = 0.55
        elif n == 3:
            n_weights[n] = 1.0
        elif n == 4:
            n_weights[n] = 1.25
        else:
            n_weights[n] = 1.35

    # Field weighting: keywords matter most, then ponder_question, then ponder_log.
    kw_w, pq_w, log_w = 2.2, 1.6, 1.0
    kw_max, pq_max, log_max = 480, 900, int(sim_max_chars)

    cur_text = _current_text_for_sim(query, current_records)
    cur_tf: Dict[int, float] = {}
    _add_hashed_char_ngrams_multi(
        cur_tf,
        cur_text,
        ns=ns,
        dim=int(sim_dim),
        max_chars=int(sim_max_chars),
        scale=1.0,
        n_weights=n_weights,
    )
    if not cur_tf:
        return pool[-take:]

    # Build TF for each doc + DF for IDF
    doc_records: List[Dict[str, Any]] = []
    doc_tfs: List[Dict[int, float]] = []
    dfs: Dict[int, int] = {}

    for r in pool:
        tf: Dict[int, float] = {}
        # keywords
        kws = r.get("keywords") or r.get("keywords_raw") or []
        if isinstance(kws, list):
            kw_s = " ".join(str(x) for x in kws if x is not None)
        else:
            kw_s = str(kws or "")
        pq = str(r.get("ponder_question", "") or "")
        plog = str(r.get("ponder_log", "") or "")

        if kw_s:
            _add_hashed_char_ngrams_multi(
                tf,
                kw_s,
                ns=ns,
                dim=int(sim_dim),
                max_chars=kw_max,
                scale=kw_w,
                n_weights=n_weights,
            )
        if pq:
            _add_hashed_char_ngrams_multi(
                tf,
                pq,
                ns=ns,
                dim=int(sim_dim),
                max_chars=pq_max,
                scale=pq_w,
                n_weights=n_weights,
            )
        if plog:
            _add_hashed_char_ngrams_multi(
                tf,
                plog,
                ns=ns,
                dim=int(sim_dim),
                max_chars=log_max,
                scale=log_w,
                n_weights=n_weights,
            )

        if not tf:
            continue
        doc_records.append(r)
        doc_tfs.append(tf)
        for k in tf.keys():
            dfs[k] = dfs.get(k, 0) + 1

    if not doc_records:
        return pool[-take:]

    n_docs = len(doc_records)
    idf: Dict[int, float] = {}
    for k, df in dfs.items():
        idf[k] = float(math.log((n_docs + 1.0) / (float(df) + 1.0)) + 1.0)

    q_vec, q_norm = _tfidf_vec(cur_tf, idf)
    if not q_vec or q_norm <= 0:
        return doc_records[-take:]

    doc_vecs: List[Dict[int, float]] = []
    doc_norms: List[float] = []
    for tf in doc_tfs:
        v, nrm = _tfidf_vec(tf, idf)
        doc_vecs.append(v)
        doc_norms.append(nrm)

    # Similarity to query
    scored: List[Tuple[float, int]] = []
    for i, dv in enumerate(doc_vecs):
        sim = _cosine_with_norm(q_vec, q_norm, dv, doc_norms[i])
        scored.append((float(sim), int(i)))

    scored.sort(key=lambda t: (t[0], t[1]))

    def _mmr_select(cand_ids: List[int], k: int, *, lam: float = 0.9) -> List[int]:
        k = max(0, int(k))
        if k <= 0 or not cand_ids:
            return []
        selected: List[int] = []
        remaining = list(cand_ids)
        while remaining and len(selected) < k:
            best_id: Optional[int] = None
            best_score = -1e9
            for cid in remaining:
                rel = float(scored_map.get(cid, 0.0))
                if not selected:
                    div = 0.0
                else:
                    div = 0.0
                    for sid in selected:
                        div = max(div, _cosine_with_norm(doc_vecs[cid], doc_norms[cid], doc_vecs[sid], doc_norms[sid]))
                mmr = float(lam) * rel - (1.0 - float(lam)) * div
                if mmr > best_score + 1e-12:
                    best_score = mmr
                    best_id = cid
                elif abs(mmr - best_score) <= 1e-12 and best_id is not None and cid > best_id:
                    # tie-break: prefer more recent (higher index in tail pool)
                    best_id = cid
            if best_id is None:
                break
            selected.append(best_id)
            remaining.remove(best_id)
        return selected

    # Map for quick access
    scored_map = {i: s for s, i in scored}
    top_sim = float(scored[-1][0]) if scored else 0.0
    rel_floor = max(0.01, top_sim * 0.35)

    if memory_retrieve == "anti":
        return [doc_records[i] for _, i in scored[:take]]

    if memory_retrieve == "similar":
        # Candidate pool for MMR
        cand_pool = min(n_docs, max(50, take * 12))
        top_ids = [i for _, i in scored[-cand_pool:]][::-1]
        good_ids = [i for i in top_ids if float(scored_map.get(i, 0.0)) >= rel_floor]
        if not good_ids:
            good_ids = top_ids[:]
        picked = _mmr_select(good_ids, min(take, len(good_ids)), lam=0.9)
        if not picked:
            picked = top_ids[:take]
        return [doc_records[i] for i in picked]

    if memory_retrieve == "mix":
        k_sim_target = max(0, min(take, int(round(take * float(mix_ratio)))))
        cand_pool = min(n_docs, max(50, k_sim_target * 12))
        top_ids = [i for _, i in scored[-cand_pool:]][::-1]
        good_ids = [i for i in top_ids if float(scored_map.get(i, 0.0)) >= rel_floor]
        if not good_ids:
            good_ids = top_ids[:]
        sim_ids = _mmr_select(good_ids, min(k_sim_target, len(good_ids)), lam=0.9)
        if not sim_ids:
            sim_ids = top_ids[:k_sim_target]

        used = set(sim_ids)
        k_anti = max(0, take - len(sim_ids))
        anti_ids = [i for _, i in scored[: max(k_anti * 3, k_anti)]]
        anti_ids2 = [i for i in anti_ids if i not in used]
        # fill if overlap (tiny pools)
        if len(anti_ids2) < k_anti:
            for _, i in scored:
                if i in used or i in anti_ids2:
                    continue
                anti_ids2.append(i)
                if len(anti_ids2) >= k_anti:
                    break

        out_ids = (sim_ids + anti_ids2[:k_anti])[:take]
        return [doc_records[i] for i in out_ids]

    raise ValueError(f"Unknown memory_retrieve: {memory_retrieve!r}")


def select_memory_records(
    hf: "LocalHFModel",
    *,
    memory_path: Path,
    current_records: Sequence[Dict[str, Any]],
    query: str,
    memory_policy: str,
    memory_retrieve: str,
    n_memory: int,
    pool_size: int,
    mix_ratio: float,
    exclude_run_id: Optional[str],
) -> List[Dict[str, Any]]:
    if memory_policy == "off":
        return []
    if memory_policy == "current_only":
        take = max(0, int(n_memory))
        if take <= 0:
            return []
        cur = list(current_records)
        return cur[-take:] if len(cur) > take else cur
    if memory_policy != "tail":
        raise ValueError(f"Unknown memory_policy: {memory_policy!r}")

    pool = tail_jsonl(memory_path, max(0, int(pool_size)))
    if not pool:
        return []

    if memory_retrieve == "tail":
        return pool[-int(n_memory) :] if n_memory > 0 else []

    cur_token_ids: List[int] = []
    for r in current_records:
        tids = r.get("token_ids")
        if isinstance(tids, list):
            for t in tids:
                try:
                    cur_token_ids.append(int(t))
                except Exception:
                    pass
    cur_vec = mean_embedding_for_token_ids(hf, cur_token_ids)
    if cur_vec is None:
        # Fallback: approximate similarity on text when embedding retrieval isn't available.
        return select_memory_records_fuzzy(
            memory_path=memory_path,
            current_records=current_records,
            query=query,
            memory_policy="tail",
            memory_retrieve=memory_retrieve,
            n_memory=n_memory,
            pool_size=pool_size,
            mix_ratio=mix_ratio,
            exclude_run_id=exclude_run_id,
        )

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for r in pool:
        if exclude_run_id and str(r.get("run_id", "")) == str(exclude_run_id):
            continue
        tids = r.get("token_ids")
        if not isinstance(tids, list) or not tids:
            continue
        vec = mean_embedding_for_token_ids(hf, [int(x) for x in tids if x is not None])
        if vec is None:
            continue
        sim = _cosine_sim(cur_vec, vec)
        scored.append((sim, r))

    if not scored:
        return pool[-int(n_memory) :] if n_memory > 0 else []

    scored.sort(key=lambda x: x[0])
    take = max(0, int(n_memory))
    if take == 0:
        return []

    if memory_retrieve == "anti":
        return [r for _, r in scored[:take]]
    if memory_retrieve == "similar":
        return [r for _, r in scored[-take:]][::-1]
    if memory_retrieve == "mix":
        k_sim = max(0, min(take, int(round(take * float(mix_ratio)))))
        k_anti = take - k_sim
        anti = [r for _, r in scored[:k_anti]]
        sim = [r for _, r in scored[-k_sim:]][::-1]
        return sim + anti

    raise ValueError(f"Unknown memory_retrieve: {memory_retrieve!r}")


def build_prompt_for_memory_remix(memory_block: str, *, lang: str) -> str:
    if lang == "ja":
        return (
            "あなたは編集者です。\n"
            "以下の ponder log の断片を素材に、結論を出さない「夢のコラージュ」を作ってください。\n"
            "制約:\n"
            "- 10〜15行、各行は - で始める\n"
            "- 断定/結論/実用的助言は禁止\n\n"
            "<ponder_memory>\n"
            f"{memory_block}\n"
            "</ponder_memory>\n"
        )
    return (
        "You are an editor.\n"
        "Using the ponder log fragments below, produce a dreamlike collage without conclusions.\n"
        "Constraints:\n"
        "- 10 to 15 lines, each line begins with -\n"
        "- No conclusions or practical advice\n\n"
        "<ponder_memory>\n"
        f"{memory_block}\n"
        "</ponder_memory>\n"
    )


def remix_memory_block(
    hf: "LocalHFModel",
    *,
    memory_block: Optional[str],
    remix: str,
    lang: str,
    keep_original: bool,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> Optional[str]:
    if not memory_block:
        return None
    mode = (remix or "off").strip()
    if mode == "off":
        return memory_block

    if mode == "shuffle":
        lines = [ln for ln in memory_block.splitlines() if ln.strip()]
        rng = random.Random(int(seed) + 31337)
        rng.shuffle(lines)
        remixed = "\n".join(lines)
        return (memory_block + "\n\n---\n\n" + remixed) if keep_original else remixed

    if mode == "compress":
        out_lines: List[str] = []
        in_log = False
        kept_bullets = 0
        for ln in memory_block.splitlines():
            s = ln.strip()
            if s.startswith("[") or s.startswith("keywords:") or s.startswith("ponder_q:"):
                out_lines.append(ln)
                in_log = False
                kept_bullets = 0
                continue
            if s.startswith("ponder_log:"):
                out_lines.append(ln)
                in_log = True
                kept_bullets = 0
                continue
            if in_log and s.startswith("-"):
                if kept_bullets < 2:
                    out_lines.append(ln)
                    kept_bullets += 1
                continue
        remixed = "\n".join(out_lines).strip()
        return (memory_block + "\n\n---\n\n" + remixed) if keep_original else remixed

    if mode == "dream":
        prompt = hf._apply_chat(build_prompt_for_memory_remix(memory_block, lang=lang), system_text=None)
        dream = hf.generate_text(
            prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=0.95,
            top_k=0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            seed=int(seed),
        )
        return (memory_block + "\n\n---\n\n" + dream) if keep_original else dream

    raise ValueError(f"Unknown memory_remix: {mode!r}")


def build_prompt_for_ensemble_summary(
    query: str,
    band_answers: Dict[str, str],
    *,
    lang: str,
) -> str:
    blocks: List[str] = []
    for k, v in band_answers.items():
        blocks.append(f"[{k}]\n{v}\n")
    joined = "\n".join(blocks).strip()
    if lang == "ja":
        return (
            "あなたは複数回答の比較編集者です。\n"
            "次の band ごとの回答を比較し、共通点・相違点・統合回答を出してください。\n"
            "出力フォーマット（タグはそのまま使う）:\n"
            "<consensus>\n- ...\n</consensus>\n"
            "<divergence>\n- near: ...\n</divergence>\n"
            "<final>\n(統合した最終回答本文)\n</final>\n\n"
            f"本題: {query}\n\n"
            "<answers>\n"
            f"{joined}\n"
            "</answers>\n"
        )
    return (
        "You are an editor comparing multiple answers.\n"
        "Compare the band-specific answers and output consensus, divergence, and a merged final answer.\n"
        "Output format (keep tags):\n"
        "<consensus>\n- ...\n</consensus>\n"
        "<divergence>\n- near: ...\n</divergence>\n"
        "<final>\n(merged final answer)\n</final>\n\n"
        f"Main question: {query}\n\n"
        "<answers>\n"
        f"{joined}\n"
        "</answers>\n"
    )


def extract_tag(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{re.escape(tag)}>\n?(.*?)\n?</{re.escape(tag)}>", text, flags=re.DOTALL)
    if not m:
        return None
    return (m.group(1) or "").strip()


# -----------------------------
# API backend (OpenAI-compatible)
# -----------------------------


def _join_url(base_url: str, path: str) -> str:
    base = (base_url or "").rstrip("/")
    p = (path or "").lstrip("/")
    return f"{base}/{p}" if p else base


def _parse_header_kv(s: str) -> Tuple[str, str]:
    if ":" not in (s or ""):
        raise ValueError(f"Invalid header (expected Key: Value): {s!r}")
    k, v = s.split(":", 1)
    k = k.strip()
    v = v.strip()
    if not k:
        raise ValueError(f"Invalid header key: {s!r}")
    return k, v


def _http_post_json(
    url: str,
    *,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: float,
    max_retries: int,
) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    req_headers.update(headers or {})

    last_err: Optional[str] = None
    tries = max(0, int(max_retries)) + 1

    for attempt in range(tries):
        req = urlrequest.Request(url, data=body, headers=req_headers, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=float(timeout)) as resp:
                raw = resp.read()
                try:
                    return json.loads(raw.decode("utf-8"))
                except Exception:
                    raise RuntimeError("Invalid JSON response")
        except urlerror.HTTPError as e:
            status = getattr(e, "code", None)
            err_text = ""
            try:
                err_text = e.read().decode("utf-8", errors="replace")
            except Exception:
                err_text = ""
            err_text = (err_text or "").strip().replace("\n", " ")
            if len(err_text) > 600:
                err_text = err_text[:600] + "…"
            last_err = f"HTTP {status}: {err_text}" if status is not None else f"HTTPError: {err_text}"

            retryable = status in (408, 409, 429, 500, 502, 503, 504)
            if attempt < tries - 1 and retryable:
                delay = min(8.0, 0.6 * (2**attempt) + random.random() * 0.2)
                time.sleep(delay)
                continue
            break
        except urlerror.URLError as e:
            last_err = f"URLError: {e}"
            if attempt < tries - 1:
                delay = min(8.0, 0.6 * (2**attempt) + random.random() * 0.2)
                time.sleep(delay)
                continue
            break

    raise RuntimeError(last_err or "Unknown HTTP error")


_GPT5_VERSIONED_RE = re.compile(r"^gpt-5\.(\d+)(?:[.\-].*)?$")


def _is_openai_official_base(base_url: str) -> bool:
    s = (base_url or "").strip().lower()
    return s.startswith("https://api.openai.com/") or s == "https://api.openai.com"


def _default_reasoning_effort_for_model(model: str) -> str:
    m = (model or "").strip().lower()
    if not m:
        return ""
    if m.startswith("gpt-5-pro"):
        return "high"
    if _GPT5_VERSIONED_RE.match(m):
        return "none"
    return ""


def _extract_reasoning_tokens(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        return 0
    try:
        return int(details.get("reasoning_tokens") or 0)
    except Exception:
        return 0


class OpenAICompatModel:
    """OpenAI-compatible Chat Completions client (stdlib only).

    Works with OpenAI-like providers that implement `POST /chat/completions`.
    """

    def __init__(
        self,
        *,
        model: str,
        api_base_url: str,
        api_key: str,
        api_chat_path: str = "/chat/completions",
        api_headers: Optional[Sequence[str]] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        api_reasoning_effort: str = "auto",
    ) -> None:
        self.model = (model or "").strip()
        self.api_base_url = (api_base_url or "").strip()
        self.api_chat_path = api_chat_path or "/chat/completions"
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.api_reasoning_effort = (api_reasoning_effort or "auto").strip().lower()
        self.last_response_meta: Dict[str, Any] = {}

        headers: Dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        for h in api_headers or []:
            try:
                k, v = _parse_header_kv(str(h))
                headers[k] = v
            except Exception as e:
                print(f"[sr_ponder] WARN: ignoring malformed --api_header {h!r}: {e}", file=sys.stderr)
                continue
        self.headers = headers

        self.url = _join_url(self.api_base_url, self.api_chat_path)

    def _apply_openai_compat_defaults(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not _is_openai_official_base(self.api_base_url):
            return payload
        effort = self.api_reasoning_effort
        if effort == "auto":
            effort = _default_reasoning_effort_for_model(self.model)
        if effort and "reasoning_effort" not in payload:
            payload["reasoning_effort"] = effort
        return payload

    def _maybe_adjust_payload_for_error(self, payload: Dict[str, Any], msg: str) -> Optional[Dict[str, Any]]:
        msg_l = (msg or "").lower()
        payload2 = dict(payload)
        changed = False

        if "max_tokens" in payload2 and ("max_tokens" in msg_l and "max_completion_tokens" in msg_l):
            v = payload2.pop("max_tokens", None)
            if v is not None and "max_completion_tokens" not in payload2:
                payload2["max_completion_tokens"] = v
                changed = True
        elif "max_completion_tokens" in payload2 and ("max_completion_tokens" in msg_l and "max_tokens" in msg_l):
            v = payload2.pop("max_completion_tokens", None)
            if v is not None and "max_tokens" not in payload2:
                payload2["max_tokens"] = v
                changed = True

        if "temperature" in payload2 and ("temperature" in msg_l) and ("unsupported" in msg_l or "default (1)" in msg_l):
            if "default (1)" in msg_l:
                if float(payload2.get("temperature", 1.0) or 1.0) != 1.0:
                    payload2["temperature"] = 1.0
                    changed = True
            else:
                payload2.pop("temperature", None)
                changed = True

        if "top_p" in payload2 and ("top_p" in msg_l) and ("unsupported" in msg_l or "not supported" in msg_l):
            payload2.pop("top_p", None)
            changed = True

        if ("logprobs" in payload2 or "top_logprobs" in payload2) and (
            ("logprobs" in msg_l or "top_logprobs" in msg_l) and ("unsupported" in msg_l or "not supported" in msg_l)
        ):
            payload2.pop("logprobs", None)
            payload2.pop("top_logprobs", None)
            changed = True

        if "reasoning_effort" in payload2 and ("reasoning_effort" in msg_l) and ("unsupported" in msg_l or "not supported" in msg_l):
            payload2.pop("reasoning_effort", None)
            changed = True

        if not changed:
            return None
        return payload2

    def _extract_text_and_meta(self, resp: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        choices = resp.get("choices") or []
        if not choices:
            raise RuntimeError("No choices in API response")
        c0 = choices[0] or {}
        msg = c0.get("message") or {}
        usage = resp.get("usage") or {}
        meta: Dict[str, Any] = {
            "response_id": resp.get("id"),
            "model": resp.get("model"),
            "finish_reason": c0.get("finish_reason"),
            "refusal": msg.get("refusal"),
            "usage": usage if isinstance(usage, dict) else None,
            "completion_tokens": int(usage.get("completion_tokens") or 0) if isinstance(usage, dict) else 0,
            "reasoning_tokens": _extract_reasoning_tokens(usage),
        }

        def _extract_parts(obj: Any) -> List[str]:
            parts: List[str] = []
            if isinstance(obj, str):
                s = obj.strip()
                if s:
                    parts.append(s)
                return parts
            if isinstance(obj, list):
                for item in obj:
                    parts.extend(_extract_parts(item))
                return parts
            if not isinstance(obj, dict):
                return parts

            text_val = obj.get("text")
            if isinstance(text_val, str) and text_val.strip():
                parts.append(text_val.strip())
            elif isinstance(text_val, dict):
                value = text_val.get("value")
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())

            for key in ("value", "output_text", "content"):
                val = obj.get(key)
                if isinstance(val, (str, list, dict)):
                    parts.extend(_extract_parts(val))
            return parts

        content = msg.get("content")
        parts = _extract_parts(content)
        if not parts:
            parts = _extract_parts(c0.get("text"))
        if not parts and isinstance(msg.get("refusal"), str) and str(msg.get("refusal")).strip():
            parts = [str(msg.get("refusal")).strip()]
            meta["used_refusal_text"] = True

        text = "\n".join(parts).strip() if parts else ""
        meta["empty_output"] = not bool(text)
        return text, meta

    def _apply_chat(self, prompt: str, *, system_text: Optional[str]) -> str:
        # Keep interface parity with LocalHFModel; API backends usually support system roles,
        # but we keep it as plain text to avoid provider differences.
        if system_text:
            return f"{system_text.strip()}\n\n{prompt}"
        return prompt

    def _chat(self, *, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": self.model, "messages": messages}
        payload.update(kwargs)
        payload = self._apply_openai_compat_defaults(payload)

        attempts = 0
        last_err: Optional[Exception] = None
        while attempts < 5:
            try:
                return _http_post_json(
                    self.url,
                    headers=self.headers,
                    payload=payload,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                )
            except RuntimeError as e:
                last_err = e
                payload2 = self._maybe_adjust_payload_for_error(payload, str(e))
                if payload2 is None or payload2 == payload:
                    raise
                payload = payload2
                attempts += 1
        if last_err is not None:
            raise last_err
        raise RuntimeError("Unknown API compatibility error")

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 768,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 0,
        seed: Optional[int] = None,
    ) -> str:
        # NOTE: We intentionally do not forward unsupported params (top_k, repetition_penalty, seed, etc.)
        # to maximize compatibility across "OpenAI-compatible" providers.
        _ = (top_k, repetition_penalty, no_repeat_ngram_size, seed)
        resp = self._chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=int(max_new_tokens),
            temperature=max(0.0, float(temperature)),
            top_p=float(top_p),
        )
        text, meta = self._extract_text_and_meta(resp)
        self.last_response_meta = meta
        return text

    def probe_top_logprobs(self, prompt: str, *, top_n: int) -> List[Dict[str, Any]]:
        """Return a list of {token, logprob, prob} for the next token distribution (top-N only)."""
        n = max(0, int(top_n))
        if n <= 0:
            return []
        resp = self._chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            logprobs=True,
            top_logprobs=n,
        )
        choices = resp.get("choices") or []
        if not choices:
            return []
        c0 = choices[0] or {}
        lp = c0.get("logprobs") or {}

        top_items: List[Dict[str, Any]] = []
        try:
            # Chat Completions style: logprobs.content[0].top_logprobs[]
            content = lp.get("content") or []
            if isinstance(content, list) and content:
                first = content[0] or {}
                tlogs = first.get("top_logprobs") or []
                if isinstance(tlogs, list):
                    for x in tlogs:
                        tok = x.get("token")
                        lpr = x.get("logprob")
                        if not isinstance(tok, str):
                            continue
                        if not isinstance(lpr, (int, float)):
                            continue
                        tok2 = clean_token_text(tok)
                        if not tok2:
                            continue
                        p = float(math.exp(float(lpr))) if float(lpr) > -1e9 else 0.0
                        top_items.append({"token": tok2, "logprob": float(lpr), "prob": p})
        except Exception:
            top_items = []

        # Some providers may return alternative shapes; best-effort fallback.
        if not top_items and isinstance(lp, dict) and "top_logprobs" in lp:
            tlogs = lp.get("top_logprobs") or []
            if isinstance(tlogs, list) and tlogs:
                first = tlogs[0]
                if isinstance(first, dict):
                    for tok, lpr in first.items():
                        if not isinstance(tok, str) or not isinstance(lpr, (int, float)):
                            continue
                        tok2 = clean_token_text(tok)
                        if not tok2:
                            continue
                        p = float(math.exp(float(lpr))) if float(lpr) > -1e9 else 0.0
                        top_items.append({"token": tok2, "logprob": float(lpr), "prob": p})

        top_items.sort(key=lambda d: float(d.get("logprob", -1e9)), reverse=True)
        # Add rank
        for i, d in enumerate(top_items):
            d["rank"] = i
        return top_items[:n]


# -----------------------------
# Model wrapper (MPS-safe + Gemma-aware)
# -----------------------------


class LocalHFModel:
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        use_chat_template: bool = True,
        force_gemma_format: bool = True,
        allocator_warmup: str = "auto",
        local_files_only: bool = True,
    ) -> None:
        self.model_path = resolve_model_ref(model_path)
        self.use_chat_template = use_chat_template
        self.force_gemma_format = force_gemma_format

        self.device_str = resolve_device(device)
        self.device = torch.device(self.device_str)
        self.torch_dtype = resolve_dtype(dtype, self.device_str)

        configure_transformers_allocator_warmup(allocator_warmup)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=bool(local_files_only),
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = dict(
            local_files_only=bool(local_files_only),
            trust_remote_code=trust_remote_code,
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=self.torch_dtype,
                **model_kwargs,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                **model_kwargs,
            )

        self.model.to(self.device)
        self.model.eval()

        self.input_device = self._infer_input_device()

        # Gemma turn tokens (if present)
        self.start_of_turn_id = self._tok_id("<start_of_turn>")
        self.end_of_turn_id = self._tok_id("<end_of_turn>")

        self._warned_prompt_trunc: bool = False

    def _max_context_tokens(self) -> Optional[int]:
        cfg = getattr(self.model, "config", None)
        cands: List[int] = []
        for k in ("max_position_embeddings", "n_positions", "n_ctx", "max_seq_len", "seq_length"):
            v = getattr(cfg, k, None) if cfg is not None else None
            if isinstance(v, int) and 32 <= v <= 262144:
                cands.append(int(v))
        tmax = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tmax, int) and 32 <= tmax <= 262144:
            cands.append(int(tmax))
        if not cands:
            return None
        # Be conservative: some tokenizers expose a larger max than the model actually supports.
        return int(min(cands))

    def _truncate_batch_left(self, batch: Dict[str, Any], *, max_len: int) -> Dict[str, Any]:
        ids = batch.get("input_ids")
        if ids is None or not torch.is_tensor(ids):
            return batch
        if ids.ndim != 2:
            return batch
        cur = int(ids.shape[1])
        if cur <= int(max_len):
            return batch
        sl = slice(cur - int(max_len), cur)
        batch["input_ids"] = ids[:, sl]
        for k in ("attention_mask", "token_type_ids"):
            x = batch.get(k)
            if x is not None and torch.is_tensor(x) and x.ndim == 2 and int(x.shape[1]) == cur:
                batch[k] = x[:, sl]
        return batch

    def _tok_id(self, token: str) -> Optional[int]:
        try:
            tid = self.tokenizer.convert_tokens_to_ids(token)
        except Exception:
            return None
        unk = getattr(self.tokenizer, "unk_token_id", None)
        if tid is None:
            return None
        if unk is not None and tid == unk:
            return None
        return int(tid)

    def _has_gemma_turn_tokens(self) -> bool:
        return self.start_of_turn_id is not None and self.end_of_turn_id is not None

    def _infer_input_device(self) -> torch.device:
        try:
            emb = self.model.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                return emb.weight.device
        except Exception:
            pass
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.device

    def _move_to_input_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.input_device)
            else:
                out[k] = v
        return out

    def _apply_gemma_turn_format(self, user_text: str) -> str:
        # Gemma prompt structure: <start_of_turn>user ... <end_of_turn>\n<start_of_turn>model
        return (
            "<start_of_turn>user\n"
            f"{user_text}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

    def _apply_chat(self, user_text: str, system_text: Optional[str] = None) -> str:
        # For Gemma IT: system role isn’t supported; inline system instructions into user turn.
        if system_text:
            user_text = system_text.rstrip() + "\n" + user_text.lstrip()

        # Prefer Gemma-native formatting if we can.
        if self.force_gemma_format and self._has_gemma_turn_tokens():
            return self._apply_gemma_turn_format(user_text)

        # Otherwise, use tokenizer chat template if available.
        if self.use_chat_template:
            tok = self.tokenizer
            try:
                if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
                    messages = [{"role": "user", "content": user_text}]
                    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        # Last-resort fallback
        return f"{user_text}\n\nAssistant:"

    @_inference_mode()
    def next_token_logits(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        max_ctx = self._max_context_tokens()
        if max_ctx is not None:
            inputs = self._truncate_batch_left(inputs, max_len=int(max_ctx))
        inputs = self._move_to_input_device(inputs)
        out = self.model(**inputs)
        return out.logits[0, -1, :].detach().float().cpu()

    def _eos_ids(self) -> List[int]:
        eos: List[int] = []
        # tokenizer/model eos
        tid = getattr(self.tokenizer, "eos_token_id", None)
        if tid is not None:
            eos.append(int(tid))
        # Gemma end-of-turn
        if self.end_of_turn_id is not None:
            eos.append(int(self.end_of_turn_id))
        # unique
        out: List[int] = []
        for x in eos:
            if x not in out:
                out.append(x)
        return out

    @_inference_mode()
    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 768,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 0,
        seed: Optional[int] = None,
    ) -> str:
        if seed is not None:
            set_all_seeds(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt")

        max_ctx = self._max_context_tokens()
        if max_ctx is not None:
            # Reserve at least 1 token for generation to avoid position-embedding overflow
            prompt_cap = max(1, int(max_ctx) - 1)
            cur = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
            if cur > prompt_cap:
                inputs = self._truncate_batch_left(inputs, max_len=prompt_cap)
                if not self._warned_prompt_trunc:
                    self._warned_prompt_trunc = True
                    print(f"[sr_ponder] [warn] prompt too long ({cur}>{prompt_cap}); truncating left to fit model context")

            cur2 = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
            room = max(1, int(max_ctx) - cur2)
            if int(max_new_tokens) > room:
                if not self._warned_prompt_trunc:
                    self._warned_prompt_trunc = True
                    print(f"[sr_ponder] [warn] capping max_new_tokens {int(max_new_tokens)} -> {room} (context limit {int(max_ctx)})")
                max_new_tokens = room

        inputs = self._move_to_input_device(inputs)

        eos_ids = self._eos_ids()
        eos_for_generate: Any = eos_ids[0] if len(eos_ids) == 1 else eos_ids

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(0.0, float(temperature)),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_for_generate,
        )
        if top_k and top_k > 0:
            gen_kwargs["top_k"] = int(top_k)
        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        gen = output_ids[0, inputs["input_ids"].shape[1] :]

        # Hard-stop at the first <end_of_turn> token to prevent transcript loops.
        if self.end_of_turn_id is not None:
            pos = (gen == int(self.end_of_turn_id)).nonzero(as_tuple=False)
            if pos.numel() > 0:
                gen = gen[: int(pos[0].item())]

        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


# -----------------------------
# Main experiment logic
# -----------------------------


@dataclasses.dataclass
class RunConfig:
    model_path: str
    memory_path: Path
    backend: str = "hf"  # hf|openai_compat
    provider: str = "hf"  # hf|openai|mistral|groq|openrouter|deepseek|custom

    # API backend (OpenAI-compatible)
    api_base_url: str = "https://api.openai.com/v1"
    api_chat_path: str = "/chat/completions"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str = ""
    api_headers: List[str] = dataclasses.field(default_factory=list)  # repeatable "Key: Value"
    api_timeout: float = 60.0
    api_max_retries: int = 2
    api_reasoning_effort: str = "auto"  # auto|none|minimal|low|medium|high|xhigh
    api_seed_method: str = "auto"  # auto|self|logprobs
    api_logprobs_top_n: int = 0  # 0 disables logprobs probing

    n_memory: int = 12
    memory_policy: str = "tail"  # tail|current_only|off
    memory_retrieve: str = "tail"  # tail|similar|anti|mix (only when memory_policy=tail)
    memory_pool: int = 200
    memory_mix_ratio: float = 0.5
    memory_exclude_current_run: bool = True
    memory_remix: str = "off"  # off|shuffle|compress|dream
    memory_remix_keep_original: bool = False
    memory_remix_max_new_tokens: int = 240
    memory_remix_temperature: float = 0.9

    rejected: RejectedTokenConfig = dataclasses.field(default_factory=RejectedTokenConfig)
    band_profile: str = "single"  # single|spectrum3
    bands: List[Dict[str, Any]] = dataclasses.field(default_factory=list)  # [{label,start_rank,end_rank}]

    answer_style: str = "plain"  # plain|surreal|metaphor|meta
    answer_max_new_tokens: int = 1550
    ponder_max_new_tokens: int = 1020
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.05
    no_repeat_ngram_size: int = 0
    seed: int = 1234

    prompt_lang: str = "en"  # resolved: en|ja
    ponder_mode: str = "assoc"  # assoc|assumption|counterexample|questions_only|metaphor (fallback)
    ponder_pipeline: List[str] = dataclasses.field(default_factory=list)  # if empty, uses ponder_mode
    pipeline_context: str = "prev"  # none|prev|all
    pipeline_context_max_chars: int = 1200
    n_ponder: int = 1  # per band
    ponder_hops: int = 1  # sequential hops per (band, band_ponder_ix)
    hop_keyword_source: str = "model"  # model|heuristic
    hop_context_max_chars: int = 900  # prev-hop context injected into hop>0 stage0

    keyword_refine: bool = False
    keyword_refine_max_new_tokens: int = 96
    keyword_refine_temperature: float = 0.3
    keyword_objective: str = "random_band"  # random_band|dissonance|unstable|random_vocab
    keyword_select_top: int = 128
    keyword_diversity: str = "off"  # off|lex|embed
    keyword_diversity_threshold: float = 0.82  # lex similarity cutoff (>= threshold => reject)
    dissonance_target: float = 0.9
    dissonance_width: float = 0.6
    dissonance_tail_k: int = 64

    prompt_jitter: int = 0  # number of paraphrases (excluding original)
    prompt_jitter_include_original: bool = True
    prompt_jitter_max_new_tokens: int = 160
    prompt_jitter_temperature: float = 0.6

    probe_top_n: int = 0
    probe_compare: bool = False
    probe_compare_stages: bool = False
    probe_compare_top_n: int = 32
    print_probe: bool = False
    interactive: bool = False
    interactive_candidates: int = 48

    answer_per_band: bool = False
    answer_ensemble: bool = False
    answer_ensemble_max_new_tokens: int = 512
    answer_ensemble_temperature: float = 0.2

    control: str = "none"  # none|no_inject|random_log|random_keywords|lens_only
    write_memory: bool = True

    device: str = "auto"
    dtype: str = "auto"
    allocator_warmup: str = "auto"  # auto|on|off (auto disables on MPS)
    trust_remote_code: bool = False
    use_chat_template: bool = True
    force_gemma_format: bool = True
    hf_local_files_only: bool = True


def api_ignored_generation_args(cfg: RunConfig) -> List[str]:
    ignored = ["--seed"]
    if int(cfg.top_k) > 0:
        ignored.append("--top_k")
    if abs(float(cfg.repetition_penalty) - 1.0) > 1e-9:
        ignored.append("--repetition_penalty")
    if int(cfg.no_repeat_ngram_size) > 0:
        ignored.append("--no_repeat_ngram_size")
    return ignored


def summarize_api_generation_meta(meta: Any) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in ("finish_reason", "response_id", "model"):
        if meta.get(key):
            out[key] = meta.get(key)
    for key in ("completion_tokens", "reasoning_tokens"):
        val = meta.get(key)
        if isinstance(val, int):
            out[key] = int(val)
    refusal = meta.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        out["refusal"] = refusal.strip()
    if isinstance(meta.get("empty_output"), bool):
        out["empty_output"] = bool(meta.get("empty_output"))
    return out


def run_baseline(
    hf: Any,
    cfg: RunConfig,
    query: str,
    *,
    trace: Optional[TraceWriter] = None,
    pack_item: Optional[str] = None,
) -> str:
    t0 = time.perf_counter()
    prompt = hf._apply_chat(
        build_prompt_for_answer(query, memory_block=None, lang=cfg.prompt_lang, style=cfg.answer_style),
        system_text=None,
    )
    ans = hf.generate_text(
        prompt,
        max_new_tokens=cfg.answer_max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        no_repeat_ngram_size=cfg.no_repeat_ngram_size,
        seed=cfg.seed,
    )
    if trace:
        trace.event(
            "baseline_answer",
            pack_item=pack_item,
            answer_style=cfg.answer_style,
            elapsed_s=float(time.perf_counter() - t0),
            answer_chars=len(ans or ""),
            answer_preview=trace.preview(ans),
        )
    return ans


def _print_probe_table(title: str, items: Sequence[Dict[str, Any]], *, limit: int) -> None:
    print(f"\n=== {title} ===\n")
    for i, x in enumerate(items[:limit], start=1):
        tok = x.get("token", "")
        tid = x.get("token_id", "?")
        rk = x.get("rank", "?")
        p = x.get("prob", None)
        p_s = f"{p:.4f}" if isinstance(p, float) else "?"
        print(f"{i:>2}. {tok!r} id={tid} rank={rk} p={p_s}")


def _compact_probe_item(item: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("token", "token_id", "rank", "prob", "logprob"):
        if key in item and item.get(key) is not None:
            out[key] = item.get(key)
    return out


def _probe_item_key(item: Dict[str, Any]) -> str:
    tid = item.get("token_id")
    if isinstance(tid, int):
        return f"id:{tid}"
    return f"tok:{str(item.get('token', '') or '')}"


def _probe_item_prob(item: Dict[str, Any]) -> float:
    p = item.get("prob")
    if isinstance(p, (int, float)):
        p2 = float(p)
        if math.isfinite(p2) and p2 >= 0.0:
            return p2
    lp = item.get("logprob")
    if isinstance(lp, (int, float)):
        lp2 = float(lp)
        if math.isfinite(lp2):
            if lp2 <= -1e9:
                return 0.0
            try:
                return float(math.exp(lp2))
            except Exception:
                return 0.0
    return 0.0


def _top_tokens_from_logits(hf: Any, logits: torch.Tensor, *, top_n: int) -> List[Dict[str, Any]]:
    n = max(0, int(top_n))
    if n <= 0:
        return []
    vocab_n = int(logits.numel())
    if vocab_n <= 0:
        return []
    n = min(n, vocab_n)
    sorted_ids = torch.argsort(logits, descending=True)
    log_z = torch.logsumexp(logits, dim=0)
    out: List[Dict[str, Any]] = []
    for rank, tid in enumerate(sorted_ids[:n].tolist()):
        out.append(
            {
                "token_id": int(tid),
                "token": _token_text(hf.tokenizer, tid),
                "rank": int(rank),
                "prob": _token_prob(logits, tid, log_z),
            }
        )
    return out


def _js_divergence_from_sparse_prob_maps(before: Dict[str, float], after: Dict[str, float]) -> float:
    keys = set(before.keys()) | set(after.keys())
    if not keys:
        return 0.0

    sum_before = float(sum(max(0.0, float(v)) for v in before.values()))
    sum_after = float(sum(max(0.0, float(v)) for v in after.values()))
    if sum_before <= 0.0 and sum_after <= 0.0:
        return 0.0

    js = 0.0
    for key in keys:
        p = max(0.0, float(before.get(key, 0.0)))
        q = max(0.0, float(after.get(key, 0.0)))
        if sum_before > 0.0:
            p /= sum_before
        else:
            p = 0.0
        if sum_after > 0.0:
            q /= sum_after
        else:
            q = 0.0
        m = 0.5 * (p + q)
        if p > 0.0 and m > 0.0:
            js += 0.5 * p * math.log(p / m)
        if q > 0.0 and m > 0.0:
            js += 0.5 * q * math.log(q / m)
    return float(js)


def _js_divergence_from_logits(before_logits: torch.Tensor, after_logits: torch.Tensor) -> float:
    log_p = torch.log_softmax(before_logits.float(), dim=0)
    log_q = torch.log_softmax(after_logits.float(), dim=0)
    p = torch.exp(log_p)
    q = torch.exp(log_q)
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp_min(1e-30))
    kl_pm = torch.sum(p * (log_p - log_m))
    kl_qm = torch.sum(q * (log_q - log_m))
    return float((0.5 * (kl_pm + kl_qm)).item())


def build_probe_compare(
    before_items: Sequence[Dict[str, Any]],
    after_items: Sequence[Dict[str, Any]],
    *,
    top_n: int,
    js_divergence: Optional[float] = None,
    js_divergence_mode: str = "topn_union_renorm",
) -> Dict[str, Any]:
    n = max(1, int(top_n))
    before = [_compact_probe_item(x) for x in list(before_items)[:n] if isinstance(x, dict)]
    after = [_compact_probe_item(x) for x in list(after_items)[:n] if isinstance(x, dict)]

    before_lookup = {_probe_item_key(x): x for x in before}
    after_lookup = {_probe_item_key(x): x for x in after}
    before_keys = set(before_lookup.keys())
    after_keys = set(after_lookup.keys())
    overlap_keys = before_keys & after_keys
    union_keys = before_keys | after_keys

    before_prob_map = {k: _probe_item_prob(v) for k, v in before_lookup.items()}
    after_prob_map = {k: _probe_item_prob(v) for k, v in after_lookup.items()}

    entered: List[Dict[str, Any]] = []
    for key in sorted(after_keys - before_keys, key=lambda k: int(after_lookup[k].get("rank", 10**9))):
        item = after_lookup[key]
        entered.append(
            {
                "token": item.get("token"),
                "token_id": item.get("token_id"),
                "rank_after": item.get("rank"),
                "prob_after": _probe_item_prob(item),
            }
        )

    exited: List[Dict[str, Any]] = []
    for key in sorted(before_keys - after_keys, key=lambda k: int(before_lookup[k].get("rank", 10**9))):
        item = before_lookup[key]
        exited.append(
            {
                "token": item.get("token"),
                "token_id": item.get("token_id"),
                "rank_before": item.get("rank"),
                "prob_before": _probe_item_prob(item),
            }
        )

    movers: List[Dict[str, Any]] = []
    for key in overlap_keys:
        before_item = before_lookup[key]
        after_item = after_lookup[key]
        try:
            rank_before = int(before_item.get("rank", 0))
            rank_after = int(after_item.get("rank", 0))
        except Exception:
            continue
        rank_shift = int(rank_before - rank_after)
        if rank_shift == 0:
            continue
        movers.append(
            {
                "token": after_item.get("token", before_item.get("token")),
                "token_id": after_item.get("token_id", before_item.get("token_id")),
                "rank_before": rank_before,
                "rank_after": rank_after,
                "rank_shift": rank_shift,
                "prob_before": _probe_item_prob(before_item),
                "prob_after": _probe_item_prob(after_item),
            }
        )
    movers.sort(key=lambda x: (-abs(int(x.get("rank_shift", 0))), int(x.get("rank_after", 10**9)), str(x.get("token", ""))))

    if js_divergence is None:
        js_divergence = _js_divergence_from_sparse_prob_maps(before_prob_map, after_prob_map)

    top1_before = before[0] if before else None
    top1_after = after[0] if after else None
    denom = max(1, min(len(before), len(after)))
    jaccard = float(len(overlap_keys) / len(union_keys)) if union_keys else 1.0

    return {
        "top_n": int(n),
        "before_count": int(len(before)),
        "after_count": int(len(after)),
        "js_divergence": float(js_divergence),
        "js_divergence_mode": str(js_divergence_mode),
        "before_observed_mass": float(sum(before_prob_map.values())),
        "after_observed_mass": float(sum(after_prob_map.values())),
        "overlap_count": int(len(overlap_keys)),
        "overlap_ratio": float(len(overlap_keys) / denom),
        "jaccard": jaccard,
        "top1_changed": bool(_probe_item_key(top1_before) != _probe_item_key(top1_after)) if top1_before and top1_after else False,
        "top1_before": top1_before,
        "top1_after": top1_after,
        "mover_count": int(len(movers)),
        "entered_count": int(len(entered)),
        "exited_count": int(len(exited)),
        "movers": movers,
        "entered": entered,
        "exited": exited,
        "top_before": before,
        "top_after": after,
    }


def make_probe_compare_timeline_entry(
    *,
    source: str,
    point: str,
    record: Optional[Dict[str, Any]] = None,
    compare_from_base: Optional[Dict[str, Any]] = None,
    compare_from_prev: Optional[Dict[str, Any]] = None,
    memory_chars: int = 0,
    prompt_chars: int = 0,
    status: str = "ok",
    reason: str = "",
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "point": str(point or "stage"),
        "source": str(source or "current_records"),
        "status": str(status or "ok"),
        "memory_chars": int(memory_chars),
        "prompt_chars": int(prompt_chars),
    }
    if reason:
        entry["reason"] = str(reason)
    if isinstance(record, dict):
        for key in ("ponder_ix", "band_label", "band_ponder_ix", "hop_ix", "pipeline_stage_ix", "ponder_mode"):
            if key in record:
                entry[key] = record.get(key)
    if compare_from_base is not None:
        entry["compare_from_base"] = compare_from_base
    if compare_from_prev is not None:
        entry["compare_from_prev"] = compare_from_prev
    return entry


def _probe_compare_trace_fields(comp: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "top_n": int(comp.get("top_n", 0) or 0),
        "js_divergence": float(comp.get("js_divergence", 0.0) or 0.0),
        "js_divergence_mode": str(comp.get("js_divergence_mode", "") or ""),
        "overlap_count": int(comp.get("overlap_count", 0) or 0),
        "jaccard": float(comp.get("jaccard", 0.0) or 0.0),
        "mover_count": int(comp.get("mover_count", 0) or 0),
        "entered_count": int(comp.get("entered_count", 0) or 0),
        "exited_count": int(comp.get("exited_count", 0) or 0),
        "top1_before": comp.get("top1_before"),
        "top1_after": comp.get("top1_after"),
    }


def _print_probe_compare_summary(title: str, comp: Dict[str, Any], *, limit: int = 8) -> None:
    js = comp.get("js_divergence")
    js_s = f"{float(js):.6f}" if isinstance(js, (int, float)) else "?"
    overlap = int(comp.get("overlap_count", 0) or 0)
    before_count = int(comp.get("before_count", 0) or 0)
    after_count = int(comp.get("after_count", 0) or 0)
    jaccard = comp.get("jaccard")
    jaccard_s = f"{float(jaccard):.3f}" if isinstance(jaccard, (int, float)) else "?"
    print(f"\n=== {title} ===\n")
    print(
        "js_divergence="
        f"{js_s} mode={comp.get('js_divergence_mode', '?')} "
        f"overlap={overlap} before={before_count} after={after_count} jaccard={jaccard_s} "
        f"top1_changed={bool(comp.get('top1_changed'))}"
    )

    movers = comp.get("movers") or []
    if isinstance(movers, list) and movers:
        print("\n[movers]")
        for item in movers[:limit]:
            tok = item.get("token", "")
            tid = item.get("token_id", "?")
            rb = item.get("rank_before", "?")
            ra = item.get("rank_after", "?")
            rs = item.get("rank_shift", 0)
            pb = item.get("prob_before")
            pa = item.get("prob_after")
            pb_s = f"{float(pb):.4f}" if isinstance(pb, (int, float)) else "?"
            pa_s = f"{float(pa):.4f}" if isinstance(pa, (int, float)) else "?"
            print(f"- {tok!r} id={tid} rank {rb}->{ra} shift={rs:+d} p {pb_s}->{pa_s}")

    entered = comp.get("entered") or []
    if isinstance(entered, list) and entered:
        print("\n[entered]")
        for item in entered[:limit]:
            tok = item.get("token", "")
            tid = item.get("token_id", "?")
            ra = item.get("rank_after", "?")
            pa = item.get("prob_after")
            pa_s = f"{float(pa):.4f}" if isinstance(pa, (int, float)) else "?"
            print(f"- {tok!r} id={tid} rank_after={ra} p={pa_s}")

    exited = comp.get("exited") or []
    if isinstance(exited, list) and exited:
        print("\n[exited]")
        for item in exited[:limit]:
            tok = item.get("token", "")
            tid = item.get("token_id", "?")
            rb = item.get("rank_before", "?")
            pb = item.get("prob_before")
            pb_s = f"{float(pb):.4f}" if isinstance(pb, (int, float)) else "?"
            print(f"- {tok!r} id={tid} rank_before={rb} p={pb_s}")


def _parse_band_spec(spec: str) -> Dict[str, Any]:
    """
    Parses:
      - "START:END"
      - "LABEL=START:END"
    END is exclusive. Ranks are 0-based (0 is the top token).
    """
    s = (spec or "").strip()
    if not s:
        raise ValueError("Empty band spec")

    label = ""
    if "=" in s:
        label, s = s.split("=", 1)
        label = label.strip()
        s = s.strip()

    if ":" not in s:
        raise ValueError(f"Invalid band spec (expected START:END): {spec!r}")
    a, b = s.split(":", 1)
    start = int(a.strip())
    end = int(b.strip())
    if start < 0 or end < 0:
        raise ValueError(f"Band ranks must be non-negative: {spec!r}")
    if end <= start:
        raise ValueError(f"Band END must be > START (end is exclusive): {spec!r}")
    if not label:
        label = f"{start}:{end}"
    return {"label": label, "start_rank": start, "end_rank": end}


def _default_bands_from_profile(profile: str) -> List[Dict[str, Any]]:
    if profile == "single":
        return []
    if profile == "spectrum3":
        # near: plausible but not the absolute top (avoid echo)
        # mid: classic rejected-token band
        # far: deeper tail for more drift (still not pure noise)
        return [
            {"label": "near", "start_rank": 8, "end_rank": 64},
            {"label": "mid", "start_rank": 80, "end_rank": 336},
            {"label": "far", "start_rank": 800, "end_rank": 1800},
        ]
    raise ValueError(f"Unknown band_profile: {profile!r}")


def _single_band_from_rejected_cfg(cfg: RunConfig, *, vocab_size: int) -> Dict[str, Any]:
    if cfg.rejected.strategy == "within_topk":
        start = max(0, min(cfg.rejected.exclude_top, cfg.rejected.top_k))
        end = max(start, cfg.rejected.top_k)
    else:
        start = cfg.rejected.top_k
        end = cfg.rejected.top_k + cfg.rejected.band_width

    end = min(end, int(vocab_size))
    if start >= end:
        start = max(0, end - 1)
        end = min(int(vocab_size), start + 1)

    label = f"{cfg.rejected.strategy}:{start}:{end}"
    return {"label": label, "start_rank": start, "end_rank": end}


def run_ponder(
    hf: LocalHFModel,
    cfg: RunConfig,
    query: str,
    *,
    trace: Optional[TraceWriter] = None,
    pack_item: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    t_total0 = time.perf_counter()
    lang = cfg.prompt_lang
    n_ponder = max(1, int(cfg.n_ponder))  # per band

    run_id = make_run_id(cfg.seed, query)
    query_sha = sha256_short(query)

    pipeline = list(cfg.ponder_pipeline) if cfg.ponder_pipeline else [cfg.ponder_mode]
    pipeline_ctx = (cfg.pipeline_context or "prev").strip()

    objective = (cfg.keyword_objective or "random_band").strip()
    control = (cfg.control or "none").strip()
    if control == "random_keywords":
        objective = "random_vocab"

    if trace:
        trace.event(
            "ponder_start",
            run_id=run_id,
            pack_item=pack_item,
            backend="hf",
            query_sha=query_sha,
            objective=objective,
            control=control,
            pipeline=pipeline,
            hops=int(cfg.ponder_hops),
            band_profile=cfg.band_profile,
            memory_policy=cfg.memory_policy,
            memory_retrieve=cfg.memory_retrieve,
            memory_remix=cfg.memory_remix,
            answer_style=cfg.answer_style,
        )

    t_probe0 = time.perf_counter()
    base_prompt = hf._apply_chat(
        build_prompt_for_answer(query, memory_block=None, lang=lang, style=cfg.answer_style),
        system_text=None,
    )
    logits = hf.next_token_logits(base_prompt)
    probe_s = float(time.perf_counter() - t_probe0)
    if trace:
        trace.event(
            "probe_done",
            run_id=run_id,
            pack_item=pack_item,
            elapsed_s=probe_s,
            prompt_chars=len(base_prompt or ""),
            vocab_size=int(logits.numel()),
        )

    sorted_ids = torch.argsort(logits, descending=True)
    ranks = torch.empty_like(sorted_ids)
    ranks[sorted_ids] = torch.arange(sorted_ids.numel(), device=sorted_ids.device, dtype=sorted_ids.dtype)
    log_z = torch.logsumexp(logits, dim=0)

    vocab_size = int(sorted_ids.numel())
    probe_compare: Optional[Dict[str, Any]] = None
    probe_compare_stages: List[Dict[str, Any]] = []
    probe_capture_n = 0
    if cfg.probe_top_n > 0:
        probe_capture_n = max(probe_capture_n, int(cfg.probe_top_n))
    if cfg.print_probe:
        probe_capture_n = max(probe_capture_n, 20)
    if cfg.probe_compare or cfg.probe_compare_stages:
        probe_capture_n = max(probe_capture_n, int(cfg.probe_compare_top_n))

    probe_top_all: List[Dict[str, Any]] = []
    if probe_capture_n > 0:
        capture_n = max(1, min(probe_capture_n, vocab_size))
        for tid in sorted_ids[:capture_n].tolist():
            probe_top_all.append(
                {
                    "token_id": int(tid),
                    "token": _token_text(hf.tokenizer, tid),
                    "rank": int(ranks[int(tid)].item()),
                    "prob": _token_prob(logits, tid, log_z),
                }
            )

    top_tokens: List[Dict[str, Any]] = []
    if cfg.probe_top_n > 0 or cfg.print_probe:
        top_n = int(cfg.probe_top_n) if cfg.probe_top_n > 0 else 20
        top_n = max(1, min(top_n, vocab_size))
        top_tokens = probe_top_all[:top_n] if probe_top_all else []
        if cfg.print_probe:
            _print_probe_table("PROBE TOP TOKENS", top_tokens, limit=top_n)

    probe_compare_before = (
        probe_top_all[: max(1, int(cfg.probe_compare_top_n))]
        if (cfg.probe_compare or cfg.probe_compare_stages) and probe_top_all
        else []
    )
    probe_compare_prev_items = list(probe_compare_before)
    probe_compare_prev_logits = logits if cfg.probe_compare_stages else None
    bands = cfg.bands or _default_bands_from_profile(cfg.band_profile)
    if not bands:
        bands = [_single_band_from_rejected_cfg(cfg, vocab_size=vocab_size)]

    special_ids: List[int] = []
    for x in (getattr(hf.tokenizer, "all_special_ids", []) or []):
        try:
            special_ids.append(int(x))
        except Exception:
            pass
    for x in (
        getattr(hf, "start_of_turn_id", None),
        getattr(hf, "end_of_turn_id", None),
        getattr(hf.tokenizer, "eos_token_id", None),
        getattr(hf.tokenizer, "pad_token_id", None),
    ):
        if x is not None:
            special_ids.append(int(x))

    query_vec: Optional[torch.Tensor] = None
    if objective == "dissonance":
        query_vec = prompt_embedding_tail(hf, base_prompt, tail_k=int(cfg.dissonance_tail_k))

    jitter_queries: List[str] = []
    jitter_logits: List[torch.Tensor] = []
    jitter_n = int(cfg.prompt_jitter)
    if objective == "unstable" and jitter_n <= 0:
        jitter_n = 3
    if jitter_n > 0:
        jitter_queries = generate_prompt_jitters(
            hf,
            query=query,
            n=jitter_n,
            lang=lang,
            include_original=bool(cfg.prompt_jitter_include_original),
            max_new_tokens=int(cfg.prompt_jitter_max_new_tokens),
            temperature=float(cfg.prompt_jitter_temperature),
            seed=int(cfg.seed) + 999,
        )
        for jq in jitter_queries:
            jp = hf._apply_chat(
                build_prompt_for_answer(jq, memory_block=None, lang=lang, style=cfg.answer_style),
                system_text=None,
            )
            jitter_logits.append(hf.next_token_logits(jp))

    records: List[Dict[str, Any]] = []
    log_ix = 0

    for band_ix, band in enumerate(bands):
        band_label = str(band.get("label", f"band{band_ix}"))
        band_start = int(band.get("start_rank", 0))
        band_end = int(band.get("end_rank", vocab_size))

        band_end = min(band_end, vocab_size)
        band_start = max(0, band_start)
        if band_start >= band_end:
            band_start = max(0, band_end - 1)
            band_end = min(vocab_size, band_start + 1)

        candidate_ranked = sorted_ids[band_start:band_end].tolist()
        if not candidate_ranked:
            candidate_ranked = sorted_ids[: min(vocab_size, 256)].tolist()

        band_dissonance: Optional[Dict[int, float]] = None
        if objective == "dissonance" and query_vec is not None:
            band_dissonance = dissonance_scores(hf, query_vec=query_vec, candidate_ids=candidate_ranked)

        band_unstable: Optional[Dict[int, float]] = None
        if objective == "unstable" and len(jitter_logits) >= 2:
            band_unstable = unstable_scores_from_jitters(candidate_ranked, jitter_logits)

        for band_ponder_ix in range(n_ponder):
            n_kw = int(cfg.rejected.n_keywords)
            rng = random.Random(int(cfg.seed) + 99991 + band_ix * 10007 + band_ponder_ix * 101 + log_ix * 37)

            token_ids = select_keyword_token_ids_hf(
                hf,
                objective=objective,
                rng=rng,
                vocab_size=vocab_size,
                candidate_pool=candidate_ranked,
                n_keywords=n_kw,
                special_ids=special_ids,
                dissonance=band_dissonance,
                dissonance_target=float(cfg.dissonance_target),
                dissonance_width=float(cfg.dissonance_width),
                unstable=band_unstable,
                select_top=int(cfg.keyword_select_top),
                diversity=str(cfg.keyword_diversity),
                diversity_threshold=float(cfg.keyword_diversity_threshold),
            )

            keywords_source = "rejected_tokens"
            if objective == "random_vocab":
                keywords_source = "random_vocab"
            if objective == "dissonance":
                keywords_source = "dissonance"
            if objective == "unstable":
                keywords_source = "unstable"

            # Interactive override
            if cfg.interactive:
                cand_items: List[Dict[str, Any]] = []
                for tid in candidate_ranked[: max(1, int(cfg.interactive_candidates))]:
                    item: Dict[str, Any] = {
                        "token_id": int(tid),
                        "token": _token_text(hf.tokenizer, tid),
                        "rank": int(ranks[int(tid)].item()),
                        "prob": _token_prob(logits, tid, log_z),
                    }
                    if band_dissonance is not None and int(tid) in band_dissonance:
                        item["dissonance"] = float(band_dissonance[int(tid)])
                    if band_unstable is not None and int(tid) in band_unstable:
                        item["unstable"] = float(band_unstable[int(tid)])
                    cand_items.append(item)
                picked = interactive_pick_token_ids(
                    prompt=f"band={band_label} band_ix={band_ix} log={band_ponder_ix}",
                    rng=rng,
                    candidates=cand_items,
                    n_keywords=n_kw,
                )
                if picked:
                    token_ids = picked
                    keywords_source = "human_pick"

            raw_keywords = decode_keyword_tokens(hf.tokenizer, token_ids)
            keywords = raw_keywords
            refined_keywords: List[str] = []
            if cfg.keyword_refine and keywords_source != "human_pick":
                refined_keywords = refine_keywords_with_model(
                    hf,
                    query=query,
                    seed_keywords=raw_keywords,
                    n_keywords=n_kw,
                    lang=lang,
                    max_new_tokens=int(cfg.keyword_refine_max_new_tokens),
                    temperature=float(cfg.keyword_refine_temperature),
                    seed=int(cfg.seed) + 200 + log_ix,
                )
                if refined_keywords:
                    keywords = refined_keywords
                    keywords_source = "model_refine"

            # Hop loop ("latent walk"): each hop derives new keywords from the previous ponder log.
            hop_n = max(1, int(cfg.ponder_hops))
            prev_hop_log: Optional[str] = None
            prev_hop_keywords: List[str] = list(keywords)

            selected_tokens0: List[Dict[str, Any]] = []
            for tid in token_ids:
                t0: Dict[str, Any] = {
                    "token_id": int(tid),
                    "token": _token_text(hf.tokenizer, tid),
                    "rank": int(ranks[int(tid)].item()),
                    "prob": _token_prob(logits, tid, log_z),
                }
                if band_dissonance is not None and int(tid) in band_dissonance:
                    t0["dissonance"] = float(band_dissonance[int(tid)])
                if band_unstable is not None and int(tid) in band_unstable:
                    t0["unstable"] = float(band_unstable[int(tid)])
                selected_tokens0.append(t0)

            for hop_ix in range(hop_n):
                hop_keyword_source: Optional[str] = None
                hop_keywords_source = keywords_source
                hop_raw_keywords = list(raw_keywords)
                hop_keywords = list(keywords)
                hop_token_ids = list(token_ids)
                hop_selected_tokens = selected_tokens0

                if hop_ix > 0:
                    hop_keyword_source = (cfg.hop_keyword_source or "model").strip()
                    if hop_keyword_source not in ("model", "heuristic"):
                        hop_keyword_source = "model"

                    hk_seed = int(cfg.seed) + 6000 + band_ix * 10007 + band_ponder_ix * 101 + hop_ix * 271
                    new_raw: List[str] = []
                    if hop_keyword_source == "model" and prev_hop_log:
                        try:
                            new_raw = extract_hop_keywords_with_model(
                                hf,
                                query=query,
                                prev_keywords=prev_hop_keywords,
                                ponder_log=prev_hop_log,
                                n=n_kw,
                                lang=lang,
                                seed=hk_seed,
                            )
                        except Exception:
                            new_raw = []

                    if not new_raw and prev_hop_log:
                        hrng = random.Random(int(cfg.seed) + 6100 + band_ix * 10007 + band_ponder_ix * 101 + hop_ix * 271)
                        new_raw = extract_hop_keywords_heuristic(
                            query=query,
                            prev_keywords=prev_hop_keywords,
                            ponder_log=prev_hop_log,
                            n=n_kw,
                            lang=lang,
                            rng=hrng,
                        )
                        if new_raw:
                            hop_keyword_source = "heuristic"

                    if not new_raw:
                        new_raw = list(prev_hop_keywords)

                    hop_raw_keywords = list(new_raw)
                    hop_keywords = list(new_raw)
                    hop_keywords_source = f"hop_{hop_keyword_source}"
                    hop_selected_tokens = []
                    hop_token_ids = token_ids_from_keywords_text(hf, hop_keywords, special_ids=special_ids)

                    if cfg.keyword_refine:
                        refined = refine_keywords_with_model(
                            hf,
                            query=query,
                            seed_keywords=hop_raw_keywords,
                            n_keywords=n_kw,
                            lang=lang,
                            max_new_tokens=int(cfg.keyword_refine_max_new_tokens),
                            temperature=float(cfg.keyword_refine_temperature),
                            seed=int(cfg.seed) + 220 + log_ix + hop_ix * 11,
                        )
                        if refined:
                            hop_keywords = refined
                            hop_keywords_source = "model_refine"
                            hop_token_ids = token_ids_from_keywords_text(hf, hop_keywords, special_ids=special_ids)

                if trace:
                    trace.event(
                        "seed_keywords",
                        run_id=run_id,
                        pack_item=pack_item,
                        band_label=band_label,
                        band_ponder_ix=int(band_ponder_ix),
                        hop_ix=int(hop_ix),
                        keywords=hop_keywords,
                        keywords_source=hop_keywords_source,
                    )

                # One ponder question per (hop, band, band_ponder_ix), reused across pipeline stages.
                if control == "lens_only":
                    ponder_q = query
                else:
                    q_rng = random.Random(int(cfg.seed) + 12345 + band_ix * 10007 + band_ponder_ix * 101 + hop_ix * 1009)
                    ponder_q = make_unrelated_question(hop_keywords, lang=lang, rng=q_rng)

                stage_logs: List[str] = []
                for stage_ix, stage_mode in enumerate(pipeline):
                    ctx_text: Optional[str] = None

                    # Hop context: feed the previous hop's log into the first stage as soft background.
                    if hop_ix > 0 and stage_ix == 0 and prev_hop_log:
                        ctx_text = prev_hop_log
                        if ctx_text and len(ctx_text) > int(cfg.hop_context_max_chars):
                            ctx_text = ctx_text[-int(cfg.hop_context_max_chars) :]

                    if pipeline_ctx == "prev" and stage_logs:
                        ctx_text = stage_logs[-1]
                    elif pipeline_ctx == "all" and stage_logs:
                        ctx_text = "\n\n".join(stage_logs)

                    if ctx_text and len(ctx_text) > int(cfg.pipeline_context_max_chars):
                        ctx_text = ctx_text[-int(cfg.pipeline_context_max_chars) :]

                    ponder_prompt = hf._apply_chat(
                        build_prompt_for_pondering(ponder_q, mode=stage_mode, lang=lang, context=ctx_text),
                        system_text=None,
                    )
                    t_gen0 = time.perf_counter()
                    ponder_log = hf.generate_text(
                        ponder_prompt,
                        max_new_tokens=cfg.ponder_max_new_tokens,
                        temperature=max(0.4, cfg.temperature),
                        top_p=cfg.top_p,
                        top_k=cfg.top_k,
                        repetition_penalty=cfg.repetition_penalty,
                        no_repeat_ngram_size=cfg.no_repeat_ngram_size,
                        seed=int(cfg.seed) + 1 + log_ix,
                    )
                    ponder_meta = summarize_api_generation_meta(getattr(hf, "last_response_meta", {}))
                    gen_s = float(time.perf_counter() - t_gen0)
                    stage_logs.append(ponder_log)
                    if trace:
                        trace.event(
                            "ponder_stage",
                            run_id=run_id,
                            pack_item=pack_item,
                            band_label=band_label,
                            band_ponder_ix=int(band_ponder_ix),
                            hop_ix=int(hop_ix),
                            stage_ix=int(stage_ix),
                            mode=stage_mode,
                            elapsed_s=gen_s,
                            text_chars=len(ponder_log or ""),
                            text_preview=trace.preview(ponder_log),
                            api_meta=ponder_meta if ponder_meta else None,
                        )

                    if cfg.print_probe and hop_selected_tokens:
                        _print_probe_table(
                            f"REJECTED TOKENS (band={band_label} hop={hop_ix} ix={log_ix} stage={stage_mode})",
                            hop_selected_tokens,
                            limit=len(hop_selected_tokens),
                        )
                        print(f"\nkeywords_source={hop_keywords_source} keywords={hop_keywords}\n")

                    record: Dict[str, Any] = {
                        "ts": now_iso(),
                        "run_id": run_id,
                        "query_sha": query_sha,
                        "prompt_lang": lang,
                        "control": control,
                        "band_profile": cfg.band_profile,
                        "band_ix": band_ix,
                        "band_label": band_label,
                        "band_ponder_ix": band_ponder_ix,
                        "hop_ix": hop_ix,
                        "hop_keyword_source": hop_keyword_source,
                        "pipeline": pipeline,
                        "pipeline_stage_ix": stage_ix,
                        "pipeline_context": pipeline_ctx,
                        "ponder_ix": log_ix,
                        "ponder_mode": stage_mode,
                        "keywords_source": hop_keywords_source,
                        "keywords_raw": hop_raw_keywords,
                        "keywords": hop_keywords,
                        "token_ids": hop_token_ids,
                        "selected_tokens": hop_selected_tokens,
                        "rejected_cfg": dataclasses.asdict(cfg.rejected),
                        "band": {"start_rank": band_start, "end_rank": band_end},
                        "ponder_question": ponder_q,
                        "ponder_log": ponder_log,
                        "gen_elapsed_s": gen_s,
                    }
                    if cfg.probe_top_n > 0 and log_ix == 0 and hop_ix == 0:
                        record["probe_top"] = top_tokens[: int(cfg.probe_top_n)]
                    if jitter_queries and log_ix == 0 and hop_ix == 0:
                        record["prompt_jitter_queries"] = jitter_queries

                    if cfg.write_memory:
                        append_jsonl(cfg.memory_path, record)
                    records.append(record)
                    if cfg.probe_compare_stages and probe_compare_before:
                        compare_top_n = max(1, int(cfg.probe_compare_top_n))
                        stage_memory_block = build_memory_block(records, max_chars_per_log=700) if records else None
                        stage_prompt = hf._apply_chat(
                            build_prompt_for_answer(query, memory_block=stage_memory_block, lang=lang, style=cfg.answer_style),
                            system_text=None,
                        )
                        stage_logits = hf.next_token_logits(stage_prompt)
                        stage_items = _top_tokens_from_logits(hf, stage_logits, top_n=compare_top_n)
                        comp_base = build_probe_compare(
                            probe_compare_before,
                            stage_items,
                            top_n=compare_top_n,
                            js_divergence=_js_divergence_from_logits(logits, stage_logits),
                            js_divergence_mode="full_vocab",
                        )
                        comp_prev = None
                        if probe_compare_prev_items and probe_compare_prev_logits is not None:
                            comp_prev = build_probe_compare(
                                probe_compare_prev_items,
                                stage_items,
                                top_n=compare_top_n,
                                js_divergence=_js_divergence_from_logits(probe_compare_prev_logits, stage_logits),
                                js_divergence_mode="full_vocab",
                            )
                        stage_entry = make_probe_compare_timeline_entry(
                            source="current_records",
                            point="stage",
                            record=record,
                            compare_from_base=comp_base,
                            compare_from_prev=comp_prev,
                            memory_chars=len(stage_memory_block or ""),
                            prompt_chars=len(stage_prompt or ""),
                        )
                        probe_compare_stages.append(stage_entry)
                        if trace:
                            trace.event(
                                "probe_compare_stage",
                                run_id=run_id,
                                pack_item=pack_item,
                                source="current_records",
                                band_label=band_label,
                                band_ponder_ix=int(band_ponder_ix),
                                hop_ix=int(hop_ix),
                                stage_ix=int(stage_ix),
                                ponder_ix=int(record.get("ponder_ix", 0) or 0),
                                ponder_mode=stage_mode,
                                memory_chars=len(stage_memory_block or ""),
                                prompt_chars=len(stage_prompt or ""),
                                prev_js_divergence=float(comp_prev.get("js_divergence", 0.0)) if isinstance(comp_prev, dict) else None,
                                **_probe_compare_trace_fields(comp_base),
                            )
                        if cfg.print_probe:
                            _print_probe_compare_summary(
                                f"PROBE STAGE (band={band_label} hop={hop_ix} stage={stage_mode} ix={log_ix})",
                                comp_base,
                                limit=min(8, compare_top_n),
                            )
                        probe_compare_prev_items = stage_items
                        probe_compare_prev_logits = stage_logits
                    log_ix += 1

                prev_hop_log = "\n\n".join(stage_logs).strip() if stage_logs else prev_hop_log
                prev_hop_keywords = list(hop_keywords)

    # Select memory records for final answer injection
    exclude_run_id = run_id if cfg.memory_exclude_current_run and cfg.memory_policy == "tail" else None
    mem_records = select_memory_records(
        hf,
        memory_path=cfg.memory_path,
        current_records=records,
        query=query,
        memory_policy=cfg.memory_policy,
        memory_retrieve=cfg.memory_retrieve,
        n_memory=cfg.n_memory,
        pool_size=cfg.memory_pool,
        mix_ratio=cfg.memory_mix_ratio,
        exclude_run_id=exclude_run_id,
    )

    if control == "no_inject":
        mem_records = []

    memory_selected_meta = [
        {
            "ts": r.get("ts"),
            "run_id": r.get("run_id"),
            "band_label": r.get("band_label"),
            "ponder_ix": r.get("ponder_ix"),
            "ponder_mode": r.get("ponder_mode"),
        }
        for r in (mem_records or [])
    ]
    if trace:
        trace.event(
            "memory_selected",
            run_id=run_id,
            pack_item=pack_item,
            count=int(len(mem_records or [])),
            memory_policy=cfg.memory_policy,
            memory_retrieve=cfg.memory_retrieve,
            memory_remix=cfg.memory_remix,
            selected=memory_selected_meta[:12],
        )

    memory_block = build_memory_block(mem_records, max_chars_per_log=700) if mem_records else None
    t_remix0 = time.perf_counter()
    memory_block = remix_memory_block(
        hf,
        memory_block=memory_block,
        remix=cfg.memory_remix,
        lang=lang,
        keep_original=cfg.memory_remix_keep_original,
        max_new_tokens=cfg.memory_remix_max_new_tokens,
        temperature=cfg.memory_remix_temperature,
        seed=int(cfg.seed) + 777,
    )
    remix_s = float(time.perf_counter() - t_remix0)
    if trace:
        trace.event(
            "memory_remix_done",
            run_id=run_id,
            pack_item=pack_item,
            remix=cfg.memory_remix,
            elapsed_s=remix_s,
            text_chars=len(memory_block or ""),
            text_preview=trace.preview(memory_block),
        )

    random_log_text: Optional[str] = None
    if control == "random_log":
        rng = random.Random(int(cfg.seed) + 424242)
        rand_ids = select_token_ids_by_objective(
            objective="random_vocab",
            rng=rng,
            vocab_size=vocab_size,
            candidate_pool=[],
            n_keywords=int(cfg.rejected.n_keywords),
            special_ids=special_ids,
        )
        rand_kw = decode_keyword_tokens(hf.tokenizer, rand_ids)
        q_rng = random.Random(int(cfg.seed) + 424243)
        rand_q = make_unrelated_question(rand_kw, lang=lang, rng=q_rng)
        rp = hf._apply_chat(build_prompt_for_pondering(rand_q, mode="assoc", lang=lang), system_text=None)
        random_log_text = hf.generate_text(
            rp,
            max_new_tokens=int(cfg.ponder_max_new_tokens),
            temperature=max(0.6, float(cfg.temperature)),
            top_p=float(cfg.top_p),
            top_k=int(cfg.top_k),
            repetition_penalty=float(cfg.repetition_penalty),
            no_repeat_ngram_size=int(cfg.no_repeat_ngram_size),
            seed=int(cfg.seed) + 424244,
        )
        memory_block = random_log_text

    # Band answers (sensitivity analysis)
    band_answers: Dict[str, str] = {}
    if cfg.answer_per_band or cfg.answer_ensemble:
        for band in bands:
            bl = str(band.get("label", "band"))
            band_recs = [r for r in records if str(r.get("band_label", "")) == bl]
            if not band_recs:
                continue
            band_mem = None if control == "no_inject" else build_memory_block(band_recs, max_chars_per_log=700)
            band_mem = remix_memory_block(
                hf,
                memory_block=band_mem,
                remix=cfg.memory_remix,
                lang=lang,
                keep_original=False,
                max_new_tokens=cfg.memory_remix_max_new_tokens,
                temperature=cfg.memory_remix_temperature,
                seed=int(cfg.seed) + 9000 + stable_hash_mod(bl, 1000),
            )
            fp = hf._apply_chat(
                build_prompt_for_answer(query, memory_block=band_mem, lang=lang, style=cfg.answer_style),
                system_text=None,
            )
            band_ans = hf.generate_text(
                fp,
                max_new_tokens=cfg.answer_max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                repetition_penalty=cfg.repetition_penalty,
                no_repeat_ngram_size=cfg.no_repeat_ngram_size,
                seed=int(cfg.seed) + 10101,
            )
            band_answers[bl] = band_ans

    ensemble_raw: Optional[str] = None
    ensemble_final: Optional[str] = None
    ensemble_consensus: Optional[str] = None
    ensemble_divergence: Optional[str] = None
    if cfg.answer_ensemble and len(band_answers) >= 2:
        ep = hf._apply_chat(build_prompt_for_ensemble_summary(query, band_answers, lang=lang), system_text=None)
        ensemble_raw = hf.generate_text(
            ep,
            max_new_tokens=int(cfg.answer_ensemble_max_new_tokens),
            temperature=float(cfg.answer_ensemble_temperature),
            top_p=0.95,
            top_k=0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            seed=int(cfg.seed) + 20202,
        )
        ensemble_consensus = extract_tag(ensemble_raw, "consensus")
        ensemble_divergence = extract_tag(ensemble_raw, "divergence")
        ensemble_final = extract_tag(ensemble_raw, "final") or ensemble_raw.strip()

    final_answer_block = memory_block
    if control == "no_inject":
        final_answer_block = None
    final_prompt = hf._apply_chat(
        build_prompt_for_answer(query, memory_block=final_answer_block, lang=lang, style=cfg.answer_style),
        system_text=None,
    )

    if cfg.probe_compare:
        compare_top_n = max(1, int(cfg.probe_compare_top_n))
        final_logits = hf.next_token_logits(final_prompt)
        probe_compare_after = _top_tokens_from_logits(hf, final_logits, top_n=compare_top_n)
        probe_compare = build_probe_compare(
            probe_compare_before,
            probe_compare_after,
            top_n=compare_top_n,
            js_divergence=_js_divergence_from_logits(logits, final_logits),
            js_divergence_mode="full_vocab",
        )
        if trace:
            trace.event(
                "probe_compare",
                run_id=run_id,
                pack_item=pack_item,
                **_probe_compare_trace_fields(probe_compare),
                movers=(probe_compare.get("movers") or [])[:8],
                entered=(probe_compare.get("entered") or [])[:8],
                exited=(probe_compare.get("exited") or [])[:8],
            )
        if cfg.print_probe:
            _print_probe_table("PROBE TOP TOKENS (FINAL)", probe_compare_after, limit=compare_top_n)
            _print_probe_compare_summary("PROBE COMPARE", probe_compare, limit=min(8, compare_top_n))

    answer_s = 0.0
    final_answer_meta: Dict[str, Any] = {}
    if cfg.answer_ensemble and ensemble_final:
        answer = ensemble_final.strip()
    else:
        t_answer0 = time.perf_counter()
        answer = hf.generate_text(
            final_prompt,
            max_new_tokens=cfg.answer_max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repetition_penalty=cfg.repetition_penalty,
            no_repeat_ngram_size=cfg.no_repeat_ngram_size,
            seed=cfg.seed,
        )
        answer_s = float(time.perf_counter() - t_answer0)

    if trace:
        trace.event(
            "answer_done",
            run_id=run_id,
            pack_item=pack_item,
            elapsed_s=float(answer_s),
            answer_chars=len(answer or ""),
            answer_preview=trace.preview(answer),
        )

    total_s = float(time.perf_counter() - t_total0)

    extras: Dict[str, Any] = {
        "run_id": run_id,
        "control": control,
        "pipeline": pipeline,
        "answer_style": cfg.answer_style,
        "memory_policy": cfg.memory_policy,
        "memory_retrieve": cfg.memory_retrieve,
        "memory_remix": cfg.memory_remix,
        "memory_selected": memory_selected_meta,
        "band_answers": band_answers if band_answers else None,
        "ensemble": {
            "raw": ensemble_raw,
            "consensus": ensemble_consensus,
            "divergence": ensemble_divergence,
            "final": ensemble_final,
        }
        if ensemble_raw
        else None,
        "probe_compare": probe_compare,
        "probe_compare_stages": probe_compare_stages if probe_compare_stages else None,
        "random_log": random_log_text,
        "prompt_jitter_queries": jitter_queries if jitter_queries else None,
        "timings": {"total_s": total_s, "probe_s": float(probe_s), "memory_remix_s": float(remix_s), "answer_s": float(answer_s)},
    }

    if trace:
        trace.event("ponder_end", run_id=run_id, pack_item=pack_item, elapsed_s=total_s)

    return answer, records, extras


def run_ponder_api(
    hf: OpenAICompatModel,
    cfg: RunConfig,
    query: str,
    *,
    trace: Optional[TraceWriter] = None,
    pack_item: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    t_total0 = time.perf_counter()
    lang = cfg.prompt_lang
    n_ponder = max(1, int(cfg.n_ponder))  # per band

    run_id = make_run_id(cfg.seed, query)
    query_sha = sha256_short(query)

    pipeline = list(cfg.ponder_pipeline) if cfg.ponder_pipeline else [cfg.ponder_mode]
    pipeline_ctx = (cfg.pipeline_context or "prev").strip()

    objective = (cfg.keyword_objective or "random_band").strip()
    control = (cfg.control or "none").strip()
    if control == "random_keywords":
        objective = "random_vocab"

    if trace:
        trace.event(
            "ponder_start",
            run_id=run_id,
            pack_item=pack_item,
            backend="openai_compat",
            query_sha=query_sha,
            objective=objective,
            control=control,
            pipeline=pipeline,
            hops=int(cfg.ponder_hops),
            band_profile=cfg.band_profile,
            memory_policy=cfg.memory_policy,
            memory_retrieve=cfg.memory_retrieve,
            memory_remix=cfg.memory_remix,
            answer_style=cfg.answer_style,
        )

    # Seed probe (optional): OpenAI-compatible top-logprobs for the *next* token.
    t_probe0 = time.perf_counter()
    base_prompt = hf._apply_chat(
        build_prompt_for_answer(query, memory_block=None, lang=lang, style=cfg.answer_style),
        system_text=None,
    )

    seed_method = (cfg.api_seed_method or "auto").strip()
    top_n = int(cfg.api_logprobs_top_n)
    use_logprobs = seed_method == "logprobs" or (seed_method == "auto" and top_n > 0)
    if seed_method == "logprobs" and top_n <= 0:
        top_n = 128

    probe_compare: Optional[Dict[str, Any]] = None
    probe_compare_stages: List[Dict[str, Any]] = []
    probe_compare_n = max(1, int(cfg.probe_compare_top_n)) if (cfg.probe_compare or cfg.probe_compare_stages) else 0
    probe_top: List[Dict[str, Any]] = []
    probe_n = max(
        top_n if use_logprobs and top_n > 0 else 0,
        int(cfg.probe_top_n) if cfg.probe_top_n > 0 else 0,
        probe_compare_n,
        20 if cfg.print_probe else 0,
    )
    if use_logprobs or cfg.print_probe or cfg.probe_top_n > 0 or cfg.probe_compare or cfg.probe_compare_stages:
        try:
            probe_top = hf.probe_top_logprobs(base_prompt, top_n=probe_n)
        except Exception:
            probe_top = []
    probe_s = float(time.perf_counter() - t_probe0)
    if trace:
        trace.event(
            "probe_done",
            run_id=run_id,
            pack_item=pack_item,
            elapsed_s=probe_s,
            prompt_chars=len(base_prompt or ""),
            use_logprobs=bool(use_logprobs),
            top_n=int(top_n),
            returned=int(len(probe_top or [])),
        )

    if cfg.print_probe and probe_top:
        items = [
            {"token_id": None, "token": x.get("token", ""), "rank": int(x.get("rank", 0)), "prob": float(x.get("prob", 0.0))}
            for x in probe_top
        ]
        _print_probe_table("PROBE TOP TOKENS (API)", items, limit=min(len(items), 50))

    probe_compare_before = probe_top[:probe_compare_n] if (cfg.probe_compare or cfg.probe_compare_stages) and probe_top else []
    probe_compare_prev_items = list(probe_compare_before)

    # Unstable objective (logprobs-only): compute per-token stddev across prompt jitters.
    jitter_queries: List[str] = []
    jitter_maps: List[Dict[str, float]] = []
    if use_logprobs and objective == "unstable":
        jitter_n = int(cfg.prompt_jitter)
        if jitter_n <= 0:
            jitter_n = 3
        jitter_queries = generate_prompt_jitters(
            hf,
            query=query,
            n=jitter_n,
            lang=lang,
            include_original=bool(cfg.prompt_jitter_include_original),
            max_new_tokens=int(cfg.prompt_jitter_max_new_tokens),
            temperature=float(cfg.prompt_jitter_temperature),
            seed=int(cfg.seed) + 999,
        )
        probe_n = max(32, top_n)
        for jq in jitter_queries:
            jp = hf._apply_chat(
                build_prompt_for_answer(jq, memory_block=None, lang=lang, style=cfg.answer_style),
                system_text=None,
            )
            try:
                tops = hf.probe_top_logprobs(jp, top_n=probe_n)
            except Exception:
                tops = []
            jitter_maps.append({str(x.get("token", "")): float(x.get("logprob", -1e9)) for x in tops if x.get("token")})

    unstable_scores: Dict[str, float] = {}
    if jitter_maps and probe_top:
        toks = [str(x.get("token", "")) for x in probe_top if isinstance(x.get("token"), str)]
        for t in toks:
            vals = [m[t] for m in jitter_maps if t in m]
            if len(vals) < 2:
                continue
            mu = sum(vals) / float(len(vals))
            var = sum((x - mu) ** 2 for x in vals) / float(len(vals))
            unstable_scores[t] = float(math.sqrt(var))

    records: List[Dict[str, Any]] = []
    log_ix = 0

    bands = cfg.bands or _default_bands_from_profile(cfg.band_profile)
    if not bands:
        bands = [{"label": "single", "start_rank": 0, "end_rank": 0}]

    api_warnings: List[str] = []
    warned_probe_short = set()
    warned_stage_probe_failure = False
    probe_depth = len(probe_top)
    if use_logprobs and probe_depth <= 0:
        api_warnings.append("API logprobs probing returned no usable tokens; keyword seeding will fall back to self-generation.")
    stage_probe_active = bool(cfg.probe_compare_stages and probe_compare_before)
    if cfg.probe_compare_stages and not stage_probe_active:
        msg = "Stage probe compare requested, but the API did not return a usable base prompt top-logprobs distribution."
        api_warnings.append(msg)
        print(f"[sr_ponder] WARN: {msg}", file=sys.stderr)
        probe_compare_stages.append(
            make_probe_compare_timeline_entry(
                source="current_records",
                point="stage",
                status="unavailable",
                reason=msg,
            )
        )
        if trace:
            trace.event("probe_compare_stage", run_id=run_id, pack_item=pack_item, status="unavailable", reason=msg)

    for band_ix, band in enumerate(bands):
        band_label = str(band.get("label", f"band{band_ix}"))
        band_start = int(band.get("start_rank", 0))
        band_end = int(band.get("end_rank", 0))

        for band_ponder_ix in range(n_ponder):
            n_kw = int(cfg.rejected.n_keywords)
            rng = random.Random(int(cfg.seed) + 99991 + band_ix * 10007 + band_ponder_ix * 101 + log_ix * 37)

            keywords_source = "api_self"
            raw_keywords: List[str] = []
            picked_items: List[Dict[str, Any]] = []
            seed_fallback_reason: Optional[str] = None
            api_seed_method_used = "self"
            api_band_probe_truncated = False

            # Attempt logprobs-based seed selection when available.
            if use_logprobs and probe_top:
                pool = probe_top
                if band_end > band_start:
                    api_band_probe_truncated = probe_depth < band_end
                    if band_start >= probe_depth:
                        pool = []
                        seed_fallback_reason = f"probe_depth={probe_depth} < band_start={band_start}"
                    elif band_end > probe_depth:
                        pool = pool[band_start:probe_depth]
                        seed_fallback_reason = f"probe_depth={probe_depth} < band_end={band_end}"
                    else:
                        pool = pool[band_start:band_end]
                if seed_fallback_reason:
                    warn_key = (band_label, band_start, band_end, probe_depth)
                    if warn_key not in warned_probe_short:
                        warned_probe_short.add(warn_key)
                        if band_start >= probe_depth:
                            msg = (
                                f"API logprobs depth {probe_depth} does not reach band {band_label!r} "
                                f"({band_start}:{band_end}); falling back to self-seeded keywords for this band."
                            )
                        else:
                            msg = (
                                f"API logprobs depth {probe_depth} only partially covers band {band_label!r} "
                                f"({band_start}:{band_end}); using truncated logprobs and self-seeded fallback if needed."
                            )
                        api_warnings.append(msg)
                        print(f"[sr_ponder] WARN: {msg}", file=sys.stderr)
                pool = [x for x in pool if isinstance(x.get("token"), str) and str(x.get("token")).strip()]
                obj_mode = objective

                tokens = [str(x.get("token", "")).strip() for x in pool]
                # de-dupe preserve order
                seen = set()
                tokens2: List[str] = []
                for t in tokens:
                    if not t:
                        continue
                    if t in seen:
                        continue
                    seen.add(t)
                    tokens2.append(t)

                dissonance_scores: Dict[str, float] = {}
                preserve = False
                if obj_mode == "unstable" and unstable_scores:
                    tokens2.sort(key=lambda t: float(unstable_scores.get(t, 0.0)), reverse=True)
                    tokens2 = tokens2[: max(1, int(cfg.keyword_select_top))]
                    preserve = True
                elif obj_mode == "dissonance":
                    dissonance_scores = api_lex_dissonance_scores(query, tokens2, lang=lang)
                    lo = float(cfg.dissonance_target) - float(cfg.dissonance_width) / 2.0
                    hi = float(cfg.dissonance_target) + float(cfg.dissonance_width) / 2.0
                    filtered = [t for t in tokens2 if lo <= float(dissonance_scores.get(t, 0.0)) <= hi]
                    if filtered:
                        filtered.sort(key=lambda t: abs(float(dissonance_scores.get(t, 0.0)) - float(cfg.dissonance_target)))
                        tokens2 = filtered[: max(1, int(cfg.keyword_select_top))]
                    else:
                        tokens2.sort(key=lambda t: float(dissonance_scores.get(t, 0.0)), reverse=True)
                        tokens2 = tokens2[: max(1, int(cfg.keyword_select_top))]
                    preserve = True

                if len(tokens2) >= n_kw:
                    if str(cfg.keyword_diversity).strip() != "off":
                        picked = pick_diverse_strings(
                            tokens2,
                            rng=rng,
                            n=n_kw,
                            threshold=float(cfg.keyword_diversity_threshold),
                            preserve_order=preserve,
                        )
                    else:
                        picked = rng.sample(tokens2, k=n_kw)
                else:
                    picked = []

                if picked:
                    keywords_source = "api_logprobs"
                    api_seed_method_used = "logprobs"
                    seed_fallback_reason = None
                    raw_keywords = picked
                    lookup = {str(x.get("token", "")).strip(): x for x in probe_top}
                    for t in picked:
                        x = lookup.get(t) or {}
                        picked_items.append(
                            {
                                "token_id": None,
                                "token": t,
                                "rank": x.get("rank"),
                                "prob": x.get("prob"),
                                "logprob": x.get("logprob"),
                                "dissonance": dissonance_scores.get(t) if dissonance_scores else None,
                                "unstable": unstable_scores.get(t) if unstable_scores else None,
                            }
                        )

            # Self-seeded keywords (fallback / default).
            if not raw_keywords:
                raw_keywords = generate_seed_keywords_self(
                    hf,
                    query=query,
                    n_keywords=n_kw,
                    lang=lang,
                    band_label=band_label,
                    objective=objective,
                    max_new_tokens=160,
                    temperature=max(0.2, float(cfg.temperature)),
                    seed=int(cfg.seed) + 5000 + log_ix,
                )
                keywords_source = "api_self"
                api_seed_method_used = "self"
                picked_items = [{"token_id": None, "token": t, "rank": None, "prob": None} for t in raw_keywords]

            keywords = raw_keywords
            refined_keywords: List[str] = []
            if cfg.keyword_refine and keywords_source != "human_pick":
                refined_keywords = refine_keywords_with_model(
                    hf,
                    query=query,
                    seed_keywords=raw_keywords,
                    n_keywords=n_kw,
                    lang=lang,
                    max_new_tokens=int(cfg.keyword_refine_max_new_tokens),
                    temperature=float(cfg.keyword_refine_temperature),
                    seed=int(cfg.seed) + 200 + log_ix,
                )
                if refined_keywords:
                    keywords = refined_keywords
                    keywords_source = "model_refine"

            if cfg.interactive:
                cand_items = picked_items or [{"token_id": None, "token": t, "rank": i, "prob": None} for i, t in enumerate(keywords)]
                picked = interactive_pick_keywords(
                    prompt=f"band={band_label} band_ix={band_ix} log={band_ponder_ix}",
                    rng=rng,
                    candidates=cand_items,
                    n_keywords=n_kw,
                )
                if picked:
                    keywords = picked
                    keywords_source = "human_pick"

            hop_n = max(1, int(cfg.ponder_hops))
            prev_hop_log: Optional[str] = None
            prev_hop_keywords: List[str] = list(keywords)

            for hop_ix in range(hop_n):
                hop_keyword_source: Optional[str] = None
                hop_keywords_source = keywords_source
                hop_raw_keywords = list(raw_keywords)
                hop_keywords = list(keywords)
                hop_picked_items = list(picked_items)

                if hop_ix > 0:
                    hop_keyword_source = (cfg.hop_keyword_source or "model").strip()
                    if hop_keyword_source not in ("model", "heuristic"):
                        hop_keyword_source = "model"

                    hk_seed = int(cfg.seed) + 6000 + band_ix * 10007 + band_ponder_ix * 101 + hop_ix * 271
                    new_raw: List[str] = []
                    if hop_keyword_source == "model" and prev_hop_log:
                        try:
                            new_raw = extract_hop_keywords_with_model(
                                hf,
                                query=query,
                                prev_keywords=prev_hop_keywords,
                                ponder_log=prev_hop_log,
                                n=n_kw,
                                lang=lang,
                                seed=hk_seed,
                            )
                        except Exception:
                            new_raw = []

                    if not new_raw and prev_hop_log:
                        hrng = random.Random(int(cfg.seed) + 6100 + band_ix * 10007 + band_ponder_ix * 101 + hop_ix * 271)
                        new_raw = extract_hop_keywords_heuristic(
                            query=query,
                            prev_keywords=prev_hop_keywords,
                            ponder_log=prev_hop_log,
                            n=n_kw,
                            lang=lang,
                            rng=hrng,
                        )
                        if new_raw:
                            hop_keyword_source = "heuristic"

                    if not new_raw:
                        new_raw = list(prev_hop_keywords)

                    hop_raw_keywords = list(new_raw)
                    hop_keywords = list(new_raw)
                    hop_keywords_source = f"hop_{hop_keyword_source}"
                    hop_picked_items = []

                    if cfg.keyword_refine:
                        refined = refine_keywords_with_model(
                            hf,
                            query=query,
                            seed_keywords=hop_raw_keywords,
                            n_keywords=n_kw,
                            lang=lang,
                            max_new_tokens=int(cfg.keyword_refine_max_new_tokens),
                            temperature=float(cfg.keyword_refine_temperature),
                            seed=int(cfg.seed) + 220 + log_ix + hop_ix * 11,
                        )
                        if refined:
                            hop_keywords = refined
                            hop_keywords_source = "model_refine"

                if trace:
                    trace.event(
                        "seed_keywords",
                        run_id=run_id,
                        pack_item=pack_item,
                        band_label=band_label,
                        band_ponder_ix=int(band_ponder_ix),
                        hop_ix=int(hop_ix),
                        keywords=hop_keywords,
                        keywords_source=hop_keywords_source,
                    )

                # One ponder question per (hop, band, band_ponder_ix), reused across pipeline stages.
                if control == "lens_only":
                    ponder_q = query
                else:
                    q_rng = random.Random(int(cfg.seed) + 12345 + band_ix * 10007 + band_ponder_ix * 101 + hop_ix * 1009)
                    ponder_q = make_unrelated_question(hop_keywords, lang=lang, rng=q_rng)

                stage_logs: List[str] = []
                for stage_ix, stage_mode in enumerate(pipeline):
                    ctx_text: Optional[str] = None
                    if hop_ix > 0 and stage_ix == 0 and prev_hop_log:
                        ctx_text = prev_hop_log
                        if ctx_text and len(ctx_text) > int(cfg.hop_context_max_chars):
                            ctx_text = ctx_text[-int(cfg.hop_context_max_chars) :]

                    if pipeline_ctx == "prev" and stage_logs:
                        ctx_text = stage_logs[-1]
                    elif pipeline_ctx == "all" and stage_logs:
                        ctx_text = "\n\n".join(stage_logs)

                    if ctx_text and len(ctx_text) > int(cfg.pipeline_context_max_chars):
                        ctx_text = ctx_text[-int(cfg.pipeline_context_max_chars) :]

                    ponder_prompt = hf._apply_chat(
                        build_prompt_for_pondering(ponder_q, mode=stage_mode, lang=lang, context=ctx_text),
                        system_text=None,
                    )
                    t_gen0 = time.perf_counter()
                    ponder_log = hf.generate_text(
                        ponder_prompt,
                        max_new_tokens=cfg.ponder_max_new_tokens,
                        temperature=max(0.4, cfg.temperature),
                        top_p=cfg.top_p,
                        top_k=0,
                        repetition_penalty=1.0,
                        no_repeat_ngram_size=0,
                        seed=int(cfg.seed) + 1 + log_ix,
                    )
                    ponder_meta = summarize_api_generation_meta(getattr(hf, "last_response_meta", {}))
                    gen_s = float(time.perf_counter() - t_gen0)
                    stage_logs.append(ponder_log)
                    if trace:
                        trace.event(
                            "ponder_stage",
                            run_id=run_id,
                            pack_item=pack_item,
                            band_label=band_label,
                            band_ponder_ix=int(band_ponder_ix),
                            hop_ix=int(hop_ix),
                            stage_ix=int(stage_ix),
                            mode=stage_mode,
                            elapsed_s=gen_s,
                            text_chars=len(ponder_log or ""),
                            text_preview=trace.preview(ponder_log),
                            api_meta=ponder_meta if ponder_meta else None,
                        )

                    if cfg.print_probe:
                        _print_probe_table(
                            f"SEED KEYWORDS (API band={band_label} hop={hop_ix} ix={log_ix} stage={stage_mode})",
                            [{"token_id": None, "token": t, "rank": None, "prob": None} for t in (hop_keywords or [])],
                            limit=len(hop_keywords or []),
                        )
                        print(f"\nkeywords_source={hop_keywords_source} keywords={hop_keywords}\n")

                    record: Dict[str, Any] = {
                        "ts": now_iso(),
                        "run_id": run_id,
                        "query_sha": query_sha,
                        "prompt_lang": lang,
                        "backend": "openai_compat",
                        "control": control,
                        "band_profile": cfg.band_profile,
                        "band_ix": band_ix,
                        "band_label": band_label,
                        "band_ponder_ix": band_ponder_ix,
                        "hop_ix": hop_ix,
                        "hop_keyword_source": hop_keyword_source,
                        "pipeline": pipeline,
                        "pipeline_stage_ix": stage_ix,
                        "pipeline_context": pipeline_ctx,
                        "ponder_ix": log_ix,
                        "ponder_mode": stage_mode,
                        "keywords_source": hop_keywords_source,
                        "keywords_raw": hop_raw_keywords,
                        "keywords": hop_keywords,
                        "token_ids": [],
                        "selected_tokens": hop_picked_items,
                        "rejected_cfg": dataclasses.asdict(cfg.rejected),
                        "band": {"start_rank": band_start, "end_rank": band_end},
                        "api_seed_method_requested": seed_method,
                        "api_seed_method_used": api_seed_method_used,
                        "api_probe_depth": probe_depth,
                        "api_band_probe_truncated": api_band_probe_truncated,
                        "api_seed_fallback_reason": seed_fallback_reason,
                        "api_generation": ponder_meta if ponder_meta else None,
                        "ponder_question": ponder_q,
                        "ponder_log": ponder_log,
                        "gen_elapsed_s": gen_s,
                    }
                    if cfg.probe_top_n > 0 and log_ix == 0 and hop_ix == 0 and probe_top:
                        record["probe_top"] = probe_top[: int(cfg.probe_top_n)]
                    if jitter_queries and log_ix == 0 and hop_ix == 0:
                        record["prompt_jitter_queries"] = jitter_queries

                    if cfg.write_memory:
                        append_jsonl(cfg.memory_path, record)
                    records.append(record)
                    if not (ponder_log or "").strip():
                        finish_reason = ponder_meta.get("finish_reason")
                        reasoning_tokens = ponder_meta.get("reasoning_tokens")
                        completion_tokens = ponder_meta.get("completion_tokens")
                        refusal = ponder_meta.get("refusal")
                        msg = (
                            f"API ponder stage returned empty text (band={band_label}, hop={hop_ix}, stage={stage_mode}, "
                            f"finish_reason={finish_reason!r}, completion_tokens={completion_tokens!r}, reasoning_tokens={reasoning_tokens!r})."
                        )
                        if refusal:
                            msg += " refusal text was present."
                        if msg not in api_warnings:
                            api_warnings.append(msg)
                            print(f"[sr_ponder] WARN: {msg}", file=sys.stderr)
                    if stage_probe_active:
                        stage_memory_block = build_memory_block(records, max_chars_per_log=700) if records else None
                        stage_prompt = hf._apply_chat(
                            build_prompt_for_answer(query, memory_block=stage_memory_block, lang=lang, style=cfg.answer_style),
                            system_text=None,
                        )
                        stage_items: List[Dict[str, Any]] = []
                        stage_reason = ""
                        try:
                            stage_items = hf.probe_top_logprobs(stage_prompt, top_n=probe_compare_n)
                        except Exception as e:
                            stage_reason = str(e)
                            stage_items = []

                        if stage_items:
                            comp_base = build_probe_compare(
                                probe_compare_before,
                                stage_items,
                                top_n=probe_compare_n,
                                js_divergence_mode="topn_union_renorm",
                            )
                            comp_prev = None
                            if probe_compare_prev_items:
                                comp_prev = build_probe_compare(
                                    probe_compare_prev_items,
                                    stage_items,
                                    top_n=probe_compare_n,
                                    js_divergence_mode="topn_union_renorm",
                                )
                            stage_entry = make_probe_compare_timeline_entry(
                                source="current_records",
                                point="stage",
                                record=record,
                                compare_from_base=comp_base,
                                compare_from_prev=comp_prev,
                                memory_chars=len(stage_memory_block or ""),
                                prompt_chars=len(stage_prompt or ""),
                            )
                            probe_compare_stages.append(stage_entry)
                            if trace:
                                trace.event(
                                    "probe_compare_stage",
                                    run_id=run_id,
                                    pack_item=pack_item,
                                    source="current_records",
                                    band_label=band_label,
                                    band_ponder_ix=int(band_ponder_ix),
                                    hop_ix=int(hop_ix),
                                    stage_ix=int(stage_ix),
                                    ponder_ix=int(record.get("ponder_ix", 0) or 0),
                                    ponder_mode=stage_mode,
                                    memory_chars=len(stage_memory_block or ""),
                                    prompt_chars=len(stage_prompt or ""),
                                    prev_js_divergence=float(comp_prev.get("js_divergence", 0.0)) if isinstance(comp_prev, dict) else None,
                                    **_probe_compare_trace_fields(comp_base),
                                )
                            if cfg.print_probe:
                                _print_probe_compare_summary(
                                    f"PROBE STAGE (API band={band_label} hop={hop_ix} stage={stage_mode} ix={log_ix})",
                                    comp_base,
                                    limit=min(8, probe_compare_n),
                                )
                            probe_compare_prev_items = list(stage_items)
                        else:
                            msg = "Stage probe compare requested, but the API did not return a usable stage top-logprobs distribution."
                            if stage_reason:
                                msg = f"{msg} ({stage_reason})"
                            if not warned_stage_probe_failure:
                                warned_stage_probe_failure = True
                                api_warnings.append(msg)
                                print(f"[sr_ponder] WARN: {msg}", file=sys.stderr)
                            stage_entry = make_probe_compare_timeline_entry(
                                source="current_records",
                                point="stage",
                                record=record,
                                memory_chars=len(stage_memory_block or ""),
                                prompt_chars=len(stage_prompt or ""),
                                status="unavailable",
                                reason=msg,
                            )
                            probe_compare_stages.append(stage_entry)
                            if trace:
                                trace.event(
                                    "probe_compare_stage",
                                    run_id=run_id,
                                    pack_item=pack_item,
                                    source="current_records",
                                    band_label=band_label,
                                    band_ponder_ix=int(band_ponder_ix),
                                    hop_ix=int(hop_ix),
                                    stage_ix=int(stage_ix),
                                    ponder_ix=int(record.get("ponder_ix", 0) or 0),
                                    ponder_mode=stage_mode,
                                    status="unavailable",
                                    reason=msg,
                                )
                    log_ix += 1

                prev_hop_log = "\n\n".join(stage_logs).strip() if stage_logs else prev_hop_log
                prev_hop_keywords = list(hop_keywords)

    # Select memory records for final answer injection (API: fuzzy retrieval via hashed char n-grams)
    exclude_run_id = run_id if cfg.memory_exclude_current_run and cfg.memory_policy == "tail" else None
    mem_records = select_memory_records_fuzzy(
        memory_path=cfg.memory_path,
        current_records=records,
        query=query,
        memory_policy=cfg.memory_policy,
        memory_retrieve=cfg.memory_retrieve,
        n_memory=cfg.n_memory,
        pool_size=cfg.memory_pool,
        mix_ratio=cfg.memory_mix_ratio,
        exclude_run_id=exclude_run_id,
    )
    if control == "no_inject":
        mem_records = []

    memory_block = build_memory_block(mem_records, max_chars_per_log=700) if mem_records else None

    memory_selected_meta = [
        {
            "ts": r.get("ts"),
            "run_id": r.get("run_id"),
            "band_label": r.get("band_label"),
            "ponder_ix": r.get("ponder_ix"),
            "ponder_mode": r.get("ponder_mode"),
        }
        for r in (mem_records or [])
    ]
    if trace:
        trace.event(
            "memory_selected",
            run_id=run_id,
            pack_item=pack_item,
            count=int(len(mem_records or [])),
            memory_policy=cfg.memory_policy,
            memory_retrieve=cfg.memory_retrieve,
            memory_remix=cfg.memory_remix,
            selected=memory_selected_meta[:12],
        )

    t_remix0 = time.perf_counter()
    memory_block = remix_memory_block(
        hf,
        memory_block=memory_block,
        remix=cfg.memory_remix,
        lang=lang,
        keep_original=cfg.memory_remix_keep_original,
        max_new_tokens=cfg.memory_remix_max_new_tokens,
        temperature=cfg.memory_remix_temperature,
        seed=int(cfg.seed) + 777,
    )
    remix_s = float(time.perf_counter() - t_remix0)
    if trace:
        trace.event(
            "memory_remix_done",
            run_id=run_id,
            pack_item=pack_item,
            remix=cfg.memory_remix,
            elapsed_s=remix_s,
            text_chars=len(memory_block or ""),
            text_preview=trace.preview(memory_block),
        )

    random_log_text: Optional[str] = None
    if control == "random_log":
        rng = random.Random(int(cfg.seed) + 424242)
        rand_kw = generate_seed_keywords_self(
            hf,
            query=query,
            n_keywords=int(cfg.rejected.n_keywords),
            lang=lang,
            band_label="random",
            objective="random_vocab",
            max_new_tokens=120,
            temperature=0.8,
            seed=int(cfg.seed) + 424242,
        )
        q_rng = random.Random(int(cfg.seed) + 424243)
        rand_q = make_unrelated_question(rand_kw, lang=lang, rng=q_rng)
        rp = hf._apply_chat(build_prompt_for_pondering(rand_q, mode="assoc", lang=lang), system_text=None)
        random_log_text = hf.generate_text(
            rp,
            max_new_tokens=int(cfg.ponder_max_new_tokens),
            temperature=max(0.6, float(cfg.temperature)),
            top_p=float(cfg.top_p),
            top_k=0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            seed=int(cfg.seed) + 424244,
        )
        memory_block = random_log_text

    # Band answers (sensitivity analysis)
    band_answers: Dict[str, str] = {}
    if cfg.answer_per_band or cfg.answer_ensemble:
        for band in bands:
            bl = str(band.get("label", "band"))
            band_recs = [r for r in records if str(r.get("band_label", "")) == bl]
            if not band_recs:
                continue
            band_mem = None if control == "no_inject" else build_memory_block(band_recs, max_chars_per_log=700)
            band_mem = remix_memory_block(
                hf,
                memory_block=band_mem,
                remix=cfg.memory_remix,
                lang=lang,
                keep_original=False,
                max_new_tokens=cfg.memory_remix_max_new_tokens,
                temperature=cfg.memory_remix_temperature,
                seed=int(cfg.seed) + 9000 + stable_hash_mod(bl, 1000),
            )
            fp = hf._apply_chat(
                build_prompt_for_answer(query, memory_block=band_mem, lang=lang, style=cfg.answer_style),
                system_text=None,
            )
            band_ans = hf.generate_text(
                fp,
                max_new_tokens=cfg.answer_max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=0,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                seed=int(cfg.seed) + 10101,
            )
            band_answers[bl] = band_ans

    ensemble_raw: Optional[str] = None
    ensemble_final: Optional[str] = None
    ensemble_consensus: Optional[str] = None
    ensemble_divergence: Optional[str] = None
    if cfg.answer_ensemble and len(band_answers) >= 2:
        ep = hf._apply_chat(build_prompt_for_ensemble_summary(query, band_answers, lang=lang), system_text=None)
        ensemble_raw = hf.generate_text(
            ep,
            max_new_tokens=int(cfg.answer_ensemble_max_new_tokens),
            temperature=float(cfg.answer_ensemble_temperature),
            top_p=0.95,
            top_k=0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            seed=int(cfg.seed) + 20202,
        )
        ensemble_consensus = extract_tag(ensemble_raw, "consensus")
        ensemble_divergence = extract_tag(ensemble_raw, "divergence")
        ensemble_final = extract_tag(ensemble_raw, "final") or ensemble_raw.strip()

    final_answer_block = memory_block
    if control == "no_inject":
        final_answer_block = None
    final_prompt = hf._apply_chat(
        build_prompt_for_answer(query, memory_block=final_answer_block, lang=lang, style=cfg.answer_style),
        system_text=None,
    )

    if cfg.probe_compare:
        if not probe_compare_before:
            msg = "Probe compare requested, but the API did not return a usable base prompt top-logprobs distribution."
            api_warnings.append(msg)
            print(f"[sr_ponder] WARN: {msg}", file=sys.stderr)
            probe_compare = {"top_n": int(probe_compare_n), "status": "unavailable", "reason": msg}
            if trace:
                trace.event("probe_compare", run_id=run_id, pack_item=pack_item, status="unavailable", reason=msg)
        else:
            probe_compare_after: List[Dict[str, Any]] = []
            compare_error: Optional[str] = None
            try:
                probe_compare_after = hf.probe_top_logprobs(final_prompt, top_n=probe_compare_n)
            except Exception as e:
                compare_error = str(e)
            if probe_compare_after:
                if cfg.print_probe:
                    items = [
                        {
                            "token_id": None,
                            "token": x.get("token", ""),
                            "rank": int(x.get("rank", 0)),
                            "prob": float(x.get("prob", 0.0)),
                        }
                        for x in probe_compare_after
                    ]
                    _print_probe_table("PROBE TOP TOKENS (API FINAL)", items, limit=min(len(items), probe_compare_n))
                probe_compare = build_probe_compare(
                    probe_compare_before,
                    probe_compare_after,
                    top_n=probe_compare_n,
                    js_divergence_mode="topn_union_renorm",
                )
                if trace:
                    trace.event(
                        "probe_compare",
                        run_id=run_id,
                        pack_item=pack_item,
                        **_probe_compare_trace_fields(probe_compare),
                        movers=(probe_compare.get("movers") or [])[:8],
                        entered=(probe_compare.get("entered") or [])[:8],
                        exited=(probe_compare.get("exited") or [])[:8],
                    )
                if cfg.print_probe:
                    _print_probe_compare_summary("PROBE COMPARE (API)", probe_compare, limit=min(8, probe_compare_n))
            else:
                msg = "Probe compare requested, but the API did not return a usable final prompt top-logprobs distribution."
                if compare_error:
                    msg = f"{msg} ({compare_error})"
                api_warnings.append(msg)
                print(f"[sr_ponder] WARN: {msg}", file=sys.stderr)
                probe_compare = {"top_n": int(probe_compare_n), "status": "unavailable", "reason": msg}
                if trace:
                    trace.event("probe_compare", run_id=run_id, pack_item=pack_item, status="unavailable", reason=msg)

    answer_s = 0.0
    if cfg.answer_ensemble and ensemble_final:
        answer = ensemble_final.strip()
    else:
        t_answer0 = time.perf_counter()
        answer = hf.generate_text(
            final_prompt,
            max_new_tokens=cfg.answer_max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            seed=cfg.seed,
        )
        final_answer_meta = summarize_api_generation_meta(getattr(hf, "last_response_meta", {}))
        answer_s = float(time.perf_counter() - t_answer0)
        if not (answer or "").strip():
            finish_reason = final_answer_meta.get("finish_reason")
            reasoning_tokens = final_answer_meta.get("reasoning_tokens")
            completion_tokens = final_answer_meta.get("completion_tokens")
            refusal = final_answer_meta.get("refusal")
            msg = (
                f"API final answer returned empty text (finish_reason={finish_reason!r}, "
                f"completion_tokens={completion_tokens!r}, reasoning_tokens={reasoning_tokens!r})."
            )
            if refusal:
                msg += " refusal text was present."
            if msg not in api_warnings:
                api_warnings.append(msg)
                print(f"[sr_ponder] WARN: {msg}", file=sys.stderr)

    if trace:
        trace.event(
            "answer_done",
            run_id=run_id,
            pack_item=pack_item,
            elapsed_s=float(answer_s),
            answer_chars=len(answer or ""),
            answer_preview=trace.preview(answer),
            api_meta=final_answer_meta if final_answer_meta else None,
        )

    total_s = float(time.perf_counter() - t_total0)

    extras: Dict[str, Any] = {
        "run_id": run_id,
        "control": control,
        "pipeline": pipeline,
        "answer_style": cfg.answer_style,
        "memory_policy": cfg.memory_policy,
        "memory_retrieve": cfg.memory_retrieve,
        "memory_remix": cfg.memory_remix,
        "memory_selected": memory_selected_meta,
        "band_answers": band_answers if band_answers else None,
        "ensemble": {
            "raw": ensemble_raw,
            "consensus": ensemble_consensus,
            "divergence": ensemble_divergence,
            "final": ensemble_final,
        }
        if ensemble_raw
        else None,
        "probe_compare": probe_compare,
        "probe_compare_stages": probe_compare_stages if probe_compare_stages else None,
        "random_log": random_log_text,
        "prompt_jitter_queries": jitter_queries if jitter_queries else None,
        "api_probe_top": probe_top[:50] if probe_top else None,
        "api_final_generation": final_answer_meta if final_answer_meta else None,
        "timings": {"total_s": total_s, "probe_s": float(probe_s), "memory_remix_s": float(remix_s), "answer_s": float(answer_s)},
        "api_warnings": api_warnings if api_warnings else None,
    }

    if trace:
        trace.event("ponder_end", run_id=run_id, pack_item=pack_item, elapsed_s=total_s)

    return answer, records, extras


def run_ponder_dispatch(
    hf: Any,
    cfg: RunConfig,
    query: str,
    *,
    trace: Optional[TraceWriter] = None,
    pack_item: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    if (cfg.backend or "hf").strip() == "openai_compat":
        return run_ponder_api(hf, cfg, query, trace=trace, pack_item=pack_item)
    return run_ponder(hf, cfg, query, trace=trace, pack_item=pack_item)


def main() -> None:
    # Windows console defaults (e.g., cp932) can crash on arbitrary Unicode coming from model outputs.
    # Prefer UTF-8 with replacement to keep the CLI usable.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    class _Fmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    # Optional JSON config (applies defaults; CLI flags still override).
    config_path = ""
    config_defaults: Dict[str, Any] = {}
    try:
        cp = argparse.ArgumentParser(add_help=False)
        cp.add_argument("--config", default="")
        pre, _ = cp.parse_known_args()
        config_path = str(getattr(pre, "config", "") or "").strip()
    except Exception:
        config_path = ""

    if config_path:
        try:
            # Windows editors often write UTF-8 with BOM; accept both.
            raw = Path(config_path).read_text(encoding="utf-8-sig")
            config_defaults = json.loads(raw)
        except Exception as e:
            raise SystemExit(f"[sr_ponder] ERROR: failed to read --config {config_path!r}: {e}")
        if not isinstance(config_defaults, dict):
            raise SystemExit(
                f"[sr_ponder] ERROR: --config must be a JSON object (dict), got {type(config_defaults).__name__}"
            )

    ap = argparse.ArgumentParser(
        description="Pondering machine (local HF + provider presets) - rejected-token + lens + bands + memory.",
        formatter_class=_Fmt,
    )

    g_core = ap.add_argument_group("Core")
    g_ponder = ap.add_argument_group("Ponder")
    g_pipeline = ap.add_argument_group("Pipeline")
    g_bands = ap.add_argument_group("Bands")
    g_keywords = ap.add_argument_group("Keywords")
    g_jitter = ap.add_argument_group("Prompt Jitter")
    g_memory = ap.add_argument_group("Memory")
    g_answer = ap.add_argument_group("Answers")
    g_interactive = ap.add_argument_group("Interactive")
    g_controls = ap.add_argument_group("Controls / Pack")
    g_observe = ap.add_argument_group("Observability / Artifacts")
    g_runtime = ap.add_argument_group("Runtime")
    g_api = ap.add_argument_group("API (OpenAI-compatible)")
    g_gen = ap.add_argument_group("Generation")

    g_core.add_argument(
        "--provider",
        choices=["auto", "hf", "openai", "mistral", "groq", "openrouter", "deepseek", "custom"],
        default="auto",
        help="High-level provider preset; API presets auto-fill base URL and key env",
    )
    g_core.add_argument(
        "--backend",
        choices=["hf", "openai_compat"],
        default="hf",
        help="Low-level backend override (usually unnecessary when --provider is set)",
    )
    g_core.add_argument(
        "--model",
        required=True,
        help="hf/provider=hf: local model dir (cache roots auto-resolve latest snapshot); API providers: model name",
    )
    g_core.add_argument("--query", required=True, help="User query (main question)")
    g_core.add_argument("--memory", default="ponder_logs.jsonl", help="Path to JSONL memory log")
    g_core.add_argument("--mode", choices=["baseline", "ponder", "both"], default="both", help="Run baseline, ponder, or both (comparison prints in both)")
    g_core.add_argument("--prompt_lang", choices=["auto", "en", "ja"], default="auto", help="Prompt language")
    g_core.add_argument("--preset", choices=["none", "surreal"], default="none", help="Apply curated settings")
    g_core.add_argument("--config", default=config_path, help="Optional JSON config file (sets defaults)")

    g_ponder.add_argument(
        "--ponder_mode",
        choices=["assoc", "assumption", "counterexample", "questions_only", "metaphor"],
        default="assoc",
        help="Ponder lens (used when --ponder_pipeline is empty)",
    )
    g_ponder.add_argument("--n_ponder", type=int, default=1, help="Number of ponder logs per band")
    g_ponder.add_argument("--ponder_hops", type=int, default=1, help="Sequential hops per ponder log (latent walk)")
    g_ponder.add_argument(
        "--hop_keyword_source",
        choices=["model", "heuristic"],
        default="model",
        help="How to derive next-hop keywords from the previous hop's ponder log",
    )
    g_ponder.add_argument("--hop_context_max_chars", type=int, default=900, help="Prev-hop context size for hop>0")
    g_ponder.add_argument(
        "--control",
        choices=list(_CONTROL_VARIANTS),
        default="none",
        help="Control variant for A/B testing",
    )
    g_ponder.add_argument("--no_write_memory", action="store_true", help="Do not append to --memory JSONL")

    g_pipeline.add_argument(
        "--ponder_pipeline",
        default="",
        help="Comma/space-separated lenses to chain (e.g. assumption,counterexample,questions_only,metaphor).",
    )
    g_pipeline.add_argument("--pipeline_context", choices=["none", "prev", "all"], default="prev")
    g_pipeline.add_argument("--pipeline_context_max_chars", type=int, default=1200)

    g_bands.add_argument("--band_profile", choices=["single", "spectrum3"], default="single", help="Rank-band profile")
    g_bands.add_argument(
        "--band",
        action="append",
        default=[],
        help='Custom band "START:END" or "LABEL=START:END" (repeatable; END exclusive).',
    )

    g_keywords.add_argument("--strategy", choices=["within_topk", "outside_topk"], default="outside_topk")
    g_keywords.add_argument("--top_k_rejected", type=int, default=80)
    g_keywords.add_argument("--exclude_top", type=int, default=8)
    g_keywords.add_argument("--band_width", type=int, default=256)
    g_keywords.add_argument("--n_keywords", type=int, default=6)
    g_keywords.add_argument("--keyword_refine", action="store_true", help="Model rewrites token fragments into keywords")
    g_keywords.add_argument("--keyword_refine_max_new_tokens", type=int, default=96)
    g_keywords.add_argument("--keyword_refine_temperature", type=float, default=0.3)
    g_keywords.add_argument(
        "--keyword_objective",
        choices=["random_band", "dissonance", "unstable", "random_vocab"],
        default="random_band",
        help="How to pick keyword token_ids inside each band",
    )
    g_keywords.add_argument("--keyword_select_top", type=int, default=128, help="Top candidates to sample from")
    g_keywords.add_argument(
        "--keyword_diversity",
        choices=["off", "lex", "embed"],
        default="off",
        help="Encourage diverse keywords (lexical or embedding) instead of pure random sampling",
    )
    g_keywords.add_argument(
        "--keyword_diversity_threshold",
        type=float,
        default=0.82,
        help="Lexical similarity cutoff (>= threshold => reject as too similar)",
    )
    g_keywords.add_argument("--dissonance_target", type=float, default=0.9)
    g_keywords.add_argument("--dissonance_width", type=float, default=0.6)
    g_keywords.add_argument("--dissonance_tail_k", type=int, default=64)

    g_jitter.add_argument("--prompt_jitter", type=int, default=0, help="Number of paraphrases (excluding original)")
    g_jitter.add_argument("--no_prompt_jitter_include_original", action="store_true", help="Do not include original query")
    g_jitter.add_argument("--prompt_jitter_max_new_tokens", type=int, default=160)
    g_jitter.add_argument("--prompt_jitter_temperature", type=float, default=0.6)

    g_memory.add_argument("--n_memory", type=int, default=6)
    g_memory.add_argument(
        "--memory_policy",
        choices=["tail", "current_only", "off"],
        default="tail",
        help="Which ponder logs are eligible to inject",
    )
    g_memory.add_argument(
        "--memory_retrieve",
        choices=["tail", "similar", "anti", "mix"],
        default="tail",
        help="How to pick memory records when --memory_policy=tail",
    )
    g_memory.add_argument("--memory_pool", type=int, default=200, help="Pool size for retrieval (tail records)")
    g_memory.add_argument("--memory_mix_ratio", type=float, default=0.5, help="similar/(similar+anti) when mix")
    g_memory.add_argument(
        "--memory_include_current_run",
        action="store_true",
        help="Allow selecting current run records when retrieving from tail",
    )
    g_memory.add_argument("--memory_remix", choices=["off", "shuffle", "compress", "dream"], default="off")
    g_memory.add_argument("--memory_remix_keep_original", action="store_true")
    g_memory.add_argument("--memory_remix_max_new_tokens", type=int, default=240)
    g_memory.add_argument("--memory_remix_temperature", type=float, default=0.9)

    g_answer.add_argument(
        "--answer_style",
        choices=["plain", "surreal", "metaphor", "meta"],
        default="plain",
        help="Prompt-only answer style guidance",
    )
    g_answer.add_argument("--answer_max_new_tokens", type=int, default=256)
    g_answer.add_argument("--answer_per_band", action="store_true", help="Generate per-band answers (sensitivity)")
    g_answer.add_argument("--answer_ensemble", action="store_true", help="Merge per-band answers into a final answer")
    g_answer.add_argument("--answer_ensemble_max_new_tokens", type=int, default=512)
    g_answer.add_argument("--answer_ensemble_temperature", type=float, default=0.2)

    g_interactive.add_argument("--interactive", action="store_true", help="Pick keyword tokens interactively")
    g_interactive.add_argument("--interactive_candidates", type=int, default=48)

    g_controls.add_argument(
        "--pack",
        choices=["none", "controls", "surreal"],
        default="none",
        help="Run a pack of variants",
    )
    g_controls.add_argument("--pack_file", default="", help="Run a custom pack from a JSON file")
    g_controls.add_argument("--pack_out", default="", help="Optional JSON output for pack results")
    g_controls.add_argument(
        "--pack_resume",
        action="store_true",
        help="If the pack output JSON already exists, reuse completed items and skip re-running them",
    )
    g_controls.add_argument("--pack_write_memory", action="store_true", help="Allow pack runs to write to memory JSONL")
    g_observe.add_argument("--print_config", action="store_true", help="Print resolved RunConfig JSON")
    g_observe.add_argument("--print_config_only", action="store_true", help="Print resolved RunConfig JSON and exit")
    g_observe.add_argument(
        "--print_compare",
        choices=["auto", "none", "json"],
        default="auto",
        help="Print baseline vs ponder comparison summary",
    )
    g_observe.add_argument(
        "--print_ponder",
        choices=["auto", "none", "full"],
        default="auto",
        help="Print human-readable ponder logs to stdout",
    )
    g_observe.add_argument(
        "--print_records",
        choices=["auto", "none", "all"],
        default="none",
        help="Debug: print raw ponder record JSON",
    )
    g_observe.add_argument("--json_out", default="", help="Write full run (or pack) results to JSON")
    g_observe.add_argument("--trace_out", default="", help="Write JSONL trace events")
    g_observe.add_argument("--trace_preview_chars", type=int, default=180, help="Trace preview length (0 disables)")
    g_observe.add_argument("--trace_report_out", default="", help="Write an HTML trace report (requires --trace_out file)")
    g_observe.add_argument(
        "--trace_report_max_records",
        type=int,
        default=0,
        help="Max JSONL records to read for trace report (0=all)",
    )
    g_observe.add_argument("--trace_report_session_id", default="", help="Filter trace report to a session_id (default: current run)")
    g_observe.add_argument("--out_dir", default="", help="If set, auto-fill --json_out/--trace_out/--trace_report_out into this dir")
    g_observe.add_argument("--run_name", default="", help="Optional label used in artifact filenames")

    g_runtime.add_argument("--device", default="auto", help="auto|mps|cpu|cuda|cuda:0 ...")
    g_runtime.add_argument("--dtype", default="auto", help="auto|float16|bfloat16|float32")
    g_runtime.add_argument(
        "--allocator_warmup",
        choices=["auto", "on", "off"],
        default="auto",
        help="Transformers caching allocator warmup (auto disables on MPS)",
    )
    g_runtime.add_argument("--trust_remote_code", action="store_true")
    g_runtime.add_argument("--hf_online", action="store_true", help="Allow downloading HF models (local_files_only=False)")
    g_runtime.add_argument("--no_chat_template", action="store_true")
    g_runtime.add_argument("--no_gemma_format", action="store_true", help="Disable Gemma-native turn formatting")
    g_runtime.add_argument("--probe_top_n", type=int, default=0, help="Store probe top-N tokens in record (0=off)")
    g_runtime.add_argument(
        "--probe_compare",
        action="store_true",
        help="Compare pre/post-ponder next-token distributions and save rank/JS deltas",
    )
    g_runtime.add_argument(
        "--probe_compare_stages",
        action="store_true",
        help="Capture a base->stage->final probe timeline (extra forward/API calls)",
    )
    g_runtime.add_argument(
        "--probe_compare_top_n",
        type=int,
        default=32,
        help="Top-N tokens used by --probe_compare / --probe_compare_stages",
    )
    g_runtime.add_argument("--print_probe", action="store_true", help="Print probe tables to stdout")

    g_api.add_argument("--api_base_url", default="https://api.openai.com/v1", help="API base URL (usually auto-filled by --provider)")
    g_api.add_argument("--api_chat_path", default="/chat/completions", help="Chat Completions path")
    g_api.add_argument("--api_key_env", default="OPENAI_API_KEY", help="Env var for API key (usually auto-filled by --provider)")
    g_api.add_argument("--api_key", default="", help="API key (discouraged; prefer env)")
    g_api.add_argument("--api_header", action="append", default=[], help='Extra header "Key: Value" (repeatable)')
    g_api.add_argument("--api_timeout", type=float, default=60.0, help="HTTP timeout (seconds)")
    g_api.add_argument("--api_max_retries", type=int, default=2, help="Retry count for 429/5xx")
    g_api.add_argument(
        "--api_reasoning_effort",
        choices=["auto", "none", "minimal", "low", "medium", "high", "xhigh"],
        default="auto",
        help="OpenAI GPT-5 reasoning effort hint (auto applies compatibility defaults where supported)",
    )
    g_api.add_argument(
        "--api_seed_method",
        choices=["auto", "self", "logprobs"],
        default="auto",
        help="API keyword seeding: self (prompted) or logprobs (if supported)",
    )
    g_api.add_argument(
        "--api_logprobs_top_n",
        type=int,
        default=0,
        help="Top-N logprobs to request (0=off; provider-capped, shallow probes warn and fall back to self-seeded keywords)",
    )

    g_gen.add_argument("--ponder_max_new_tokens", type=int, default=160)
    g_gen.add_argument("--temperature", type=float, default=0.7)
    g_gen.add_argument("--top_p", type=float, default=0.95)
    g_gen.add_argument("--top_k", type=int, default=0)
    g_gen.add_argument("--repetition_penalty", type=float, default=1.05)
    g_gen.add_argument("--no_repeat_ngram_size", type=int, default=0)
    g_gen.add_argument("--seed", type=int, default=1234)

    explicit_dests = explicit_dests_from_argv(ap, sys.argv)

    config_append_defaults: Dict[str, Any] = {}
    if config_defaults:
        known = {a.dest for a in ap._actions}
        unknown = sorted([k for k in config_defaults.keys() if k not in known])
        if unknown:
            raise SystemExit(f"[sr_ponder] ERROR: unknown keys in --config: {', '.join(unknown)}")

        # For append-style flags (e.g., --band, --api_header), argparse merges defaults + CLI values.
        # We want explicit CLI flags to override config, so apply append defaults after parsing.
        append_dests: set[str] = set()
        _AppendAction = getattr(argparse, "_AppendAction", None)
        if _AppendAction is not None:
            try:
                append_dests = {a.dest for a in ap._actions if isinstance(a, _AppendAction)}
            except Exception:
                append_dests = set()

        config_non_append = {k: v for k, v in config_defaults.items() if k not in append_dests}
        config_append_defaults = {k: v for k, v in config_defaults.items() if k in append_dests}
        ap.set_defaults(**config_non_append)

    args = ap.parse_args()

    for k, v in (config_append_defaults or {}).items():
        if k in explicit_dests:
            continue
        try:
            setattr(args, k, v)
        except Exception:
            pass

    pack_file_s = _expand_path_str(str(getattr(args, "pack_file", "") or ""))
    args.pack_file = pack_file_s
    if pack_file_s and str(getattr(args, "pack", "none") or "none") != "none":
        raise SystemExit("[sr_ponder] ERROR: use either --pack or --pack_file (not both)")

    apply_preset_inplace(args, explicit_dests=explicit_dests)
    resolved_provider = apply_provider_defaults_inplace(args, explicit_dests=explicit_dests)

    # Convenience: write artifacts into a directory with stable filenames.
    out_dir_s = _expand_path_str(str(getattr(args, "out_dir", "") or ""))
    if out_dir_s:
        args.out_dir = out_dir_s
        od = Path(out_dir_s)
        od.mkdir(parents=True, exist_ok=True)
        stamp = now_slug()
        run_name_s = str(getattr(args, "run_name", "") or "").strip()
        suffix = ("_" + slugify_filename(run_name_s)) if run_name_s else ""
        out_dir_is_explicit = "out_dir" in explicit_dests
        json_out_is_explicit = "json_out" in explicit_dests
        trace_out_is_explicit = "trace_out" in explicit_dests
        trace_report_out_is_explicit = "trace_report_out" in explicit_dests
        if bool(getattr(args, "print_config_only", False)):
            kind = "config"
        elif str(getattr(args, "pack", "none") or "none") != "none":
            kind = f"pack_{str(getattr(args, 'pack', 'pack')).strip()}"
        elif pack_file_s:
            kind = f"pack_{slugify_filename(Path(pack_file_s).stem)}"
        else:
            kind = "run"
        base = f"{kind}_{stamp}{suffix}"
        # If --out_dir is explicitly provided, prefer it over config defaults
        # unless the artifact paths are also explicitly specified on the CLI.
        if (not json_out_is_explicit) and (out_dir_is_explicit or (not str(getattr(args, "json_out", "") or "").strip())):
            args.json_out = str(od / f"{base}.json")
        if (not trace_out_is_explicit) and (out_dir_is_explicit or (not str(getattr(args, "trace_out", "") or "").strip())):
            args.trace_out = str(od / f"{base}.trace.jsonl")
        if (not trace_report_out_is_explicit) and (
            out_dir_is_explicit or (not str(getattr(args, "trace_report_out", "") or "").strip())
        ):
            args.trace_report_out = str(od / f"{base}.trace.html")

    # Normalize common path-like args (expand env vars / ~, keep '-' intact).
    try:
        args.json_out = _expand_path_str(str(getattr(args, "json_out", "") or ""))
        args.trace_out = _expand_path_str(str(getattr(args, "trace_out", "") or ""))
        args.trace_report_out = _expand_path_str(str(getattr(args, "trace_report_out", "") or ""))
        args.pack_out = _expand_path_str(str(getattr(args, "pack_out", "") or ""))
        args.pack_file = _expand_path_str(str(getattr(args, "pack_file", "") or ""))
    except Exception:
        pass

    prompt_lang = resolve_prompt_lang(args.prompt_lang, args.query)
    bands = [_parse_band_spec(x) for x in (args.band or [])]
    pipeline = parse_ponder_pipeline(args.ponder_pipeline, fallback_mode=args.ponder_mode)

    write_memory = not bool(args.no_write_memory)
    if (args.pack != "none" or pack_file_s) and not bool(args.pack_write_memory):
        write_memory = False

    cfg = RunConfig(
        model_path=args.model,
        memory_path=Path(args.memory),
        backend=args.backend,
        provider=resolved_provider,
        api_base_url=args.api_base_url,
        api_chat_path=args.api_chat_path,
        api_key_env=args.api_key_env,
        api_key=args.api_key,
        api_headers=list(args.api_header or []),
        api_timeout=float(args.api_timeout),
        api_max_retries=int(args.api_max_retries),
        api_reasoning_effort=args.api_reasoning_effort,
        api_seed_method=args.api_seed_method,
        api_logprobs_top_n=int(args.api_logprobs_top_n),
        n_memory=args.n_memory,
        memory_policy=args.memory_policy,
        memory_retrieve=args.memory_retrieve,
        memory_pool=args.memory_pool,
        memory_mix_ratio=args.memory_mix_ratio,
        memory_exclude_current_run=not bool(args.memory_include_current_run),
        memory_remix=args.memory_remix,
        memory_remix_keep_original=bool(args.memory_remix_keep_original),
        memory_remix_max_new_tokens=args.memory_remix_max_new_tokens,
        memory_remix_temperature=args.memory_remix_temperature,
        rejected=RejectedTokenConfig(
            top_k=args.top_k_rejected,
            strategy=args.strategy,
            exclude_top=args.exclude_top,
            band_width=args.band_width,
            n_keywords=args.n_keywords,
        ),
        band_profile=args.band_profile,
        bands=bands,
        answer_style=args.answer_style,
        answer_max_new_tokens=args.answer_max_new_tokens,
        ponder_max_new_tokens=args.ponder_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        seed=args.seed,
        prompt_lang=prompt_lang,
        ponder_mode=args.ponder_mode,
        ponder_pipeline=pipeline,
        pipeline_context=args.pipeline_context,
        pipeline_context_max_chars=args.pipeline_context_max_chars,
        n_ponder=args.n_ponder,
        ponder_hops=args.ponder_hops,
        hop_keyword_source=args.hop_keyword_source,
        hop_context_max_chars=args.hop_context_max_chars,
        keyword_refine=args.keyword_refine,
        keyword_refine_max_new_tokens=args.keyword_refine_max_new_tokens,
        keyword_refine_temperature=args.keyword_refine_temperature,
        keyword_objective=args.keyword_objective,
        keyword_select_top=args.keyword_select_top,
        keyword_diversity=args.keyword_diversity,
        keyword_diversity_threshold=args.keyword_diversity_threshold,
        dissonance_target=args.dissonance_target,
        dissonance_width=args.dissonance_width,
        dissonance_tail_k=args.dissonance_tail_k,
        prompt_jitter=args.prompt_jitter,
        prompt_jitter_include_original=not bool(args.no_prompt_jitter_include_original),
        prompt_jitter_max_new_tokens=args.prompt_jitter_max_new_tokens,
        prompt_jitter_temperature=args.prompt_jitter_temperature,
        probe_top_n=args.probe_top_n,
        probe_compare=bool(args.probe_compare),
        probe_compare_stages=bool(args.probe_compare_stages),
        probe_compare_top_n=int(args.probe_compare_top_n),
        print_probe=args.print_probe,
        interactive=bool(args.interactive),
        interactive_candidates=args.interactive_candidates,
        answer_per_band=bool(args.answer_per_band),
        answer_ensemble=bool(args.answer_ensemble),
        answer_ensemble_max_new_tokens=args.answer_ensemble_max_new_tokens,
        answer_ensemble_temperature=args.answer_ensemble_temperature,
        control=args.control,
        write_memory=write_memory,
        device=args.device,
        dtype=args.dtype,
        allocator_warmup=args.allocator_warmup,
        trust_remote_code=args.trust_remote_code,
        use_chat_template=not args.no_chat_template,
        force_gemma_format=not args.no_gemma_format,
        hf_local_files_only=not bool(args.hf_online),
    )

    trace: Optional[Any] = None
    trace_path_s = str(getattr(args, "trace_out", "") or "").strip()
    if trace_path_s:
        session_id = sha256_short(f"{now_iso()}|{cfg.seed}|{cfg.backend}|{cfg.model_path}|{args.query}")
        if trace_path_s == "-":
            # Stream traces to stderr to avoid mixing with the main output stream.
            trace = StreamTraceWriter(
                sys.stderr,
                session_id=session_id,
                preview_chars=int(getattr(args, "trace_preview_chars", 0) or 0),
                label="-",
            )
        else:
            trace = TraceWriter(
                Path(trace_path_s),
                session_id=session_id,
                preview_chars=int(getattr(args, "trace_preview_chars", 0) or 0),
            )
        trace.event(
            "session_start",
            query=args.query,
            model=cfg.model_path,
            backend=cfg.backend,
            provider=cfg.provider,
            pack=args.pack,
            pack_file=pack_file_s,
            preset=args.preset,
            config=str(getattr(args, "config", "") or ""),
            pid=os.getpid(),
            versions={"python": sys.version, "torch": getattr(torch, "__version__", None), "transformers": getattr(transformers, "__version__", None)},
        )

    if bool(args.print_config) or bool(args.print_config_only):
        cfg_dump = {
            "ts": now_iso(),
            "kind": "config",
            "query": args.query,
            "preset": args.preset,
            "config": str(getattr(args, "config", "") or ""),
            "cfg": sanitize_cfg_dict(dataclasses.asdict(cfg)),
        }
        print(json.dumps(_jsonable(cfg_dump), ensure_ascii=False, indent=2))
        out_s = str(getattr(args, "json_out", "") or "").strip()
        if out_s and bool(args.print_config_only):
            out_path = write_json_dest(out_s, cfg_dump)
            if out_path is not None:
                print(f"\n[json_out] wrote {out_path}")
            else:
                print("\n[json_out] wrote -")
        if bool(args.print_config_only):
            if trace:
                trace.event("config_only_exit")
            return

    hf: Any
    if cfg.backend == "hf":
        if torch is None or transformers is None or AutoModelForCausalLM is None or AutoTokenizer is None:
            raise SystemExit(
                "[sr_ponder] ERROR: backend=hf requires torch + transformers. "
                "Install deps (example): pip install torch transformers"
            )
        try:
            hf = LocalHFModel(
                cfg.model_path,
                device=cfg.device,
                dtype=cfg.dtype,
                trust_remote_code=cfg.trust_remote_code,
                use_chat_template=cfg.use_chat_template,
                force_gemma_format=cfg.force_gemma_format,
                allocator_warmup=cfg.allocator_warmup,
                local_files_only=cfg.hf_local_files_only,
            )
        except ValueError as e:
            raise SystemExit(f"[sr_ponder] ERROR: {e}") from e
        print(
            f"[sr_ponder] backend=hf provider={cfg.provider} model={cfg.model_path!r} "
            f"device={hf.device_str} input_device={hf.input_device} dtype={hf.torch_dtype} "
            f"alloc_warmup={cfg.allocator_warmup} "
            f"gemma_turn_tokens={hf._has_gemma_turn_tokens()} lang={cfg.prompt_lang} "
            f"band_profile={cfg.band_profile} bands={len(cfg.bands) if cfg.bands else 'profile'} "
            f"objective={cfg.keyword_objective} diversity={cfg.keyword_diversity} hops={cfg.ponder_hops} answer_style={cfg.answer_style} "
            f"hf_local_only={cfg.hf_local_files_only} control={cfg.control} "
            f"pipeline={','.join(cfg.ponder_pipeline) if cfg.ponder_pipeline else cfg.ponder_mode} "
            f"memory={cfg.memory_policy}/{cfg.memory_retrieve}/{cfg.memory_remix} write_memory={cfg.write_memory}"
        )
    elif cfg.backend == "openai_compat":
        api_key = (cfg.api_key or os.environ.get(cfg.api_key_env, "") or "").strip()
        if not api_key:
            raise SystemExit(
                f"[sr_ponder] ERROR: missing API key. Set env {cfg.api_key_env} or pass --api_key."
            )
        hf = OpenAICompatModel(
            model=cfg.model_path,
            api_base_url=cfg.api_base_url,
            api_key=api_key,
            api_chat_path=cfg.api_chat_path,
            api_headers=cfg.api_headers,
            timeout=cfg.api_timeout,
            max_retries=cfg.api_max_retries,
            api_reasoning_effort=cfg.api_reasoning_effort,
        )
        print(
            f"[sr_ponder] backend=openai_compat provider={cfg.provider} model={cfg.model_path!r} base_url={cfg.api_base_url} "
            f"reasoning_effort={cfg.api_reasoning_effort} seed_method={cfg.api_seed_method} logprobs_top_n={cfg.api_logprobs_top_n} "
            f"lang={cfg.prompt_lang} band_profile={cfg.band_profile} bands={len(cfg.bands) if cfg.bands else 'profile'} "
            f"objective={cfg.keyword_objective} diversity={cfg.keyword_diversity} hops={cfg.ponder_hops} answer_style={cfg.answer_style} control={cfg.control} "
            f"pipeline={','.join(cfg.ponder_pipeline) if cfg.ponder_pipeline else cfg.ponder_mode} "
            f"memory={cfg.memory_policy}/{cfg.memory_retrieve}/{cfg.memory_remix} write_memory={cfg.write_memory}"
        )
        ignored_args = api_ignored_generation_args(cfg)
        if ignored_args:
            print(
                "[sr_ponder] NOTE: openai_compat does not currently forward "
                + ", ".join(ignored_args)
                + " (provider-compatibility fallback).",
                file=sys.stderr,
            )
        if cfg.api_seed_method in ("auto", "logprobs"):
            band_end_max = max((int(b.get("end_rank", 0)) for b in (cfg.bands or _default_bands_from_profile(cfg.band_profile))), default=0)
            if band_end_max > 0 and int(cfg.api_logprobs_top_n) > 0 and int(cfg.api_logprobs_top_n) < band_end_max:
                print(
                    f"[sr_ponder] WARN: --api_logprobs_top_n={int(cfg.api_logprobs_top_n)} is shallower than the "
                    f"largest band end ({band_end_max}); some API bands will fall back to self-seeded keywords.",
                    file=sys.stderr,
                )
    else:
        raise SystemExit(f"[sr_ponder] ERROR: unknown backend: {cfg.backend!r}")

    if args.pack != "none" or pack_file_s:
        pack_cfg = dataclasses.replace(cfg, interactive=False, answer_per_band=False, answer_ensemble=False)
        pack_id = str(getattr(args, "pack", "none") or "none")
        pack_file_path: Optional[Path] = None
        items: List[Tuple[str, Dict[str, Any]]] = []
        if pack_file_s:
            pack_file_path = Path(pack_file_s)
            pack_id, items = load_pack_file(pack_file_path)

        results: Dict[str, Any] = {
            "ts": now_iso(),
            "kind": "pack",
            "provider": cfg.provider,
            "model": cfg.model_path,
            "query": args.query,
            "pack": pack_id,
            "pack_file": str(pack_file_path) if pack_file_path else "",
            "preset": args.preset,
            "config": str(getattr(args, "config", "") or ""),
            "env": {
                "python": sys.version,
                "platform": sys.platform,
                "torch": getattr(torch, "__version__", None),
                "transformers": getattr(transformers, "__version__", None),
            },
            "base_cfg": sanitize_cfg_dict(dataclasses.asdict(pack_cfg)),
            "items": [],
        }

        if not items and args.pack == "controls":
            items = [
                ("baseline", {"kind": "baseline"}),
                ("ponder", {"kind": "ponder", "control": "none"}),
                ("no_inject", {"kind": "ponder", "control": "no_inject"}),
                ("random_keywords", {"kind": "ponder", "control": "random_keywords"}),
                ("random_log", {"kind": "ponder", "control": "random_log"}),
                ("lens_only", {"kind": "ponder", "control": "lens_only"}),
            ]
        elif not items and args.pack == "surreal":
            surreal_walk_cfg: Dict[str, Any] = {
                "answer_style": "surreal",
                "band_profile": "spectrum3",
                "bands": [],
                "ponder_pipeline": ["metaphor", "metaphor"],
                "pipeline_context": "prev",
                "n_ponder": 1,
                "ponder_hops": 3,
                "hop_keyword_source": "model",
                "keyword_objective": "dissonance",
                "keyword_diversity": "embed",
                "memory_policy": "current_only",
                "memory_remix": "dream",
                "prompt_jitter": 2,
                "keyword_refine": True,
            }
            surreal_unstable_cfg: Dict[str, Any] = {
                "answer_style": "surreal",
                "band_profile": "spectrum3",
                "bands": [],
                "ponder_pipeline": ["metaphor", "metaphor"],
                "pipeline_context": "prev",
                "n_ponder": 1,
                "ponder_hops": 2,
                "hop_keyword_source": "model",
                "keyword_objective": "unstable",
                "keyword_diversity": "lex",
                "memory_policy": "current_only",
                "memory_remix": "compress",
                "prompt_jitter": 4,
                "keyword_refine": False,
            }
            surreal_questions_cfg: Dict[str, Any] = {
                "answer_style": "meta",
                "band_profile": "spectrum3",
                "bands": [],
                "ponder_pipeline": ["questions_only", "metaphor"],
                "pipeline_context": "prev",
                "n_ponder": 1,
                "ponder_hops": 2,
                "hop_keyword_source": "model",
                "keyword_objective": "dissonance",
                "keyword_diversity": "embed",
                "memory_policy": "current_only",
                "memory_remix": "dream",
                "prompt_jitter": 1,
                "keyword_refine": False,
            }
            items = [
                ("baseline_plain", {"kind": "baseline", "cfg": {"answer_style": "plain"}}),
                ("baseline_surreal", {"kind": "baseline", "cfg": {"answer_style": "surreal"}}),
                ("walk_dissonance", {"kind": "ponder", "control": "none", "cfg": surreal_walk_cfg}),
                ("walk_unstable", {"kind": "ponder", "control": "none", "cfg": surreal_unstable_cfg}),
                ("questions_to_metaphor", {"kind": "ponder", "control": "none", "cfg": surreal_questions_cfg}),
                ("lens_only_metaphor", {"kind": "ponder", "control": "lens_only", "cfg": surreal_walk_cfg}),
            ]
        elif not items:
            raise SystemExit(f"[sr_ponder] ERROR: unknown pack: {args.pack!r}")

        # Output selection for pack runs:
        # - Prefer --json_out when explicitly provided (or when --out_dir is explicitly provided),
        #   even if pack_out is set via config defaults. This makes packs consistent with runs.
        # - Otherwise, use --pack_out if set, falling back to --json_out.
        out_dir_is_explicit = "out_dir" in explicit_dests
        json_out_is_explicit = "json_out" in explicit_dests
        pack_out_is_explicit = "pack_out" in explicit_dests

        pack_out_s = (args.pack_out or "").strip()
        json_out_s = str(getattr(args, "json_out", "") or "").strip()

        prefer_json = (json_out_is_explicit and not pack_out_is_explicit) or (out_dir_is_explicit and not pack_out_is_explicit)
        if prefer_json and json_out_s:
            out_label = "json_out"
            out_s = json_out_s
        elif pack_out_s:
            out_label = "pack_out"
            out_s = pack_out_s
        elif json_out_s:
            out_label = "json_out"
            out_s = json_out_s
        else:
            out_label = ""
            out_s = ""

        # Optional resume: if an output file already exists, reuse compatible items and skip them.
        prev_by_name: Dict[str, Dict[str, Any]] = {}
        resume_path: Optional[Path] = None
        if bool(getattr(args, "pack_resume", False)) and out_s and out_s != "-":
            resume_path = Path(out_s)
            if resume_path.exists():
                try:
                    prev = json.loads(resume_path.read_text(encoding="utf-8"))
                except Exception as e:
                    raise SystemExit(f"[sr_ponder] ERROR: failed to read --pack_resume file {str(resume_path)!r}: {e}")
                if not isinstance(prev, dict) or str(prev.get("kind") or "").strip() != "pack":
                    raise SystemExit(f"[sr_ponder] ERROR: --pack_resume expects a pack results JSON (kind=pack): {str(resume_path)!r}")
                if str(prev.get("pack") or "") != str(pack_id):
                    raise SystemExit(
                        f"[sr_ponder] ERROR: --pack_resume pack mismatch: file has pack={prev.get('pack')!r}, current pack={pack_id!r}"
                    )
                if str(prev.get("query") or "") != str(args.query or ""):
                    raise SystemExit("[sr_ponder] ERROR: --pack_resume query mismatch (refusing to mix results)")
                prev_items = prev.get("items")
                if prev_items and not isinstance(prev_items, list):
                    raise SystemExit("[sr_ponder] ERROR: --pack_resume file has invalid items (expected array)")
                for it in (prev_items or []):
                    if not isinstance(it, dict):
                        continue
                    nm = str(it.get("name") or "").strip()
                    if nm:
                        prev_by_name[nm] = it

        results["artifacts"] = {
            "out_label": out_label if out_s else "",
            "out": out_s,
            "trace_out": str(trace.path) if trace else "",
            "trace_report_out": str(getattr(args, "trace_report_out", "") or "").strip(),
            "out_dir": str(getattr(args, "out_dir", "") or ""),
            "resume_from": str(resume_path) if resume_path and resume_path.exists() else "",
        }

        if trace:
            trace.event(
                "pack_start",
                pack=pack_id,
                pack_file=str(pack_file_path) if pack_file_path else "",
                items=int(len(items)),
                out_path=out_s,
                out_label=out_label if out_s else "",
                trace_out=str(trace.path),
                trace_report_out=str(getattr(args, "trace_report_out", "") or "").strip(),
                resume_from=str(resume_path) if resume_path and resume_path.exists() else "",
            )

        for name, spec in items:
            print(f"\n=== PACK: {name} ===\n")
            cfg_overrides = dict(spec.get("cfg", {}) or {})
            cfg_overrides_safe = sanitize_cfg_dict(cfg_overrides)
            item_query = args.query
            qv = spec.get("query")
            if isinstance(qv, str):
                item_query = qv

            # Resume/skip if this item is already present and compatible.
            if prev_by_name:
                prev_it = prev_by_name.get(str(name))
                if isinstance(prev_it, dict):
                    want_kind = str(spec.get("kind") or "").strip()
                    want_kind = "baseline" if want_kind == "baseline" else "ponder"
                    prev_kind = str(prev_it.get("kind") or "").strip()
                    prev_control = str(prev_it.get("control") or "").strip()
                    want_control = str(spec.get("control", "none") or "none").strip()
                    prev_query = str(prev_it.get("query") or args.query)
                    prev_overrides = prev_it.get("cfg_overrides")
                    if not isinstance(prev_overrides, dict):
                        prev_overrides = {}
                    match = (
                        (prev_kind == want_kind)
                        and (want_kind == "baseline" or prev_control == want_control)
                        and (prev_query == str(item_query))
                        and (dict(prev_overrides) == dict(cfg_overrides_safe) if cfg_overrides else dict(prev_overrides) == {})
                    )
                    if match:
                        print(f"[sr_ponder] [resume] skipping {name!r} (already in {str(resume_path)!r})")
                        results["items"].append(prev_it)
                        if trace:
                            trace.event("pack_item_skip", pack=pack_id, item=name)
                        continue

            if trace:
                trace.event(
                    "pack_item_start",
                    pack=pack_id,
                    item=name,
                    kind=str(spec.get("kind")),
                    cfg_overrides=cfg_overrides_safe,
                )
            t0 = time.perf_counter()
            if spec["kind"] == "baseline":
                try:
                    cfg2 = dataclasses.replace(pack_cfg, **cfg_overrides) if cfg_overrides else pack_cfg
                except TypeError as e:
                    raise SystemExit(f"[sr_ponder] ERROR: pack item {name!r} has invalid cfg overrides: {e}")
                ans = run_baseline(hf, cfg2, item_query, trace=trace, pack_item=name)
                print(ans)
                item: Dict[str, Any] = {"name": name, "kind": "baseline", "answer": ans}
                if item_query != args.query:
                    item["query"] = item_query
                if cfg_overrides:
                    item["cfg_overrides"] = cfg_overrides_safe
                item["metrics"] = {"answer_chars": len(ans or ""), "elapsed_s": float(time.perf_counter() - t0)}
                results["items"].append(item)
                if trace:
                    trace.event("pack_item_end", pack=pack_id, item=name, elapsed_s=float(time.perf_counter() - t0))
                continue
            try:
                cfg2 = dataclasses.replace(pack_cfg, control=str(spec.get("control", "none")), **cfg_overrides)
            except TypeError as e:
                raise SystemExit(f"[sr_ponder] ERROR: pack item {name!r} has invalid cfg overrides: {e}")
            ans, recs, extras = run_ponder_dispatch(hf, cfg2, item_query, trace=trace, pack_item=name)
            print(ans)
            item = {"name": name, "kind": "ponder", "control": cfg2.control, "answer": ans, "extras": extras}
            if item_query != args.query:
                item["query"] = item_query
            if cfg_overrides:
                item["cfg_overrides"] = cfg_overrides_safe
            item["metrics"] = {"answer_chars": len(ans or ""), "records": int(len(recs)), "elapsed_s": float(time.perf_counter() - t0)}
            results["items"].append(item)
            if trace:
                trace.event("pack_item_end", pack=pack_id, item=name, elapsed_s=float(time.perf_counter() - t0))

        # Pack summary (human-readable).
        try:
            rows: List[Dict[str, Any]] = []
            for it in results.get("items") or []:
                if not isinstance(it, dict):
                    continue
                m = it.get("metrics") if isinstance(it.get("metrics"), dict) else {}
                kind = str(it.get("kind") or "")
                ctrl = str(it.get("control") or "") if kind == "ponder" else ""
                rows.append(
                    {
                        "name": str(it.get("name") or ""),
                        "kind": kind,
                        "control": ctrl,
                        "chars": m.get("answer_chars"),
                        "records": m.get("records"),
                        "elapsed_s": m.get("elapsed_s"),
                    }
                )
            if rows:
                print("\n=== PACK SUMMARY ===\n")
                cols = ["name", "kind", "control", "chars", "records", "elapsed_s"]
                widths = {c: len(c) for c in cols}
                for r in rows:
                    for c in cols:
                        widths[c] = max(widths[c], len(str(r.get(c, "") if r.get(c, "") is not None else "")))
                widths["name"] = min(widths["name"], 42)
                header = "  ".join(c.ljust(widths[c]) for c in cols)
                print(header)
                print("  ".join("-" * widths[c] for c in cols))
                for r in rows:
                    line_parts: List[str] = []
                    for c in cols:
                        v = r.get(c, "")
                        s = "" if v is None else str(v)
                        if c == "name" and len(s) > widths["name"]:
                            s = s[: max(0, widths["name"] - 1)] + "…"
                        line_parts.append(s.ljust(widths[c]))
                    print("  ".join(line_parts))
        except Exception:
            pass

        if out_s:
            out_path = write_json_dest(out_s, results)
            if out_path is not None:
                print(f"\n[{out_label}] wrote {out_path}")
            else:
                print(f"\n[{out_label}] wrote -")

        trace_report_s = str(getattr(args, "trace_report_out", "") or "").strip()
        if trace_report_s and trace:
            sid_filter = str(getattr(args, "trace_report_session_id", "") or "").strip() or session_id
            rep_path = maybe_write_trace_report(
                trace=trace,
                dest=trace_report_s,
                session_id=session_id,
                max_records=int(getattr(args, "trace_report_max_records", 0) or 0),
                session_filter=sid_filter,
            )
            if rep_path is not None:
                print(f"\n[trace_report_out] wrote {rep_path}")
                trace.event("trace_report_out", path=str(rep_path), session_id_filter=sid_filter)

        if trace:
            trace.event("pack_end", pack=pack_id, items=int(len(items)))
            trace.event("session_end")
        return

    baseline_ans: Optional[str] = None
    ponder_ans: Optional[str] = None
    ponder_recs: List[Dict[str, Any]] = []
    ponder_extras: Dict[str, Any] = {}
    comparison: Dict[str, Any] = {}

    if args.mode in ("baseline", "both"):
        baseline_ans = run_baseline(hf, cfg, args.query, trace=trace)
        print("\n=== BASELINE ===\n")
        print(baseline_ans)

    if args.mode in ("ponder", "both"):
        ponder_ans, ponder_recs, ponder_extras = run_ponder_dispatch(hf, cfg, args.query, trace=trace)

        _print_ponder_logs(ponder_recs, mode=str(getattr(args, "print_ponder", "auto") or "auto"))

        if args.print_records != "none":
            if args.print_records == "all" or (args.print_records == "auto" and len(ponder_recs) <= 3):
                payload: Any = ponder_recs if len(ponder_recs) > 1 else ponder_recs[0]
            elif args.print_records == "auto":
                payload = ponder_recs[0] if ponder_recs else []
            else:
                payload = []
            if payload:
                print("\n=== PONDER RECORD(S) (just written) ===\n")
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                if args.print_records == "auto" and len(ponder_recs) > 3:
                    print(f"\n[info] records={len(ponder_recs)} (use --print_records all to dump everything)")

        # Extras (band answers + ensemble)
        band_answers = ponder_extras.get("band_answers") if isinstance(ponder_extras, dict) else None
        if isinstance(band_answers, dict) and band_answers:
            print("\n=== BAND ANSWERS ===\n")
            for bl, txt in band_answers.items():
                print(f"\n--- {bl} ---\n")
                print(txt)

        ens = ponder_extras.get("ensemble") if isinstance(ponder_extras, dict) else None
        if isinstance(ens, dict) and ens.get("raw"):
            print("\n=== ENSEMBLE (raw) ===\n")
            print(ens.get("raw"))

        if ponder_extras.get("random_log"):
            print("\n=== CONTROL: RANDOM LOG ===\n")
            print(ponder_extras.get("random_log"))

        print("\n=== PONDERED ANSWER ===\n")
        print(ponder_ans)

    comparison = build_run_comparison(
        query=args.query,
        baseline_answer=baseline_ans,
        ponder_answer=ponder_ans,
        records=ponder_recs,
        extras=ponder_extras if ponder_extras else None,
    )
    if baseline_ans is not None and ponder_ans is not None:
        _print_run_comparison(comparison, mode=str(getattr(args, "print_compare", "auto") or "auto"))

    out_s = str(getattr(args, "json_out", "") or "").strip()
    if out_s:
        payload = {
            "ts": now_iso(),
            "kind": "run",
            "provider": cfg.provider,
            "query": args.query,
            "preset": args.preset,
            "config": str(getattr(args, "config", "") or ""),
            "artifacts": {
                "json_out": out_s,
                "trace_out": str(trace.path) if trace else "",
                "trace_report_out": str(getattr(args, "trace_report_out", "") or "").strip(),
                "out_dir": str(getattr(args, "out_dir", "") or ""),
            },
            "env": {
                "python": sys.version,
                "platform": sys.platform,
                "torch": getattr(torch, "__version__", None),
                "transformers": getattr(transformers, "__version__", None),
            },
            "cfg": sanitize_cfg_dict(dataclasses.asdict(cfg)),
            "baseline": baseline_ans,
            "ponder": ponder_ans,
            "records": ponder_recs,
            "extras": ponder_extras if ponder_extras else None,
            "comparison": comparison if comparison else None,
            "metrics": compute_run_metrics(
                query=args.query,
                baseline_answer=baseline_ans,
                ponder_answer=ponder_ans,
                records=ponder_recs,
                extras=ponder_extras if ponder_extras else None,
            ),
        }
        out_path = write_json_dest(out_s, payload)
        if out_path is not None:
            print(f"\n[json_out] wrote {out_path}")
        else:
            print("\n[json_out] wrote -")
        if trace:
            trace.event("json_out", path=str(out_path) if out_path is not None else "-")

    trace_report_s = str(getattr(args, "trace_report_out", "") or "").strip()
    if trace_report_s and trace:
        sid_filter = str(getattr(args, "trace_report_session_id", "") or "").strip() or session_id
        rep_path = maybe_write_trace_report(
            trace=trace,
            dest=trace_report_s,
            session_id=session_id,
            max_records=int(getattr(args, "trace_report_max_records", 0) or 0),
            session_filter=sid_filter,
        )
        if rep_path is not None:
            print(f"\n[trace_report_out] wrote {rep_path}")
            trace.event("trace_report_out", path=str(rep_path), session_id_filter=sid_filter)

    if trace:
        trace.event("session_end")


if __name__ == "__main__":
    main()
