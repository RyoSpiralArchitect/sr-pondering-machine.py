#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from sr_memory_capsule import hash_charngram_embed_4096, pack_memory_slots
from sr_pondering_machine import (
    LocalHFModel,
    RejectedTokenConfig,
    RunConfig,
    append_jsonl,
    iter_jsonl,
    now_iso,
    parse_ponder_pipeline,
    resolve_prompt_lang,
    run_ponder_dispatch,
    safe_mkdir,
    select_memory_records,
    sha256_short,
)


_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u3040-\u30ff\u3400-\u9fff]{2,}")


def parse_csv_items(text: str) -> List[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def load_queries(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"[sr_ponder_bench] ERROR: queries file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        out: List[Dict[str, str]] = []
        for ix, row in enumerate(iter_jsonl(path), start=1):
            q = str(row.get("query") or row.get("text") or "").strip()
            if not q:
                continue
            out.append({"id": str(row.get("id") or f"q{ix}"), "query": q})
        return out

    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        out = []
        if isinstance(raw, list):
            for ix, item in enumerate(raw, start=1):
                if isinstance(item, str):
                    q = item.strip()
                    if q:
                        out.append({"id": f"q{ix}", "query": q})
                elif isinstance(item, dict):
                    q = str(item.get("query") or item.get("text") or "").strip()
                    if q:
                        out.append({"id": str(item.get("id") or f"q{ix}"), "query": q})
        return out

    out = []
    for ix, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        q = line.strip()
        if not q or q.startswith("#"):
            continue
        out.append({"id": f"q{ix}", "query": q})
    return out


def tokenize_ids(tokenizer: Any, text: str, *, max_tokens: int = 96) -> List[int]:
    s = str(text or "").strip()
    if not s:
        return []
    try:
        enc = tokenizer(s, add_special_tokens=False, truncation=True, max_length=max(1, int(max_tokens)))
        ids = enc.get("input_ids") or []
    except Exception:
        return []
    out: List[int] = []
    for item in ids:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def derive_keywords(text: str, *, limit: int = 8) -> List[str]:
    out: List[str] = []
    seen = set()
    for tok in _WORD_RE.findall(str(text or "").lower()):
        if len(tok) <= 1:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= limit:
            break
    return out


def preferred_capsule_slots(memory: Dict[str, Any], explicit: Sequence[str]) -> List[str]:
    if explicit:
        return [str(s) for s in explicit if str(s)]
    preferred = ["persona", "rules", "task", "task_raw", "query", "angles", "ponder", "answer"]
    out = [s for s in preferred if str(memory.get(s, "") or "").strip()]
    if out:
        return out
    return [str(k) for k, v in memory.items() if str(v or "").strip()]


def canonical_log_records(rows: Sequence[Dict[str, Any]], tokenizer: Any, *, max_token_ids: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ix, row in enumerate(rows):
        ponder_q = str(row.get("ponder_question") or "").strip()
        ponder_log = str(row.get("ponder_log") or "").strip()
        if not ponder_q and not ponder_log:
            continue
        token_ids = row.get("token_ids") if isinstance(row.get("token_ids"), list) else None
        if not token_ids:
            token_ids = tokenize_ids(tokenizer, f"{ponder_q}\n{ponder_log}", max_tokens=max_token_ids)
        out.append(
            {
                "ts": str(row.get("ts") or row.get("created_at") or now_iso()),
                "run_id": "log:" + str(row.get("run_id") or sha256_short(f"log|{ix}|{ponder_q}|{ponder_log}")),
                "band_label": str(row.get("band_label") or ""),
                "ponder_ix": int(row.get("ponder_ix") or ix),
                "ponder_mode": str(row.get("ponder_mode") or "assoc"),
                "keywords": row.get("keywords") if isinstance(row.get("keywords"), list) else derive_keywords(ponder_q or ponder_log),
                "ponder_question": ponder_q,
                "ponder_log": ponder_log,
                "token_ids": [int(x) for x in token_ids if x is not None],
            }
        )
    out.sort(key=lambda r: (str(r.get("ts") or ""), str(r.get("run_id") or ""), int(r.get("ponder_ix") or 0)))
    return out


def canonical_capsule_records(
    rows: Sequence[Dict[str, Any]],
    tokenizer: Any,
    *,
    explicit_slots: Sequence[str],
    max_token_ids: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ix, row in enumerate(rows):
        memory = row.get("memory") or {}
        if not isinstance(memory, dict):
            continue
        slots = preferred_capsule_slots(memory, explicit_slots)
        packed = pack_memory_slots({str(k): str(v or "") for k, v in memory.items()}, slots)
        if not packed:
            continue
        task = str(memory.get("task_raw") or memory.get("task") or memory.get("query") or "").strip()
        question = task or packed
        mode = "capsule"
        meta = row.get("meta") or {}
        if isinstance(meta, dict) and meta.get("kind"):
            mode = str(meta.get("kind"))
        token_ids = tokenize_ids(tokenizer, f"{question}\n{packed}", max_tokens=max_token_ids)
        out.append(
            {
                "ts": str(row.get("created_at") or row.get("ts") or now_iso()),
                "run_id": "capsule:" + str((meta.get("run_id") if isinstance(meta, dict) else None) or sha256_short(f"capsule|{ix}|{question}|{packed}")),
                "band_label": str(mode),
                "ponder_ix": int(ix),
                "ponder_mode": "capsule",
                "keywords": derive_keywords(f"{question}\n{packed}"),
                "ponder_question": question,
                "ponder_log": packed,
                "token_ids": token_ids,
            }
        )
    out.sort(key=lambda r: (str(r.get("ts") or ""), str(r.get("run_id") or ""), int(r.get("ponder_ix") or 0)))
    return out


def alternating_hybrid(log_rows: Sequence[Dict[str, Any]], capsule_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not log_rows:
        return list(capsule_rows)
    if not capsule_rows:
        return list(log_rows)
    keep = min(len(log_rows), len(capsule_rows))
    left = list(log_rows[-keep:])
    right = list(capsule_rows[-keep:])
    out: List[Dict[str, Any]] = []
    for a, b in zip(left, right):
        out.append(a)
        out.append(b)
    return out


def write_canonical_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    safe_mkdir(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def memory_text(rec: Dict[str, Any]) -> str:
    kws = rec.get("keywords") or []
    kw_s = " ".join(str(x) for x in kws) if isinstance(kws, list) else str(kws or "")
    return "\n".join(
        [
            kw_s.strip(),
            str(rec.get("ponder_question") or "").strip(),
            str(rec.get("ponder_log") or "").strip(),
        ]
    ).strip()


def mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs]
    if not vals:
        return 0.0
    return float(sum(vals) / float(len(vals)))


def relevance_and_diversity(query: str, rows: Sequence[Dict[str, Any]], cache: Dict[str, torch.Tensor]) -> Tuple[float, float]:
    device = torch.device("cpu")

    def embed(text: str) -> torch.Tensor:
        key = text or ""
        if key not in cache:
            cache[key] = hash_charngram_embed_4096(key, device=device)
        return cache[key]

    qv = embed(query)
    texts = [memory_text(r) for r in rows if memory_text(r)]
    if not texts:
        return 0.0, 0.0
    vecs = [embed(t) for t in texts]
    rel = [float(torch.dot(qv, v).item()) for v in vecs]
    if len(vecs) < 2:
        return mean(rel), 0.0
    pair_sims: List[float] = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            pair_sims.append(float(torch.dot(vecs[i], vecs[j]).item()))
    diversity = 1.0 - mean(pair_sims)
    return mean(rel), float(diversity)


def summarize_results(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("memory_source") or ""), str(row.get("memory_backend") or ""))
        groups.setdefault(key, []).append(row)

    items: List[Dict[str, Any]] = []
    for (source, backend), xs in sorted(groups.items()):
        metrics = [dict(r.get("metrics") or {}) for r in xs]
        items.append(
            {
                "memory_source": source,
                "memory_backend": backend,
                "runs": len(xs),
                "avg_retrieved_count": mean([m.get("retrieved_count", 0.0) for m in metrics]),
                "avg_relevance": mean([m.get("avg_relevance", 0.0) for m in metrics]),
                "avg_diversity": mean([m.get("diversity", 0.0) for m in metrics]),
                "avg_answer_chars": mean([m.get("answer_chars", 0.0) for m in metrics]),
            }
        )
    return {"created_at": now_iso(), "items": items}


def summary_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# SR Ponder Memory Bench",
        "",
        "| source | backend | runs | avg_retrieved | avg_relevance | avg_diversity | avg_answer_chars |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary.get("items") or []:
        lines.append(
            "| {memory_source} | {memory_backend} | {runs} | {avg_retrieved_count:.2f} | {avg_relevance:.4f} | {avg_diversity:.4f} | {avg_answer_chars:.1f} |".format(
                **item
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark sr_pondering_machine memory backends on canonicalized log/capsule corpora.")
    ap.add_argument("--model", required=True, help="Local HF model directory")
    ap.add_argument("--queries", required=True, help="TXT/JSON/JSONL queries file")
    ap.add_argument("--log_memory", default="", help="Raw ponder JSONL memory")
    ap.add_argument("--capsule_store", default="", help="sr_memory_capsule gateway store JSONL")
    ap.add_argument("--capsule_slots", default="", help="Comma-separated slots to render from capsule rows")
    ap.add_argument("--sources", default="log,capsule,hybrid")
    ap.add_argument("--memory_backends", default="embed,fuzzy")
    ap.add_argument("--memory_retrieve", choices=["tail", "similar", "anti", "mix"], default="similar")
    ap.add_argument("--n_memory", type=int, default=6)
    ap.add_argument("--memory_pool", type=int, default=200)
    ap.add_argument("--memory_mix_ratio", type=float, default=0.5)
    ap.add_argument("--seeds", default="1234")
    ap.add_argument("--prompt_lang", choices=["auto", "en", "ja"], default="auto")
    ap.add_argument("--ponder_mode", choices=["assoc", "assumption", "counterexample", "questions_only", "metaphor"], default="assoc")
    ap.add_argument("--ponder_pipeline", default="")
    ap.add_argument("--pipeline_context", choices=["none", "prev", "all"], default="prev")
    ap.add_argument("--pipeline_context_max_chars", type=int, default=1200)
    ap.add_argument("--answer_max_new_tokens", type=int, default=256)
    ap.add_argument("--ponder_max_new_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--allocator_warmup", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--no_chat_template", action="store_true")
    ap.add_argument("--no_gemma_format", action="store_true")
    ap.add_argument("--top_k_rejected", type=int, default=80)
    ap.add_argument("--exclude_top", type=int, default=8)
    ap.add_argument("--band_width", type=int, default=256)
    ap.add_argument("--n_keywords", type=int, default=6)
    ap.add_argument("--keyword_objective", choices=["random_band", "dissonance", "unstable", "random_vocab"], default="random_band")
    ap.add_argument("--keyword_select_top", type=int, default=128)
    ap.add_argument("--max_token_ids", type=int, default=96)
    ap.add_argument("--out_jsonl", default="sr_ponder_memory_bench.jsonl")
    ap.add_argument("--out_summary_json", default="sr_ponder_memory_bench.summary.json")
    ap.add_argument("--out_summary_md", default="sr_ponder_memory_bench.summary.md")
    args = ap.parse_args()

    query_items = load_queries(Path(args.queries).expanduser())
    if not query_items:
        raise SystemExit("[sr_ponder_bench] ERROR: no queries loaded")

    sources = parse_csv_items(args.sources)
    if not sources:
        raise SystemExit("[sr_ponder_bench] ERROR: --sources is empty")
    backends = parse_csv_items(args.memory_backends)
    if not backends:
        raise SystemExit("[sr_ponder_bench] ERROR: --memory_backends is empty")
    seeds = [int(x) for x in parse_csv_items(args.seeds)]
    if not seeds:
        raise SystemExit("[sr_ponder_bench] ERROR: --seeds is empty")

    hf = LocalHFModel(
        args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
        use_chat_template=not bool(args.no_chat_template),
        force_gemma_format=not bool(args.no_gemma_format),
        allocator_warmup=args.allocator_warmup,
    )
    tokenizer = hf.tokenizer

    log_rows: List[Dict[str, Any]] = []
    capsule_rows: List[Dict[str, Any]] = []

    if "log" in sources or "hybrid" in sources:
        log_path = Path(args.log_memory).expanduser()
        if not str(args.log_memory or "").strip():
            raise SystemExit("[sr_ponder_bench] ERROR: --log_memory is required for source=log|hybrid")
        log_rows = canonical_log_records(iter_jsonl(log_path), tokenizer, max_token_ids=int(args.max_token_ids))
        if not log_rows:
            raise SystemExit(f"[sr_ponder_bench] ERROR: no canonical log rows from {log_path}")

    if "capsule" in sources or "hybrid" in sources:
        capsule_path = Path(args.capsule_store).expanduser()
        if not str(args.capsule_store or "").strip():
            raise SystemExit("[sr_ponder_bench] ERROR: --capsule_store is required for source=capsule|hybrid")
        capsule_rows = canonical_capsule_records(
            iter_jsonl(capsule_path),
            tokenizer,
            explicit_slots=parse_csv_items(args.capsule_slots),
            max_token_ids=int(args.max_token_ids),
        )
        if not capsule_rows:
            raise SystemExit(f"[sr_ponder_bench] ERROR: no canonical capsule rows from {capsule_path}")

    with tempfile.TemporaryDirectory(prefix="sr_ponder_memory_bench_") as tmpdir:
        tmp = Path(tmpdir)
        source_paths: Dict[str, Path] = {}
        if "log" in sources:
            p = tmp / "log.canonical.jsonl"
            write_canonical_jsonl(p, log_rows)
            source_paths["log"] = p
        if "capsule" in sources:
            p = tmp / "capsule.canonical.jsonl"
            write_canonical_jsonl(p, capsule_rows)
            source_paths["capsule"] = p
        if "hybrid" in sources:
            p = tmp / "hybrid.canonical.jsonl"
            write_canonical_jsonl(p, alternating_hybrid(log_rows, capsule_rows))
            source_paths["hybrid"] = p

        out_jsonl = Path(args.out_jsonl).expanduser()
        safe_mkdir(out_jsonl)
        out_jsonl.write_text("", encoding="utf-8")

        pipeline = parse_ponder_pipeline(args.ponder_pipeline, fallback_mode=args.ponder_mode)
        base_cfg = RunConfig(
            model_path=args.model,
            memory_path=out_jsonl,
            backend="hf",
            n_memory=int(args.n_memory),
            memory_format="ponder_jsonl",
            memory_policy="tail",
            memory_retrieve=args.memory_retrieve,
            memory_backend="embed",
            memory_pool=int(args.memory_pool),
            memory_mix_ratio=float(args.memory_mix_ratio),
            memory_exclude_current_run=True,
            rejected=RejectedTokenConfig(
                top_k=int(args.top_k_rejected),
                strategy="outside_topk",
                exclude_top=int(args.exclude_top),
                band_width=int(args.band_width),
                n_keywords=int(args.n_keywords),
            ),
            band_profile="single",
            answer_max_new_tokens=int(args.answer_max_new_tokens),
            ponder_max_new_tokens=int(args.ponder_max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            repetition_penalty=float(args.repetition_penalty),
            no_repeat_ngram_size=int(args.no_repeat_ngram_size),
            prompt_lang="en",
            ponder_mode=args.ponder_mode,
            ponder_pipeline=pipeline,
            pipeline_context=args.pipeline_context,
            pipeline_context_max_chars=int(args.pipeline_context_max_chars),
            n_ponder=1,
            keyword_objective=args.keyword_objective,
            keyword_select_top=int(args.keyword_select_top),
            control="none",
            write_memory=False,
            device=args.device,
            dtype=args.dtype,
            allocator_warmup=args.allocator_warmup,
            trust_remote_code=bool(args.trust_remote_code),
            use_chat_template=not bool(args.no_chat_template),
            force_gemma_format=not bool(args.no_gemma_format),
        )

        rows_out: List[Dict[str, Any]] = []
        embed_cache: Dict[str, torch.Tensor] = {}

        for q in query_items:
            query_id = str(q.get("id") or "")
            query = str(q.get("query") or "").strip()
            if not query:
                continue
            lang = resolve_prompt_lang(args.prompt_lang, query)
            for source in sources:
                memory_path = source_paths.get(source)
                if memory_path is None:
                    continue
                for backend in backends:
                    for seed in seeds:
                        cfg = dataclasses.replace(
                            base_cfg,
                            memory_path=memory_path,
                            memory_backend=backend,
                            seed=int(seed),
                            prompt_lang=lang,
                        )
                        answer, records, extras = run_ponder_dispatch(hf, cfg, query)
                        selected_records, backend_used = select_memory_records(
                            hf,
                            memory_path=memory_path,
                            current_records=records,
                            query=query,
                            memory_policy=cfg.memory_policy,
                            memory_retrieve=cfg.memory_retrieve,
                            memory_backend=cfg.memory_backend,
                            n_memory=cfg.n_memory,
                            pool_size=cfg.memory_pool,
                            mix_ratio=cfg.memory_mix_ratio,
                            exclude_run_id=None,
                        )
                        avg_rel, div = relevance_and_diversity(query, selected_records, embed_cache)
                        row = {
                            "ts": now_iso(),
                            "query_id": query_id,
                            "query": query,
                            "seed": int(seed),
                            "memory_source": source,
                            "memory_backend": backend,
                            "memory_backend_used": backend_used,
                            "memory_path": str(memory_path),
                            "answer": answer,
                            "records": records,
                            "extras": extras,
                            "metrics": {
                                "retrieved_count": len(selected_records),
                                "avg_relevance": avg_rel,
                                "diversity": div,
                                "answer_chars": len(answer),
                            },
                        }
                        append_jsonl(out_jsonl, row)
                        rows_out.append(row)
                        print(
                            f"[sr_ponder_bench] query={query_id} source={source} backend={backend} seed={seed} "
                            f"retrieved={len(selected_records)} rel={avg_rel:.4f} div={div:.4f}"
                        )

        summary = summarize_results(rows_out)
        out_summary_json = Path(args.out_summary_json).expanduser()
        out_summary_md = Path(args.out_summary_md).expanduser()
        safe_mkdir(out_summary_json)
        safe_mkdir(out_summary_md)
        out_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        out_summary_md.write_text(summary_markdown(summary), encoding="utf-8")
        print(f"[sr_ponder_bench] wrote {out_jsonl}")
        print(f"[sr_ponder_bench] wrote {out_summary_json}")
        print(f"[sr_ponder_bench] wrote {out_summary_md}")


if __name__ == "__main__":
    main()
