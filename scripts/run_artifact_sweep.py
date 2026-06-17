#!/usr/bin/env python3
"""Run a small Claude/Gemini artifact sweep for sr_pondering_machine.py."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


QUERIES = [
    {
        "id": "non_relative_not_absolute",
        "text": "非相対的ではあるが絶対的ではない状態とは？",
    },
    {
        "id": "reality_interface",
        "text": "現実って何のインターフェース？",
    },
    {
        "id": "wander_before_deciding",
        "text": "What changes when an answer is allowed to wander before deciding?",
    },
]


def now_slug() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def slugify(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return out.strip("_") or "item"


PROFILES: Dict[str, Dict[str, Any]] = {
    "standard": {
        "answer_max_new_tokens": "256",
        "ponder_max_new_tokens": "192",
        "scaffold_token_target": "180",
        "api_timeout": "180",
        "claude_budget_usd": "0.08",
        "providers": {
            "gemini": {"model": "gemini-3.5-flash", "reasoning_effort": "auto"},
            "claude": {"model": "haiku", "reasoning_effort": "auto"},
        },
    },
    "strong_deep": {
        "answer_max_new_tokens": "2048",
        "ponder_max_new_tokens": "1024",
        "scaffold_token_target": "320",
        "api_timeout": "900",
        "claude_budget_usd": "0.45",
        "providers": {
            "gemini": {"model": "gemini-3.1-pro-preview", "reasoning_effort": "high"},
            "claude": {"model": "opus", "reasoning_effort": "xhigh"},
        },
    },
}


def provider_args(provider: str, profile: Dict[str, Any]) -> List[str]:
    provider_cfg = profile.get("providers", {}).get(provider, {})
    if provider == "gemini":
        return [
            "--backend",
            "openai_compat",
            "--provider",
            "custom",
            "--api_base_url",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "--api_key_env",
            "GEMINI_API_KEY",
            "--model",
            str(provider_cfg.get("model") or "gemini-3.5-flash"),
            "--api_reasoning_effort",
            str(provider_cfg.get("reasoning_effort") or "auto"),
        ]
    if provider == "claude":
        return [
            "--provider",
            "claude",
            "--model",
            str(provider_cfg.get("model") or "haiku"),
            "--api_reasoning_effort",
            str(provider_cfg.get("reasoning_effort") or "auto"),
        ]
    raise ValueError(f"unknown provider: {provider}")


def build_cmd(repo: Path, provider: str, query: Dict[str, str], out_dir: Path, profile: Dict[str, Any]) -> Dict[str, Any]:
    stem = f"{provider}_{slugify(query['id'])}"
    json_out = out_dir / f"{stem}.json"
    trace_out = out_dir / f"{stem}.trace.jsonl"
    trace_report_out = out_dir / f"{stem}.trace.html"
    matrix_report_out = out_dir / f"{stem}.matrix.html"
    cmd = [
        sys.executable,
        str(repo / "sr_pondering_machine.py"),
        *provider_args(provider, profile),
        "--query",
        query["text"],
        "--lab_matrix",
        "scaffold_abcd",
        "--memory_policy",
        "current_only",
        "--no_write_memory",
        "--answer_max_new_tokens",
        str(profile.get("answer_max_new_tokens") or "256"),
        "--ponder_max_new_tokens",
        str(profile.get("ponder_max_new_tokens") or "192"),
        "--scaffold_token_target",
        str(profile.get("scaffold_token_target") or "180"),
        "--compare_semantic",
        "hash",
        "--compare_judge",
        "off",
        "--compare_stance",
        "auto",
        "--compare_spatial_metaphor",
        "auto",
        "--compare_token_budget",
        "auto",
        "--print_compare",
        "none",
        "--print_ponder",
        "none",
        "--api_timeout",
        str(profile.get("api_timeout") or "180"),
        "--api_max_retries",
        "1",
        "--pack_resume",
        "--json_out",
        str(json_out),
        "--trace_out",
        str(trace_out),
        "--trace_report_out",
        str(trace_report_out),
        "--matrix_report_out",
        str(matrix_report_out),
    ]
    return {
        "provider": provider,
        "query_id": query["id"],
        "query": query["text"],
        "cmd": cmd,
        "json_out": str(json_out),
        "trace_out": str(trace_out),
        "trace_report_out": str(trace_report_out),
        "matrix_report_out": str(matrix_report_out),
        "stdout": str(out_dir / f"{stem}.stdout.txt"),
        "stderr": str(out_dir / f"{stem}.stderr.txt"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--providers", default="gemini,claude")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="standard")
    ap.add_argument("--limit_queries", type=int, default=0)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir or (repo / "artifacts" / f"sweep_{now_slug()}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    queries = QUERIES[: max(0, int(args.limit_queries)) or len(QUERIES)]
    profile = PROFILES[args.profile]

    env = os.environ.copy()
    if Path("/etc/ssl/cert.pem").exists():
        env.setdefault("SSL_CERT_FILE", "/etc/ssl/cert.pem")
    env.setdefault("SR_CLAUDE_MAX_BUDGET_USD", str(profile.get("claude_budget_usd") or "0.08"))

    manifest: Dict[str, Any] = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo": str(repo),
        "out_dir": str(out_dir),
        "providers": providers,
        "queries": queries,
        "profile": args.profile,
        "profile_config": profile,
        "runs": [],
    }

    for provider in providers:
        for query in queries:
            spec = build_cmd(repo, provider, query, out_dir, profile)
            spec["profile"] = args.profile
            print(f"[sweep] start {provider}/{query['id']}")
            t0 = time.perf_counter()
            proc = subprocess.run(spec["cmd"], cwd=str(repo), env=env, text=True, capture_output=True, check=False)
            elapsed = time.perf_counter() - t0
            Path(spec["stdout"]).write_text(proc.stdout or "", encoding="utf-8")
            Path(spec["stderr"]).write_text(proc.stderr or "", encoding="utf-8")
            spec.update(
                {
                    "returncode": proc.returncode,
                    "elapsed_s": elapsed,
                    "ok": proc.returncode == 0,
                }
            )
            manifest["runs"].append(spec)
            (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[sweep] done {provider}/{query['id']} rc={proc.returncode} elapsed={elapsed:.1f}s")

    return 0 if all(run.get("ok") for run in manifest["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
