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
from typing import Any, Dict, List, Sequence


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
    "dose_ladder": {
        "answer_max_new_tokens": "2048",
        "ponder_max_new_tokens": "1024",
        "scaffold_token_target": "0",
        "api_timeout": "900",
        "claude_budget_usd": "0.45",
        "matrix": "dose_ladder",
        "dose_values": [0, 32, 64, 128, 256, 512, 1024],
        "dose_conditions": ["assoc", "random", "facts", "isomorphic"],
        "providers": {
            "gemini": {"model": "gemini-3.1-pro-preview", "reasoning_effort": "high"},
            "claude": {"model": "opus", "reasoning_effort": "xhigh"},
        },
    },
    "gemini_closure_contract": {
        "answer_max_new_tokens": "2048",
        "ponder_max_new_tokens": "1024",
        "scaffold_token_target": "0",
        "api_timeout": "900",
        "claude_budget_usd": "0.45",
        "matrix": "closure_contract",
        "dose_values": [128, 512],
        "dose_conditions": ["facts", "isomorphic"],
        "output_contracts": [
            "none",
            "final_closure",
            "log_closure",
            "log_final_closure",
            "log_skeleton_closure",
            "log_skeleton_final_closure",
        ],
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


def parse_int_csv(value: str, *, default: Sequence[int]) -> List[int]:
    if not str(value or "").strip():
        return [int(x) for x in default]
    out: List[int] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if raw:
            out.append(max(0, int(raw)))
    return sorted(set(out))


def parse_str_csv(value: str, *, default: Sequence[str]) -> List[str]:
    if not str(value or "").strip():
        return [str(x) for x in default]
    return [x.strip() for x in str(value).split(",") if x.strip()]


def write_dose_ladder_matrix(
    out_dir: Path,
    query: Dict[str, str],
    *,
    dose_values: Sequence[int],
    dose_conditions: Sequence[str],
) -> Path:
    """Write a per-query scaffold dose ladder matrix consumed by --lab_matrix."""
    matrix_dir = out_dir / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = matrix_dir / f"scaffold_dose_ladder_{slugify(query['id'])}.json"
    items: List[Dict[str, Any]] = []
    for condition in dose_conditions:
        condition_s = str(condition).strip()
        if not condition_s:
            continue
        for dose in dose_values:
            dose_i = int(dose)
            if dose_i <= 0:
                continue
            items.append(
                {
                    "name": f"{condition_s}_dose_{dose_i}",
                    "kind": "ponder",
                    "control": "none",
                    "cfg": {
                        "memory_policy": "current_only",
                        "scaffold_condition": condition_s,
                        "scaffold_token_target": dose_i,
                    },
                }
            )

    spec = {
        "name": f"scaffold_dose_ladder_{slugify(query['id'])}",
        "description": "Scaffold condition x injected-token dose ladder. Dose 0 is the shared baseline row.",
        "include_baseline": True,
        "items": items,
    }
    matrix_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return matrix_path


def write_closure_contract_matrix(
    out_dir: Path,
    query: Dict[str, str],
    *,
    dose_values: Sequence[int],
    dose_conditions: Sequence[str],
    output_contracts: Sequence[str],
) -> Path:
    """Write a per-query matrix varying output-closure contracts."""
    matrix_dir = out_dir / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = matrix_dir / f"gemini_closure_contract_{slugify(query['id'])}.json"
    items: List[Dict[str, Any]] = []
    for condition in dose_conditions:
        condition_s = str(condition).strip()
        if not condition_s:
            continue
        for dose in dose_values:
            dose_i = int(dose)
            if dose_i <= 0:
                continue
            for contract in output_contracts:
                contract_s = str(contract).strip() or "none"
                items.append(
                    {
                        "name": f"{condition_s}_dose_{dose_i}_{contract_s}",
                        "kind": "ponder",
                        "control": "none",
                        "cfg": {
                            "memory_policy": "current_only",
                            "scaffold_condition": condition_s,
                            "scaffold_token_target": dose_i,
                            "output_contract": contract_s,
                        },
                    }
                )

    spec = {
        "name": f"gemini_closure_contract_{slugify(query['id'])}",
        "description": "Output-closure contract x scaffold condition x injected-token dose. Baseline is shared and has no output contract.",
        "include_baseline": True,
        "items": items,
    }
    matrix_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return matrix_path


def build_cmd(
    repo: Path,
    provider: str,
    query: Dict[str, str],
    out_dir: Path,
    profile: Dict[str, Any],
    *,
    lab_matrix: str = "scaffold_abcd",
) -> Dict[str, Any]:
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
        lab_matrix,
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
        "lab_matrix_arg": lab_matrix,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--providers", default="gemini,claude")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="standard")
    ap.add_argument("--limit_queries", type=int, default=0)
    ap.add_argument("--dose_values", default="", help="Comma-separated scaffold token targets for dose_ladder profile")
    ap.add_argument("--dose_conditions", default="", help="Comma-separated scaffold conditions for dose_ladder profile")
    ap.add_argument("--output_contracts", default="", help="Comma-separated output contracts for closure-contract profile")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir or (repo / "artifacts" / f"sweep_{now_slug()}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    queries = QUERIES[: max(0, int(args.limit_queries)) or len(QUERIES)]
    profile = PROFILES[args.profile]
    dose_values = parse_int_csv(str(args.dose_values or ""), default=profile.get("dose_values") or [])
    dose_conditions = parse_str_csv(str(args.dose_conditions or ""), default=profile.get("dose_conditions") or [])
    output_contracts = parse_str_csv(str(args.output_contracts or ""), default=profile.get("output_contracts") or ["none"])

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
        "dose_values": dose_values if profile.get("matrix") in ("dose_ladder", "closure_contract") else [],
        "dose_conditions": dose_conditions if profile.get("matrix") in ("dose_ladder", "closure_contract") else [],
        "output_contracts": output_contracts if profile.get("matrix") == "closure_contract" else [],
        "runs": [],
    }

    for provider in providers:
        for query in queries:
            lab_matrix = "scaffold_abcd"
            if profile.get("matrix") == "dose_ladder":
                lab_matrix = str(
                    write_dose_ladder_matrix(
                        out_dir,
                        query,
                        dose_values=dose_values,
                        dose_conditions=dose_conditions,
                    )
                )
            elif profile.get("matrix") == "closure_contract":
                lab_matrix = str(
                    write_closure_contract_matrix(
                        out_dir,
                        query,
                        dose_values=dose_values,
                        dose_conditions=dose_conditions,
                        output_contracts=output_contracts,
                    )
                )
            spec = build_cmd(repo, provider, query, out_dir, profile, lab_matrix=lab_matrix)
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
