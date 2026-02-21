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
import dataclasses
import datetime as dt
import difflib
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Utilities
# -----------------------------


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_iso() -> str:
    # timezone-aware UTC timestamp (avoids datetime.utcnow() deprecation)
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


_JA_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")  # hiragana/katakana/CJK


def is_japanese(text: str) -> bool:
    return _JA_RE.search(text or "") is not None


def resolve_prompt_lang(prompt_lang: str, query: str) -> str:
    if prompt_lang in ("en", "ja"):
        return prompt_lang
    return "ja" if is_japanese(query) else "en"


def sha256_short(s: str, n: int = 12) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:n]


def safe_mkdir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    safe_mkdir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def tail_jsonl(path: Path, n: int) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    out: List[Dict[str, Any]] = []
    for line in lines[-n:]:
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
            raise SystemExit(f"[sr_ponder] ERROR: not a HF model dir (missing config.json): {p}")
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
            for dev in (expanded_device_map or {}).values():
                if dev in ("disk", "cpu", "meta"):
                    continue
                try:
                    tdev = torch.device(dev)
                except Exception:
                    continue
                if tdev.type not in ("cuda", "xpu"):
                    return

        return getattr(mu, "_sr_ponder_allocator_warmup_orig")(model, expanded_device_map, hf_quantizer)

    setattr(mu, "caching_allocator_warmup", _patched_caching_allocator_warmup)
    setattr(mu, "_sr_ponder_allocator_warmup_patched", True)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype != "auto":
        return getattr(torch, dtype)
    if device in ("mps", "cuda"):
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


def build_prompt_for_answer(query: str, memory_block: Optional[str], *, lang: str) -> str:
    if memory_block:
        if lang == "ja":
            return (
                f"{default_system_text(lang)}\n\n"
                "以下は最近生成された「本題と直接関係しない Ponder Log」です。"
                "ただし、隠れた前提や別の切り口に気づく助けになるなら軽く参照してもよい。\n"
                "<ponder_log>\n"
                f"{memory_block}\n"
                "</ponder_log>\n\n"
                "本題の質問:\n"
                f"{query}\n\n"
                "出力は回答本文のみ（見出しや引用は不要）。\n"
            )
        return (
            f"{default_system_text(lang)}\n\n"
            "The following is a recently generated Ponder Log Not Directly Related to the Main Topic."
            "But you may use it casually if it helps you notice hidden assumptions or alternative framings.\n"
            "<ponder_log>\n"
            f"{memory_block}\n"
            "</ponder_log>\n\n"
            "Actual Question:\n"
            f"{query}\n\n"
            "Write only the answer in your output (headings and quotes are not needed).\n"
        )
    if lang == "ja":
        return (
            f"{default_system_text(lang)}\n\n"
            "本題の質問:\n"
            f"{query}\n\n"
            "出力は回答本文のみ（見出しや引用は不要）。\n"
        )
    return (
        f"{default_system_text(lang)}\n\n"
        "Actual Question:\n"
        f"{query}\n\n"
        "Write only the answer in your output (headings and quotes are not needed).\n"
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


def refine_keywords_with_model(
    hf: "LocalHFModel",
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
    idx = torch.tensor([int(x) for x in candidate_ids], dtype=torch.long)
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
        tries = 0
        while len(out) < n and tries < n * 50:
            tries += 1
            tid = int(rng.randrange(0, int(vocab_size)))
            if tid in special:
                continue
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


def _cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    aa = a.detach().float().cpu()
    bb = b.detach().float().cpu()
    denom = (torch.linalg.norm(aa) * torch.linalg.norm(bb)).item() + eps
    if denom <= 0:
        return 0.0
    return float(torch.dot(aa, bb).item() / denom)


def select_memory_records(
    hf: "LocalHFModel",
    *,
    memory_path: Path,
    current_records: Sequence[Dict[str, Any]],
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
        return list(current_records)
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
        return pool[-int(n_memory) :] if n_memory > 0 else []

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
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = dict(
            local_files_only=True,
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

    @torch.inference_mode()
    def next_token_logits(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer(prompt, return_tensors="pt")
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

    @torch.inference_mode()
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

    keyword_refine: bool = False
    keyword_refine_max_new_tokens: int = 96
    keyword_refine_temperature: float = 0.3
    keyword_objective: str = "random_band"  # random_band|dissonance|unstable|random_vocab
    keyword_select_top: int = 128
    dissonance_target: float = 0.9
    dissonance_width: float = 0.6
    dissonance_tail_k: int = 64

    prompt_jitter: int = 0  # number of paraphrases (excluding original)
    prompt_jitter_include_original: bool = True
    prompt_jitter_max_new_tokens: int = 160
    prompt_jitter_temperature: float = 0.6

    probe_top_n: int = 0
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


def run_baseline(hf: LocalHFModel, cfg: RunConfig, query: str) -> str:
    prompt = hf._apply_chat(build_prompt_for_answer(query, memory_block=None, lang=cfg.prompt_lang), system_text=None)
    return hf.generate_text(
        prompt,
        max_new_tokens=cfg.answer_max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        no_repeat_ngram_size=cfg.no_repeat_ngram_size,
        seed=cfg.seed,
    )


def _print_probe_table(title: str, items: Sequence[Dict[str, Any]], *, limit: int) -> None:
    print(f"\n=== {title} ===\n")
    for i, x in enumerate(items[:limit], start=1):
        tok = x.get("token", "")
        tid = x.get("token_id", "?")
        rk = x.get("rank", "?")
        p = x.get("prob", None)
        p_s = f"{p:.4f}" if isinstance(p, float) else "?"
        print(f"{i:>2}. {tok!r} id={tid} rank={rk} p={p_s}")


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


def run_ponder(hf: LocalHFModel, cfg: RunConfig, query: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    lang = cfg.prompt_lang
    n_ponder = max(1, int(cfg.n_ponder))  # per band

    run_id = sha256_short(f"{now_iso()}|{cfg.seed}|{query}")
    query_sha = sha256_short(query)

    pipeline = list(cfg.ponder_pipeline) if cfg.ponder_pipeline else [cfg.ponder_mode]
    pipeline_ctx = (cfg.pipeline_context or "prev").strip()

    objective = (cfg.keyword_objective or "random_band").strip()
    control = (cfg.control or "none").strip()
    if control == "random_keywords":
        objective = "random_vocab"

    base_prompt = hf._apply_chat(build_prompt_for_answer(query, memory_block=None, lang=lang), system_text=None)
    logits = hf.next_token_logits(base_prompt)

    sorted_ids = torch.argsort(logits, descending=True)
    ranks = torch.empty_like(sorted_ids)
    ranks[sorted_ids] = torch.arange(sorted_ids.numel(), device=sorted_ids.device, dtype=sorted_ids.dtype)
    log_z = torch.logsumexp(logits, dim=0)

    top_tokens: List[Dict[str, Any]] = []
    if cfg.probe_top_n > 0 or cfg.print_probe:
        top_n = int(cfg.probe_top_n) if cfg.probe_top_n > 0 else 20
        top_n = max(1, min(top_n, int(sorted_ids.numel())))
        for tid in sorted_ids[:top_n].tolist():
            top_tokens.append(
                {
                    "token_id": int(tid),
                    "token": _token_text(hf.tokenizer, tid),
                    "rank": int(ranks[int(tid)].item()),
                    "prob": _token_prob(logits, tid, log_z),
                }
            )
        if cfg.print_probe:
            _print_probe_table("PROBE TOP TOKENS", top_tokens, limit=top_n)

    vocab_size = int(sorted_ids.numel())
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
            jp = hf._apply_chat(build_prompt_for_answer(jq, memory_block=None, lang=lang), system_text=None)
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

            token_ids = select_token_ids_by_objective(
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

            # One ponder question per (band, band_ponder_ix), reused across pipeline stages.
            if control == "lens_only":
                ponder_q = query
            else:
                q_rng = random.Random(int(cfg.seed) + 12345 + band_ix * 10007 + band_ponder_ix * 101)
                ponder_q = make_unrelated_question(keywords, lang=lang, rng=q_rng)

            stage_logs: List[str] = []
            for stage_ix, stage_mode in enumerate(pipeline):
                ctx_text: Optional[str] = None
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
                stage_logs.append(ponder_log)

                selected_tokens = []
                for tid in token_ids:
                    t: Dict[str, Any] = {
                        "token_id": int(tid),
                        "token": _token_text(hf.tokenizer, tid),
                        "rank": int(ranks[int(tid)].item()),
                        "prob": _token_prob(logits, tid, log_z),
                    }
                    if band_dissonance is not None and int(tid) in band_dissonance:
                        t["dissonance"] = float(band_dissonance[int(tid)])
                    if band_unstable is not None and int(tid) in band_unstable:
                        t["unstable"] = float(band_unstable[int(tid)])
                    selected_tokens.append(t)

                if cfg.print_probe:
                    _print_probe_table(
                        f"REJECTED TOKENS (band={band_label} ix={log_ix} stage={stage_mode})",
                        selected_tokens,
                        limit=len(selected_tokens),
                    )
                    print(f"\nkeywords_source={keywords_source} keywords={keywords}\n")

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
                    "pipeline": pipeline,
                    "pipeline_stage_ix": stage_ix,
                    "pipeline_context": pipeline_ctx,
                    "ponder_ix": log_ix,
                    "ponder_mode": stage_mode,
                    "keywords_source": keywords_source,
                    "keywords_raw": raw_keywords,
                    "keywords": keywords,
                    "token_ids": token_ids,
                    "selected_tokens": selected_tokens,
                    "rejected_cfg": dataclasses.asdict(cfg.rejected),
                    "band": {"start_rank": band_start, "end_rank": band_end},
                    "ponder_question": ponder_q,
                    "ponder_log": ponder_log,
                }
                if cfg.probe_top_n > 0 and log_ix == 0:
                    record["probe_top"] = top_tokens[: int(cfg.probe_top_n)]
                if jitter_queries and log_ix == 0:
                    record["prompt_jitter_queries"] = jitter_queries

                if cfg.write_memory:
                    append_jsonl(cfg.memory_path, record)
                records.append(record)
                log_ix += 1

    # Select memory records for final answer injection
    exclude_run_id = run_id if cfg.memory_exclude_current_run and cfg.memory_policy == "tail" else None
    mem_records = select_memory_records(
        hf,
        memory_path=cfg.memory_path,
        current_records=records,
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
                seed=int(cfg.seed) + 9000 + hash(bl) % 1000,
            )
            fp = hf._apply_chat(build_prompt_for_answer(query, memory_block=band_mem, lang=lang), system_text=None)
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
    if cfg.answer_ensemble and ensemble_final:
        answer = ensemble_final.strip()
    else:
        final_prompt = hf._apply_chat(build_prompt_for_answer(query, memory_block=final_answer_block, lang=lang), system_text=None)
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

    extras: Dict[str, Any] = {
        "run_id": run_id,
        "control": control,
        "pipeline": pipeline,
        "memory_policy": cfg.memory_policy,
        "memory_retrieve": cfg.memory_retrieve,
        "memory_remix": cfg.memory_remix,
        "memory_selected": [
            {
                "ts": r.get("ts"),
                "run_id": r.get("run_id"),
                "band_label": r.get("band_label"),
                "ponder_ix": r.get("ponder_ix"),
                "ponder_mode": r.get("ponder_mode"),
            }
            for r in (mem_records or [])
        ],
        "band_answers": band_answers if band_answers else None,
        "ensemble": {
            "raw": ensemble_raw,
            "consensus": ensemble_consensus,
            "divergence": ensemble_divergence,
            "final": ensemble_final,
        }
        if ensemble_raw
        else None,
        "random_log": random_log_text,
        "prompt_jitter_queries": jitter_queries if jitter_queries else None,
    }

    return answer, records, extras


def main() -> None:
    class _Fmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    ap = argparse.ArgumentParser(
        description="Local pondering machine (HF causal LM) — rejected-token + lens + bands + memory.",
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
    g_runtime = ap.add_argument_group("Runtime")
    g_gen = ap.add_argument_group("Generation")

    g_core.add_argument("--model", required=True, help="Local model directory (offline)")
    g_core.add_argument("--query", required=True, help="User query (main question)")
    g_core.add_argument("--memory", default="ponder_logs.jsonl", help="Path to JSONL memory log")
    g_core.add_argument("--mode", choices=["baseline", "ponder", "both"], default="both")
    g_core.add_argument("--prompt_lang", choices=["auto", "en", "ja"], default="auto", help="Prompt language")

    g_ponder.add_argument(
        "--ponder_mode",
        choices=["assoc", "assumption", "counterexample", "questions_only", "metaphor"],
        default="assoc",
        help="Ponder lens (used when --ponder_pipeline is empty)",
    )
    g_ponder.add_argument("--n_ponder", type=int, default=1, help="Number of ponder logs per band")
    g_ponder.add_argument(
        "--control",
        choices=["none", "no_inject", "random_log", "random_keywords", "lens_only"],
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

    g_answer.add_argument("--answer_max_new_tokens", type=int, default=256)
    g_answer.add_argument("--answer_per_band", action="store_true", help="Generate per-band answers (sensitivity)")
    g_answer.add_argument("--answer_ensemble", action="store_true", help="Merge per-band answers into a final answer")
    g_answer.add_argument("--answer_ensemble_max_new_tokens", type=int, default=512)
    g_answer.add_argument("--answer_ensemble_temperature", type=float, default=0.2)

    g_interactive.add_argument("--interactive", action="store_true", help="Pick keyword tokens interactively")
    g_interactive.add_argument("--interactive_candidates", type=int, default=48)

    g_controls.add_argument(
        "--pack",
        choices=["none", "controls"],
        default="none",
        help="Run a pack of control variants",
    )
    g_controls.add_argument("--pack_out", default="", help="Optional JSON output for pack results")
    g_controls.add_argument("--pack_write_memory", action="store_true", help="Allow pack runs to write to memory JSONL")
    g_controls.add_argument(
        "--print_records",
        choices=["auto", "none", "all"],
        default="auto",
        help="Print ponder records JSON",
    )

    g_runtime.add_argument("--device", default="auto", help="auto|mps|cpu|cuda|cuda:0 ...")
    g_runtime.add_argument("--dtype", default="auto", help="auto|float16|bfloat16|float32")
    g_runtime.add_argument(
        "--allocator_warmup",
        choices=["auto", "on", "off"],
        default="auto",
        help="Transformers caching allocator warmup (auto disables on MPS)",
    )
    g_runtime.add_argument("--trust_remote_code", action="store_true")
    g_runtime.add_argument("--no_chat_template", action="store_true")
    g_runtime.add_argument("--no_gemma_format", action="store_true", help="Disable Gemma-native turn formatting")
    g_runtime.add_argument("--probe_top_n", type=int, default=0, help="Store probe top-N tokens in record (0=off)")
    g_runtime.add_argument("--print_probe", action="store_true", help="Print probe tables to stdout")

    g_gen.add_argument("--ponder_max_new_tokens", type=int, default=160)
    g_gen.add_argument("--temperature", type=float, default=0.7)
    g_gen.add_argument("--top_p", type=float, default=0.95)
    g_gen.add_argument("--top_k", type=int, default=0)
    g_gen.add_argument("--repetition_penalty", type=float, default=1.05)
    g_gen.add_argument("--no_repeat_ngram_size", type=int, default=0)
    g_gen.add_argument("--seed", type=int, default=1234)

    args = ap.parse_args()

    prompt_lang = resolve_prompt_lang(args.prompt_lang, args.query)
    bands = [_parse_band_spec(x) for x in (args.band or [])]
    pipeline = parse_ponder_pipeline(args.ponder_pipeline, fallback_mode=args.ponder_mode)

    write_memory = not bool(args.no_write_memory)
    if args.pack != "none" and not bool(args.pack_write_memory):
        write_memory = False

    cfg = RunConfig(
        model_path=args.model,
        memory_path=Path(args.memory),
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
        keyword_refine=args.keyword_refine,
        keyword_refine_max_new_tokens=args.keyword_refine_max_new_tokens,
        keyword_refine_temperature=args.keyword_refine_temperature,
        keyword_objective=args.keyword_objective,
        keyword_select_top=args.keyword_select_top,
        dissonance_target=args.dissonance_target,
        dissonance_width=args.dissonance_width,
        dissonance_tail_k=args.dissonance_tail_k,
        prompt_jitter=args.prompt_jitter,
        prompt_jitter_include_original=not bool(args.no_prompt_jitter_include_original),
        prompt_jitter_max_new_tokens=args.prompt_jitter_max_new_tokens,
        prompt_jitter_temperature=args.prompt_jitter_temperature,
        probe_top_n=args.probe_top_n,
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
    )

    hf = LocalHFModel(
        cfg.model_path,
        device=cfg.device,
        dtype=cfg.dtype,
        trust_remote_code=cfg.trust_remote_code,
        use_chat_template=cfg.use_chat_template,
        force_gemma_format=cfg.force_gemma_format,
        allocator_warmup=cfg.allocator_warmup,
    )
    print(
        f"[sr_ponder] device={hf.device_str} input_device={hf.input_device} dtype={hf.torch_dtype} "
        f"alloc_warmup={cfg.allocator_warmup} "
        f"gemma_turn_tokens={hf._has_gemma_turn_tokens()} lang={cfg.prompt_lang} "
        f"band_profile={cfg.band_profile} bands={len(cfg.bands) if cfg.bands else 'profile'} "
        f"objective={cfg.keyword_objective} control={cfg.control} "
        f"pipeline={','.join(cfg.ponder_pipeline) if cfg.ponder_pipeline else cfg.ponder_mode} "
        f"memory={cfg.memory_policy}/{cfg.memory_retrieve}/{cfg.memory_remix} write_memory={cfg.write_memory}"
    )

    if args.pack != "none":
        pack_cfg = dataclasses.replace(cfg, interactive=False, answer_per_band=False, answer_ensemble=False)
        results: Dict[str, Any] = {
            "ts": now_iso(),
            "model": cfg.model_path,
            "query": args.query,
            "pack": args.pack,
            "items": [],
        }

        items: List[Tuple[str, Dict[str, Any]]] = [
            ("baseline", {"kind": "baseline"}),
            ("ponder", {"kind": "ponder", "control": "none"}),
            ("no_inject", {"kind": "ponder", "control": "no_inject"}),
            ("random_keywords", {"kind": "ponder", "control": "random_keywords"}),
            ("random_log", {"kind": "ponder", "control": "random_log"}),
            ("lens_only", {"kind": "ponder", "control": "lens_only"}),
        ]

        for name, spec in items:
            print(f"\n=== PACK: {name} ===\n")
            if spec["kind"] == "baseline":
                ans = run_baseline(hf, pack_cfg, args.query)
                print(ans)
                results["items"].append({"name": name, "kind": "baseline", "answer": ans})
                continue
            cfg2 = dataclasses.replace(pack_cfg, control=str(spec.get("control", "none")))
            ans, _, extras = run_ponder(hf, cfg2, args.query)
            print(ans)
            results["items"].append({"name": name, "kind": "ponder", "control": cfg2.control, "answer": ans, "extras": extras})

        if (args.pack_out or "").strip():
            out_path = Path((args.pack_out or "").strip())
            safe_mkdir(out_path)
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n[pack] wrote {out_path}")
        return

    if args.mode in ("baseline", "both"):
        ans = run_baseline(hf, cfg, args.query)
        print("\n=== BASELINE ===\n")
        print(ans)

    if args.mode in ("ponder", "both"):
        ans, recs, extras = run_ponder(hf, cfg, args.query)

        if args.print_records != "none":
            if args.print_records == "all" or (args.print_records == "auto" and len(recs) <= 3):
                payload: Any = recs if len(recs) > 1 else recs[0]
            elif args.print_records == "auto":
                payload = recs[0] if recs else []
            else:
                payload = []
            if payload:
                print("\n=== PONDER RECORD(S) (just written) ===\n")
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                if args.print_records == "auto" and len(recs) > 3:
                    print(f"\n[info] records={len(recs)} (use --print_records all to dump everything)")

        # Extras (band answers + ensemble)
        band_answers = extras.get("band_answers") if isinstance(extras, dict) else None
        if isinstance(band_answers, dict) and band_answers:
            print("\n=== BAND ANSWERS ===\n")
            for bl, txt in band_answers.items():
                print(f"\n--- {bl} ---\n")
                print(txt)

        ens = extras.get("ensemble") if isinstance(extras, dict) else None
        if isinstance(ens, dict) and ens.get("raw"):
            print("\n=== ENSEMBLE (raw) ===\n")
            print(ens.get("raw"))

        if extras.get("random_log"):
            print("\n=== CONTROL: RANDOM LOG ===\n")
            print(extras.get("random_log"))

        print("\n=== PONDERED ANSWER ===\n")
        print(ans)


if __name__ == "__main__":
    main()
