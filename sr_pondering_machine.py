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
import hashlib
import json
import os
import random
import re
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


def build_prompt_for_pondering(ponder_q: str, *, mode: str, lang: str) -> str:
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
        return body

    # English
    if mode == "assoc":
        return (
            "Create a brief ponder log for the following question without drawing any conclusions.\n"
            "Conditions:\n"
            "- Do not provide practical advice or reach a final conclusion.\n"
            "- Mix assumptions, counterexamples, and analogies.\n"
            "- 10 to 15 lines. Each line begins with - .\n\n"
            f"Question: {ponder_q}\n"
        )
    if mode == "assumption":
        return (
            "List the hidden assumptions behind the following question, and suggest alternative assumptions.\n"
            "Conditions:\n"
            "- No conclusion.\n"
            "- 10 to 15 lines. Each line begins with - .\n\n"
            f"Question: {ponder_q}\n"
        )
    if mode == "counterexample":
        return (
            "Generate counterexamples, edge cases, and exceptions related to the following question.\n"
            "Conditions:\n"
            "- No conclusion.\n"
            "- 10 to 15 lines. Each line begins with - .\n\n"
            f"Question: {ponder_q}\n"
        )
    if mode == "questions_only":
        return (
            "Output only questions derived from the following question.\n"
            "Conditions:\n"
            "- No answers or assertions.\n"
            "- 10 to 15 lines. Each line begins with - .\n\n"
            f"Question: {ponder_q}\n"
        )
    if mode == "metaphor":
        return (
            "Write a ponder log using only metaphors, analogies, and image fragments inspired by the following question.\n"
            "Conditions:\n"
            "- No conclusion.\n"
            "- 10 to 15 lines. Each line begins with - .\n\n"
            f"Question: {ponder_q}\n"
        )
    raise ValueError(f"Unknown ponder_mode: {mode!r}")


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


def _token_prob(logits_1d: torch.Tensor, token_id: int, log_z: torch.Tensor) -> float:
    return float(torch.exp(logits_1d[int(token_id)] - log_z).item())


def _token_text(tokenizer, token_id: int) -> str:
    try:
        s = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    except Exception:
        s = str(token_id)
    s = clean_token_text(s)
    return s


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
    ) -> None:
        self.model_path = model_path
        self.use_chat_template = use_chat_template
        self.force_gemma_format = force_gemma_format

        self.device_str = resolve_device(device)
        self.device = torch.device(self.device_str)
        self.torch_dtype = resolve_dtype(dtype, self.device_str)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
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
                model_path,
                dtype=self.torch_dtype,
                **model_kwargs,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
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
    ponder_mode: str = "assoc"  # assoc|assumption|counterexample|questions_only|metaphor
    n_ponder: int = 1

    keyword_refine: bool = False
    keyword_refine_max_new_tokens: int = 96
    keyword_refine_temperature: float = 0.3

    probe_top_n: int = 0
    print_probe: bool = False

    device: str = "auto"
    dtype: str = "auto"
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


def run_ponder(hf: LocalHFModel, cfg: RunConfig, query: str) -> Tuple[str, List[Dict[str, Any]]]:
    lang = cfg.prompt_lang
    n_ponder = max(1, int(cfg.n_ponder))  # per band

    run_id = sha256_short(f"{now_iso()}|{cfg.seed}|{query}")
    query_sha = sha256_short(query)

    base_prompt = hf._apply_chat(build_prompt_for_answer(query, memory_block=None, lang=lang), system_text=None)
    logits = hf.next_token_logits(base_prompt)

    sorted_ids = torch.argsort(logits, descending=True)
    ranks = torch.empty_like(sorted_ids)
    ranks[sorted_ids] = torch.arange(sorted_ids.numel())
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

    records: List[Dict[str, Any]] = []
    log_ix = 0
    for band_ix, band in enumerate(bands):
        band_label = str(band.get("label", f"band{band_ix}"))
        band_start = int(band.get("start_rank", 0))
        band_end = int(band.get("end_rank", vocab_size))

        band_end = min(band_end, vocab_size)
        if band_start < 0:
            band_start = 0
        if band_start >= band_end:
            band_start = max(0, band_end - 1)
            band_end = min(vocab_size, band_start + 1)

        candidate_pool = sorted_ids[band_start:band_end].tolist()
        if not candidate_pool:
            # fallback: allow something, even if the band was degenerate
            candidate_pool = sorted_ids[: min(vocab_size, 256)].tolist()

        pool_rng = random.Random(cfg.seed + 99991 + band_ix * 10007)
        pool_rng.shuffle(candidate_pool)
        pool_pos = 0

        for band_ponder_ix in range(n_ponder):
            n_kw = int(cfg.rejected.n_keywords)

            token_ids = candidate_pool[pool_pos : pool_pos + n_kw]
            pool_pos += n_kw
            if len(token_ids) < n_kw:
                pool_rng.shuffle(candidate_pool)
                pool_pos = 0
                token_ids = candidate_pool[pool_pos : pool_pos + n_kw]
                pool_pos += n_kw
            if len(token_ids) < n_kw and candidate_pool:
                while len(token_ids) < n_kw:
                    token_ids.append(pool_rng.choice(candidate_pool))

            raw_keywords = decode_keyword_tokens(hf.tokenizer, token_ids)
            keywords = raw_keywords
            keywords_source = "rejected_tokens"
            refined_keywords: List[str] = []
            if cfg.keyword_refine:
                refined_keywords = refine_keywords_with_model(
                    hf,
                    query=query,
                    seed_keywords=raw_keywords,
                    n_keywords=n_kw,
                    lang=lang,
                    max_new_tokens=int(cfg.keyword_refine_max_new_tokens),
                    temperature=float(cfg.keyword_refine_temperature),
                    seed=cfg.seed + 200 + log_ix,
                )
                if refined_keywords:
                    keywords = refined_keywords
                    keywords_source = "model_refine"

            q_rng = random.Random(cfg.seed + 12345 + log_ix)
            ponder_q = make_unrelated_question(keywords, lang=lang, rng=q_rng)
            ponder_prompt = hf._apply_chat(
                build_prompt_for_pondering(ponder_q, mode=cfg.ponder_mode, lang=lang),
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
                seed=cfg.seed + 1 + log_ix,
            )

            selected_tokens = [
                {
                    "token_id": int(tid),
                    "token": _token_text(hf.tokenizer, tid),
                    "rank": int(ranks[int(tid)].item()),
                    "prob": _token_prob(logits, tid, log_z),
                }
                for tid in token_ids
            ]
            if cfg.print_probe:
                _print_probe_table(
                    f"REJECTED TOKENS (band={band_label} ix={log_ix})",
                    selected_tokens,
                    limit=len(selected_tokens),
                )
                print(f"\nkeywords_source={keywords_source} keywords={keywords}\n")

            record: Dict[str, Any] = {
                "ts": now_iso(),
                "run_id": run_id,
                "query_sha": query_sha,
                "prompt_lang": lang,
                "band_profile": cfg.band_profile,
                "band_ix": band_ix,
                "band_label": band_label,
                "band_ponder_ix": band_ponder_ix,
                "ponder_ix": log_ix,
                "ponder_mode": cfg.ponder_mode,
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

            append_jsonl(cfg.memory_path, record)
            records.append(record)
            log_ix += 1

    if cfg.memory_policy == "tail":
        mem_records = tail_jsonl(cfg.memory_path, cfg.n_memory)
    elif cfg.memory_policy == "current_only":
        mem_records = records
    elif cfg.memory_policy == "off":
        mem_records = []
    else:
        raise ValueError(f"Unknown memory_policy: {cfg.memory_policy!r}")

    memory_block = build_memory_block(mem_records, max_chars_per_log=700) if mem_records else None
    final_prompt = hf._apply_chat(build_prompt_for_answer(query, memory_block=memory_block, lang=lang), system_text=None)
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

    return answer, records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Local model directory (offline)")
    ap.add_argument("--query", required=True, help="User query (main question)")
    ap.add_argument("--memory", default="ponder_logs.jsonl", help="Path to JSONL memory log")
    ap.add_argument("--mode", choices=["baseline", "ponder", "both"], default="both")

    ap.add_argument("--prompt_lang", choices=["auto", "en", "ja"], default="auto", help="Prompt language: auto|en|ja")
    ap.add_argument(
        "--ponder_mode",
        choices=["assoc", "assumption", "counterexample", "questions_only", "metaphor"],
        default="assoc",
        help="Ponder log lens",
    )
    ap.add_argument("--n_ponder", type=int, default=1, help="Number of ponder logs per band")
    ap.add_argument(
        "--memory_policy",
        choices=["tail", "current_only", "off"],
        default="tail",
        help="Which ponder logs to inject into the final answer",
    )
    ap.add_argument("--keyword_refine", action="store_true", help="Let the model rewrite token fragments into keywords")
    ap.add_argument("--keyword_refine_max_new_tokens", type=int, default=96)
    ap.add_argument("--keyword_refine_temperature", type=float, default=0.3)
    ap.add_argument("--probe_top_n", type=int, default=0, help="Store probe top-N tokens in the JSONL record (0=off)")
    ap.add_argument("--print_probe", action="store_true", help="Print probe tables to stdout")

    ap.add_argument("--band_profile", choices=["single", "spectrum3"], default="single", help="Rank-band profile")
    ap.add_argument(
        "--band",
        action="append",
        default=[],
        help='Custom rank band "START:END" or "LABEL=START:END" (repeatable; END is exclusive).',
    )

    ap.add_argument("--device", default="auto", help="auto|mps|cpu|cuda|cuda:0 ...")
    ap.add_argument("--dtype", default="auto", help="auto|float16|bfloat16|float32")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--no_chat_template", action="store_true")
    ap.add_argument("--no_gemma_format", action="store_true", help="Disable Gemma-native turn formatting")

    ap.add_argument("--strategy", choices=["within_topk", "outside_topk"], default="outside_topk")
    ap.add_argument("--top_k_rejected", type=int, default=80)
    ap.add_argument("--exclude_top", type=int, default=8)
    ap.add_argument("--band_width", type=int, default=256)
    ap.add_argument("--n_keywords", type=int, default=6)
    ap.add_argument("--n_memory", type=int, default=6)

    ap.add_argument("--answer_max_new_tokens", type=int, default=256)
    ap.add_argument("--ponder_max_new_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)

    args = ap.parse_args()

    prompt_lang = resolve_prompt_lang(args.prompt_lang, args.query)
    bands = [_parse_band_spec(x) for x in (args.band or [])]
    cfg = RunConfig(
        model_path=args.model,
        memory_path=Path(args.memory),
        n_memory=args.n_memory,
        memory_policy=args.memory_policy,
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
        n_ponder=args.n_ponder,
        keyword_refine=args.keyword_refine,
        keyword_refine_max_new_tokens=args.keyword_refine_max_new_tokens,
        keyword_refine_temperature=args.keyword_refine_temperature,
        probe_top_n=args.probe_top_n,
        print_probe=args.print_probe,
        device=args.device,
        dtype=args.dtype,
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
    )
    print(
        f"[sr_ponder] device={hf.device_str} input_device={hf.input_device} dtype={hf.torch_dtype} "
        f"gemma_turn_tokens={hf._has_gemma_turn_tokens()} lang={cfg.prompt_lang} ponder_mode={cfg.ponder_mode} "
        f"band_profile={cfg.band_profile} custom_bands={len(cfg.bands)}"
    )

    if args.mode in ("baseline", "both"):
        ans = run_baseline(hf, cfg, args.query)
        print("\n=== BASELINE ===\n")
        print(ans)

    if args.mode in ("ponder", "both"):
        ans, recs = run_ponder(hf, cfg, args.query)
        print("\n=== PONDER RECORD(S) (just written) ===\n")
        print(json.dumps(recs if len(recs) > 1 else recs[0], ensure_ascii=False, indent=2))
        print("\n=== PONDERED ANSWER ===\n")
        print(ans)


if __name__ == "__main__":
    main()
