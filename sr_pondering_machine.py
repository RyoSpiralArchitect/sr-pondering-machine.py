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


def safe_mkdir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    safe_mkdir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def tail_jsonl(path: Path, n: int) -> List[Dict[str, Any]]:
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
    if logits_1d.dim() != 1:
        raise ValueError(f"logits_1d must be 1D, got shape={tuple(logits_1d.shape)}")

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
    random.shuffle(candidates)
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


_PONDER_TEMPLATES = [
    "Using {a} and {b} and the other words, what do you freely associate without any specific purpose?",
]


def make_unrelated_question(keywords: Sequence[str]) -> str:
    if len(keywords) >= 2:
        a, b = keywords[0], keywords[1]
    elif len(keywords) == 1:
        a, b = keywords[0], "Emptiness"
    else:
        a, b = " Unknown", "Not-defined"
    tmpl = random.choice(_PONDER_TEMPLATES)
    return tmpl.format(a=a, b=b)


def build_memory_block(records: Sequence[Dict[str, Any]], *, max_chars_per_log: int = 600) -> str:
    chunks: List[str] = []
    for r in records:
        ts = r.get("ts", "?")
        kw = r.get("keywords", [])
        kw_s = ", ".join(kw) if isinstance(kw, list) else str(kw)
        pq = r.get("ponder_question", "")
        plog = (r.get("ponder_log", "") or "")[:max_chars_per_log].rstrip()
        chunks.append(
            f"[{ts}] keywords: {kw_s}\n"
            f"ponder_q: {pq}\n"
            f"ponder_log:\n{plog}\n"
        )
    return "\n".join(chunks).strip()


def default_system_text() -> str:
    # Gemma IT does not support a dedicated system role; keep this inside the user turn.
    return "After pondering the question, you provide an answer."


def build_prompt_for_answer(query: str, memory_block: Optional[str]) -> str:
    if memory_block:
        return (
            f"{default_system_text()}\n\n"
            "The following is a recently generated Ponder Log Not Directly Related to the Main Topic."
            "But you may use it casually if it helps you notice hidden assumptions or alternative framings.\n"
            "<ponder_log>\n"
            f"{memory_block}\n"
            "</ponder_log>\n\n"
            "Actual Question:\n"
            f"{query}\n\n"
            "Write only the answer in your output (headings and quotes are not needed).\n"
        )
    return (
        f"{default_system_text()}\n\n"
        "Actual Question:\n"
        f"{query}\n\n"
        "Write only the answer in your output (headings and quotes are not needed).\n"
    )


def build_prompt_for_pondering(ponder_q: str) -> str:
    return (
        "Create a brief ponder log for the following question without drawing any conclusions.\n"
        "Conditions:\n"
        "- Do not provide practical advice or reach a final conclusion.\n"
        "- Mixing assumptions, counterexamples, and analogies\n"
        "- 10 to 15 lines. Each line begins with - .\n\n"
        f" Question: {ponder_q}\n"
    )


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

    rejected: RejectedTokenConfig = dataclasses.field(default_factory=RejectedTokenConfig)

    answer_max_new_tokens: int = 1550
    ponder_max_new_tokens: int = 1020
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.05
    no_repeat_ngram_size: int = 0
    seed: int = 1234

    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    use_chat_template: bool = True
    force_gemma_format: bool = True


def run_baseline(hf: LocalHFModel, cfg: RunConfig, query: str) -> str:
    prompt = hf._apply_chat(build_prompt_for_answer(query, memory_block=None), system_text=None)
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


def run_ponder(hf: LocalHFModel, cfg: RunConfig, query: str) -> Tuple[str, Dict[str, Any]]:
    base_prompt = hf._apply_chat(build_prompt_for_answer(query, memory_block=None), system_text=None)
    logits = hf.next_token_logits(base_prompt)

    token_ids = choose_rejected_token_ids(logits, cfg.rejected)
    keywords = decode_keyword_tokens(hf.tokenizer, token_ids)

    ponder_q = make_unrelated_question(keywords)
    ponder_prompt = hf._apply_chat(build_prompt_for_pondering(ponder_q), system_text=None)
    ponder_log = hf.generate_text(
        ponder_prompt,
        max_new_tokens=cfg.ponder_max_new_tokens,
        temperature=max(0.4, cfg.temperature),
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        no_repeat_ngram_size=cfg.no_repeat_ngram_size,
        seed=cfg.seed + 1,
    )

    record: Dict[str, Any] = {
        "ts": now_iso(),
        "keywords": keywords,
        "token_ids": token_ids,
        "rejected_cfg": dataclasses.asdict(cfg.rejected),
        "ponder_question": ponder_q,
        "ponder_log": ponder_log,
    }
    append_jsonl(cfg.memory_path, record)

    records = tail_jsonl(cfg.memory_path, cfg.n_memory)
    memory_block = build_memory_block(records, max_chars_per_log=700)
    final_prompt = hf._apply_chat(build_prompt_for_answer(query, memory_block=memory_block), system_text=None)
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

    return answer, record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Local model directory (offline)")
    ap.add_argument("--query", required=True, help="User query (main question)")
    ap.add_argument("--memory", default="ponder_logs.jsonl", help="Path to JSONL memory log")
    ap.add_argument("--mode", choices=["baseline", "ponder", "both"], default="both")

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

    cfg = RunConfig(
        model_path=args.model,
        memory_path=Path(args.memory),
        n_memory=args.n_memory,
        rejected=RejectedTokenConfig(
            top_k=args.top_k_rejected,
            strategy=args.strategy,
            exclude_top=args.exclude_top,
            band_width=args.band_width,
            n_keywords=args.n_keywords,
        ),
        answer_max_new_tokens=args.answer_max_new_tokens,
        ponder_max_new_tokens=args.ponder_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        seed=args.seed,
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
        f"gemma_turn_tokens={hf._has_gemma_turn_tokens()}"
    )

    if args.mode in ("baseline", "both"):
        ans = run_baseline(hf, cfg, args.query)
        print("\n=== BASELINE ===\n")
        print(ans)

    if args.mode in ("ponder", "both"):
        ans, rec = run_ponder(hf, cfg, args.query)
        print("\n=== PONDER RECORD (just written) ===\n")
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        print("\n=== PONDERED ANSWER ===\n")
        print(ans)


if __name__ == "__main__":
    main()
