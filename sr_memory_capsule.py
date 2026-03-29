#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌀 SR Memory Capsule (One-file)
- Default: STRICT OFFLINE (no HF hub)
- Subcommands:
    check-model
    capsule encode|decode|render
    chat repl
    dataforge build
    b1plus train|infer

Core idea:
Session_t -> (HOT baton, COLD archive) -> Session_{t+1}
- HOT baton: SMEM1 (int8 quantized projected embedding) base64
- COLD: lzma + optional AES-GCM base64
- Slots: persona/rules/task (extendable)

Requirements:
  pip install torch transformers
Optional:
  pip install cryptography           (AES-GCM)
  pip install sentence-transformers  (if you enable embed backend "st" & model present locally)

Notes on OFFLINE:
- By default this script sets:
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    HF_HUB_DISABLE_TELEMETRY=1
  To allow online, set env:
    SPIRAL_ALLOW_ONLINE=1
  BEFORE running the script.
"""

from __future__ import annotations

# -----------------------------
# HARD OFFLINE GUARD (before importing transformers)
# -----------------------------
import os as _os
if str(_os.environ.get("SPIRAL_ALLOW_ONLINE", "0")).lower() not in ("1", "true", "yes"):
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    _os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# -----------------------------
# Imports
# -----------------------------
import os
import argparse
import base64
import dataclasses
import hashlib
import io
import json
import lzma
import re
import struct
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from transformers.generation.logits_process import LogitsProcessorList  # type: ignore
except Exception:
    try:
        from transformers.generation import LogitsProcessorList  # type: ignore
    except Exception:
        class LogitsProcessorList(list):  # type: ignore
            def __call__(self, input_ids, scores):
                for p in self:
                    scores = p(input_ids, scores)
                return scores


class SanitizeLogitsProcessor:
    def __init__(self, eos_token_id: Optional[int]):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # Run softmax sampling in fp32 and avoid NaN/Inf poisoning.
        scores = scores.to(dtype=torch.float32)
        if not torch.isfinite(scores).all():
            scores = scores.clone()
            scores[~torch.isfinite(scores)] = -1e4
        if scores.dim() == 2:
            ok = torch.isfinite(scores).any(dim=-1)
            if not bool(torch.all(ok)):
                scores = scores.clone()
                idx = int(self.eos_token_id) if self.eos_token_id is not None else 0
                for b in range(scores.shape[0]):
                    if not bool(ok[b]):
                        scores[b, idx] = 0.0
        return scores


# =============================
# Utils
# =============================
def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def write_text(path: str, s: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

def append_text(path: str, s: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        f.write(s)

def write_json(path: str, obj: Any, indent: int = 2):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)

def append_jsonl(path: str, obj: Any):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_jsonl_lines(path: str, limit: Optional[int] = None) -> List[bytes]:
    lines: List[bytes] = []
    with open(path, "rb") as f:
        for ln in f:
            if not ln.strip():
                continue
            lines.append(ln.rstrip(b"\r\n"))
            if limit and limit > 0 and len(lines) >= limit:
                break
    return lines

def _dedup_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        k = x.strip()
        if not k:
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


# =============================
# Crypto (optional AES-GCM)
# =============================
def _try_aesgcm():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
        return AESGCM
    except Exception:
        return None

AESGCM = _try_aesgcm()

def _kdf_pbkdf2(passphrase: str, salt: bytes, length: int = 32) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, 200_000, dklen=length)

def encrypt_bytes(plain: bytes, passphrase: Optional[str]) -> Dict[str, Any]:
    if not passphrase:
        return {"enc": "none", "nonce_b64": None, "salt_b64": None, "ct_b64": base64.b64encode(plain).decode("ascii")}
    if AESGCM is None:
        # fallback: store plaintext
        return {"enc": "none", "nonce_b64": None, "salt_b64": None, "ct_b64": base64.b64encode(plain).decode("ascii")}

    salt = os.urandom(16)
    key = _kdf_pbkdf2(passphrase, salt, 32)
    nonce = os.urandom(12)
    aes = AESGCM(key)
    ct = aes.encrypt(nonce, plain, associated_data=b"spiral")
    return {
        "enc": "aesgcm",
        "salt_b64": base64.b64encode(salt).decode("ascii"),
        "nonce_b64": base64.b64encode(nonce).decode("ascii"),
        "ct_b64": base64.b64encode(ct).decode("ascii"),
    }

def decrypt_bytes(payload: Dict[str, Any], passphrase: Optional[str]) -> bytes:
    enc = payload.get("enc", "none")
    if enc == "none":
        return base64.b64decode(payload["ct_b64"])
    if enc == "aesgcm":
        if AESGCM is None:
            raise RuntimeError("cryptography not installed; cannot decrypt AES-GCM.")
        if not passphrase:
            raise RuntimeError("passphrase required for AES-GCM decryption.")
        salt = base64.b64decode(payload["salt_b64"])
        nonce = base64.b64decode(payload["nonce_b64"])
        ct = base64.b64decode(payload["ct_b64"])
        key = _kdf_pbkdf2(passphrase, salt, 32)
        aes = AESGCM(key)
        return aes.decrypt(nonce, ct, associated_data=b"spiral")
    raise RuntimeError(f"Unknown enc: {enc}")


# =============================
# SMEM1 Baton Codec
# =============================
def pack_smem1_int8(q: torch.Tensor, scale: float) -> bytes:
    """
    SMEM1:
      magic b"SMEM1" (5)
      dim u16 big-endian (2)
      scale f32 little-endian (4)
      q int8[dim]
    """
    q = q.flatten().contiguous()
    dim = int(q.numel())
    if dim > 65535:
        raise ValueError("SMEM1 dim too large (>65535).")
    header = b"SMEM1" + struct.pack(">H", dim) + struct.pack("<f", float(scale))
    return header + q.cpu().numpy().tobytes()

def unpack_smem1_int8(b64: str, device: torch.device) -> torch.Tensor:
    buf = base64.b64decode(b64)
    if buf[:5] != b"SMEM1":
        raise ValueError("Bad baton magic (expected SMEM1).")
    dim = int.from_bytes(buf[5:7], "big")
    scale = struct.unpack("<f", buf[7:11])[0]
    raw = buf[11:11+dim]
    q = torch.frombuffer(memoryview(bytearray(raw)), dtype=torch.int8).to(torch.float32)
    z = q * float(scale)
    return z.to(device)

def quantize_int8(vec: torch.Tensor) -> Tuple[torch.Tensor, float]:
    v = vec.to(torch.float32)
    mx = float(torch.max(torch.abs(v)).item()) + 1e-12
    scale = mx / 127.0
    q = torch.clamp(torch.round(v / scale), -127, 127).to(torch.int8)
    return q, float(scale)

EMBED_BACKEND_HASH4096 = "hash4096"
EMBED_BACKEND_CHAR_NGRAM4096 = "charngram4096"
EMBED_BACKEND_CHOICES = (EMBED_BACKEND_HASH4096, EMBED_BACKEND_CHAR_NGRAM4096)

def hash_embed_4096(text: str, device: torch.device) -> torch.Tensor:
    text = text or ""
    h0 = hashlib.sha256(text.encode("utf-8")).digest()
    buf = bytearray()
    cur = h0
    while len(buf) < 4096:
        cur = hashlib.sha256(cur + h0).digest()
        buf.extend(cur)
    x = torch.tensor(list(buf[:4096]), dtype=torch.float32, device=device)
    x = (x - 127.5) / 127.5
    x = F.normalize(x, p=2, dim=0)
    return x

def hash_charngram_embed_4096(
    text: str,
    device: torch.device,
    dim: int = 4096,
    n_min: int = 3,
    n_max: int = 5,
    max_features: int = 20000,
) -> torch.Tensor:
    t = (text or "").lower()
    x = torch.zeros(dim, dtype=torch.float32, device=device)
    L = len(t)
    feats = 0
    for n in range(n_min, n_max + 1):
        if L < n:
            continue
        for i in range(L - n + 1):
            gram = t[i:i+n]
            h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % dim
            sign = 1.0 if (h[4] & 1) == 0 else -1.0
            x[idx] += sign
            feats += 1
            if feats >= max_features:
                break
        if feats >= max_features:
            break

    # Keep deterministic non-zero behavior for empty/very short text.
    if feats == 0:
        return hash_embed_4096(t, device=device)

    x = F.normalize(x, p=2, dim=0)
    if not bool(torch.isfinite(x).all()):
        return hash_embed_4096(t, device=device)
    return x

def normalize_embed_backend(name: Optional[str]) -> str:
    raw = (name or EMBED_BACKEND_HASH4096).strip().lower()
    aliases = {
        "hash": EMBED_BACKEND_HASH4096,
        "hashproj": EMBED_BACKEND_HASH4096,
        "charngram": EMBED_BACKEND_CHAR_NGRAM4096,
        "char_ngram": EMBED_BACKEND_CHAR_NGRAM4096,
        "ngram": EMBED_BACKEND_CHAR_NGRAM4096,
    }
    mapped = aliases.get(raw, raw)
    if mapped in EMBED_BACKEND_CHOICES:
        return mapped
    return EMBED_BACKEND_HASH4096

def embed_text_4096(text: str, device: torch.device, embed_backend: str) -> torch.Tensor:
    backend = normalize_embed_backend(embed_backend)
    if backend == EMBED_BACKEND_CHAR_NGRAM4096:
        return hash_charngram_embed_4096(text, device=device, dim=4096)
    return hash_embed_4096(text, device=device)

def make_proj_matrix(in_dim: int, out_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) & 0xFFFFFFFF)
    A = torch.randn((out_dim, in_dim), generator=g, dtype=torch.float32, device="cpu")
    A = F.normalize(A, p=2, dim=1)
    return A.to(device)

def slot_seed(base_seed: int, slot: str) -> int:
    h = hashlib.sha256(slot.encode("utf-8")).digest()
    add = int.from_bytes(h[:2], "big")
    return int(base_seed) + add

def encode_baton_hashproj(
    text: str,
    baton_dim: int,
    proj_seed: int,
    device: torch.device,
    proj_cache: Dict[Tuple[str, str, int, int], torch.Tensor],
    slot: str = "default",
    embed_backend: str = EMBED_BACKEND_HASH4096,
) -> str:
    backend = normalize_embed_backend(embed_backend)
    e = embed_text_4096(text, device=device, embed_backend=backend)
    in_dim = int(e.numel())
    sseed = slot_seed(proj_seed, slot)
    key = (backend, slot, in_dim, baton_dim)
    if key not in proj_cache:
        proj_cache[key] = make_proj_matrix(in_dim, baton_dim, seed=sseed, device=device)
    A = proj_cache[key]
    z = (A @ e).to(torch.float32)
    z = F.normalize(z, p=2, dim=0)
    q, scale = quantize_int8(z)
    b = pack_smem1_int8(q, scale)
    return base64.b64encode(b).decode("ascii")

def render_baton_tagged(b64: str, chunk: int = 80) -> str:
    parts = [b64[i:i+chunk] for i in range(0, len(b64), chunk)]
    return "<SPIRAL_BATON v=SMEM1>\n" + "\n".join(parts) + "\n</SPIRAL_BATON>"


# =============================
# Capsule (HOT + COLD)
# =============================
@dataclass
class Capsule:
    created_at: str
    hot_baton_b64: str
    hot_meta: Dict[str, Any]
    cold_archive: Optional[Dict[str, Any]] = None
    notes: Optional[Dict[str, Any]] = None

def compress_lzma(data: bytes, preset: int = 6) -> bytes:
    return lzma.compress(data, preset=preset)

def decompress_lzma(data: bytes) -> bytes:
    return lzma.decompress(data)

def make_cold_archive(text: str, store: bool, passphrase: Optional[str]) -> Optional[Dict[str, Any]]:
    if not store:
        return None
    raw = text.encode("utf-8")
    comp = compress_lzma(raw, preset=6)
    enc = encrypt_bytes(comp, passphrase=passphrase)
    return {"comp": "lzma", **enc}

def recover_cold_text(cold: Optional[Dict[str, Any]], passphrase: Optional[str]) -> Optional[str]:
    if not cold:
        return None
    comp = cold.get("comp", "none")
    if comp == "none":
        return None
    b = decrypt_bytes(cold, passphrase=passphrase)
    if comp == "lzma":
        raw = decompress_lzma(b)
    else:
        raw = b
    return raw.decode("utf-8", errors="replace")


# =============================
# Local model loading (NO HUB)
# =============================
def resolve_local_model_dir(path: str) -> str:
    """
    Make sure path is a local directory. If path looks like HF cache root, auto-pick a snapshot.
    Raises a clean error before transformers touches hub validators.
    """
    p = os.path.expanduser(path)
    p = os.path.abspath(p)

    if not os.path.exists(p):
        raise SystemExit(
            f"[model_path error] path does not exist: {p}\n"
            f"Hint: you might have meant a relative path like './model/llama-3.2-3b' instead of '/model/...'."
        )
    if os.path.isfile(p):
        # common mistake: pointing to a single weights file or gguf
        if p.lower().endswith(".gguf"):
            raise SystemExit(
                f"[model_path error] '{p}' is a .gguf file.\n"
                f"Transformers cannot load GGUF directly. Use llama.cpp/llama-cpp-python backend instead."
            )
        raise SystemExit(f"[model_path error] expected a DIRECTORY, got a FILE: {p}")

    # If config.json exists, good.
    if os.path.exists(os.path.join(p, "config.json")):
        return p

    # If HF cache-style: .../snapshots/<hash>/config.json
    snap = os.path.join(p, "snapshots")
    if os.path.isdir(snap):
        candidates = []
        for name in os.listdir(snap):
            d = os.path.join(snap, name)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json")):
                candidates.append(d)
        if candidates:
            # pick latest modified snapshot
            candidates.sort(key=lambda d: os.path.getmtime(d), reverse=True)
            return candidates[0]

    # Otherwise fail with guidance
    raise SystemExit(
        f"[model_path error] directory exists but config.json not found: {p}\n"
        f"Expected HF-style folder containing at least config.json + tokenizer files + weights.\n"
        f"If you downloaded via huggingface cache, pass the snapshot directory:\n"
        f"  ~/.cache/huggingface/hub/.../snapshots/<hash>/\n"
    )

def pick_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def pick_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    if dtype == "bf16":
        # mps bf16 is shaky; user can override
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    # auto
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32

def load_local_llm(model_path: str, device: torch.device, dtype: torch.dtype, use_fast: bool) -> Tuple[Any, Any]:
    mp = resolve_local_model_dir(model_path)
    try:
        tok = AutoTokenizer.from_pretrained(mp, local_files_only=True, use_fast=use_fast)
    except Exception:
        tok = AutoTokenizer.from_pretrained(mp, local_files_only=True, use_fast=False)

    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    # Load model offline
    if device.type == "cuda":
        # try device_map auto
        try:
            model = AutoModelForCausalLM.from_pretrained(
                mp, local_files_only=True, torch_dtype=dtype, device_map="auto"
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(mp, local_files_only=True, torch_dtype=dtype)
            model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(mp, local_files_only=True, torch_dtype=dtype)
        model.to(device)

    model.eval()
    return tok, model


# =============================
# Chat template helpers
# =============================
@dataclass
class Msg:
    role: str
    content: str

def heuristic_task_only(history: List[Msg], last_n: int) -> str:
    user_msgs = [m.content.strip() for m in history if m.role == "user" and m.content.strip()]
    tail = user_msgs[-last_n:] if last_n > 0 else user_msgs
    return "\n".join(tail).strip()

def build_input_ids(tok, messages: List[Dict[str, str]]) -> torch.Tensor:
    try:
        res = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(res, "input_ids"):
            res = res.input_ids
        elif isinstance(res, dict) and "input_ids" in res:
            res = res["input_ids"]
        elif isinstance(res, list):
            res = torch.tensor([res], dtype=torch.long)
        if not isinstance(res, torch.Tensor):
            res = torch.tensor(res, dtype=torch.long)
        if res.dtype != torch.long:
            res = res.to(dtype=torch.long)
        if res.dim() == 1:
            res = res.unsqueeze(0)
        return res
    except Exception:
        # fallback plain
        txt = ""
        for m in messages:
            txt += f"{m['role'].upper()}:\n{m['content']}\n\n"
        txt += "ASSISTANT:\n"
        res = tok(txt, return_tensors="pt").input_ids
        if res.dtype != torch.long:
            res = res.to(dtype=torch.long)
        if res.dim() == 1:
            res = res.unsqueeze(0)
        return res

def pack_memory_slots(
    memory: Dict[str, str],
    slots: List[str],
    slot_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    min_confidence: float = 0.0,
) -> str:
    parts = []
    fallback_task = ""
    for s in slots:
        v = (memory.get(s) or "").strip()
        if not v:
            continue
        if s == "task":
            fallback_task = v
        if slot_meta is not None and float(min_confidence) > 0.0:
            conf = float((slot_meta.get(s, {}) or {}).get("confidence", 0.0))
            if conf < float(min_confidence):
                continue
        parts.append(f"[{s.upper()}]\n{v}")
    if not parts and fallback_task:
        # Keep at least current objective in prompt when confidence filtering is too strict.
        parts.append(f"[TASK]\n{fallback_task}")
    return "\n\n".join(parts).strip()

def build_system_text(
    base_system: Optional[str],
    memory: Dict[str, str],
    slots: List[str],
    slot_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    min_confidence: float = 0.0,
) -> Optional[str]:
    guard = (
        "Important: Do not mention internal memory slots/extraction/batons or this script unless the user explicitly asks. "
        "Never respond as a 'memory slot extractor'; answer naturally as a helpful assistant."
    )
    mem = pack_memory_slots(memory, slots, slot_meta=slot_meta, min_confidence=float(min_confidence))
    runtime_directives = build_runtime_directives(memory)
    parts = []
    if base_system and base_system.strip():
        parts.append(base_system.strip())
    if mem:
        parts.append(mem)
    if runtime_directives:
        parts.append(runtime_directives)
    parts.append(guard)
    s = "\n\n".join([p for p in parts if p]).strip()
    return s if s else None

_ANTI_EMOTION_ASSUMPTION_HINTS = [
    "dont assume",
    "don't assume",
    "stop assumption",
    "assumption of feelings",
    "judge my emotion",
    "not frustrated",
    "cant stop assumption",
    "can't stop assumption",
    "感情を決めつけ",
    "感情を推測",
    "推測しないで",
]

def has_no_emotion_inference_signal(memory: Dict[str, str]) -> bool:
    blob = "\n".join([
        str(memory.get("rules", "") or ""),
        str(memory.get("task", "") or ""),
        str(memory.get("task_raw", "") or ""),
    ]).lower()
    return any(h in blob for h in _ANTI_EMOTION_ASSUMPTION_HINTS)

def build_runtime_directives(memory: Dict[str, str]) -> Optional[str]:
    directives: List[str] = []
    if has_no_emotion_inference_signal(memory):
        directives.append("Do not infer, label, or paraphrase the user's emotions unless the user explicitly states them.")
        directives.append("Avoid reflective templates like \"It sounds like you're feeling...\".")
        directives.append("When uncertain, ask one short neutral clarification question instead of emotion summaries.")
    if not directives:
        return None
    return "[RUNTIME_DIRECTIVES]\n" + "\n".join(f"- {d}" for d in directives)

def infer_context_window_tokens(model, tok=None, fallback: int = 4096) -> int:
    vals: List[int] = []
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for k in (
            "max_position_embeddings",
            "n_positions",
            "seq_length",
            "max_seq_len",
            "max_sequence_length",
            "context_length",
            "n_ctx",
            "sliding_window",
        ):
            v = getattr(cfg, k, None)
            if isinstance(v, int) and 128 <= v <= 262144:
                vals.append(int(v))
        rope = getattr(cfg, "rope_scaling", None)
        if isinstance(rope, dict):
            for k in ("original_max_position_embeddings", "max_position_embeddings"):
                v = rope.get(k)
                if isinstance(v, int) and 128 <= v <= 262144:
                    vals.append(int(v))

    if tok is not None:
        v = getattr(tok, "model_max_length", None)
        if isinstance(v, int) and 128 <= v <= 262144:
            vals.append(int(v))

    if not vals:
        return int(max(128, fallback))
    return int(max(128, min(vals)))

def resolve_input_budget_tokens(
    model,
    tok,
    max_new_tokens: int,
    max_input_tokens: int,
    safety_tokens: int,
) -> int:
    ctx = infer_context_window_tokens(model, tok=tok, fallback=4096)
    if int(max_input_tokens) > 0:
        return int(max(128, min(int(max_input_tokens), ctx)))
    budget = int(ctx) - int(max_new_tokens) - int(max(16, safety_tokens))
    if budget >= 128:
        return int(budget)
    return int(max(128, ctx // 2))

def _build_generation_messages(
    system_text: Optional[str],
    context_messages: List[Msg],
    user_text: str,
) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system_text:
        msgs.append({"role": "system", "content": system_text})
    for m in context_messages:
        if m.role in ("user", "assistant"):
            msgs.append({"role": m.role, "content": m.content})
    msgs.append({"role": "user", "content": user_text})
    return msgs

@torch.no_grad()
def generate_reply(
    model,
    tok,
    base_system: Optional[str],
    memory: Dict[str, str],
    slots: List[str],
    context_messages: List[Msg],
    user_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    slot_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    slot_confidence_threshold: float = 0.0,
    max_input_tokens: int = 0,
    input_safety_tokens: int = 96,
    slot_runtime_max_chars: int = 900,
) -> str:
    dev = next(model.parameters()).device
    context_work = list(context_messages or [])
    mem_work = {k: (memory.get(k, "") if memory else "") for k in slots}
    budget = resolve_input_budget_tokens(
        model, tok,
        max_new_tokens=int(max_new_tokens),
        max_input_tokens=int(max_input_tokens),
        safety_tokens=int(input_safety_tokens),
    )

    # Keep generation stable on smaller local models by fitting prompt to a token budget:
    # 1) trim oldest raw context, 2) trim slot text, 3) drop lowest-priority slots.
    input_ids = None
    for _ in range(16):
        system_text = build_system_text(
            base_system, mem_work, slots,
            slot_meta=slot_meta,
            min_confidence=float(slot_confidence_threshold),
        )
        msgs = _build_generation_messages(system_text, context_work, user_text)
        cand_ids = build_input_ids(tok, msgs).to(dev)
        if int(cand_ids.shape[1]) <= budget:
            input_ids = cand_ids
            break

        shrunk = False
        if context_work:
            drop_n = max(1, len(context_work) // 3)
            context_work = context_work[drop_n:]
            shrunk = True
            continue

        if int(slot_runtime_max_chars) > 0:
            changed = False
            limit = int(slot_runtime_max_chars)
            for s in slots:
                v = (mem_work.get(s) or "").strip()
                if len(v) > limit:
                    mem_work[s] = v[-limit:]
                    changed = True
            if changed:
                slot_runtime_max_chars = max(160, limit // 2)
                shrunk = True
                continue

        non_empty = [s for s in slots if (mem_work.get(s) or "").strip()]
        for s in reversed(non_empty):
            if s not in ("task", "rules", "persona"):
                mem_work[s] = ""
                shrunk = True
                break
        if not shrunk:
            for s in ("persona", "rules"):
                if s in non_empty:
                    mem_work[s] = ""
                    shrunk = True
                    break

        if not shrunk:
            input_ids = cand_ids[:, -budget:] if int(cand_ids.shape[1]) > budget else cand_ids
            break

    if input_ids is None:
        system_text = build_system_text(
            base_system, mem_work, slots,
            slot_meta=slot_meta,
            min_confidence=float(slot_confidence_threshold),
        )
        msgs = _build_generation_messages(system_text, context_work, user_text)
        input_ids = build_input_ids(tok, msgs).to(dev)
        if int(input_ids.shape[1]) > budget:
            input_ids = input_ids[:, -budget:]

    attn = torch.ones_like(input_ids, dtype=torch.long)

    do_sample = temperature > 0
    if not do_sample:
        temperature = 1.0
        top_p = 1.0

    logits_processor = LogitsProcessorList([SanitizeLogitsProcessor(tok.eos_token_id)])
    try:
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=float(temperature),
            top_p=float(top_p),
            logits_processor=logits_processor,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
    except RuntimeError as e:
        msg = str(e)
        if "probability tensor contains either" in msg or "multinomial" in msg:
            # Sampling can become numerically unstable on some backends (e.g., MPS/fp16).
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                logits_processor=logits_processor,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )
        else:
            raise
    new_ids = out[0, input_ids.shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True).strip()


# =============================
# Slot extraction
# =============================
def heuristic_extract_slots(history: List[Msg], slots: List[str], task_last_messages: int = 8) -> Dict[str, str]:
    persona_lines = []
    rules_lines = []
    rule_kw = ["must","should","don't","do not","never","always","ルール","必ず","禁止","しないで","従って","フォーマット","形式","条件"]
    persona_kw = [
        "name", "my name", "名前",
        "私は", "ぼくは", "俺は",
        "I am", "I'm", "im ",
        "I like", "I love", "I prefer", "I enjoy", "I hate",
        "i like", "i love", "i prefer", "i enjoy", "i hate",
        "i'm into", "im into",
        "好み", "好き", "嫌い",
        "role", "職", "仕事",
    ]

    user_msgs = [m for m in history if m.role == "user"]
    sys_msgs = [m for m in history if m.role == "system"]

    # Persona: only from user-provided self-descriptions (avoid greeting noise)
    for m in user_msgs:
        for ln in m.content.splitlines():
            if any(k.lower() in ln.lower() for k in persona_kw):
                persona_lines.append(ln.strip())

    for m in sys_msgs:
        rules_lines.append(m.content.strip())

    # Rules: prefer system/user directives (avoid assistant self-description pollution)
    for m in user_msgs:
        for ln in m.content.splitlines():
            if any(k.lower() in ln.lower() for k in rule_kw):
                rules_lines.append(ln.strip())

    tail_user = user_msgs[-task_last_messages:] if task_last_messages > 0 else user_msgs
    task_text = "\n".join([m.content.strip() for m in tail_user if m.content.strip()]).strip()

    out = {s: "" for s in slots}
    if "persona" in out:
        out["persona"] = "\n".join(_dedup_keep_order([x for x in persona_lines if x])).strip()
    if "rules" in out:
        out["rules"] = "\n".join(_dedup_keep_order([x for x in rules_lines if x])).strip()
    if "task" in out:
        out["task"] = task_text
    return out

def extract_json_object_loose(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    cand = text[start:end+1].strip()
    cand = re.sub(r",\s*([}\]])", r"\1", cand)
    cand = cand.replace("“","\"").replace("”","\"").replace("’","'")
    try:
        return json.loads(cand)
    except Exception:
        if cand.count("'") > cand.count("\""):
            cand2 = re.sub(r"'", "\"", cand)
            cand2 = re.sub(r",\s*([}\]])", r"\1", cand2)
            try:
                return json.loads(cand2)
            except Exception:
                return None
        return None

def parse_sectioned_slots(text: str, slots: List[str]) -> Optional[Dict[str, str]]:
    """
    Parse sectioned text like:
      [PERSONA]
      ...
      [RULES]
      ...
    """
    if not text:
        return None
    allow = {s.lower(): s for s in slots}
    heading = re.compile(r"^\s*\[([A-Za-z0-9_]+)\]\s*$")
    cur: Optional[str] = None
    bucket: Dict[str, List[str]] = {s: [] for s in slots}
    seen = set()

    for ln in text.splitlines():
        m = heading.match(ln.strip())
        if m:
            k = m.group(1).strip().lower()
            cur = allow.get(k)
            if cur is not None:
                seen.add(cur)
            continue
        if cur is not None:
            bucket[cur].append(ln.rstrip())

    if not seen:
        return None
    out = {s: "\n".join(bucket[s]).strip() for s in slots}
    return out

def parse_tagged_slot_object(text: str, slots: List[str]) -> Optional[Dict[str, str]]:
    """
    Parse strict tagged format:
      <<<persona>>>
      ...
      <<</persona>>>
    """
    if not text:
        return None
    out: Dict[str, str] = {}
    hit = 0
    for s in slots:
        pat = re.compile(
            r"<<<\s*" + re.escape(s) + r"\s*>>>(.*?)<<<\s*/\s*" + re.escape(s) + r"\s*>>>",
            re.IGNORECASE | re.DOTALL,
        )
        m = pat.search(text)
        if m:
            out[s] = m.group(1).strip()
            hit += 1
        else:
            out[s] = ""
    if hit <= 0:
        return None
    return out

def parse_slots_loose(text: str, slots: List[str]) -> Optional[Dict[str, str]]:
    """
    Multi-strategy parser:
      1) strict tagged blocks
      2) JSON object
      3) [SECTION] blocks
    """
    tagged = parse_tagged_slot_object(text, slots)
    if tagged is not None:
        return tagged

    obj = extract_json_object_loose(text)
    if isinstance(obj, dict):
        res: Dict[str, str] = {}
        for s in slots:
            v = obj.get(s, "")
            if not isinstance(v, str):
                v = json.dumps(v, ensure_ascii=False)
            res[s] = v.strip()
        return res

    sec = parse_sectioned_slots(text, slots)
    if sec is not None:
        return sec
    return None

def normalize_slots(mem: Dict[str, str], slots: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    rule_kw = [
        "must", "should", "do not", "don't", "never", "always",
        "format", "output", "return", "respond", "response", "use", "please",
        "rule", "rules",
        "assum", "emotion", "feeling", "judge", "manipulat",
        "ルール", "必ず", "禁止", "しないで", "従って", "フォーマット", "形式", "条件", "してください",
        "感情", "推測", "決めつけ", "前提",
    ]
    greet = {"hi", "hello", "hey", "thanks", "thank you", "thx", "ok", "okay", "cool", "nicely done", "good job"}
    always_verbs = {"respond", "reply", "answer", "output", "return", "format", "use", "write", "speak", "avoid", "include", "exclude", "be"}

    for k in slots:
        t = (mem.get(k) or "").strip()

        # Remove role prefixes (line-wise)
        t = re.sub(r"(?im)^\s*(assistant|user|system)\s*[:：]\s*", "", t).strip()

        # Normalize common placeholders
        if t.lower() in ("none", "null", "n/a", "na"):
            t = ""

        if k == "persona":
            lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
            t = (lines[0] if lines else "").strip()
            if t.lower() in ("assistant", "system", "user"):
                t = ""
            if t.lower() in greet:
                t = ""
            if re.match(r"(?i)^\s*(hi|hello|hey|thanks|thank you|thx)\b", t):
                t = ""
            if "memory slot extractor" in t.lower():
                t = ""
            t = t[:200]

        if k == "rules":
            t = strip_reflective_template_lines(t)
            kept = []
            for ln in t.splitlines():
                s = ln.strip()
                low = s.lower()
                if re.match(r"^\s*[-*•]\s+", s) or re.match(r"^\s*\d+[.)]\s+", s):
                    kept.append(s)
                    continue
                if "memory slot extractor" in low or "return only one json" in low or "exact keys" in low:
                    continue
                if low.startswith(("i ", "i'm ", "im ", "as an ai", "as a machine")):
                    # Usually self-description; not a user rule.
                    continue
                if "always" in low and not any(v in low for v in always_verbs):
                    continue
                if any(x in low for x in rule_kw):
                    kept.append(s)
            t = "\n".join(_dedup_keep_order([x for x in kept if x])).strip()[:800]
            if t.lower() in ("none", "null"):
                t = ""

        if k in ("task", "task_sum"):
            t = re.sub(r"(?im)^i understand[^\n]*\n?", "", t).strip()
            t = strip_reflective_template_lines(t)
            # Keep tail: task is "current objective / recent context"
            if len(t) > 800:
                t = t[-800:]

        if k == "task_raw":
            t = strip_reflective_template_lines(t)
            # Raw should stay user-centric and bounded (kept out of slot_only injection by default).
            if len(t) > 1400:
                t = t[-1400:]

        out[k] = t.strip()

    return out

# -----------------------------
# Slot gating + merge policy
# -----------------------------
_BAD_ROLE_TAG = re.compile(r"\b(USER|ASSISTANT|SYSTEM)\s*[:：]\s*", re.IGNORECASE)
_BAD_META = re.compile(
    r"\b(as an ai|i am a machine|context window|summari(z|s)ation|i don't have feelings|memory slot extractor)\b",
    re.IGNORECASE,
)

_SLOT_BUDGET_CHARS: Dict[str, Tuple[int, int]] = {
    # Keep mins low to support short JP constraints like「日本語で」.
    "persona": (2, 240),
    "rules": (2, 900),
    "task": (1, 900),
    "task_sum": (1, 900),
    "task_raw": (1, 1400),
}

_TASK_CONSTRAINT_HINTS = [
    "must", "should", "don't", "do not", "never", "always", "need to", "needs to",
    "format", "output", "return", "respond", "json", "yaml", "csv",
    "deadline", "due", "by ", "within", "at least", "at most", "exactly",
    "必ず", "禁止", "しないで", "従って", "条件", "要件", "形式", "期限", "まで", "以内", "以上", "以下", "優先",
]

def slot_candidate_ok(slot: str, cand: str) -> bool:
    t = (cand or "").strip()
    if not t:
        return False
    mn, mx = _SLOT_BUDGET_CHARS.get(slot, (1, 4000))
    if len(t) < mn or len(t) > mx:
        return False
    if _BAD_ROLE_TAG.search(t):
        return False
    if slot in ("persona", "rules") and _BAD_META.search(t):
        return False
    return True

def _split_task_fragments(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    tmp = re.sub(r"(?im)^\s*(goal|constraints?|open)\s*:\s*", "", t)
    parts: List[str] = []
    for ln in tmp.splitlines():
        s = ln.strip()
        if not s:
            continue
        chunks = [x.strip() for x in s.split(" / ") if x.strip()]
        if chunks:
            parts.extend(chunks)
        else:
            parts.append(s)
    return _dedup_keep_order(parts)

def _looks_task_constraint(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    low = t.lower()
    if any(k in low for k in _TASK_CONSTRAINT_HINTS):
        return True
    if re.search(r"\d", t):
        return True
    if ("?" in t) or ("？" in t):
        return True
    return False

def merge_task_snapshots(prev_task: str, cand_task: str, max_chars: int = 900) -> str:
    pv = (prev_task or "").strip()
    cv = (cand_task or "").strip()
    if not pv:
        return cv
    if not cv:
        return pv

    cur = _split_task_fragments(cv)
    old = _split_task_fragments(pv)
    merged = list(cur)
    for part in old:
        if part in merged:
            continue
        if _looks_task_constraint(part) or len(merged) < 8:
            merged.append(part)
        if len(" / ".join(merged)) >= max_chars:
            break

    out = " / ".join(_dedup_keep_order(merged)).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rstrip(" /")
    return out

def merge_slots(prev: Dict[str, str], cand: Dict[str, str], slots: List[str]) -> Dict[str, str]:
    """
    Conservative merge:
      - persona/rules: only overwrite when candidate passes gating
      - task(+ task_sum/task_raw): allow overwrite when candidate is sane; otherwise keep previous

    The caller should normalize `cand` before merging.
    """
    out: Dict[str, str] = dict(prev or {})
    for s in slots:
        pv = (out.get(s) or "").strip()
        cv = (cand.get(s) or "").strip()
        if not cv:
            continue

        if s in ("persona", "rules"):
            # persona/rules never bypass slot gating.
            if not slot_candidate_ok(s, cv):
                continue
        else:
            # task/task_sum/task_raw can be bootstrapped once when previous is empty.
            if not (slot_candidate_ok(s, cv) or (not pv and not _BAD_ROLE_TAG.search(cv))):
                continue

        if s == "task" and pv:
            _, mx = _SLOT_BUDGET_CHARS.get("task", (1, 900))
            out[s] = merge_task_snapshots(pv, cv, max_chars=mx)
        else:
            out[s] = cv

    # Preserve extra (non-slot) keys we may keep for debugging/hierarchical task handling.
    for extra in ("task_raw", "task_sum"):
        pv = (out.get(extra) or "").strip()
        cv = (cand.get(extra) or "").strip()
        if cv and slot_candidate_ok(extra, cv):
            out[extra] = cv
        elif pv:
            out[extra] = pv

    return out

def _clip01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v

def init_slot_meta(slots: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for s in slots:
        out[s] = {
            "confidence": 0.0,
            "version": 0,
            "updated_at": None,
            "source": "init",
            "source_turn_ids": [],
        }
    return out

def normalize_slot_meta(slot_meta: Optional[Dict[str, Any]], slots: List[str]) -> Dict[str, Dict[str, Any]]:
    base = init_slot_meta(slots)
    if not isinstance(slot_meta, dict):
        return base
    for s in slots:
        raw = slot_meta.get(s, {})
        if not isinstance(raw, dict):
            continue
        conf = _clip01(raw.get("confidence", 0.0))
        try:
            version = max(0, int(raw.get("version", 0)))
        except Exception:
            version = 0
        source = str(raw.get("source", "unknown") or "unknown")[:64]
        updated_at_raw = raw.get("updated_at", None)
        updated_at = str(updated_at_raw) if updated_at_raw else None
        ids_raw = raw.get("source_turn_ids", [])
        ids: List[int] = []
        if isinstance(ids_raw, list):
            for x in ids_raw:
                try:
                    ids.append(int(x))
                except Exception:
                    continue
        ids = _dedup_keep_order([str(i) for i in ids])[-8:]
        ids_i = [int(x) for x in ids]
        base[s] = {
            "confidence": conf,
            "version": version,
            "updated_at": updated_at,
            "source": source,
            "source_turn_ids": ids_i,
        }
    return base

def estimate_slot_confidences(cand: Dict[str, str], slots: List[str], extractor: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    ex = (extractor or "").strip().lower()
    for s in slots:
        t = (cand.get(s) or "").strip()
        if not t:
            out[s] = 0.0
            continue

        if s == "persona":
            score = 0.78 if ex == "heuristic" else 0.86
        elif s == "rules":
            score = 0.74 if ex == "heuristic" else 0.82
        elif s == "task":
            score = 0.68 if ex == "heuristic" else 0.76
            if "Goal:" in t:
                score += 0.05
            if "Constraints:" in t:
                score += 0.05
            if "Open:" in t:
                score += 0.03
        else:
            score = 0.62 if ex == "heuristic" else 0.70

        if slot_candidate_ok(s, t):
            score += 0.05
        if _BAD_ROLE_TAG.search(t):
            score -= 0.20
        if _BAD_META.search(t):
            score -= 0.25
        if len(t) < 6:
            score -= 0.12

        out[s] = _clip01(score)
    return out

def merge_confidence_value(
    prev_conf: float,
    incoming_conf: float,
    mode: str = "balanced",
    decay: float = 0.55,
) -> float:
    prev = _clip01(prev_conf)
    inc = _clip01(incoming_conf)
    d = _clip01(decay)
    m = (mode or "balanced").strip().lower()

    if m == "replace":
        return inc
    if m == "conservative":
        # Slow updates: keep historical confidence unless incoming is clearly stronger.
        return _clip01(max(inc * 0.70, prev * max(0.70, d)))
    if m == "aggressive":
        # Fast updates: favor newly extracted confidence.
        return _clip01(max(inc, prev * min(0.45, max(0.10, d * 0.75))))
    # balanced(default)
    return _clip01(max(inc, prev * d))

def update_slot_meta_after_merge(
    slot_meta: Dict[str, Dict[str, Any]],
    prev_memory: Dict[str, str],
    new_memory: Dict[str, str],
    cand_conf: Dict[str, float],
    slots: List[str],
    source: str,
    turn_idx: int,
    updated_at: str,
    conf_mode: str = "balanced",
    conf_floor: float = 0.35,
    conf_decay: float = 0.55,
    conf_merge_bonus: float = 0.05,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    meta = normalize_slot_meta(slot_meta, slots)
    updates: Dict[str, Dict[str, Any]] = {}

    for s in slots:
        pv = (prev_memory.get(s) or "").strip()
        nv = (new_memory.get(s) or "").strip()
        m = dict(meta.get(s, {}))
        if not nv:
            m["confidence"] = 0.0
            meta[s] = m
            continue
        if nv == pv:
            continue

        prev_conf = _clip01(m.get("confidence", 0.0))
        floor = _clip01(conf_floor)
        incoming = _clip01(cand_conf.get(s, max(floor, prev_conf * 0.9)))
        if pv and pv in nv:
            incoming = _clip01(max(incoming, min(0.97, prev_conf + _clip01(conf_merge_bonus))))
        conf = merge_confidence_value(
            prev_conf=prev_conf,
            incoming_conf=incoming,
            mode=conf_mode,
            decay=conf_decay,
        )

        try:
            version = int(m.get("version", 0))
        except Exception:
            version = 0
        ids = m.get("source_turn_ids", [])
        if not isinstance(ids, list):
            ids = []
        ids2: List[int] = []
        for x in ids:
            try:
                ids2.append(int(x))
            except Exception:
                continue
        ids2.append(int(turn_idx))
        ids2 = [int(x) for x in _dedup_keep_order([str(i) for i in ids2])[-8:]]

        m["confidence"] = conf
        m["version"] = max(0, version) + 1
        m["updated_at"] = updated_at
        m["source"] = str(source or "unknown")[:64]
        m["source_turn_ids"] = ids2
        meta[s] = m
        updates[s] = m

    return meta, updates

def compute_memory_delta(prev_mem: Dict[str, str], cur_mem: Dict[str, str], slots: List[str]) -> Dict[str, Dict[str, Any]]:
    changes: Dict[str, Dict[str, Any]] = {}
    for s in slots:
        pv = str(prev_mem.get(s, "") or "")
        cv = str(cur_mem.get(s, "") or "")
        if pv == cv:
            continue
        if not cv:
            changes[s] = {"op": "clear"}
            continue
        if pv and cv.startswith(pv):
            tail = cv[len(pv):]
            if tail and len(tail) <= max(320, len(cv) // 2):
                changes[s] = {"op": "append", "text": tail}
                continue
        if pv and cv.endswith(pv):
            head = cv[:-len(pv)]
            if head and len(head) <= max(320, len(cv) // 2):
                changes[s] = {"op": "prepend", "text": head}
                continue
        if s == "task" and pv:
            old_parts = _split_task_fragments(pv)
            new_parts = _split_task_fragments(cv)
            added = [x for x in new_parts if x not in old_parts]
            if added and len(" / ".join(added)) <= max(160, len(cv) // 2):
                changes[s] = {"op": "task_add", "items": added}
                continue
        changes[s] = {"op": "replace", "text": cv}
    return changes

def apply_memory_delta(base_mem: Dict[str, str], changes: Dict[str, Any], slots: List[str]) -> Dict[str, str]:
    out = {s: str(base_mem.get(s, "") or "") for s in slots}
    if not isinstance(changes, dict):
        return out
    for s in slots:
        spec = changes.get(s)
        if not isinstance(spec, dict):
            continue
        op = str(spec.get("op", "replace"))
        if op == "clear":
            out[s] = ""
        elif op == "append":
            out[s] = out.get(s, "") + str(spec.get("text", "") or "")
        elif op == "prepend":
            out[s] = str(spec.get("text", "") or "") + out.get(s, "")
        elif op == "task_add":
            items = spec.get("items", [])
            parts = _split_task_fragments(out.get(s, ""))
            if isinstance(items, list):
                for x in items:
                    sx = str(x).strip()
                    if sx:
                        parts.append(sx)
            out[s] = " / ".join(_dedup_keep_order(parts))
        else:
            out[s] = str(spec.get("text", "") or "")
    return out

def compact_relay_log(path: str, max_entries: int):
    if max_entries <= 0 or (not os.path.exists(path)):
        return
    lines = [ln for ln in read_text(path).splitlines() if ln.strip()]
    if len(lines) <= max_entries:
        return
    objs: List[Dict[str, Any]] = []
    for ln in lines:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if isinstance(obj, dict):
            objs.append(obj)
    if len(objs) <= max_entries:
        return

    start = max(0, len(objs) - max_entries)
    key_idx = None
    for i in range(start, -1, -1):
        if str(objs[i].get("kind", "")) == "keyframe":
            key_idx = i
            break
    if key_idx is None:
        key_idx = start
    kept = objs[key_idx:]
    if len(kept) > max_entries:
        kept = kept[-max_entries:]
    write_text(path, "\n".join(json.dumps(x, ensure_ascii=False) for x in kept) + "\n")

def append_relay_event(path: str, event: Dict[str, Any], max_entries: int):
    append_jsonl(path, event)
    compact_relay_log(path, max_entries=max_entries)

def load_relay_state(path: str, slots: List[str]) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, Dict[str, Any]]]]:
    if not os.path.exists(path):
        return None, None
    lines = [ln for ln in read_text(path).splitlines() if ln.strip()]
    if not lines:
        return None, None
    events: List[Dict[str, Any]] = []
    for ln in lines:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    if not events:
        return None, None

    start = 0
    for i, ev in enumerate(events):
        if str(ev.get("kind", "")) == "keyframe":
            start = i
    mem = {s: "" for s in slots}
    meta = init_slot_meta(slots)

    for ev in events[start:]:
        kind = str(ev.get("kind", ""))
        if kind == "keyframe":
            m = ev.get("memory", {})
            if isinstance(m, dict):
                mem = {s: str(m.get(s, "") or "") for s in slots}
            sm = ev.get("slot_meta", {})
            meta = normalize_slot_meta(sm if isinstance(sm, dict) else {}, slots)
            continue
        if kind != "delta":
            continue
        changes = ev.get("changes", {})
        if isinstance(changes, dict):
            mem = apply_memory_delta(mem, changes, slots)
        smu = ev.get("slot_meta_updates", {})
        if isinstance(smu, dict):
            for s in slots:
                if s not in smu or not isinstance(smu[s], dict):
                    continue
                one = normalize_slot_meta({s: smu[s]}, [s])
                meta[s] = one[s]

    mem = normalize_slots(mem, slots)
    if not any((mem.get(s) or "").strip() for s in slots):
        return None, None
    return mem, normalize_slot_meta(meta, slots)

_SCRIPT_INVOKE_LINE = re.compile(r"^\s*python\d*(\s+-\S+)*\s+sr_memory_capsule\.py\b", re.IGNORECASE)
_SHELL_PROMPT_LINE = re.compile(r"^\s*[^@\s]+@[^%\s]+\s+.*%\s+", re.IGNORECASE)
_CODE_FENCE_LINE = re.compile(r"^\s*```")
_REFLECTIVE_TEMPLATE_LINE = re.compile(
    r"(it sounds like you('| a)re feeling|you('| a)re saying that|you('| a)re pointing out that|is that (right|correct|a fair summary))",
    re.IGNORECASE,
)
_QUOTE_PREFIX_LINE = re.compile(r"^\s*>\s*")

def strip_command_like_lines(text: str) -> str:
    """
    Remove common accidental log/command lines from user-provided text blocks.
    Keep it conservative: only strip obvious shell prompts or `sr_memory_capsule.py` invocations.
    """
    if not text:
        return ""
    out_lines: List[str] = []
    in_fence = False
    for ln in text.splitlines():
        if _CODE_FENCE_LINE.match(ln):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if _SCRIPT_INVOKE_LINE.match(ln):
            continue
        if _SHELL_PROMPT_LINE.match(ln):
            continue
        out_lines.append(ln)
    return "\n".join(out_lines).strip()

def strip_reflective_template_lines(text: str) -> str:
    if not text:
        return ""
    out_lines: List[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        t = _QUOTE_PREFIX_LINE.sub("", s)
        if _REFLECTIVE_TEMPLATE_LINE.search(t):
            continue
        out_lines.append(s)
    return "\n".join(out_lines).strip()

def summarize_task_from_raw(task_raw: str, max_chars: int = 520) -> str:
    """
    Produce a compact "current objective" with enough detail for memory relay:
    goal + constraints + open questions.
    """
    t = strip_command_like_lines(task_raw)
    if not t:
        return ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return ""

    greet = {"hi", "hello", "hey", "yo", "thanks", "thank you", "thx", "ok", "okay"}
    def is_greeting_line(s: str) -> bool:
        low = s.lower().strip(" !?.")
        return (low in greet) or low.startswith(("hi ", "hello ", "hey "))

    goal = ""
    constraints_rev: List[str] = []
    open_rev: List[str] = []

    for ln in reversed(lines):
        if is_greeting_line(ln):
            continue
        if len(ln) < 4:
            continue
        if (not goal) and len(ln) >= 8:
            goal = ln
        if ("?" in ln or "？" in ln) and len(open_rev) < 2:
            open_rev.append(ln)
        if _looks_task_constraint(ln) and len(constraints_rev) < 4:
            constraints_rev.append(ln)
        if goal and len(open_rev) >= 2 and len(constraints_rev) >= 4:
            break

    constraints = list(reversed(_dedup_keep_order(list(reversed(constraints_rev)))))
    open_items = list(reversed(_dedup_keep_order(list(reversed(open_rev)))))
    parts: List[str] = []
    if goal:
        parts.append(f"Goal: {goal}")
    if constraints:
        parts.append("Constraints: " + " / ".join(constraints[:3]))
    if open_items:
        parts.append("Open: " + " / ".join(open_items[:2]))
    if not parts:
        parts.append(lines[-1])

    s = "\n".join(parts).strip()
    if len(s) > max_chars:
        s = s[:max_chars].rstrip()
    return s

@torch.no_grad()
def llm_extract_slots(
    model,
    tok,
    history_text: str,
    slots: List[str],
    max_new_tokens: int = 400,
    retries: int = 0,
    debug: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, str]]:
    slot_list = ", ".join(slots)
    slots_block = "\n".join([f"<<<{s}>>>\n...\n<<</{s}>>>" for s in slots])
    instruction_tagged = (
        "You are a memory slot extractor.\n"
        f"Extract the conversation memory into these slots: {slot_list}.\n"
        "Return ONLY tagged blocks, in this exact format:\n"
        f"{slots_block}\n"
        "Rules:\n"
        "- Keep concise factual memory only\n"
        "- If a slot is empty, keep the block with no content\n"
        "- No JSON, no markdown fences, no explanations\n"
    )
    instruction_json = (
        "You are a memory slot extractor.\n"
        f"Extract the conversation memory into these slots: {slot_list}.\n"
        "Return ONLY one JSON object with EXACT keys and string values.\n"
        "No extra text.\n"
    )

    if debug is not None:
        debug.clear()
        debug.update({"tagged_attempts": 0, "json_attempts": 0, "mode": None, "parsed": False})

    def _run_once(instruction: str, tail: str) -> Optional[Dict[str, str]]:
        try:
            msgs = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": tail},
            ]
            input_ids = build_input_ids(tok, msgs)
        except Exception:
            prompt = instruction + "\n\n" + tail
            input_ids = tok(prompt, return_tensors="pt").input_ids

        input_ids = input_ids.to(next(model.parameters()).device)
        attn = torch.ones_like(input_ids, dtype=torch.long)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
        new_ids = out[0, input_ids.shape[1]:]
        txt = tok.decode(new_ids, skip_special_tokens=True).strip()
        return parse_slots_loose(txt, slots)

    tagged_tail = (
        "Conversation:\n"
        f"```text\n{history_text}\n```\n\n"
        "Write tagged slot blocks now."
    )
    json_tail = (
        "Conversation:\n"
        f"```text\n{history_text}\n```\n\n"
        "JSON:"
    )

    n = max(1, int(retries) + 1)
    for _ in range(n):
        if debug is not None:
            debug["tagged_attempts"] = int(debug.get("tagged_attempts", 0)) + 1
            debug["mode"] = "tagged"
        res = _run_once(instruction_tagged, tagged_tail)
        if res is not None:
            if debug is not None:
                debug["parsed"] = True
            return res
    for _ in range(max(1, n // 2)):
        if debug is not None:
            debug["json_attempts"] = int(debug.get("json_attempts", 0)) + 1
            debug["mode"] = "json"
        res = _run_once(instruction_json, json_tail)
        if res is not None:
            if debug is not None:
                debug["parsed"] = True
            return res
    return None


# =============================
# Gateway helpers
# =============================
def heuristic_summarize_for_inject(text: str, max_chars: int = 2000) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = t[-max_chars:]
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    rule_kw = ["must","should","don't","do not","never","always","ルール","必ず","禁止","しないで","従って","フォーマット","形式","条件"]
    picks = [ln for ln in lines if any(k.lower() in ln.lower() for k in rule_kw)]
    picks = _dedup_keep_order(picks)[:40]
    head = "[-] Restored log summary (heuristic)\n"
    body = "\n".join(f"- {p}" for p in picks[:30])
    tail = "\n\n[-] Recent tail\n" + "\n".join(lines[-10:])
    out = (head + (body if body else "- (no explicit constraints found)") + tail).strip()
    return out[:max_chars]

@torch.no_grad()
def llm_summarize_for_inject(
    model,
    tok,
    text: str,
    max_new_tokens: int = 220,
    target_chars: int = 1500,
) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = t[-8000:]
    instruction = (
        "Summarize the conversation log into a compact memory injection.\n"
        "Format:\n"
        "[PERSONA]\n...\n[RULES]\n...\n[TASK]\n...\n[FACTS]\n...\n"
        f"Keep it under ~{target_chars} characters. No preamble.\n"
    )
    try:
        msgs = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"LOG:\n```text\n{t}\n```\n\nWrite the memory injection now."},
        ]
        input_ids = build_input_ids(tok, msgs)
    except Exception:
        prompt = instruction + "\n\n" + t + "\n\nMEMORY INJECTION:\n"
        input_ids = tok(prompt, return_tensors="pt").input_ids

    input_ids = input_ids.to(next(model.parameters()).device)
    attn = torch.ones_like(input_ids, dtype=torch.long)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    new_ids = out[0, input_ids.shape[1]:]
    s = tok.decode(new_ids, skip_special_tokens=True).strip()
    if len(s) > target_chars:
        s = s[:target_chars]
    return s

def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return float(F.cosine_similarity(a, b, dim=0).item())

def gateway_retrieve(store_path: str, query_baton: Dict[str, str], slots: List[str], top_k: int, device: torch.device) -> List[Dict[str, Any]]:
    if not os.path.exists(store_path):
        return []
    q_parts = [unpack_smem1_int8(query_baton[s], device=device) for s in slots if s in query_baton]
    if not q_parts:
        return []
    qv = torch.cat(q_parts)
    cands = []
    for ln in read_text(store_path).splitlines():
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        b = obj.get("baton", {})
        try:
            v_parts = [unpack_smem1_int8(b[s], device=device) for s in slots if s in b]
            if not v_parts:
                continue
            v = torch.cat(v_parts)
            sim = cosine(qv, v)
            cands.append((sim, obj))
        except Exception:
            continue
    cands.sort(key=lambda x: x[0], reverse=True)
    return [o for _, o in cands[:top_k]]

def trim_inject(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if max_chars > 0 and len(t) > max_chars:
        return t[-max_chars:]
    return t

def build_external_prompt(base_system: Optional[str], inject_text: str, user: str) -> str:
    sys_part = (base_system or "").strip()
    if inject_text.strip():
        sys_part = (sys_part + "\n\n" + inject_text).strip() if sys_part else inject_text.strip()
    return f"SYSTEM:\n{sys_part}\n\nUSER:\n{user}\n\nASSISTANT:\n"

def build_openai_chat_completions_request(
    *,
    model: str,
    system_text: str,
    user_text: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    prefer_developer_role: bool = True,
) -> dict:
    role = "developer" if prefer_developer_role else "system"
    messages = []
    if (system_text or "").strip():
        messages.append({"role": role, "content": system_text})
    messages.append({"role": "user", "content": user_text})

    return {
        "method": "POST",
        "url": "https://api.openai.com/v1/chat/completions",
        "headers": {
            "Authorization": "Bearer $OPENAI_API_KEY",
            "Content-Type": "application/json",
        },
        "json": {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        },
    }

def build_openai_responses_request(
    *,
    model: str,
    instructions: str,
    user_text: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
) -> dict:
    return {
        "method": "POST",
        "url": "https://api.openai.com/v1/responses",
        "headers": {
            "Authorization": "Bearer $OPENAI_API_KEY",
            "Content-Type": "application/json",
        },
        "json": {
            "model": model,
            "instructions": instructions,
            "input": user_text,
            "max_output_tokens": int(max_output_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        },
    }

def build_anthropic_messages_request(
    *,
    model: str,
    system_text: str,
    user_text: str,
    max_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    anthropic_version: str = "2023-06-01",
) -> dict:
    body = {
        "model": model,
        "max_tokens": int(max_tokens),
        "system": system_text,
        "messages": [{"role": "user", "content": user_text}],
    }
    if temperature is not None:
        body["temperature"] = float(temperature)
    if top_p is not None:
        body["top_p"] = float(top_p)

    return {
        "method": "POST",
        "url": "https://api.anthropic.com/v1/messages",
        "headers": {
            "x-api-key": "$ANTHROPIC_API_KEY",
            "anthropic-version": anthropic_version,
            "content-type": "application/json",
        },
        "json": body,
    }


# =============================
# check-model command
# =============================
def cmd_check_model(args):
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    mp = resolve_local_model_dir(args.model_path)
    print("[ok] resolved model dir:", mp)
    print("[env] HF_HUB_OFFLINE=", os.environ.get("HF_HUB_OFFLINE"))
    print("[env] TRANSFORMERS_OFFLINE=", os.environ.get("TRANSFORMERS_OFFLINE"))
    # quick file check
    essentials = ["config.json"]
    found = {x: os.path.exists(os.path.join(mp, x)) for x in essentials}
    print("[files]", found)
    tok, model = load_local_llm(mp, device=device, dtype=dtype, use_fast=not args.slow_tokenizer)
    print("[ok] tokenizer:", type(tok).__name__)
    print("[ok] model:", type(model).__name__)
    print("[ok] hidden_size:", getattr(model.config, "hidden_size", None))
    print("[ok] eos_token_id:", tok.eos_token_id)


# =============================
# capsule commands
# =============================
def cmd_capsule_encode(args):
    device = torch.device("cpu")
    proj_cache: Dict[Tuple[str, str, int, int], torch.Tensor] = {}
    embed_backend = normalize_embed_backend(args.embed_backend)

    text = read_text(args.in_path) if args.in_path else sys.stdin.read()
    slot = args.slot or "default"
    baton = encode_baton_hashproj(
        text=text,
        baton_dim=int(args.proj_dim),
        proj_seed=int(args.proj_seed),
        device=device,
        proj_cache=proj_cache,
        slot=slot,
        embed_backend=embed_backend,
    )
    cold = make_cold_archive(text, store=bool(args.store_archive), passphrase=args.passphrase)

    cap = Capsule(
        created_at=now_str(),
        hot_baton_b64=baton,
        hot_meta={
            "format": "SMEM1",
            "embed_backend": embed_backend,
            "proj_dim": int(args.proj_dim),
            "proj_seed": int(args.proj_seed),
            "slot": slot,
        },
        cold_archive=cold,
        notes={"aes_available": bool(AESGCM is not None)},
    )
    write_json(args.out_path, dataclasses.asdict(cap), indent=2)
    print(f"[ok] wrote capsule -> {args.out_path}")
    print(f"[hot] b64chars={len(baton)}")
    if cold:
        print(f"[cold] enc={cold.get('enc')} b64chars={len(cold.get('ct_b64',''))}")
    else:
        print("[cold] disabled")

def cmd_capsule_decode(args):
    cap = Capsule(**json.loads(read_text(args.capsule)))
    # HOT preview
    z = unpack_smem1_int8(cap.hot_baton_b64, device=torch.device("cpu"))
    recovered = recover_cold_text(cap.cold_archive, passphrase=args.passphrase)
    out = {
        "hot_vector_dim": int(z.numel()),
        "hot_vector_preview": z[:8].tolist(),
        "cold_recovered_text": recovered,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

def cmd_capsule_render(args):
    cap = Capsule(**json.loads(read_text(args.capsule)))
    print(render_baton_tagged(cap.hot_baton_b64, chunk=int(args.chunk)))


# =============================
# chat repl command
# =============================
HELP = """Commands:
  /help
  /exit
  /save
  /slots
  /slots_meta
  /relay_state
  /conf show
  /conf mode balanced|conservative|aggressive|replace
  /conf threshold <0..1>
  /capsule
  /mode full|slot|slot_only
  /recent N
  /extract heuristic|llm
  /extract_now
  /system show
  /system set <text>
"""

def render_transcript_txt(base_system: Optional[str], messages: List[Msg]) -> str:
    parts = []
    if base_system:
        parts.append("system: " + base_system.strip())
    for m in messages:
        parts.append(f"{m.role}: {m.content}".strip())
    return "\n\n".join(parts).strip() + "\n"

def cmd_chat_repl(args):
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)

    slots = [s.strip() for s in args.slots.split(",") if s.strip()]
    session_dir = args.session_dir or os.path.join("sessions", time.strftime("%Y%m%d_%H%M%S"))
    ensure_dir(session_dir)

    path_transcript = os.path.join(session_dir, "transcript.txt")
    path_session = os.path.join(session_dir, "session.json")
    path_capsule = os.path.join(session_dir, "capsule_latest.json")
    path_train = os.path.join(session_dir, "train.jsonl")
    path_relay = os.path.join(session_dir, "memory_relay.jsonl")

    base_system = None
    if args.system_file:
        base_system = read_text(args.system_file).strip() or None
    elif args.system_text is not None:
        base_system = args.system_text.strip() or None

    messages: List[Msg] = []
    config = {
        "context_mode": args.context_mode,
        "recent_messages": args.recent_messages,
        "extractor": args.extractor,
        "embed_backend": normalize_embed_backend(args.embed_backend),
        "extract_llm_every": args.extract_llm_every,
        "extract_history_max_chars": args.extract_history_max_chars,
        "extract_llm_max_new_tokens": args.extract_llm_max_new_tokens,
        "task_last_messages": args.task_last_messages,
        "slots": slots,
        "baton_dim": args.baton_dim,
        "proj_seed": args.proj_seed,
        "max_input_tokens": args.max_input_tokens,
        "input_safety_tokens": args.input_safety_tokens,
        "slot_runtime_max_chars": args.slot_runtime_max_chars,
        "slot_confidence_threshold": args.slot_confidence_threshold,
        "slot_conf_mode": args.slot_conf_mode,
        "slot_conf_floor": args.slot_conf_floor,
        "slot_conf_decay": args.slot_conf_decay,
        "slot_conf_merge_bonus": args.slot_conf_merge_bonus,
        "relay_keyframe_every": args.relay_keyframe_every,
        "relay_max_entries": args.relay_max_entries,
    }

    # resume
    if args.resume and os.path.exists(os.path.join(args.resume, "session.json")):
        session_dir = args.resume
        path_transcript = os.path.join(session_dir, "transcript.txt")
        path_session = os.path.join(session_dir, "session.json")
        path_capsule = os.path.join(session_dir, "capsule_latest.json")
        path_train = os.path.join(session_dir, "train.jsonl")
        path_relay = os.path.join(session_dir, "memory_relay.jsonl")
        obj = json.loads(read_text(path_session))
        base_system = obj.get("base_system", base_system)
        messages = [Msg(**m) for m in obj.get("messages", [])]
        config.update(obj.get("config", {}))

    config.setdefault("slot_confidence_threshold", args.slot_confidence_threshold)
    config.setdefault("slot_conf_mode", args.slot_conf_mode)
    config.setdefault("slot_conf_floor", args.slot_conf_floor)
    config.setdefault("slot_conf_decay", args.slot_conf_decay)
    config.setdefault("slot_conf_merge_bonus", args.slot_conf_merge_bonus)
    config.setdefault("relay_keyframe_every", args.relay_keyframe_every)
    config.setdefault("relay_max_entries", args.relay_max_entries)
    config["embed_backend"] = normalize_embed_backend(config.get("embed_backend"))

    tok, model = load_local_llm(args.model_path, device=device, dtype=dtype, use_fast=not args.slow_tokenizer)
    proj_cache: Dict[Tuple[str, str, int, int], torch.Tensor] = {}
    turn_counter = 0
    slot_state = {s: "" for s in slots}
    slot_meta_state = init_slot_meta(slots)
    last_meta_updates: Dict[str, Dict[str, Any]] = {}
    last_merge_source = "init"

    ctx_limit = infer_context_window_tokens(model, tok=tok, fallback=4096)
    effective_budget = resolve_input_budget_tokens(
        model, tok,
        max_new_tokens=int(args.max_new_tokens),
        max_input_tokens=int(config.get("max_input_tokens", 0)),
        safety_tokens=int(config.get("input_safety_tokens", 96)),
    )

    def _try_load_capsule_state() -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, Dict[str, Any]]]]:
        if not os.path.exists(path_capsule):
            return None, None
        try:
            cap_obj = json.loads(read_text(path_capsule))
        except Exception:
            return None, None
        mem_obj = cap_obj.get("memory")
        if not isinstance(mem_obj, dict):
            return None, None
        mem = {s: str(mem_obj.get(s, "") or "") for s in slots}
        mem = normalize_slots(mem, slots)
        if not any((mem.get(s) or "").strip() for s in slots):
            return None, None
        meta_obj = cap_obj.get("memory_meta", {})
        meta = normalize_slot_meta(meta_obj if isinstance(meta_obj, dict) else {}, slots)
        return mem, meta

    def history_all(extra: Optional[List[Msg]] = None) -> List[Msg]:
        hist = []
        if base_system:
            hist.append(Msg(role="system", content=base_system))
        hist.extend(messages)
        if extra:
            hist.extend(extra)
        return hist

    def _compute_task_raw(pending_user_text: Optional[str]) -> str:
        extra = [Msg(role="user", content=pending_user_text)] if pending_user_text else None
        raw = heuristic_task_only(history_all(extra), int(config.get("task_last_messages", 8)))
        raw = strip_command_like_lines(raw)
        return normalize_slots({"task_raw": raw}, ["task_raw"]).get("task_raw", "")

    def run_llm_extract(pending_user_text: Optional[str] = None) -> Dict[str, str]:
        extra = [Msg(role="user", content=pending_user_text)] if pending_user_text else None
        hist = history_all(extra)
        hist_text = "\n\n---\n\n".join([f"{m.role.upper()}:\n{m.content}" for m in hist]).strip()
        max_chars = int(config.get("extract_history_max_chars", 0))
        if max_chars > 0 and len(hist_text) > max_chars:
            hist_text = hist_text[-max_chars:]
        mem = None
        if hist_text.strip():
            mem = llm_extract_slots(
                model, tok, hist_text,
                slots=slots,
                max_new_tokens=int(config.get("extract_llm_max_new_tokens", 400)),
                retries=2,
            )

        def _looks_like_extraction_artifact(m: Dict[str, str]) -> bool:
            blob = "\n".join([(m.get(s, "") or "") for s in slots]).lower()
            bad = [
                "memory slot extractor",
                "extract conversation memory",
                "extract the conversation memory",
                "return only one json",
                "return only",
                "exact keys",
                "json:",
                "slots:",
            ]
            return any(b in blob for b in bad)

        if mem is None or _looks_like_extraction_artifact(mem):
            mem = heuristic_extract_slots(hist, slots=slots, task_last_messages=int(config.get("task_last_messages", 8)))
        mem = normalize_slots(mem, slots)
        mem["task_raw"] = _compute_task_raw(pending_user_text)
        # Always keep `task` as a compact summary for slot_only injection.
        task_sum = summarize_task_from_raw(mem["task_raw"])
        if task_sum:
            mem["task"] = normalize_slots({"task": task_sum}, ["task"]).get("task", "")
        elif pending_user_text:
            mem["task"] = normalize_slots({"task": pending_user_text}, ["task"]).get("task", "")
        return mem

    def run_heuristic_extract(pending_user_text: Optional[str] = None) -> Dict[str, str]:
        extra = [Msg(role="user", content=pending_user_text)] if pending_user_text else None
        hist = history_all(extra)
        mem = heuristic_extract_slots(hist, slots=slots, task_last_messages=int(config.get("task_last_messages", 8)))
        mem = normalize_slots(mem, slots)
        mem["task_raw"] = _compute_task_raw(pending_user_text)
        task_sum = summarize_task_from_raw(mem["task_raw"])
        if task_sum:
            mem["task"] = normalize_slots({"task": task_sum}, ["task"]).get("task", "")
        elif pending_user_text:
            mem["task"] = normalize_slots({"task": pending_user_text}, ["task"]).get("task", "")
        return mem

    def _merge_and_track(cand: Dict[str, str], source: str):
        nonlocal slot_state, slot_meta_state, last_meta_updates, last_merge_source
        prev = dict(slot_state)
        slot_state = merge_slots(slot_state, cand, slots)
        cand_conf = estimate_slot_confidences(cand, slots, extractor=source)
        slot_meta_state, last_meta_updates = update_slot_meta_after_merge(
            slot_meta_state,
            prev_memory=prev,
            new_memory=slot_state,
            cand_conf=cand_conf,
            slots=slots,
            source=source,
            turn_idx=turn_counter,
            updated_at=now_str(),
            conf_mode=str(config.get("slot_conf_mode", "balanced")),
            conf_floor=float(config.get("slot_conf_floor", 0.35)),
            conf_decay=float(config.get("slot_conf_decay", 0.55)),
            conf_merge_bonus=float(config.get("slot_conf_merge_bonus", 0.05)),
        )
        last_merge_source = source

    def _build_capsule_obj(memory: Dict[str, str], baton: Dict[str, str], cold_archive: Optional[Dict[str, Any]], relay_kind: str) -> Dict[str, Any]:
        return {
            "created_at": now_str(),
            "base_system": base_system,
            "slots": slots,
            "memory": memory,
            "memory_meta": slot_meta_state,
            "baton": baton,
            "baton_meta": {
                "format": "SMEM1",
                "embed_backend": str(config.get("embed_backend", EMBED_BACKEND_HASH4096)),
                "proj_seed": config["proj_seed"],
                "baton_dim": config["baton_dim"],
            },
            "cold_archive": cold_archive,
            "notes": {
                "aes_available": bool(AESGCM is not None),
                "relay_last_kind": relay_kind,
                "relay_keyframe_every": int(config.get("relay_keyframe_every", 0)),
                "slot_conf_mode": str(config.get("slot_conf_mode", "balanced")),
                "slot_confidence_threshold": float(config.get("slot_confidence_threshold", 0.0)),
            },
        }

    def _emit_relay(prev_memory: Dict[str, str], cur_memory: Dict[str, str], force_keyframe: bool = False):
        every = int(config.get("relay_keyframe_every", 0))
        if every <= 0:
            return
        kind = "delta"
        if force_keyframe or turn_counter <= 1 or (turn_counter % every == 0):
            kind = "keyframe"
            event = {
                "created_at": now_str(),
                "kind": "keyframe",
                "turn": int(turn_counter),
                "source": last_merge_source,
                "memory": {s: str(cur_memory.get(s, "") or "") for s in slots},
                "slot_meta": slot_meta_state,
            }
        else:
            changes = compute_memory_delta(prev_memory, cur_memory, slots)
            if not changes:
                return
            event = {
                "created_at": now_str(),
                "kind": "delta",
                "turn": int(turn_counter),
                "source": last_merge_source,
                "changes": changes,
                "slot_meta_updates": last_meta_updates,
            }
        append_relay_event(path_relay, event, max_entries=int(config.get("relay_max_entries", 600)))

    def update_slots_before_generation(pending_user_text: str) -> Dict[str, str]:
        nonlocal turn_counter, slot_state
        if config.get("extractor", "heuristic") == "heuristic":
            cand = run_heuristic_extract(pending_user_text)
            _merge_and_track(cand, source="heuristic")
            return slot_state

        def _slots_empty(s: Dict[str, str]) -> bool:
            return all((not (s.get(k) or "").strip()) for k in slots)

        every = int(config.get("extract_llm_every", 1))
        if _slots_empty(slot_state):
            # Seed quickly to avoid heavy "extract+reply" on the very first turn.
            # If user explicitly wants LLM extraction every turn, honor it.
            cand = run_llm_extract(pending_user_text) if every <= 1 else run_heuristic_extract(pending_user_text)
            _merge_and_track(cand, source="llm" if every <= 1 else "heuristic")
            return slot_state

        if every <= 1 or (turn_counter % every == 0):
            cand = run_llm_extract(pending_user_text)
            _merge_and_track(cand, source="llm")
        else:
            # Cheap update: keep persona/rules, but always refresh task from recent user turns.
            task_raw = _compute_task_raw(pending_user_text)
            task_sum = summarize_task_from_raw(task_raw) or task_raw
            task_norm = normalize_slots({"task": task_sum}, ["task"]).get("task", "")
            _merge_and_track({"task": task_norm, "task_raw": task_raw}, source="task_refresh")
        return slot_state

    def make_baton(memory: Dict[str, str]) -> Dict[str, str]:
        b = {}
        dev = next(model.parameters()).device
        for s in slots:
            b[s] = encode_baton_hashproj(
                text=memory.get(s, ""),
                baton_dim=int(config["baton_dim"]),
                proj_seed=int(config["proj_seed"]),
                device=dev,
                proj_cache=proj_cache,
                slot=s,
                embed_backend=str(config.get("embed_backend", EMBED_BACKEND_HASH4096)),
            )
        return b

    def save_all(capsule_obj: Optional[Dict[str, Any]] = None):
        write_text(path_transcript, render_transcript_txt(base_system, messages))
        write_json(path_session, {
            "base_system": base_system,
            "messages": [dataclasses.asdict(m) for m in messages],
            "config": config,
            "slot_meta": slot_meta_state,
        })
        if capsule_obj is not None:
            write_json(path_capsule, capsule_obj)

    # Seed slot_state/meta (prefer relay log, then capsule, then session slot_meta).
    relay_mem, relay_meta = load_relay_state(path_relay, slots)
    if relay_mem is not None:
        slot_state = relay_mem
    else:
        cap_mem, cap_meta = _try_load_capsule_state()
        if cap_mem is not None:
            slot_state = cap_mem
        if cap_meta is not None:
            slot_meta_state = cap_meta

    if relay_meta is not None:
        slot_meta_state = relay_meta
    if os.path.exists(path_session):
        try:
            sess_obj = json.loads(read_text(path_session))
            sess_meta = sess_obj.get("slot_meta", None)
            if isinstance(sess_meta, dict) and len(sess_meta) > 0:
                slot_meta_state = normalize_slot_meta(sess_meta, slots)
        except Exception:
            pass

    turn_counter = sum(1 for m in messages if m.role == "user")

    # Backward compatibility: old sessions may have memory but no memory_meta.
    # Seed confidence from current slot text so confidence filtering can work immediately.
    if any((slot_state.get(s) or "").strip() for s in slots):
        conf_vals = [float((slot_meta_state.get(s, {}) or {}).get("confidence", 0.0)) for s in slots]
        if all(c <= 0.0 for c in conf_vals):
            seed_conf = estimate_slot_confidences(slot_state, slots, extractor="resume_seed")
            slot_meta_state, _ = update_slot_meta_after_merge(
                slot_meta_state,
                prev_memory={s: "" for s in slots},
                new_memory=slot_state,
                cand_conf=seed_conf,
                slots=slots,
                source="resume_seed",
                turn_idx=max(0, turn_counter),
                updated_at=now_str(),
                conf_mode=str(config.get("slot_conf_mode", "balanced")),
                conf_floor=float(config.get("slot_conf_floor", 0.35)),
                conf_decay=float(config.get("slot_conf_decay", 0.55)),
                conf_merge_bonus=float(config.get("slot_conf_merge_bonus", 0.05)),
            )

    if messages and not any((slot_state.get(s) or "").strip() for s in slots):
        if config.get("extractor", "heuristic") == "llm" and int(config.get("extract_llm_every", 1)) <= 1:
            cand = run_llm_extract()
            _merge_and_track(cand, source="llm")
        else:
            cand = run_heuristic_extract()
            _merge_and_track(cand, source="heuristic")
        _emit_relay({s: "" for s in slots}, slot_state, force_keyframe=True)

    # If legacy/poisoned resume leaves task empty after sanitization, refresh once from transcript.
    if messages and ("task" in slots) and (not (slot_state.get("task") or "").strip()):
        cand = run_heuristic_extract()
        _merge_and_track(cand, source="resume_reextract")
        _emit_relay({s: "" for s in slots}, slot_state, force_keyframe=True)

    # initial save
    save_all()

    print(f"[session] {session_dir}")
    print("[offline]", os.environ.get("HF_HUB_OFFLINE"), os.environ.get("TRANSFORMERS_OFFLINE"))
    print(f"[context] model_ctx~{ctx_limit} input_budget~{effective_budget}")
    print(
        f"[memory] conf_mode={config.get('slot_conf_mode', 'balanced')}"
        f" conf_th={float(config.get('slot_confidence_threshold', 0.0)):.2f}"
        f" relay_every={int(config.get('relay_keyframe_every', 0))}"
    )
    print("Type /help for commands.\n")

    while True:
        try:
            line = input("you> ").rstrip("\n")
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]")
            break
        if not line.strip():
            continue

        if line.startswith("/"):
            cmd = line.strip()
            if cmd == "/help":
                print(HELP); continue
            if cmd in ("/exit","/quit"):
                break
            if cmd.startswith("/mode "):
                m = cmd.split(" ", 1)[1].strip()
                if m in ("full","slot","slot_only"):
                    config["context_mode"] = m
                    print("[ok] mode=", m)
                else:
                    print("[err] mode must be full|slot|slot_only")
                continue
            if cmd.startswith("/recent "):
                try:
                    config["recent_messages"] = max(0, int(cmd.split(" ",1)[1].strip()))
                    print("[ok] recent_messages=", config["recent_messages"])
                except Exception:
                    print("[err] /recent N")
                continue
            if cmd.startswith("/extract "):
                e = cmd.split(" ",1)[1].strip()
                if e in ("heuristic","llm"):
                    config["extractor"] = e
                    print("[ok] extractor=", e)
                else:
                    print("[err] extractor must be heuristic|llm")
                continue
            if cmd == "/extract_now":
                cand = run_llm_extract() if config.get("extractor","heuristic") == "llm" else run_heuristic_extract()
                _merge_and_track(cand, source=config.get("extractor","heuristic"))
                print("[ok] extracted now")
                continue
            if cmd.startswith("/system "):
                rest = cmd.split(" ",1)[1].strip()
                if rest == "show":
                    print(base_system or "")
                    continue
                if rest.startswith("set "):
                    base_system = rest[4:].strip() or None
                    print("[ok] system updated")
                    continue
                print("[err] /system show | /system set <text>")
                continue
            if cmd == "/slots":
                print(json.dumps(slot_state, ensure_ascii=False, indent=2))
                continue
            if cmd == "/slots_meta":
                print(json.dumps(slot_meta_state, ensure_ascii=False, indent=2))
                continue
            if cmd == "/relay_state":
                snap_mem, snap_meta = load_relay_state(path_relay, slots)
                out = {
                    "relay_path": path_relay,
                    "has_snapshot": bool(snap_mem is not None),
                    "memory": snap_mem if snap_mem is not None else slot_state,
                    "slot_meta": snap_meta if snap_meta is not None else slot_meta_state,
                }
                print(json.dumps(out, ensure_ascii=False, indent=2))
                continue
            if cmd.startswith("/conf "):
                rest = cmd.split(" ", 1)[1].strip()
                if rest == "show":
                    out = {
                        "slot_conf_mode": str(config.get("slot_conf_mode", "balanced")),
                        "slot_confidence_threshold": float(config.get("slot_confidence_threshold", 0.0)),
                        "slot_conf_floor": float(config.get("slot_conf_floor", 0.35)),
                        "slot_conf_decay": float(config.get("slot_conf_decay", 0.55)),
                        "slot_conf_merge_bonus": float(config.get("slot_conf_merge_bonus", 0.05)),
                    }
                    print(json.dumps(out, ensure_ascii=False, indent=2))
                    continue
                if rest.startswith("mode "):
                    m = rest.split(" ", 1)[1].strip().lower()
                    if m in ("balanced", "conservative", "aggressive", "replace"):
                        config["slot_conf_mode"] = m
                        print("[ok] slot_conf_mode=", m)
                    else:
                        print("[err] /conf mode balanced|conservative|aggressive|replace")
                    continue
                if rest.startswith("threshold "):
                    try:
                        v = _clip01(float(rest.split(" ", 1)[1].strip()))
                        config["slot_confidence_threshold"] = v
                        print("[ok] slot_confidence_threshold=", v)
                    except Exception:
                        print("[err] /conf threshold <0..1>")
                    continue
                print("[err] /conf show | /conf mode ... | /conf threshold <0..1>")
                continue
            if cmd == "/capsule" or cmd == "/save":
                mem = slot_state
                baton = make_baton(mem)
                tr = render_transcript_txt(base_system, messages)
                cold = make_cold_archive(tr, store=bool(args.store_archive), passphrase=args.archive_passphrase)
                _emit_relay(mem, mem, force_keyframe=True)
                capsule_obj = _build_capsule_obj(mem, baton, cold_archive=cold, relay_kind="manual_save")
                save_all(capsule_obj=capsule_obj)
                print("[saved]", session_dir)
                continue

            print("[err] unknown command. /help")
            continue

        # Normal chat turn
        user_text = line
        turn_counter += 1
        prev_memory = dict(slot_state)
        mem = update_slots_before_generation(user_text)
        baton = make_baton(mem)
        _emit_relay(prev_memory, mem, force_keyframe=False)

        mode = config.get("context_mode","slot")
        recent_n = int(config.get("recent_messages", 6))
        if mode == "full":
            context = messages[:]
        elif mode == "slot":
            context = messages[-recent_n:] if recent_n > 0 else []
        else:
            context = []

        assistant = generate_reply(
            model=model, tok=tok,
            base_system=base_system,
            memory=mem, slots=slots,
            context_messages=context,
            user_text=user_text,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            slot_meta=slot_meta_state,
            slot_confidence_threshold=float(config.get("slot_confidence_threshold", 0.0)),
            max_input_tokens=int(config.get("max_input_tokens", 0)),
            input_safety_tokens=int(config.get("input_safety_tokens", 96)),
            slot_runtime_max_chars=int(config.get("slot_runtime_max_chars", 900)),
        )

        messages.append(Msg(role="user", content=user_text))
        messages.append(Msg(role="assistant", content=assistant))

        print(f"ai> {assistant}\n")

        # save transcript/session/capsule
        tr = render_transcript_txt(base_system, messages)
        cold = make_cold_archive(tr, store=bool(args.store_archive), passphrase=args.archive_passphrase)
        capsule_obj = _build_capsule_obj(mem, baton, cold_archive=cold, relay_kind=last_merge_source)
        save_all(capsule_obj=capsule_obj)

        # append jsonl training record
        rec = {
            "meta": {
                "created_at": now_str(),
                "session_dir": session_dir,
                "mode": mode,
                "recent_messages": recent_n,
                "extractor": config.get("extractor", "heuristic"),
                "slot_conf_mode": config.get("slot_conf_mode", "balanced"),
                "slot_confidence_threshold": float(config.get("slot_confidence_threshold", 0.0)),
            },
            "system": base_system,
            "user": user_text,
            "assistant": assistant,
            "memory": mem,
            "memory_meta": slot_meta_state,
            "baton": baton,
        }
        append_jsonl(path_train, rec)

    # final save
    mem = slot_state
    baton = make_baton(mem)
    _emit_relay(mem, mem, force_keyframe=True)
    tr = render_transcript_txt(base_system, messages)
    cold = make_cold_archive(tr, store=bool(args.store_archive), passphrase=args.archive_passphrase)
    capsule_obj = _build_capsule_obj(mem, baton, cold_archive=cold, relay_kind="final_save")
    save_all(capsule_obj=capsule_obj)
    print("[saved]", session_dir)


# =============================
# dataforge build command
# =============================
def parse_log_file(path: str) -> List[Msg]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".json",".jsonl"):
        if ext == ".json":
            obj = json.loads(read_text(path))
            return _parse_json_obj_as_msgs(obj)
        msgs: List[Msg] = []
        for ln in read_text(path).splitlines():
            ln = ln.strip()
            if not ln:
                continue
            msgs.extend(_parse_json_obj_as_msgs(json.loads(ln)))
        return msgs
    # txt
    return _parse_txt_as_msgs(read_text(path))

def _parse_json_obj_as_msgs(obj: Any) -> List[Msg]:
    if isinstance(obj, dict) and "messages" in obj and isinstance(obj["messages"], list):
        obj = obj["messages"]
    if isinstance(obj, list):
        out = []
        for it in obj:
            if not isinstance(it, dict):
                continue
            role = str(it.get("role","user")).lower().strip()
            role = role if role in ("system","user","assistant") else "user"
            content = str(it.get("content", it.get("text","")))
            if content.strip():
                out.append(Msg(role=role, content=content))
        return out
    return [Msg(role="user", content=json.dumps(obj, ensure_ascii=False))]

def _parse_txt_as_msgs(text: str) -> List[Msg]:
    lines = text.splitlines()
    out: List[Msg] = []
    pat = re.compile(r"^\s*(system|user|assistant)\s*[:：]\s*(.*)$", re.IGNORECASE)
    cur_role: Optional[str] = None
    buf: List[str] = []
    found = False

    def flush():
        nonlocal cur_role, buf
        if cur_role and buf:
            c = "\n".join(buf).strip()
            if c:
                out.append(Msg(role=cur_role.lower(), content=c))
        cur_role = None
        buf = []

    for ln in lines:
        m = pat.match(ln)
        if m:
            found = True
            flush()
            cur_role = m.group(1).lower()
            buf = [m.group(2)]
        else:
            if cur_role is None:
                cur_role = "user"
            buf.append(ln)
    flush()

    if found:
        return [m for m in out if m.content.strip()]
    t = text.strip()
    return [Msg(role="user", content=t)] if t else []

def collect_input_paths(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        paths = []
        for root, _, files in os.walk(input_path):
            for fn in files:
                if fn.lower().endswith((".json",".jsonl",".txt")):
                    paths.append(os.path.join(root, fn))
        paths.sort()
        return paths
    return [input_path]

def format_history(messages: List[Msg], max_chars: int) -> str:
    parts = []
    for m in messages:
        parts.append(f"{m.role.upper()}:\n{m.content}".strip())
    s = "\n\n---\n\n".join(parts).strip()
    if max_chars > 0 and len(s) > max_chars:
        s = s[-max_chars:]
    return s

def iter_user_turns(messages: List[Msg], stride: int) -> Iterable[Tuple[int, List[Msg], str, Optional[str]]]:
    user_indices = [i for i,m in enumerate(messages) if m.role == "user"]
    if stride > 1:
        user_indices = user_indices[::stride]
    n = len(messages)
    for i in user_indices:
        if i <= 0:
            continue
        history = messages[:i]
        user_text = messages[i].content
        assistant_log = None
        if i+1 < n and messages[i+1].role == "assistant":
            assistant_log = messages[i+1].content
        yield i, history, user_text, assistant_log

def get_base_system_from_history(history: List[Msg], override_system: Optional[str]) -> Optional[str]:
    if override_system is not None:
        s = override_system.strip()
        return s if s else None
    sys_msgs = [m.content.strip() for m in history if m.role == "system" and m.content.strip()]
    if not sys_msgs:
        return None
    return "\n\n".join(_dedup_keep_order(sys_msgs)).strip() or None

def truncate_slot_texts(memory: Dict[str, str], slot_max_chars: int) -> Dict[str, str]:
    if slot_max_chars <= 0:
        return memory
    out = {}
    for k,v in memory.items():
        v = (v or "").strip()
        if len(v) > slot_max_chars:
            v = v[-slot_max_chars:]
        out[k] = v
    return out

def cmd_dataforge_build(args):
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    slots = [s.strip() for s in args.slots.split(",") if s.strip()]
    embed_backend = normalize_embed_backend(args.embed_backend)

    tok, model = load_local_llm(args.teacher_path, device=device, dtype=dtype, use_fast=not args.slow_tokenizer)
    proj_cache: Dict[Tuple[str, str, int, int], torch.Tensor] = {}

    override_system = None
    if args.system_file:
        override_system = read_text(args.system_file)
    elif args.system_text is not None:
        override_system = args.system_text

    paths = collect_input_paths(args.input)
    if not paths:
        raise SystemExit("No logs found.")

    if (not args.append) and os.path.exists(args.out_jsonl):
        os.remove(args.out_jsonl)

    total = 0
    t0 = time.time()
    for p in paths:
        msgs = parse_log_file(p)
        if not msgs:
            continue
        for idx, history, user_text, assistant_log in iter_user_turns(msgs, stride=args.stride):
            hist_text = format_history(history, max_chars=args.history_max_chars)
            if len(hist_text) < args.min_history_chars:
                continue

            base_system = get_base_system_from_history(history, override_system)
            if args.extractor == "llm":
                mem = llm_extract_slots(model, tok, hist_text, slots=slots, max_new_tokens=400)
                if mem is None:
                    mem = heuristic_extract_slots(history, slots=slots, task_last_messages=args.task_last_messages)
            else:
                mem = heuristic_extract_slots(history, slots=slots, task_last_messages=args.task_last_messages)

            mem = truncate_slot_texts(mem, slot_max_chars=args.slot_max_chars)

            dev = next(model.parameters()).device
            baton = {}
            for s in slots:
                baton[s] = encode_baton_hashproj(
                    text=mem.get(s,""),
                    baton_dim=int(args.baton_dim),
                    proj_seed=int(args.proj_seed),
                    device=dev,
                    proj_cache=proj_cache,
                    slot=s,
                    embed_backend=embed_backend,
                )

            # teacher response
            assistant_teacher = generate_reply(
                model=model, tok=tok,
                base_system=base_system,
                memory=mem, slots=slots,
                context_messages=[],  # for dataforge, we don't include raw context; memory covers it
                user_text=user_text,
                max_new_tokens=int(args.teacher_max_new_tokens),
                temperature=float(args.teacher_temperature),
                top_p=float(args.teacher_top_p),
            )

            if args.label_source == "log" and assistant_log:
                assistant = assistant_log
            else:
                assistant = assistant_teacher

            rec = {
                "meta": {"source_path": p, "user_index": idx, "created_at": now_str()},
                "system": base_system,
                "user": user_text,
                "assistant": assistant,
                "assistant_teacher": assistant_teacher,
                "assistant_log": assistant_log,
                "memory": mem,
                "baton": baton,
                "baton_meta": {
                    "format": "SMEM1",
                    "embed_backend": embed_backend,
                    "proj_seed": int(args.proj_seed),
                    "baton_dim": int(args.baton_dim),
                    "in_dim": 4096,
                },
            }
            append_jsonl(args.out_jsonl, rec)
            total += 1
            if args.verbose:
                print(f"[ok] {os.path.basename(p)} idx={idx} -> #{total}")
            if args.max_samples > 0 and total >= args.max_samples:
                break
        if args.max_samples > 0 and total >= args.max_samples:
            break

    dt = time.time() - t0
    print(f"[done] wrote {total} samples -> {args.out_jsonl} (elapsed {dt:.1f}s)")


# =============================
# B1+ (KD + variable prefix + slots)
# =============================
class SlotPrefixNet(nn.Module):
    def __init__(self, baton_dim: int, hidden_size: int, n_prefix_max: int, mlp_width: int = 2048, min_prefix: int = 0):
        super().__init__()
        self.n_prefix_max = int(n_prefix_max)
        self.min_prefix = int(min_prefix)
        self.trunk = nn.Sequential(nn.Linear(baton_dim, mlp_width), nn.SiLU())
        self.emb_head = nn.Linear(mlp_width, self.n_prefix_max * hidden_size)
        self.gate_head = nn.Linear(mlp_width, self.n_prefix_max)
        self.ln = nn.LayerNorm(hidden_size)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(z)
        emb = self.emb_head(h).view(z.shape[0], self.n_prefix_max, -1)
        emb = self.ln(emb)
        gate_logits = self.gate_head(h)
        p = torch.sigmoid(gate_logits)
        mask = torch.cumprod(p, dim=-1)
        if self.min_prefix > 0:
            mask[:, :self.min_prefix] = 1.0
        prefix = self.alpha * emb * mask.unsqueeze(-1)
        expected_len = mask.sum(dim=-1)
        return prefix, mask, expected_len

class MultiSlotPrefix(nn.Module):
    def __init__(self, slots: List[str], baton_dim: int, hidden_size: int, slot_prefix_max: Dict[str,int], mlp_width: int, min_prefix: int):
        super().__init__()
        self.slots = slots
        self.nets = nn.ModuleDict({
            s: SlotPrefixNet(baton_dim, hidden_size, int(slot_prefix_max[s]), mlp_width=mlp_width, min_prefix=min_prefix)
            for s in slots
        })

    def forward(self, z_by_slot: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefixes, masks = [], []
        expected = 0.0
        for s in self.slots:
            pref, m, el = self.nets[s](z_by_slot[s])
            prefixes.append(pref); masks.append(m); expected = expected + el
        return torch.cat(prefixes, dim=1), torch.cat(masks, dim=1), expected

def kd_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, tau: float) -> torch.Tensor:
    s = F.log_softmax(student_logits / tau, dim=-1)
    t = F.softmax(teacher_logits / tau, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (tau * tau)

def pack_memory_for_teacher(memory: Dict[str,str], slots: List[str]) -> str:
    return pack_memory_slots(memory, slots)

def apply_chat_prompt_str(tok, system: Optional[str], user: str) -> str:
    msgs = []
    if system:
        msgs.append({"role":"system","content":system})
    msgs.append({"role":"user","content":user})
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        sys_part = f"SYSTEM:\n{system}\n\n" if system else ""
        return sys_part + f"USER:\n{user}\n\nASSISTANT:\n"

def cmd_b1plus_train(args):
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)

    slots = [s.strip() for s in args.slots.split(",") if s.strip()]

    tok_s, student = load_local_llm(args.student_path, device=device, dtype=dtype, use_fast=not args.slow_tokenizer)
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    student.config.use_cache = False

    teacher_shared = False
    if args.teacher_path is None or os.path.abspath(args.teacher_path) == os.path.abspath(args.student_path):
        teacher = student
        tok_t = tok_s
        teacher_shared = True
    else:
        tok_t, teacher = load_local_llm(args.teacher_path, device=device, dtype=dtype, use_fast=not args.slow_tokenizer)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.config.use_cache = False

    hidden = int(student.config.hidden_size)

    slot_prefix_max = {s: int(args.n_prefix_max) for s in slots}
    if args.slot_prefix_max:
        for item in args.slot_prefix_max.split(","):
            if not item.strip():
                continue
            k,v = item.split("=")
            slot_prefix_max[k.strip()] = int(v.strip())

    prefix_net = MultiSlotPrefix(
        slots=slots,
        baton_dim=int(args.baton_dim),
        hidden_size=hidden,
        slot_prefix_max=slot_prefix_max,
        mlp_width=int(args.mlp_width),
        min_prefix=int(args.min_prefix),
    ).to(device)

    opt = torch.optim.AdamW(prefix_net.parameters(), lr=float(args.lr), weight_decay=0.01)

    # load jsonl (streamed, avoids large full-file read)
    lines = read_jsonl_lines(args.train_jsonl, limit=int(args.limit) if args.limit else None)
    # simple shuffle
    import random
    random.seed(args.seed)
    random.shuffle(lines)

    step = 0
    for epoch in range(int(args.epochs)):
        for ln in lines:
            step += 1
            ex = json.loads(ln)
            baton = ex["baton"]
            memory = ex["memory"]
            user = ex["user"]
            assistant = ex["assistant"]

            # z per slot
            z_by_slot = {}
            for s in slots:
                z_by_slot[s] = unpack_smem1_int8(baton[s], device=device).unsqueeze(0)

            prefix_embeds, _, expected_len = prefix_net(z_by_slot)

            # build prompts
            base_system = ex.get("system", None)
            mem_pack = pack_memory_for_teacher(memory, slots)
            teacher_system = (base_system.strip() + "\n\n" + mem_pack).strip() if base_system and mem_pack else (mem_pack or base_system)
            teacher_prompt = apply_chat_prompt_str(tok_t, teacher_system, user)
            student_prompt = apply_chat_prompt_str(tok_s, base_system, user)

            assistant_text = assistant + (tok_s.eos_token or "")
            assistant_ids = tok_s(assistant_text, return_tensors="pt").input_ids.to(device)

            t_prompt_ids = tok_t(teacher_prompt, return_tensors="pt").input_ids.to(device)
            s_prompt_ids = tok_s(student_prompt, return_tensors="pt").input_ids.to(device)

            t_full_ids = torch.cat([t_prompt_ids, assistant_ids], dim=1)
            s_full_ids = torch.cat([s_prompt_ids, assistant_ids], dim=1)

            with torch.no_grad():
                t_out = teacher(input_ids=t_full_ids, use_cache=False, return_dict=True)
                t_logits = t_out.logits  # [1, Lt, V]

            embed = student.get_input_embeddings()
            tok_embeds = embed(s_full_ids)
            prefix_embeds = prefix_embeds.to(dtype=tok_embeds.dtype)
            inputs_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)

            labels = s_full_ids.clone()
            labels[:, :s_prompt_ids.shape[1]] = -100
            prefix_ignore = torch.full((1, prefix_embeds.shape[1]), -100, dtype=torch.long, device=device)
            labels = torch.cat([prefix_ignore, labels], dim=1)

            s_out = student(
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones((1, inputs_embeds.shape[1]), dtype=torch.long, device=device),
                labels=labels if args.ce_alpha > 0 else None,
                use_cache=False,
                return_dict=True,
            )
            s_logits = s_out.logits
            ce_loss = s_out.loss if args.ce_alpha > 0 else torch.tensor(0.0, device=device)

            A = assistant_ids.shape[1]
            tp = t_prompt_ids.shape[1]
            sp = s_prompt_ids.shape[1]
            N = prefix_embeds.shape[1]
            t_pos = torch.arange(tp - 1, tp - 1 + A, device=device)
            s_pos = torch.arange(N + sp - 1, N + sp - 1 + A, device=device)

            t_slice = t_logits[0, t_pos, :]
            s_slice = s_logits[0, s_pos, :]

            kd_loss = kd_kl(s_slice, t_slice, tau=float(args.tau)) if args.kd_alpha > 0 else torch.tensor(0.0, device=device)
            len_loss = expected_len.mean()

            loss = float(args.ce_alpha) * ce_loss + float(args.kd_alpha) * kd_loss + float(args.len_alpha) * len_loss
            loss.backward()

            if step % int(args.grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(prefix_net.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if step % int(args.log_every) == 0:
                print(f"epoch={epoch+1} step={step} loss={float(loss.item()):.4f} kd={float(kd_loss.item()):.4f} ce={float(ce_loss.item()):.4f} elen={float(len_loss.item()):.2f}")

            if args.save_every > 0 and step % int(args.save_every) == 0:
                ensure_dir(args.out_dir)
                ckpt = {
                    "student_path": args.student_path,
                    "teacher_path": args.teacher_path,
                    "slots": slots,
                    "baton_dim": int(args.baton_dim),
                    "slot_prefix_max": slot_prefix_max,
                    "mlp_width": int(args.mlp_width),
                    "min_prefix": int(args.min_prefix),
                    "state_dict": prefix_net.state_dict(),
                }
                path = os.path.join(args.out_dir, f"prefix_step{step}.pt")
                torch.save(ckpt, path)
                print("[saved]", path)

        if not teacher_shared and device.type == "cuda":
            torch.cuda.empty_cache()

    ensure_dir(args.out_dir)
    final_path = os.path.join(args.out_dir, "prefix_final.pt")
    torch.save({
        "student_path": args.student_path,
        "teacher_path": args.teacher_path,
        "slots": slots,
        "baton_dim": int(args.baton_dim),
        "slot_prefix_max": slot_prefix_max,
        "mlp_width": int(args.mlp_width),
        "min_prefix": int(args.min_prefix),
        "state_dict": prefix_net.state_dict(),
    }, final_path)
    print("[done] saved ->", final_path)

@torch.no_grad()
def cmd_b1plus_infer(args):
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)

    ckpt = torch.load(args.prefix_ckpt, map_location="cpu")
    student_path = ckpt["student_path"]

    tok, model = load_local_llm(student_path, device=device, dtype=dtype, use_fast=not args.slow_tokenizer)
    model.eval()

    slots = ckpt["slots"]
    slot_prefix_max = ckpt["slot_prefix_max"]
    prefix_net = MultiSlotPrefix(
        slots=slots,
        baton_dim=int(ckpt["baton_dim"]),
        hidden_size=int(model.config.hidden_size),
        slot_prefix_max=slot_prefix_max,
        mlp_width=int(ckpt["mlp_width"]),
        min_prefix=int(ckpt["min_prefix"]),
    ).to(device)
    prefix_net.load_state_dict(ckpt["state_dict"], strict=True)
    prefix_net.eval()

    baton_obj = json.loads(args.baton_json) if args.baton_json.strip().startswith("{") else json.loads(read_text(args.baton_json))
    baton = baton_obj["baton"] if isinstance(baton_obj, dict) and "baton" in baton_obj else baton_obj
    baton_meta = baton_obj.get("baton_meta") if isinstance(baton_obj, dict) else None
    baton_dim = int(baton_meta.get("baton_dim")) if isinstance(baton_meta, dict) and baton_meta.get("baton_dim") else int(ckpt["baton_dim"])

    z_by_slot = {}
    for s in slots:
        if isinstance(baton, dict) and s in baton:
            z_by_slot[s] = unpack_smem1_int8(baton[s], device=device).unsqueeze(0)
        else:
            z_by_slot[s] = torch.zeros((1, baton_dim), dtype=torch.float32, device=device)

    prefix_embeds, mask_cont, _ = prefix_net(z_by_slot)

    # optional truncation
    if args.truncate_threshold is not None:
        m = mask_cont[0].detach().cpu()
        K = 0
        for i in range(m.numel()):
            if float(m[i]) >= float(args.truncate_threshold):
                K = i + 1
            else:
                break
        K = max(K, 1)
        prefix_embeds = prefix_embeds[:, :K, :]

    # standard generate with inputs_embeds is usually OK for inference here
    prompt = apply_chat_prompt_str(tok, args.system, args.user)
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    embed = model.get_input_embeddings()
    prompt_embeds = embed(prompt_ids)
    prefix_embeds = prefix_embeds.to(dtype=prompt_embeds.dtype)
    inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)

    attn = torch.ones((1, inputs_embeds.shape[1]), dtype=torch.long, device=device)
    out = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        max_new_tokens=int(args.max_new_tokens),
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    # decode only new tokens after prefix+prompt length is tricky with inputs_embeds; easiest: decode full and strip prompt text
    full_text = tok.decode(out[0], skip_special_tokens=True)
    # best-effort: print tail
    print(full_text)


# =============================
# Gateway command
# =============================
def cmd_gateway(args):
    """
    POST /chat
      {
        "session_id": "demo",
        "user": "....",
        "capsule": {...} | null,
        "external_reply": "...." | null,
        "openai_model": "gpt-5" | null,
        "anthropic_model": "claude-..." | null
      }
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)

    slots = [s.strip() for s in args.slots.split(",") if s.strip()]
    embed_backend = normalize_embed_backend(args.embed_backend)
    proj_cache: Dict[Tuple[str, str, int, int], torch.Tensor] = {}

    # summarizer/extractor model (optional)
    sum_tok = sum_model = None
    if args.summarize_mode == "llm" or args.extractor == "llm":
        if not args.summarizer_model_path:
            raise SystemExit("--summarize_mode llm or --extractor llm requires --summarizer_model_path")
        sum_tok, sum_model = load_local_llm(
            args.summarizer_model_path, device=device, dtype=dtype, use_fast=not args.slow_tokenizer
        )

    ensure_dir(args.store_dir)

    def latest_capsule_path(session_id: str) -> str:
        return os.path.join(args.store_dir, f"{session_id}.latest_capsule.json")

    def transcript_path(session_id: str) -> str:
        return os.path.join(args.store_dir, f"{session_id}.transcript.txt")

    def store_jsonl_path(session_id: str) -> str:
        return os.path.join(args.store_dir, f"{session_id}.jsonl")

    def load_latest_capsule(session_id: str) -> Optional[Dict[str, Any]]:
        p = latest_capsule_path(session_id)
        if os.path.exists(p):
            return json.loads(read_text(p))
        return None

    def save_latest_capsule(session_id: str, cap: Dict[str, Any]):
        write_json(latest_capsule_path(session_id), cap)

    def append_transcript(session_id: str, user_text: str, assistant_text: str):
        tp = transcript_path(session_id)
        prev = read_text(tp) if os.path.exists(tp) else ""
        block = f"\n\nUSER:\n{user_text}\n\nASSISTANT:\n{assistant_text}\n"
        write_text(tp, (prev + block).strip() + "\n")

    def update_store(session_id: str, memory: Dict[str, str], baton: Dict[str, str]):
        append_jsonl(store_jsonl_path(session_id), {"created_at": now_str(), "memory": memory, "baton": baton})

    def extract_slots_from_text(log_text: str) -> Dict[str, str]:
        tail = log_text[-int(args.extract_history_max_chars):] if args.extract_history_max_chars > 0 else log_text
        msgs = _parse_txt_as_msgs(tail)
        task_raw = normalize_slots({"task_raw": strip_command_like_lines(heuristic_task_only(msgs, int(getattr(args, "task_last_messages", 8))))}, ["task_raw"]).get("task_raw", "")
        task_sum = summarize_task_from_raw(task_raw) or task_raw
        task_norm = normalize_slots({"task": task_sum}, ["task"]).get("task", "")

        if args.extractor == "llm" and sum_model is not None:
            mem = llm_extract_slots(
                sum_model, sum_tok, tail, slots=slots,
                max_new_tokens=int(args.extract_llm_max_new_tokens),
                retries=2,
            )
            if mem is not None:
                base = normalize_slots({s: str(mem.get(s, "") or "") for s in slots}, slots)
                if "task" in slots and task_norm:
                    base["task"] = task_norm
                if task_raw:
                    base["task_raw"] = task_raw
                return base

        mem2 = heuristic_extract_slots(msgs, slots=slots, task_last_messages=int(getattr(args, "task_last_messages", 8)))
        base2 = normalize_slots({s: str(mem2.get(s, "") or "") for s in slots}, slots)
        if "task" in slots and task_norm:
            base2["task"] = task_norm
        if task_raw:
            base2["task_raw"] = task_raw
        return base2

    def task_only_from_log(log_text: str) -> str:
        tail = log_text[-int(args.extract_history_max_chars):] if args.extract_history_max_chars > 0 else log_text
        msgs = _parse_txt_as_msgs(tail)
        raw = heuristic_task_only(msgs, int(getattr(args, "task_last_messages", 8))) if msgs else ""
        return strip_command_like_lines(raw)

    def make_baton_from_memory(memory: Dict[str, str]) -> Dict[str, str]:
        b = {}
        for s in slots:
            b[s] = encode_baton_hashproj(
                memory.get(s, ""),
                baton_dim=int(args.baton_dim),
                proj_seed=int(args.proj_seed),
                device=device,
                proj_cache=proj_cache,
                slot=s,
                embed_backend=embed_backend,
            )
        return b

    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: Dict[str, Any]):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/chat":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n).decode("utf-8", errors="replace")
                req = json.loads(raw)
            except Exception as e:
                return self._send(400, {"error": f"bad json: {e}"})

            session_id = str(req.get("session_id", "default"))
            user = str(req.get("user", "") or "")
            external_reply = req.get("external_reply", None)
            capsule_in = req.get("capsule", None)

            openai_model = str(req.get("openai_model", args.openai_model))
            anthropic_model = str(req.get("anthropic_model", args.anthropic_model))

            cap = capsule_in or load_latest_capsule(session_id) or {}
            base_system = (cap.get("base_system") or args.system_text or "").strip() or None

            # 1) restore transcript from cold if missing
            tp = transcript_path(session_id)
            if (not os.path.exists(tp)) and cap.get("cold_archive"):
                restored = recover_cold_text(cap.get("cold_archive"), passphrase=args.archive_passphrase)
                if restored:
                    write_text(tp, restored)

            # 2) if external_reply present: append transcript and auto-update
            if isinstance(external_reply, str) and external_reply.strip():
                u = user.strip() or "(user omitted)"
                append_transcript(session_id, user_text=u, assistant_text=external_reply.strip())

            log_text = read_text(tp) if os.path.exists(tp) else ""

            # 3) compute memory/baton
            memory_in = cap.get("memory") if isinstance(cap.get("memory"), dict) else None
            baton = cap.get("baton") if isinstance(cap.get("baton"), dict) else None
            prev_notes = cap.get("notes") if isinstance(cap.get("notes"), dict) else {}
            ext_turns = int(prev_notes.get("external_reply_turns", 0))
            has_external = isinstance(external_reply, str) and external_reply.strip()
            if has_external:
                ext_turns += 1

            full_every = max(1, int(args.external_full_extract_every))

            full_extract = False
            if bool(args.reextract_each_call) or (memory_in is None) or (baton is None):
                full_extract = True
            elif has_external and (full_every <= 1 or (ext_turns % full_every == 0)):
                full_extract = True

            memory_prev = normalize_slots({s: str((memory_in or {}).get(s, "") or "") for s in slots}, slots)

            if full_extract:
                cand = extract_slots_from_text(log_text) if log_text.strip() else {s: "" for s in slots}
                memory = merge_slots(memory_prev, cand, slots)
            elif has_external:
                # cheap update: only refresh task; keep persona/rules
                memory = dict(memory_prev or {s: "" for s in slots})
                task_text = task_only_from_log(log_text)
                if task_text:
                    task_raw = normalize_slots({"task_raw": task_text}, ["task_raw"]).get("task_raw", "")
                    task_sum = summarize_task_from_raw(task_raw) or task_raw
                    task_norm = normalize_slots({"task": task_sum}, ["task"]).get("task", "")
                    memory = merge_slots(memory, {"task": task_norm, "task_raw": task_raw}, slots)
            else:
                memory = dict(memory_prev)

            # Always normalize + recompute baton for consistency (cheap + prevents dirty memory amplification)
            memory = normalize_slots({s: str((memory or {}).get(s, "") or "") for s in slots}, slots)
            baton = make_baton_from_memory(memory)

            assert isinstance(memory, dict)
            assert isinstance(baton, dict)

            # 4) retrieve similar past memories
            retrieved = gateway_retrieve(store_jsonl_path(session_id), baton, slots, top_k=int(args.top_k), device=device) if args.top_k > 0 else []

            # 5) cold summarize (optional)
            cold_summary = ""
            if args.summarize_mode != "none" and log_text.strip():
                if args.summarize_mode == "heuristic":
                    cold_summary = heuristic_summarize_for_inject(log_text, max_chars=int(args.summarize_max_chars))
                else:
                    cold_summary = llm_summarize_for_inject(
                        sum_model, sum_tok, log_text,
                        max_new_tokens=int(args.summarize_llm_max_new_tokens),
                        target_chars=int(args.summarize_max_chars),
                    )

            # 6) build injection text
            inject_parts = [pack_memory_slots(memory, slots)]
            for item in retrieved:
                inject_parts.append(pack_memory_slots(item.get("memory", {}), slots))
            if cold_summary.strip():
                inject_parts.append("[COLD_SUMMARY]\n" + cold_summary.strip())

            inject_text = "\n\n".join([p for p in inject_parts if p.strip()])
            inject_text = trim_inject(inject_text, max_chars=int(args.inject_max_chars))

            # 7) ready-to-send requests
            sys_instructions = ((base_system or "") + ("\n\n" + inject_text if inject_text.strip() else "")).strip()
            prepared_prompt = build_external_prompt(base_system, inject_text, user) if user.strip() else ""

            if user.strip():
                openai_chat = build_openai_chat_completions_request(
                    model=openai_model,
                    system_text=sys_instructions,
                    user_text=user,
                    max_tokens=int(args.openai_max_tokens),
                    temperature=float(args.openai_temperature),
                    top_p=float(args.openai_top_p),
                    prefer_developer_role=True,
                )
                openai_chat_legacy = build_openai_chat_completions_request(
                    model=openai_model,
                    system_text=sys_instructions,
                    user_text=user,
                    max_tokens=int(args.openai_max_tokens),
                    temperature=float(args.openai_temperature),
                    top_p=float(args.openai_top_p),
                    prefer_developer_role=False,
                )
                openai_responses = build_openai_responses_request(
                    model=openai_model,
                    instructions=sys_instructions,
                    user_text=user,
                    max_output_tokens=int(args.openai_max_output_tokens),
                    temperature=float(args.openai_temperature),
                    top_p=float(args.openai_top_p),
                )
                anthropic_messages = build_anthropic_messages_request(
                    model=anthropic_model,
                    system_text=sys_instructions,
                    user_text=user,
                    max_tokens=int(args.anthropic_max_tokens),
                    temperature=float(args.anthropic_temperature) if args.anthropic_temperature is not None else None,
                    top_p=float(args.anthropic_top_p) if args.anthropic_top_p is not None else None,
                    anthropic_version=str(args.anthropic_version),
                )
            else:
                openai_chat = None
                openai_chat_legacy = None
                openai_responses = None
                anthropic_messages = None

            # 8) update store + capsule
            update_store(session_id, memory, baton)

            cold = None
            if args.store_archive and log_text.strip():
                cold = make_cold_archive(log_text, store=True, passphrase=args.archive_passphrase)

            new_cap = {
                "created_at": now_str(),
                "base_system": base_system,
                "slots": slots,
                "memory": memory,
                "baton": baton,
                "baton_meta": {
                    "format": "SMEM1",
                    "embed_backend": embed_backend,
                    "proj_seed": int(args.proj_seed),
                    "baton_dim": int(args.baton_dim),
                },
                "cold_archive": cold if cold is not None else cap.get("cold_archive"),
                "notes": {
                    "retrieved_count": len(retrieved),
                    "summarize_mode": args.summarize_mode,
                    "extractor": args.extractor,
                    "auto_updated_on_external_reply": bool(isinstance(external_reply, str) and external_reply.strip()),
                    "external_reply_turns": ext_turns,
                    "external_full_extract_every": full_every,
                    "external_full_extract": full_extract,
                },
            }
            save_latest_capsule(session_id, new_cap)

            return self._send(200, {
                "prepared_prompt": prepared_prompt,
                "inject_chars": len(inject_text),
                "retrieved_count": len(retrieved),
                "openai_chat": openai_chat,
                "openai_chat_legacy": openai_chat_legacy,
                "openai_responses": openai_responses,
                "anthropic_messages": anthropic_messages,
                "new_capsule": new_cap,
            })

    srv = HTTPServer((args.host, int(args.port)), H)
    print(f"[gateway] http://{args.host}:{args.port}  store={args.store_dir}")
    print(f"[summarize] {args.summarize_mode}  [extractor] {args.extractor}  [reextract_each_call] {args.reextract_each_call}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[gateway] stopped")


# =============================
# Bench command
# =============================
def _collect_session_transcripts(root: str) -> List[str]:
    paths: List[str] = []
    for base, _, files in os.walk(root):
        for fn in files:
            if fn == "transcript.txt":
                paths.append(os.path.join(base, fn))
    paths.sort()
    return paths

def _collect_train_jsonl(root: str) -> List[str]:
    paths: List[str] = []
    for base, _, files in os.walk(root):
        for fn in files:
            if fn == "train.jsonl":
                paths.append(os.path.join(base, fn))
    paths.sort()
    return paths

def _parse_embed_backend_list(raw: str) -> List[str]:
    items = [x.strip() for x in (raw or "").split(",") if x.strip()]
    if not items:
        return [EMBED_BACKEND_HASH4096]
    out: List[str] = []
    for it in items:
        out.append(normalize_embed_backend(it))
    return _dedup_keep_order(out)

def _is_contaminated_slot(slot: str, text: str) -> Dict[str, bool]:
    t = (text or "").strip()
    flags = {
        "nonempty": bool(t),
        "bad_role_tag": bool(_BAD_ROLE_TAG.search(t)) if t else False,
        "bad_meta": bool(_BAD_META.search(t)) if t else False,
        "reflective": bool(_REFLECTIVE_TEMPLATE_LINE.search(_QUOTE_PREFIX_LINE.sub("", t))) if t else False,
        "persona_greeting": False,
    }
    if slot == "persona" and t:
        if re.match(r"(?i)^\s*(hi|hello|hey|thanks|thank you|thx)\b", t):
            flags["persona_greeting"] = True
    return flags

def _mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))

def cmd_bench(args):
    slots = [s.strip() for s in args.slots.split(",") if s.strip()]
    embed_backends = _parse_embed_backend_list(args.embed_backend)

    mode = str(args.mode or "auto").strip().lower()
    inp = args.input
    if mode not in ("auto", "logs", "train"):
        raise SystemExit("--mode must be auto|logs|train")

    if mode == "auto":
        if os.path.isdir(inp):
            mode = "train" if _collect_train_jsonl(inp) else "logs"
        else:
            mode = "train" if str(inp).lower().endswith(".jsonl") else "logs"

    paths: List[str] = []
    if mode == "logs":
        if os.path.isdir(inp):
            paths = _collect_session_transcripts(inp)
            if not paths:
                paths = collect_input_paths(inp)
        else:
            paths = [inp]
    else:
        if os.path.isdir(inp):
            paths = _collect_train_jsonl(inp)
            if not paths:
                paths = [p for p in collect_input_paths(inp) if p.lower().endswith(".jsonl")]
        else:
            paths = [inp]

    if not paths:
        raise SystemExit("No input logs found.")

    # Load LLM only if requested (logs mode + extractor=llm).
    sum_model = sum_tok = None
    if mode == "logs" and str(args.extractor) == "llm":
        if not args.model_path:
            raise SystemExit("--extractor llm requires --model_path")
        device_llm = pick_device(args.device)
        dtype_llm = pick_dtype(args.dtype, device_llm)
        sum_tok, sum_model = load_local_llm(args.model_path, device=device_llm, dtype=dtype_llm, use_fast=not args.slow_tokenizer)

    records: List[Dict[str, Any]] = []
    max_samples = int(args.max_samples)
    history_max_chars = int(args.history_max_chars)
    min_history_chars = int(args.min_history_chars)
    task_last_messages = int(args.task_last_messages)

    # Build per-turn memory snapshots.
    for p in paths:
        if max_samples > 0 and len(records) >= max_samples:
            break

        if mode == "train":
            # jsonl records: expect {"memory": {...}, ...}
            for ln in read_jsonl_lines(p, limit=None if max_samples <= 0 else (max_samples - len(records))):
                if max_samples > 0 and len(records) >= max_samples:
                    break
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                mem_obj = obj.get("memory", None)
                if not isinstance(mem_obj, dict):
                    continue
                mem = {s: str(mem_obj.get(s, "") or "") for s in slots}
                mem = normalize_slots(mem, slots)
                if len(pack_memory_slots(mem, slots)) <= 0:
                    continue
                query_text = str(mem_obj.get("task_raw") or mem.get("task") or obj.get("user") or "")
                rec = {
                    "source_path": p,
                    "user_index": int(obj.get("meta", {}).get("user_index", -1)) if isinstance(obj.get("meta", {}), dict) else -1,
                    "history_chars": None,
                    "memory_pack_chars": int(len(pack_memory_slots(mem, slots))),
                    "compression_ratio": None,
                    "memory": mem,
                    "query_text": query_text,
                }
                records.append(rec)
            continue

        # logs mode
        msgs = parse_log_file(p)
        if not msgs:
            continue

        slot_state = {s: "" for s in slots}
        for idx, history, user_text, _assistant_log in iter_user_turns(msgs, stride=1):
            if max_samples > 0 and len(records) >= max_samples:
                break
            hist_plus = history + [Msg(role="user", content=user_text)]
            hist_text = format_history(hist_plus, max_chars=history_max_chars)
            if min_history_chars > 0 and len(hist_text) < min_history_chars:
                continue

            if str(args.extractor) == "llm" and sum_model is not None:
                dbg: Dict[str, Any] = {}
                mem = llm_extract_slots(
                    sum_model, sum_tok, hist_text, slots=slots,
                    max_new_tokens=int(args.extract_llm_max_new_tokens),
                    retries=int(args.extract_retries),
                    debug=dbg,
                )
                if mem is None:
                    mem = heuristic_extract_slots(hist_plus, slots=slots, task_last_messages=task_last_messages)
                src = {"extractor": "llm", "llm_debug": dbg, "llm_ok": bool(mem is not None and dbg.get("parsed"))}
            else:
                mem = heuristic_extract_slots(hist_plus, slots=slots, task_last_messages=task_last_messages)
                src = {"extractor": "heuristic"}

            mem = normalize_slots(mem, slots)
            task_raw = heuristic_task_only(hist_plus, last_n=task_last_messages)
            task_raw = strip_command_like_lines(task_raw)
            task_raw = normalize_slots({"task_raw": task_raw}, ["task_raw"]).get("task_raw", "")
            mem["task_raw"] = task_raw
            task_sum = summarize_task_from_raw(task_raw)
            if task_sum:
                mem["task"] = normalize_slots({"task": task_sum}, ["task"]).get("task", "")
            elif user_text:
                mem["task"] = normalize_slots({"task": user_text}, ["task"]).get("task", "")

            slot_state = merge_slots(slot_state, mem, slots)
            snapshot = {s: str(slot_state.get(s, "") or "") for s in slots}
            snapshot = normalize_slots(snapshot, slots)

            mem_pack = pack_memory_slots(snapshot, slots)
            rec = {
                "source_path": p,
                "user_index": int(idx),
                "history_chars": int(len(hist_text)),
                "memory_pack_chars": int(len(mem_pack)),
                "compression_ratio": (float(len(hist_text)) / float(max(1, len(mem_pack)))) if mem_pack else None,
                "memory": snapshot,
                "query_text": str(task_raw or snapshot.get("task") or user_text or ""),
                "source": src,
            }
            records.append(rec)

    if len(records) < 2:
        raise SystemExit(f"Not enough samples for retrieval metrics (n={len(records)}).")

    # Query embedding for "silver relevance" (char n-gram similarity).
    cpu = torch.device("cpu")
    q_embeds = [embed_text_4096(str(r.get("query_text","") or ""), device=cpu, embed_backend=EMBED_BACKEND_CHAR_NGRAM4096) for r in records]
    E = torch.stack(q_embeds, dim=0)  # [N, 4096]

    # Contamination stats (slot-level)
    contam_totals: Dict[str, Dict[str, int]] = {s: {} for s in slots}
    for r in records:
        mem = r["memory"]
        for s in slots:
            flags = _is_contaminated_slot(s, mem.get(s, ""))
            for k,v in flags.items():
                contam_totals[s][k] = int(contam_totals[s].get(k, 0)) + (1 if v else 0)

    # Build per-backend baton vectors and retrieval metrics.
    window = int(args.window)
    top_k = int(args.top_k)
    rel_mode = str(args.relevance_mode or "topk").strip().lower()
    rel_top = int(args.relevance_top)
    rel_thr = float(args.relevance_threshold)
    baton_dim = int(args.baton_dim)
    proj_seed = int(args.proj_seed)

    backend_summaries: Dict[str, Any] = {}
    per_backend_arrays: Dict[str, Dict[str, List[Any]]] = {}
    proj_cache: Dict[Tuple[str, str, int, int], torch.Tensor] = {}

    for backend in embed_backends:
        vecs: List[torch.Tensor] = []
        hot_chars: List[int] = []
        for r in records:
            mem = r["memory"]
            parts = []
            total_hot = 0
            for s in slots:
                b64 = encode_baton_hashproj(
                    text=str(mem.get(s, "") or ""),
                    baton_dim=baton_dim,
                    proj_seed=proj_seed,
                    device=cpu,
                    proj_cache=proj_cache,
                    slot=s,
                    embed_backend=backend,
                )
                total_hot += len(b64)
                parts.append(unpack_smem1_int8(b64, device=cpu))
            v = torch.cat(parts, dim=0)
            v = F.normalize(v, p=2, dim=0)
            vecs.append(v)
            hot_chars.append(int(total_hot))

        V = torch.stack(vecs, dim=0)  # [N, slot_dim]

        n_eval = 0
        precision_sum = 0.0
        recall_sum = 0.0
        mrr_sum = 0.0
        relevant_sum = 0.0
        k_eff_sum = 0.0
        hits_sum = 0.0

        per_record_prec: List[Optional[float]] = [None] * len(records)
        per_record_rr: List[Optional[float]] = [None] * len(records)
        per_record_rel_n: List[int] = [0] * len(records)
        per_record_hits: List[int] = [0] * len(records)

        for i in range(len(records)):
            start = 0 if window <= 0 else max(0, i - window)
            if start >= i:
                continue

            sim_rel = (E[start:i] @ E[i]).to(torch.float32)
            if rel_mode == "topk":
                k_rel = min(max(1, rel_top), int(sim_rel.numel()))
                rel_idxs = torch.topk(sim_rel, k=k_rel, largest=True).indices
                relevant = torch.zeros_like(sim_rel, dtype=torch.bool)
                relevant[rel_idxs] = True
                rel_n = int(k_rel)
            else:
                relevant = sim_rel >= rel_thr
                rel_n = int(relevant.sum().item())
            per_record_rel_n[i] = rel_n
            if rel_n <= 0:
                continue

            sim = (V[start:i] @ V[i]).to(torch.float32)
            k = min(top_k, int(sim.numel()))
            if k <= 0:
                continue
            idxs = torch.topk(sim, k=k, largest=True).indices
            rel_at = relevant[idxs]
            hits = int(rel_at.sum().item())
            per_record_hits[i] = hits

            precision = float(hits) / float(k)
            recall = float(hits) / float(rel_n)
            rr = 0.0
            if hits > 0:
                first = int(torch.nonzero(rel_at, as_tuple=False)[0].item())
                rr = 1.0 / float(first + 1)

            per_record_prec[i] = precision
            per_record_rr[i] = rr

            n_eval += 1
            precision_sum += precision
            recall_sum += recall
            mrr_sum += rr
            relevant_sum += float(rel_n)
            k_eff_sum += float(k)
            hits_sum += float(hits)

        backend_summaries[backend] = {
            "n_records": int(len(records)),
            "n_eval": int(n_eval),
            "window": int(window),
            "top_k": int(top_k),
            "relevance_mode": rel_mode,
            "relevance_top": int(rel_top),
            "relevance_threshold": float(rel_thr),
            "hot_b64chars_mean": _mean([float(x) for x in hot_chars]),
            "hot_b64chars_min": int(min(hot_chars)) if hot_chars else None,
            "hot_b64chars_max": int(max(hot_chars)) if hot_chars else None,
            "precision_at_k_mean": (precision_sum / float(max(1, n_eval))) if n_eval > 0 else None,
            "recall_at_k_mean": (recall_sum / float(max(1, n_eval))) if n_eval > 0 else None,
            "mrr_mean": (mrr_sum / float(max(1, n_eval))) if n_eval > 0 else None,
            "avg_relevant_per_query": (relevant_sum / float(max(1, n_eval))) if n_eval > 0 else None,
            "avg_hits_at_k": (hits_sum / float(max(1, n_eval))) if n_eval > 0 else None,
            "avg_k_eff": (k_eff_sum / float(max(1, n_eval))) if n_eval > 0 else None,
        }
        per_backend_arrays[backend] = {
            "hot_b64chars": hot_chars,
            "precision_at_k": per_record_prec,
            "rr": per_record_rr,
            "relevant_n": per_record_rel_n,
            "hits": per_record_hits,
        }

    compression = [float(r["compression_ratio"]) for r in records if isinstance(r.get("compression_ratio"), (int,float))]
    out = {
        "created_at": now_str(),
        "mode": mode,
        "input": inp,
        "paths_count": int(len(paths)),
        "n_records": int(len(records)),
        "slots": slots,
        "embed_backends": embed_backends,
        "baton_dim": baton_dim,
        "proj_seed": proj_seed,
        "relevance_backend": EMBED_BACKEND_CHAR_NGRAM4096,
        "relevance_mode": rel_mode,
        "relevance_top": int(rel_top),
        "relevance_threshold": float(rel_thr),
        "compression_ratio_mean": _mean(compression) if compression else None,
        "compression_ratio_min": float(min(compression)) if compression else None,
        "compression_ratio_max": float(max(compression)) if compression else None,
        "contamination_totals": contam_totals,
        "backend_metrics": backend_summaries,
    }

    if args.out_json:
        write_json(args.out_json, out, indent=2)

    if args.out_jsonl:
        # per-record output for plotting
        if (not args.append) and os.path.exists(args.out_jsonl):
            os.remove(args.out_jsonl)
        for i, r in enumerate(records):
            row = {
                "i": int(i),
                "source_path": r.get("source_path"),
                "user_index": int(r.get("user_index", -1)),
                "history_chars": r.get("history_chars"),
                "memory_pack_chars": int(r.get("memory_pack_chars", 0)),
                "compression_ratio": r.get("compression_ratio"),
                "memory": r.get("memory"),
                "query_text": r.get("query_text"),
                "backend": {b: {
                    "hot_b64chars": per_backend_arrays[b]["hot_b64chars"][i],
                    "relevant_n": per_backend_arrays[b]["relevant_n"][i],
                    "hits": per_backend_arrays[b]["hits"][i],
                    "precision_at_k": per_backend_arrays[b]["precision_at_k"][i],
                    "rr": per_backend_arrays[b]["rr"][i],
                } for b in embed_backends},
            }
            append_jsonl(args.out_jsonl, row)

    if args.verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"[done] n_records={len(records)} backends={embed_backends} -> {args.out_json}")


# =============================
# Main CLI
# =============================


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # check-model
    c = sub.add_parser("check-model")
    c.add_argument("--model_path", required=True)
    c.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"])
    c.add_argument("--dtype", default="auto", choices=["auto","bf16","fp16","fp32"])
    c.add_argument("--slow_tokenizer", action="store_true")
    c.set_defaults(func=cmd_check_model)

    # capsule
    cap = sub.add_parser("capsule")
    cap_sub = cap.add_subparsers(dest="cap_cmd", required=True)

    e = cap_sub.add_parser("encode")
    e.add_argument("--in", dest="in_path", default=None, help="input text file (default stdin)")
    e.add_argument("--out", dest="out_path", required=True, help="output capsule json")
    e.add_argument("--proj_dim", type=int, default=256)
    e.add_argument("--proj_seed", type=int, default=0)
    e.add_argument("--embed_backend", default=EMBED_BACKEND_HASH4096, choices=list(EMBED_BACKEND_CHOICES))
    e.add_argument("--slot", default="default")
    e.add_argument("--store_archive", action="store_true")
    e.add_argument("--passphrase", default=None)
    e.set_defaults(func=cmd_capsule_encode)

    d = cap_sub.add_parser("decode")
    d.add_argument("--capsule", required=True)
    d.add_argument("--passphrase", default=None)
    d.set_defaults(func=cmd_capsule_decode)

    r = cap_sub.add_parser("render")
    r.add_argument("--capsule", required=True)
    r.add_argument("--chunk", type=int, default=80)
    r.set_defaults(func=cmd_capsule_render)

    # chat
    ch = sub.add_parser("chat")
    ch_sub = ch.add_subparsers(dest="chat_cmd", required=True)

    repl = ch_sub.add_parser("repl")
    repl.add_argument("--model_path", required=True)
    repl.add_argument("--session_dir", default=None)
    repl.add_argument("--resume", default=None)
    repl.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"])
    repl.add_argument("--dtype", default="auto", choices=["auto","bf16","fp16","fp32"])
    repl.add_argument("--slow_tokenizer", action="store_true")

    repl.add_argument("--slots", default="persona,rules,task")
    repl.add_argument("--extractor", default="heuristic", choices=["heuristic","llm"])
    repl.add_argument("--extract_llm_every", type=int, default=5)
    repl.add_argument("--extract_history_max_chars", type=int, default=12000)
    repl.add_argument("--extract_llm_max_new_tokens", type=int, default=400)
    repl.add_argument("--task_last_messages", type=int, default=8)

    repl.add_argument("--context_mode", default="slot", choices=["full","slot","slot_only"])
    repl.add_argument("--recent_messages", type=int, default=6)

    repl.add_argument("--baton_dim", type=int, default=256)
    repl.add_argument("--proj_seed", type=int, default=0)
    repl.add_argument("--embed_backend", default=EMBED_BACKEND_HASH4096, choices=list(EMBED_BACKEND_CHOICES))

    repl.add_argument("--max_new_tokens", type=int, default=256)
    repl.add_argument("--temperature", type=float, default=0.7)
    repl.add_argument("--top_p", type=float, default=0.9)
    repl.add_argument("--max_input_tokens", type=int, default=0,
                      help="prompt token budget before generation (0=auto by model context)")
    repl.add_argument("--input_safety_tokens", type=int, default=96,
                      help="reserved token headroom when auto-budgeting input")
    repl.add_argument("--slot_runtime_max_chars", type=int, default=900,
                      help="runtime slot text trim limit used only when prompt exceeds token budget")
    repl.add_argument("--slot_confidence_threshold", type=float, default=0.25,
                      help="minimum confidence to inject a slot into prompt (0 disables filtering)")
    repl.add_argument("--slot_conf_mode", default="balanced",
                      choices=["balanced", "conservative", "aggressive", "replace"],
                      help="confidence update rule for slot metadata")
    repl.add_argument("--slot_conf_floor", type=float, default=0.35,
                      help="minimum incoming confidence floor per slot update")
    repl.add_argument("--slot_conf_decay", type=float, default=0.55,
                      help="history decay used by confidence update rules")
    repl.add_argument("--slot_conf_merge_bonus", type=float, default=0.05,
                      help="bonus when new slot value extends previous value")
    repl.add_argument("--relay_keyframe_every", type=int, default=6,
                      help="write full memory keyframe every N turns (0 disables relay log)")
    repl.add_argument("--relay_max_entries", type=int, default=600,
                      help="max entries to retain in memory relay log")

    repl.add_argument("--system_text", default=None)
    repl.add_argument("--system_file", default=None)

    repl.add_argument("--store_archive", action="store_true")
    repl.add_argument("--archive_passphrase", default=None)

    repl.set_defaults(func=cmd_chat_repl)

    # dataforge
    df = sub.add_parser("dataforge")
    df_sub = df.add_subparsers(dest="df_cmd", required=True)

    b = df_sub.add_parser("build")
    b.add_argument("--input", required=True)
    b.add_argument("--out_jsonl", required=True)

    b.add_argument("--teacher_path", required=True)
    b.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"])
    b.add_argument("--dtype", default="auto", choices=["auto","bf16","fp16","fp32"])
    b.add_argument("--slow_tokenizer", action="store_true")

    b.add_argument("--slots", default="persona,rules,task")
    b.add_argument("--extractor", default="llm", choices=["llm","heuristic"])
    b.add_argument("--history_max_chars", type=int, default=12000)
    b.add_argument("--slot_max_chars", type=int, default=6000)
    b.add_argument("--task_last_messages", type=int, default=8)

    b.add_argument("--baton_dim", type=int, default=256)
    b.add_argument("--proj_seed", type=int, default=0)
    b.add_argument("--embed_backend", default=EMBED_BACKEND_HASH4096, choices=list(EMBED_BACKEND_CHOICES))

    b.add_argument("--stride", type=int, default=1)
    b.add_argument("--max_samples", type=int, default=0)
    b.add_argument("--min_history_chars", type=int, default=200)

    b.add_argument("--system_text", default=None)
    b.add_argument("--system_file", default=None)

    b.add_argument("--teacher_max_new_tokens", type=int, default=256)
    b.add_argument("--teacher_temperature", type=float, default=0.2)
    b.add_argument("--teacher_top_p", type=float, default=0.9)

    b.add_argument("--label_source", default="teacher", choices=["teacher","log"])
    b.add_argument("--append", action="store_true")
    b.add_argument("--verbose", action="store_true")
    b.set_defaults(func=cmd_dataforge_build)

    # gateway
    gw = sub.add_parser("gateway")
    gw.add_argument("--host", default="127.0.0.1")
    gw.add_argument("--port", type=int, default=8088)
    gw.add_argument("--store_dir", default="gateway_store")
    gw.add_argument("--slots", default="persona,rules,task")
    gw.add_argument("--system_text", default=None)

    gw.add_argument("--baton_dim", type=int, default=256)
    gw.add_argument("--proj_seed", type=int, default=0)
    gw.add_argument("--embed_backend", default=EMBED_BACKEND_HASH4096, choices=list(EMBED_BACKEND_CHOICES))
    gw.add_argument("--top_k", type=int, default=3)
    gw.add_argument("--inject_max_chars", type=int, default=3500)

    gw.add_argument("--summarize_mode", default="heuristic", choices=["none","heuristic","llm"])
    gw.add_argument("--summarize_max_chars", type=int, default=1500)
    gw.add_argument("--summarize_llm_max_new_tokens", type=int, default=220)

    gw.add_argument("--extractor", default="heuristic", choices=["heuristic","llm"])
    gw.add_argument("--extract_history_max_chars", type=int, default=12000)
    gw.add_argument("--extract_llm_max_new_tokens", type=int, default=400)
    gw.add_argument("--task_last_messages", type=int, default=8)

    gw.add_argument("--summarizer_model_path", default=None)
    gw.add_argument("--store_archive", action="store_true")
    gw.add_argument("--archive_passphrase", default=None)

    # ready-to-send request defaults
    gw.add_argument("--openai_model", default="gpt-5")
    gw.add_argument("--openai_max_tokens", type=int, default=512)
    gw.add_argument("--openai_max_output_tokens", type=int, default=512)
    gw.add_argument("--openai_temperature", type=float, default=0.7)
    gw.add_argument("--openai_top_p", type=float, default=1.0)

    gw.add_argument("--anthropic_model", default="claude-sonnet-4-5")
    gw.add_argument("--anthropic_max_tokens", type=int, default=512)
    gw.add_argument("--anthropic_temperature", type=float, default=None)
    gw.add_argument("--anthropic_top_p", type=float, default=None)
    gw.add_argument("--anthropic_version", default="2023-06-01")

    gw.add_argument("--reextract_each_call", action="store_true",
                    help="always re-extract slots from transcript on every /chat call (slower but robust)")
    gw.add_argument("--external_full_extract_every", type=int, default=5,
                    help="when external_reply is provided, do full extraction every N calls; otherwise update task only")

    gw.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"])
    gw.add_argument("--dtype", default="auto", choices=["auto","bf16","fp16","fp32"])
    gw.add_argument("--slow_tokenizer", action="store_true")
    gw.set_defaults(func=cmd_gateway)

    # bench
    be = sub.add_parser("bench")
    be.add_argument("--mode", default="auto", choices=["auto","logs","train"])
    be.add_argument("--input", required=True)
    be.add_argument("--out_json", default="bench_out.json")
    be.add_argument("--out_jsonl", default=None)
    be.add_argument("--append", action="store_true")
    be.add_argument("--verbose", action="store_true")

    be.add_argument("--slots", default="persona,rules,task")
    be.add_argument("--embed_backend", default=EMBED_BACKEND_HASH4096,
                    help="comma-separated list, e.g. 'hash4096,charngram4096'")
    be.add_argument("--baton_dim", type=int, default=256)
    be.add_argument("--proj_seed", type=int, default=0)
    be.add_argument("--top_k", type=int, default=3)
    be.add_argument("--window", type=int, default=800,
                    help="candidate window for relevance/retrieval (0 = all previous)")
    be.add_argument("--relevance_mode", default="topk", choices=["topk","threshold"],
                    help="how to define silver relevance among previous turns")
    be.add_argument("--relevance_top", type=int, default=8,
                    help="relevance set size when --relevance_mode topk")
    be.add_argument("--relevance_threshold", type=float, default=0.32,
                    help="silver relevance threshold in charngram cosine space")

    # logs-mode extraction controls
    be.add_argument("--extractor", default="heuristic", choices=["heuristic","llm"])
    be.add_argument("--model_path", default=None)
    be.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"])
    be.add_argument("--dtype", default="auto", choices=["auto","bf16","fp16","fp32"])
    be.add_argument("--slow_tokenizer", action="store_true")
    be.add_argument("--history_max_chars", type=int, default=12000)
    be.add_argument("--min_history_chars", type=int, default=200)
    be.add_argument("--task_last_messages", type=int, default=8)
    be.add_argument("--extract_llm_max_new_tokens", type=int, default=220)
    be.add_argument("--extract_retries", type=int, default=1)

    be.add_argument("--max_samples", type=int, default=0)
    be.set_defaults(func=cmd_bench)

    # b1plus
    b1 = sub.add_parser("b1plus")
    b1_sub = b1.add_subparsers(dest="b1_cmd", required=True)

    t = b1_sub.add_parser("train")
    t.add_argument("--student_path", required=True)
    t.add_argument("--teacher_path", default=None)
    t.add_argument("--train_jsonl", required=True)
    t.add_argument("--out_dir", default="b1plus_out")

    t.add_argument("--slots", default="persona,rules,task")
    t.add_argument("--baton_dim", type=int, default=256)
    t.add_argument("--n_prefix_max", type=int, default=32)
    t.add_argument("--slot_prefix_max", default=None)
    t.add_argument("--min_prefix", type=int, default=0)
    t.add_argument("--mlp_width", type=int, default=2048)

    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--grad_accum", type=int, default=4)
    t.add_argument("--log_every", type=int, default=20)
    t.add_argument("--save_every", type=int, default=200)
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--seed", type=int, default=0)

    t.add_argument("--tau", type=float, default=1.5)
    t.add_argument("--kd_alpha", type=float, default=1.0)
    t.add_argument("--ce_alpha", type=float, default=0.0)
    t.add_argument("--len_alpha", type=float, default=0.01)

    t.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"])
    t.add_argument("--dtype", default="auto", choices=["auto","bf16","fp16","fp32"])
    t.add_argument("--slow_tokenizer", action="store_true")
    t.set_defaults(func=cmd_b1plus_train)

    inf = b1_sub.add_parser("infer")
    inf.add_argument("--prefix_ckpt", required=True)
    inf.add_argument("--baton_json", required=True, help="json string or file path: {slot:b64,...}")
    inf.add_argument("--user", required=True)
    inf.add_argument("--system", default=None)
    inf.add_argument("--max_new_tokens", type=int, default=200)
    inf.add_argument("--do_sample", action="store_true")
    inf.add_argument("--top_p", type=float, default=0.9)
    inf.add_argument("--temperature", type=float, default=0.8)
    inf.add_argument("--truncate_threshold", type=float, default=0.5)

    inf.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"])
    inf.add_argument("--dtype", default="auto", choices=["auto","bf16","fp16","fp32"])
    inf.add_argument("--slow_tokenizer", action="store_true")
    inf.set_defaults(func=cmd_b1plus_infer)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
