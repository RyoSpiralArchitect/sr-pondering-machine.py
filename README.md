# sr_pondering_machine.py 🐈‍⬛

A **pondering machine** for local Hugging Face causal language models — optimised for Apple Silicon (MPS) and Gemma instruction-tuned models.

## What Is a "Pondering Machine"?

Instead of answering a question directly, the pondering machine first *wanders off on a tangent* before giving the final answer.

The core idea:

1. **Probe the model** — run the user query through the model and capture the next-token logits.
2. **Pick "rejected" tokens** — select tokens that scored well but were *not* at the top of the distribution. These act as seeds for an unrelated train of thought.
3. **Generate a ponder log** — ask the model to freely associate around those keywords. The result is a short, stream-of-consciousness log with no conclusions.
4. **Answer with context** — feed the ponder log back into the model as soft background context before producing the final answer.

The hypothesis is that the tangential pondering log can surface hidden assumptions or alternative framings, leading to richer answers.

## Features

- **MPS-first** — resolves Apple Silicon device-mismatch errors automatically.
- **Gemma-aware** — natively applies the `<start_of_turn>` / `<end_of_turn>` prompt format expected by Gemma IT models and stops generation cleanly at `<end_of_turn>`.
- **Persistent memory** — ponder logs are stored in a JSONL file and the most recent entries are reused in subsequent runs.
- **Baseline mode** — run without pondering for easy A/B comparison.
- **Ponder lenses** — switch between association / assumptions / counterexamples / questions-only / metaphor via `--ponder_mode`.
- **Multi-ponder** — generate multiple ponder logs per band with `--n_ponder`.
- **Keyword refinement** — optionally rewrite token fragments into cleaner keywords via `--keyword_refine`.
- **Prompt language auto** — `--prompt_lang auto|en|ja` (auto-detects Japanese queries).
- **Probe tracing** — print/store probe token info with `--print_probe` / `--probe_top_n`.
- **Spectral bands** — run multiple rank-bands (near/mid/far) via `--band_profile spectrum3` or define custom bands with `--band`.
- **Highly configurable** — token selection strategy, generation hyperparameters, device, dtype, and more are all adjustable from the command line.

## Requirements

- Python 3.9+
- [PyTorch](https://pytorch.org/) (with MPS, CUDA, or CPU support)
- [Transformers](https://github.com/huggingface/transformers) (`pip install transformers`)
- A locally downloaded Hugging Face causal LM (e.g. `gemma-3-270m-it`)

```
pip install torch transformers
```

> **Apple Silicon tip:** if you encounter unsupported MPS ops, set:
> ```
> export PYTORCH_ENABLE_MPS_FALLBACK=1
> ```

## Quick Start

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "Explain quantum entanglement to a high school student" \
  --mode both \
  --memory ./ponder_logs.jsonl
```

This runs both the **baseline** (direct answer) and the **ponder** (wander-then-answer) modes and prints both outputs for comparison.

## Experimental Recipes

### Lens: counterexamples + multi-ponder

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "Should we optimize for accuracy or speed in LLM systems?" \
  --mode ponder \
  --ponder_mode counterexample \
  --n_ponder 3 \
  --memory_policy current_only
```

### Spectral: near/mid/far bands + current-only memory

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "Why do we overfit narratives to randomness?" \
  --mode ponder \
  --band_profile spectrum3 \
  --n_ponder 2 \
  --memory_policy current_only
```

### Custom bands (rank ranges)

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "Design a new kind of musical scale." \
  --mode ponder \
  --band "near=8:64" \
  --band "mid=80:336" \
  --band "far=1200:2400" \
  --n_ponder 1 \
  --memory_policy current_only \
  --print_probe
```

### Lens: metaphor + keyword refinement (more dreamlike)

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "量子もつれを高校生にも分かるように説明して" \
  --mode ponder \
  --prompt_lang auto \
  --ponder_mode metaphor \
  --keyword_refine \
  --print_probe
```

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | Path to a local model directory. |
| `--query` | *(required)* | The question to answer. |
| `--memory` | `ponder_logs.jsonl` | Path to the JSONL memory log. |
| `--mode` | `both` | `baseline` · `ponder` · `both` |
| `--prompt_lang` | `auto` | Prompt language: `auto` · `en` · `ja` |
| `--ponder_mode` | `assoc` | `assoc` · `assumption` · `counterexample` · `questions_only` · `metaphor` |
| `--n_ponder` | `1` | Number of ponder logs per band (total logs = `n_ponder * n_bands`). |
| `--memory_policy` | `tail` | Inject: `tail` (last `n_memory`) · `current_only` · `off` |
| `--keyword_refine` | `False` | Let the model rewrite token fragments into keywords. |
| `--keyword_refine_max_new_tokens` | `96` | Keyword-refine max new tokens. |
| `--keyword_refine_temperature` | `0.3` | Keyword-refine temperature. |
| `--probe_top_n` | `0` | Store probe top-N tokens into the JSONL record (0 = off). |
| `--print_probe` | `False` | Print probe tables to stdout. |
| `--band_profile` | `single` | Rank-band profile: `single` · `spectrum3` |
| `--band` | *(none)* | Custom band(s): `START:END` or `LABEL=START:END` (repeatable, END exclusive). |
| `--device` | `auto` | `auto` · `mps` · `cpu` · `cuda` · `cuda:0` … |
| `--dtype` | `auto` | `auto` · `float16` · `bfloat16` · `float32` |
| `--trust_remote_code` | `False` | Pass `--trust_remote_code` to enable. |
| `--no_chat_template` | `False` | Disable the tokenizer chat template. |
| `--no_gemma_format` | `False` | Disable Gemma-native turn formatting. |
| `--strategy` | `outside_topk` | Rejected-token selection strategy: `within_topk` or `outside_topk`. |
| `--top_k_rejected` | `80` | Top-K cutoff for rejected token selection. |
| `--exclude_top` | `8` | Number of top tokens to exclude (used with `within_topk`). |
| `--band_width` | `256` | Width of the candidate band (used with `outside_topk`). |
| `--n_keywords` | `6` | Number of keywords to extract. |
| `--n_memory` | `6` | Number of recent ponder records to include as memory. |
| `--answer_max_new_tokens` | `256` | Max new tokens for the answer step. |
| `--ponder_max_new_tokens` | `160` | Max new tokens for the ponder log step. |
| `--temperature` | `0.7` | Sampling temperature. |
| `--top_p` | `0.95` | Nucleus sampling probability. |
| `--top_k` | `0` | Top-K sampling (0 = disabled). |
| `--repetition_penalty` | `1.05` | Repetition penalty. |
| `--no_repeat_ngram_size` | `0` | No-repeat n-gram size (0 = disabled). |
| `--seed` | `1234` | Random seed for reproducibility. |

## Rejected-Token Selection Strategies

| Strategy | Description |
|---|---|
| `outside_topk` | Picks tokens ranked just *outside* the top-K (positions `top_k_rejected` to `top_k_rejected + band_width`). These are plausible but not dominant tokens. |
| `within_topk` | Picks tokens from *inside* the top-K window, skipping the very top `exclude_top` tokens. |

## Memory Log Format

Each ponder run appends a JSON record to the JSONL file:

```json
{
  "ts": "2025-01-01T00:00:00Z",
  "run_id": "a1b2c3d4e5f6",
  "band_profile": "single",
  "band_label": "outside_topk:80:336",
  "ponder_ix": 0,
  "prompt_lang": "en",
  "ponder_mode": "assoc",
  "keywords": ["gravity", "mirror", "sleep"],
  "keywords_raw": ["gravity", "mirror", "sleep"],
  "keywords_source": "rejected_tokens",
  "token_ids": [1234, 5678, 9012],
  "selected_tokens": [{ "token": "gravity", "token_id": 1234, "rank": 120, "prob": 0.0123 }],
  "rejected_cfg": { "top_k": 80, "strategy": "outside_topk", "exclude_top": 8, "band_width": 256, "n_keywords": 6 },
  "ponder_question": "Using gravity and mirror and the other words, what do you freely associate without any specific purpose?",
  "ponder_log": "- A mirror hanging in zero-g would still reflect ...\n- ..."
}
```

The logs injected into the final answer depend on `--memory_policy`:

- `tail`: the most recent `n_memory` records
- `current_only`: only the logs generated in the current run
- `off`: no log injection (still writes to JSONL)

## Memory Report (Visualization)

Generate a quick, dependency-free report from your JSONL memory:

```bash
python3 sr_ponder_report.py --memory ./ponder_logs.jsonl
python3 sr_ponder_report.py --memory ./ponder_logs.jsonl --out ./ponder_report.html
```

## Notes

- **Use the `-it` (instruction-tuned) variant** of Gemma for best results. The base model does not reliably follow instructions.
- The model must already be **downloaded locally**. Network access is disabled at inference time (`local_files_only=True`).
- `--keyword_refine` adds an extra generation call before the ponder step (slower, but often produces better keywords).
