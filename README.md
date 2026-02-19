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

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | Path to a local model directory. |
| `--query` | *(required)* | The question to answer. |
| `--memory` | `ponder_logs.jsonl` | Path to the JSONL memory log. |
| `--mode` | `both` | `baseline` · `ponder` · `both` |
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
  "keywords": ["gravity", "mirror", "sleep"],
  "token_ids": [1234, 5678, 9012],
  "rejected_cfg": { "top_k": 80, "strategy": "outside_topk", "exclude_top": 8, "band_width": 256, "n_keywords": 6 },
  "ponder_question": "Using gravity and mirror and the other words, what do you freely associate without any specific purpose?",
  "ponder_log": "- A mirror hanging in zero-g would still reflect ...\n- ..."
}
```

The most recent `n_memory` records are injected as soft context when generating the final pondered answer.

## Notes

- **Use the `-it` (instruction-tuned) variant** of Gemma for best results. The base model does not reliably follow instructions.
- The model must already be **downloaded locally**. Network access is disabled at inference time (`local_files_only=True`).
