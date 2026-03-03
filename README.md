# sr_pondering_machine.py 🐈‍⬛

A **pondering machine** for local Hugging Face causal language models *and* OpenAI-compatible API models — optimised for Apple Silicon (MPS) and Gemma instruction-tuned models.

## What Is a "Pondering Machine"?

Instead of answering a question directly, the pondering machine first *wanders off on a tangent* before giving the final answer.

The core idea:

1. **Probe the model** — capture the next-token distribution (local logits, or API top-logprobs when available).
2. **Pick "rejected" tokens** — select tokens that scored well but were *not* at the top of the distribution. These act as seeds for an unrelated train of thought.
3. **Generate a ponder log** — ask the model to freely associate around those keywords. The result is a short, stream-of-consciousness log with no conclusions.
4. **Answer with context** — feed the ponder log back into the model as soft background context before producing the final answer.

The hypothesis is that the tangential pondering log can surface hidden assumptions or alternative framings, leading to richer answers.

## Features

- **API backend** — `--backend openai_compat` supports OpenAI-style `POST /chat/completions` providers.
- **MPS-first** — resolves Apple Silicon device-mismatch errors automatically.
- **Gemma-aware** — natively applies the `<start_of_turn>` / `<end_of_turn>` prompt format expected by Gemma IT models and stops generation cleanly at `<end_of_turn>`.
- **Persistent memory** — ponder logs are stored in a JSONL file and the most recent entries are reused in subsequent runs.
- **Baseline mode** — run without pondering for easy A/B comparison.
- **Ponder lenses** — switch between association / assumptions / counterexamples / questions-only / metaphor via `--ponder_mode`.
- **Lens pipeline** — chain multiple lenses via `--ponder_pipeline` (with `--pipeline_context prev|all|none`).
- **Multi-ponder** — generate multiple ponder logs per band with `--n_ponder`.
- **Latent walk (hops)** — chain multiple drift steps via `--ponder_hops` (next-hop seeds from `--hop_keyword_source model|heuristic`).
- **Presets** — apply curated settings with one flag (e.g. `--preset surreal`).
- **Keyword refinement** — optionally rewrite token fragments into cleaner keywords via `--keyword_refine`.
- **Keyword objectives** — pick seeds by dissonance/instability via `--keyword_objective dissonance|unstable`.
- **Keyword diversity** — encourage more diverse seed keywords via `--keyword_diversity lex|embed`.
- **Prompt jitter** — paraphrase the query (`--prompt_jitter`) to find unstable seed tokens (sharper drift).
- **Prompt language auto** — `--prompt_lang auto|en|ja` (auto-detects Japanese queries).
- **Probe tracing** — print/store probe token info with `--print_probe` / `--probe_top_n`.
- **Spectral bands** — run multiple rank-bands (near/mid/far) via `--band_profile spectrum3` or define custom bands with `--band`.
- **Memory retrieval** — pick memory by similarity/anti-similarity via `--memory_retrieve similar|anti|mix`.
- **Memory remix** — shuffle/compress/dream the injected memory via `--memory_remix`.
- **Answer style** — steer the final answer style via `--answer_style plain|surreal|metaphor|meta`.
- **Answer ensemble** — generate per-band answers + merged answer with `--answer_per_band` / `--answer_ensemble`.
- **Interactive pick** — choose keyword tokens yourself with `--interactive`.
- **Controls pack** — run A/B packs via `--pack controls|surreal` (or bring your own with `--pack_file`).
- **Pack resume** — skip already-completed pack items with `--pack_resume`.
- **Artifacts & tracing** — export results with `--json_out` and write step-level traces with `--trace_out` (or use `--out_dir` for auto-naming).
- **Trace report** — turn `--trace_out` into an HTML timeline with `sr_trace_report.py`.
- **Config file** — set defaults from a JSON file via `--config` (CLI flags still override).
- **Highly configurable** — token selection strategy, generation hyperparameters, device, dtype, and more are all adjustable from the command line.

## Requirements

- Python 3.9+
- For `--backend hf`:
  - [PyTorch](https://pytorch.org/) (with MPS, CUDA, or CPU support)
  - [Transformers](https://github.com/huggingface/transformers)
  - A locally downloaded Hugging Face causal LM (e.g. `gemma-3-270m-it`) — or use `--hf_online` to allow Hub downloads.
- For `--backend openai_compat`:
  - Python stdlib only (no `torch`/`transformers` required)

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
  --backend hf \
  --model ./model/gemma-3-270m-it \
  --query "Explain quantum entanglement to a high school student" \
  --mode both \
  --memory ./ponder_logs.jsonl
```

This runs both the **baseline** (direct answer) and the **ponder** (wander-then-answer) modes and prints both outputs for comparison.

## API Quick Start (OpenAI-compatible)

Use `--backend openai_compat` to call an OpenAI-style API endpoint.
This backend can run without installing `torch`/`transformers`.

```bash
export OPENAI_API_KEY="..."

python3 sr_pondering_machine.py \
  --backend openai_compat \
  --api_base_url https://api.openai.com/v1 \
  --model "your-model-name" \
  --query "Should we optimize for accuracy or speed in LLM systems?" \
  --mode ponder \
  --api_seed_method self
```

If your provider supports Chat Completions logprobs, you can try the more “rejected-token-ish” seeding:

```bash
python3 sr_pondering_machine.py \
  --backend openai_compat \
  --api_base_url https://api.openai.com/v1 \
  --model "your-model-name" \
  --query "Why do we overfit narratives to randomness?" \
  --mode ponder \
  --api_seed_method logprobs \
  --api_logprobs_top_n 128
```

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

### Latent walk: hops + keyword diversity (stronger drift)

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "What is creativity, really?" \
  --mode ponder \
  --band_profile spectrum3 \
  --keyword_objective dissonance \
  --keyword_diversity embed \
  --ponder_hops 3 \
  --hop_keyword_source model \
  --memory_policy current_only
```

### Preset: surreal (one-flag setup)

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "創造性って結局なに？" \
  --mode ponder \
  --preset surreal
```

### Pack: surreal (compare a few curated weirdness variants)

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "現実って何のインターフェース？" \
  --pack surreal \
  --pack_out ./pack_surreal.json
```

Tip: pack results write to `--pack_out` (or `--json_out`), and traces write to `--trace_out` (or auto-name both via `--out_dir`).
If you re-run the same pack with the same `--pack_out`/`--json_out`, you can resume/skip completed items:

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "現実って何のインターフェース？" \
  --pack surreal \
  --pack_out ./pack_surreal.json \
  --pack_resume
```

### Pack: custom (JSON file)

Create `pack.json`:

```json
{
  "name": "surreal_lab",
  "base_cfg": {
    "band_profile": "spectrum3",
    "memory_policy": "current_only"
  },
  "items": [
    { "name": "baseline_plain", "kind": "baseline", "cfg": { "answer_style": "plain" } },
    {
      "name": "walk_dissonance",
      "kind": "ponder",
      "control": "none",
      "cfg": {
        "answer_style": "surreal",
        "ponder_hops": 3,
        "keyword_objective": "dissonance",
        "keyword_diversity": "embed",
        "ponder_pipeline": ["metaphor", "metaphor"]
      }
    },
    { "name": "lens_only_metaphor", "kind": "ponder", "control": "lens_only", "cfg": { "ponder_mode": "metaphor" } }
  ]
}
```

Run it:

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "Why do we overfit narratives to randomness?" \
  --pack_file ./pack.json \
  --out_dir ./artifacts \
  --run_name surreal_lab
```

### Artifacts: JSON output + trace (observability)

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "Why do we overfit narratives to randomness?" \
  --mode both \
  --json_out ./run.json \
  --trace_out ./trace.jsonl
```

Or let the tool name files for you:

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "Why do we overfit narratives to randomness?" \
  --mode both \
  --out_dir ./artifacts \
  --run_name overfit_ab
```

To view traces as a timeline, render an HTML report:

```bash
python3 sr_trace_report.py --trace ./trace.jsonl --out ./trace_report.html
```

Tip: `--trace_out -` writes trace JSONL events to **stderr** (useful for piping without mixing with the main stdout).

### Config: JSON defaults (CLI still overrides)

Create `config.json`:

```json
{
  "preset": "surreal",
  "answer_style": "surreal",
  "ponder_hops": 3,
  "keyword_objective": "dissonance",
  "keyword_diversity": "embed"
}
```

Then run:

```bash
python3 sr_pondering_machine.py \
  --config ./config.json \
  --model ./model/gemma-3-270m-it \
  --query "創造性って結局なに？"
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

### Pipeline: assumption → counterexample → questions_only → metaphor

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "Why do people confuse confidence with truth?" \
  --mode ponder \
  --band_profile spectrum3 \
  --ponder_pipeline "assumption,counterexample,questions_only,metaphor" \
  --pipeline_context prev \
  --memory_policy current_only
```

### Answer ensemble: per-band answers + merged final answer

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "Explain quantum entanglement to a high school student." \
  --mode ponder \
  --band_profile spectrum3 \
  --n_ponder 2 \
  --memory_policy current_only \
  --answer_per_band \
  --answer_ensemble
```

### Keyword objective: dissonance (moderate semantic drift)

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "Design a new kind of musical scale." \
  --mode ponder \
  --keyword_objective dissonance \
  --dissonance_target 0.9 \
  --dissonance_width 0.6 \
  --memory_policy current_only
```

### Keyword objective: unstable (prompt jitter)

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "What is creativity, really?" \
  --mode ponder \
  --keyword_objective unstable \
  --prompt_jitter 4 \
  --memory_policy current_only
```

### Memory retrieval + remix (dream collage)

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "Why do we overfit narratives to randomness?" \
  --mode ponder \
  --memory_policy tail \
  --memory_retrieve mix \
  --memory_pool 300 \
  --memory_mix_ratio 0.5 \
  --memory_remix dream
```

### Interactive pick (human-in-the-loop keywords)

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "Invent a new sport." \
  --mode ponder \
  --band_profile spectrum3 \
  --interactive
```

### Controls pack (A/B all at once)

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "Is free will an illusion?" \
  --pack controls \
  --pack_out ./pack_results.json
```

## CLI Reference

Run `python3 sr_pondering_machine.py --help` for the full grouped help.

### Core

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | Path to a local model directory. |
| `--query` | *(required)* | The question to answer. |
| `--memory` | `ponder_logs.jsonl` | Path to the JSONL memory log. |
| `--mode` | `both` | `baseline` · `ponder` · `both` |
| `--prompt_lang` | `auto` | `auto` · `en` · `ja` |

### Ponder + Pipeline

| Argument | Default | Description |
|---|---|---|
| `--ponder_mode` | `assoc` | `assoc` · `assumption` · `counterexample` · `questions_only` · `metaphor` |
| `--ponder_pipeline` | *(empty)* | Lens chain like `assumption,counterexample,metaphor` |
| `--pipeline_context` | `prev` | `none` · `prev` · `all` |
| `--n_ponder` | `1` | Ponder logs per band |
| `--control` | `none` | `none` · `no_inject` · `random_log` · `random_keywords` · `lens_only` |
| `--no_write_memory` | `False` | Don’t append JSONL records |

### Bands

| Argument | Default | Description |
|---|---|---|
| `--band_profile` | `single` | `single` · `spectrum3` |
| `--band` | *(none)* | `START:END` or `LABEL=START:END` (repeatable, END exclusive) |

### Keywords

| Argument | Default | Description |
|---|---|---|
| `--strategy` | `outside_topk` | `within_topk` · `outside_topk` |
| `--top_k_rejected` | `80` | Top-K cutoff for rejected tokens |
| `--exclude_top` | `8` | Exclude top tokens (within_topk) |
| `--band_width` | `256` | Band width (outside_topk) |
| `--n_keywords` | `6` | Keywords per ponder log |
| `--keyword_refine` | `False` | Model rewrites token fragments |
| `--keyword_objective` | `random_band` | `random_band` · `dissonance` · `unstable` · `random_vocab` |
| `--keyword_select_top` | `128` | Sample from top-N candidates |
| `--dissonance_target` | `0.9` | Target dissonance (1 - cosine sim) |
| `--dissonance_width` | `0.6` | Acceptable window |
| `--dissonance_tail_k` | `64` | Prompt tail tokens used for query embedding |

### Prompt jitter

| Argument | Default | Description |
|---|---|---|
| `--prompt_jitter` | `0` | Paraphrases (excluding original) |
| `--no_prompt_jitter_include_original` | `False` | Don’t include the original query |

### Memory

| Argument | Default | Description |
|---|---|---|
| `--n_memory` | `6` | Memory records to inject |
| `--memory_policy` | `tail` | `tail` · `current_only` · `off` |
| `--memory_retrieve` | `tail` | `tail` · `similar` · `anti` · `mix` |
| `--memory_pool` | `200` | Tail pool size for retrieval |
| `--memory_mix_ratio` | `0.5` | similar/(similar+anti) in mix |
| `--memory_include_current_run` | `False` | Allow selecting current run from tail |
| `--memory_remix` | `off` | `off` · `shuffle` · `compress` · `dream` |

### Answers

| Argument | Default | Description |
|---|---|---|
| `--answer_per_band` | `False` | Print per-band answers |
| `--answer_ensemble` | `False` | Merge per-band answers |

### Interactive / Pack

| Argument | Default | Description |
|---|---|---|
| `--interactive` | `False` | Pick keywords interactively |
| `--pack` | `none` | `none` · `controls` · `surreal` |
| `--pack_file` | *(empty)* | Run a custom pack from a JSON file |
| `--pack_out` | *(empty)* | Optional JSON results |

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
  "control": "none",
  "band_profile": "single",
  "band_label": "outside_topk:80:336",
  "ponder_ix": 0,
  "prompt_lang": "en",
  "pipeline": ["assoc", "metaphor"],
  "pipeline_stage_ix": 0,
  "pipeline_context": "prev",
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

Memory injection is controlled by:

- `tail`: the most recent `n_memory` records
- `current_only`: only the logs generated in the current run
- `off`: no log injection (still writes to JSONL)

With `--memory_policy tail`, you can choose retrieval:

- `--memory_retrieve tail`: literal tail
- `--memory_retrieve similar|anti|mix`:
  - `--backend hf`: cosine similarity over token-id embeddings
  - `--backend openai_compat`: approximate TF‑IDF cosine over hashed character n-grams (plus MMR-style diversity)

And optionally remix the injected text with `--memory_remix`.

## Memory Report (Visualization)

Generate a quick, dependency-free report from your JSONL memory:

```bash
python3 sr_ponder_report.py --memory ./ponder_logs.jsonl
python3 sr_ponder_report.py --memory ./ponder_logs.jsonl --out ./ponder_report.html
```

## Notes

- **Use the `-it` (instruction-tuned) variant** of Gemma for best results. The base model does not reliably follow instructions.
- For `--backend hf`, the model must already be **downloaded locally**. Network access is disabled at inference time (`local_files_only=True`).
- For `--backend openai_compat`, set an API key (default env: `OPENAI_API_KEY`) and point `--api_base_url` at an OpenAI-style provider.
- For API backends, `--memory_retrieve similar|anti|mix` uses a cheap approximate similarity (hashed character n-grams + IDF weighting). It’s not as strong as real embedding retrieval, but better than raw tail.
- If you see an error like “Repo id must be in the form …” while passing an absolute `--model` path, it usually means the directory does not exist (Transformers falls back to treating it like a Hub ID). Double-check the path and try the closest matching folder name.
- On Apple Silicon (MPS), Transformers 5.x “caching allocator warmup” can crash on large models with `RuntimeError: Invalid buffer size: ...`. The script defaults to `--allocator_warmup auto` (which disables warmup on MPS). You can also force it off with `--allocator_warmup off`.
- `--keyword_refine` adds an extra generation call before the ponder step (slower, but often produces better keywords).
