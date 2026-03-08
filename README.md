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

- **Provider presets** — run API models with `--provider openai|mistral|groq|openrouter|deepseek --model ...` instead of wiring base URLs by hand.
- **MPS-first** — resolves Apple Silicon device-mismatch errors automatically.
- **Gemma-aware** — natively applies the `<start_of_turn>` / `<end_of_turn>` prompt format expected by Gemma IT models and stops generation cleanly at `<end_of_turn>`.
- **Persistent memory** — ponder logs are stored in a JSONL file and the most recent entries are reused in subsequent runs.
- **Baseline comparison** — `--mode both` prints baseline, pondered answer, and a compact comparison summary by default.
- **Semantic comparison** — comparison output now includes a semantic similarity/alignment block (`--compare_semantic auto|hash|embed`).
- **Stance shift** — comparison output can profile `definition / framing / conditionalization / example expansion / resolution` drift with `--compare_stance auto`.
- **Spatial metaphor density** — comparison output can track spatial-metaphor density in answers and ponder logs with `--compare_spatial_metaphor auto`.
- **Token budget comparison** — comparison output can now separate visible scaffold size from API `reasoning_tokens` / `completion_tokens` with `--compare_token_budget auto`.
- **Terminal ponder logs** — human-readable ponder logs now print to the terminal by default; raw JSON records stay opt-in.
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
- **Probe compare** — measure how pondering changes the next-token distribution with `--probe_compare` and `--probe_compare_stages` (rank movers + JS divergence).
- **Spectral bands** — run multiple rank-bands (near/mid/far) via `--band_profile spectrum3` or define custom bands with `--band`.
- **Memory retrieval** — pick memory by similarity/anti-similarity via `--memory_retrieve similar|anti|mix`.
- **Memory remix** — shuffle/compress/dream the injected memory via `--memory_remix`.
- **Answer style** — steer the final answer style via `--answer_style plain|surreal|metaphor|meta`.
- **Answer ensemble** — generate per-band answers + merged answer with `--answer_per_band` / `--answer_ensemble`.
- **Interactive pick** — choose keyword tokens yourself with `--interactive`.
- **Controls pack** — run A/B packs via `--pack controls|surreal` (or bring your own with `--pack_file`).
- **Pack resume** — skip already-completed pack items with `--pack_resume`.
- **Artifacts & tracing** — export results with `--json_out`, write step-level traces with `--trace_out`, and generate an HTML timeline with `--trace_report_out` (or use `--out_dir` for auto-naming).
- **Trace report** — write a report with `--trace_report_out` or post-process traces with `sr_trace_report.py`.
- **Config file** — set defaults from a JSON file via `--config` (CLI flags still override).
- **Highly configurable** — token selection strategy, generation hyperparameters, device, dtype, and more are all adjustable from the command line.

## Requirements

- Python 3.9+
- For `--backend hf` / `--provider hf`:
  - [PyTorch](https://pytorch.org/) (with MPS, CUDA, or CPU support)
  - [Transformers](https://github.com/huggingface/transformers)
  - A locally downloaded Hugging Face causal LM (e.g. `gemma-3-270m-it`) — or use `--hf_online` to allow Hub downloads.
- For API providers (`--provider openai|mistral|groq|openrouter|deepseek|custom`):
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
  --provider hf \
  --model ./model/gemma-3-270m-it \
  --query "Explain quantum entanglement to a high school student" \
  --mode both \
  --memory ./ponder_logs.jsonl
```

This runs both the **baseline** (direct answer) and the **ponder** (wander-then-answer) modes and prints both outputs, the ponder log, and a compact comparison summary.
By default that comparison includes a semantic similarity view: local HF runs use mean token-embedding cosine, and API runs try a local encoder automatically (`./model/minilm`, `./models/minilm`, or similar MiniLM/e5/bge-style dirs) on CPU before falling back to hashed character n-gram TF-IDF cosine.

## API Quick Start (Provider presets)

Use `--provider ...` to auto-fill the API base URL and key env for common OpenAI-style providers.
This path can run without installing `torch`/`transformers`.

```bash
export OPENAI_API_KEY="..."

python3 sr_pondering_machine.py \
  --provider openai \
  --model gpt-5.4 \
  --query "Should we optimize for accuracy or speed in LLM systems?" \
  --mode ponder \
  --api_seed_method self
```

Other built-in provider presets:

- `--provider mistral` → `MISTRAL_API_KEY`
- `--provider groq` → `GROQ_API_KEY`
- `--provider openrouter` → `OPENROUTER_API_KEY`
- `--provider deepseek` → `DEEPSEEK_API_KEY`
- `--provider custom` → keep using your explicit `--api_base_url`

If your provider supports Chat Completions logprobs, you can try the more “rejected-token-ish” seeding:

```bash
python3 sr_pondering_machine.py \
  --provider openai \
  --model gpt-5.4 \
  --query "Why do we overfit narratives to randomness?" \
  --mode ponder \
  --api_seed_method logprobs \
  --api_logprobs_top_n 32
```

Note: some providers/models use `max_completion_tokens` instead of `max_tokens`. This tool retries automatically when it detects that mismatch.
Start with a small value like `32`: providers often cap `top_logprobs`, and if the returned depth is shallower than your requested band range, that band falls back to self-seeded keywords with a warning.

### OpenAI GPT-5 family notes

For `https://api.openai.com/v1`, newer GPT-5 variants can have stricter parameter compatibility than generic OpenAI-compatible providers.

- `gpt-5` may reject `max_tokens`, `temperature != 1`, or `top_p`; the client now retries with compatibility fallbacks.
- Versioned models such as `gpt-5.2` / `gpt-5.4` can work better with `--api_reasoning_effort none`.
- With `--provider openai` and `gpt-5*`, the default `--ponder_max_new_tokens` floor is raised to `384`.
- If a GPT-5 call returns empty visible text while spending the full completion budget on reasoning, the tool retries once with a larger token budget before giving up.
- Empty visible answers still record `finish_reason`, `completion_tokens`, and `reasoning_tokens` in `extras.api_final_generation` / `api_warnings` so you can tell whether the model stayed silent, refused, or spent the budget elsewhere.

Example:

```bash
PYTHONNOUSERSITE=1 python3 sr_pondering_machine.py \
  --provider openai \
  --api_reasoning_effort none \
  --model gpt-5.4 \
  --query "Should we optimize for accuracy or speed in LLM systems?" \
  --mode ponder \
  --api_seed_method self \
  --memory_policy current_only \
  --json_out ./artifacts/gpt-5.4.json \
  --trace_out ./artifacts/gpt-5.4.trace.jsonl \
  --trace_report_out ./artifacts/gpt-5.4.trace.html
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

Tip: pack results write to `--pack_out` (or `--json_out`), traces write to `--trace_out`, and reports write to `--trace_report_out` (or auto-name all via `--out_dir`).
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
  --trace_out ./trace.jsonl \
  --trace_report_out ./trace_report.html
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

This will auto-write a JSON result file, a JSONL trace, and an HTML trace report into `./artifacts`.

To view traces as a timeline, render an HTML report:

```bash
python3 sr_trace_report.py --trace ./trace.jsonl --out ./trace_report.html
```

Tip: `--trace_out -` writes trace JSONL events to **stderr** (useful for piping without mixing with the main stdout).
Tip: `--json_out -` writes the JSON payload to **stdout**.

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

### Probe compare (what changed before vs after pondering?)

```bash
python3 sr_pondering_machine.py \
  --model ./model/gemma-3-270m-it \
  --query "Should we optimize for accuracy or speed in LLM systems?" \
  --mode ponder \
  --ponder_mode counterexample \
  --n_ponder 3 \
  --memory_policy current_only \
  --probe_compare \
  --probe_compare_stages \
  --probe_compare_top_n 32 \
  --trace_out ./run.trace.jsonl \
  --json_out ./run.json
```

This stores a run-level `extras.probe_compare` block, a per-stage `extras.probe_compare_stages` timeline, and `probe_compare` / `probe_compare_stage` trace events.
`sr_trace_report.py` now summarizes those events into a final probe card plus a stage timeline table.
For `--backend hf`, JS divergence is computed over the full vocabulary. For `--backend openai_compat`, it is an approximation over the returned top-logprobs union.
`--probe_compare_stages` probes the answer prompt after every ponder stage, so it adds one extra forward pass (HF) or one extra logprobs API call per stage.

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
| `--provider` | `auto` | `auto` · `hf` · `openai` · `mistral` · `groq` · `openrouter` · `deepseek` · `custom` |
| `--backend` | `hf` | Low-level backend override; usually you just set `--provider` |
| `--model` | *(required)* | `provider=hf`: local model directory · API providers: model name |
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

### Observability / Runtime / API

| Argument | Default | Description |
|---|---|---|
| `--print_compare` | `auto` | Print a compact baseline vs ponder summary |
| `--print_ponder` | `auto` | Print human-readable ponder logs to stdout |
| `--print_records` | `none` | Debug: print raw ponder-record JSON |
| `--compare_semantic` | `auto` | `off` · `auto` · `hash` · `embed` |
| `--compare_stance` | `auto` | `off` · `auto` |
| `--compare_spatial_metaphor` | `auto` | `off` · `auto` |
| `--compare_token_budget` | `auto` | `off` · `auto` |
| `--compare_embed_model` | *(empty)* | Optional local/cached encoder for true embedding cosine; otherwise auto-discovers local MiniLM/e5/bge/gte/mpnet dirs |
| `--json_out` | *(empty)* | Write full run/pack results to JSON |
| `--trace_out` | *(empty)* | Write step-level trace JSONL |
| `--trace_report_out` | *(empty)* | Write HTML trace report |
| `--probe_top_n` | `0` | Store base probe top-N tokens in the first record |
| `--probe_compare` | `False` | Compare pre/post-ponder next-token distributions |
| `--probe_compare_stages` | `False` | Capture a base→stage→final probe timeline |
| `--probe_compare_top_n` | `32` | Top-N window used by `--probe_compare` |
| `--print_probe` | `False` | Print probe tables (and probe-compare summary) to stdout |
| `--device` | `auto` | `auto` · `mps` · `cpu` · `cuda[:N]` |
| `--dtype` | `auto` | `auto` · `float16` · `bfloat16` · `float32` |
| `--allocator_warmup` | `auto` | `auto` · `on` · `off` |
| `--api_seed_method` | `auto` | `auto` · `self` · `logprobs` |
| `--api_logprobs_top_n` | `0` | Top-logprobs depth requested from the API |
| `--api_reasoning_effort` | `auto` | `auto` · `none` · `minimal` · `low` · `medium` · `high` · `xhigh` |

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
- For `--provider hf` / `--backend hf`, the model must already be **downloaded locally**. Network access is disabled at inference time (`local_files_only=True`).
- For API providers, start with `--provider ... --model ...`; only drop down to `--backend openai_compat --api_base_url ...` when you need a custom endpoint.
- For API providers on macOS with aggressive user `sitecustomize` hooks, `PYTHONNOUSERSITE=1` can avoid unrelated import-time crashes.
- For API providers, `--seed`, `--top_k`, `--repetition_penalty`, and `--no_repeat_ngram_size` are currently not forwarded; the script prints a warning so cross-backend comparisons stay honest.
- For OpenAI GPT-5 family models, the client retries common compatibility errors (`max_tokens` ↔ `max_completion_tokens`, unsupported `temperature`, unsupported `top_p`, unsupported `logprobs`) automatically.
- For OpenAI GPT-5 family models, the client also retries one time when visible output is empty and the entire completion budget appears to have been consumed by reasoning tokens.
- For API providers, rate limits now surface as a short CLI error with any available `Retry-After` delay instead of a raw stack trace.
- For `--compare_semantic auto`, local HF runs use cosine over mean token embeddings from the active model; API runs first try a local encoder auto-discovery pass (`model/minilm`, `models/minilm`, or similar MiniLM/e5/bge/gte/mpnet dirs) on CPU and only fall back to hashed char n-gram TF-IDF cosine if none is available.
- When that API-side local encoder exists, the script prewarms it in the background while the main API generations are running, so the semantic block is less likely to become the last visible bottleneck.
- For `--compare_semantic embed`, you can set `--compare_embed_model` explicitly, or let the script auto-pick a local encoder from the same MiniLM/e5/bge/gte/mpnet search path. Set `SR_COMPARE_EMBED_MODEL` if you want to pin that default without passing the CLI flag each time.
- `--compare_stance auto` uses a built-in heuristic lexicon to estimate answer-policy drift across `definition`, `framing`, `conditionalization`, `example_expansion`, and `resolution`.
- `--compare_spatial_metaphor auto` measures spatial-metaphor density per 1k chars for the baseline answer, pondered answer, ponder questions, and ponder logs, then reports dominant groups like `path`, `stage`, `container`, and `geometry`.
- `--compare_token_budget auto` reports visible scaffold size (keywords, ponder questions/logs, injected memory) alongside API-side prompt / completion / reasoning token totals, so you can test “more scaffold” against “more internal reasoning” instead of conflating them.
- External semantic encoder loading is quiet by default to avoid noisy `from_pretrained()` progress bars and local `sitecustomize` chatter; set `SR_COMPARE_EMBED_VERBOSE=1` if you want to see that load output.
- For API backends, `--memory_retrieve similar|anti|mix` uses a cheap approximate similarity (hashed character n-grams + IDF weighting). It’s not as strong as real embedding retrieval, but better than raw tail.
- For API backends, `--api_logprobs_top_n` is provider-capped. If the returned logprob depth is shallower than a requested band (for example `spectrum3` + `far`), that band degrades to self-seeded keywords and the script warns about it.
- For `--probe_compare`, `--backend hf` computes JS divergence on the full vocab; `--backend openai_compat` computes an approximate JS divergence on the observed top-logprobs union and reports the observed mass.
- `--probe_compare_stages` uses the current run’s accumulated ponder logs as the injected memory source for each timeline point. This is diagnostic by design; the final answer may still use a different selected/remixed memory block.
- If you see an error like “Repo id must be in the form …” while passing an absolute `--model` path, it usually means the directory does not exist (Transformers falls back to treating it like a Hub ID). Double-check the path and try the closest matching folder name.
- On Apple Silicon (MPS), Transformers 5.x “caching allocator warmup” can crash on large models with `RuntimeError: Invalid buffer size: ...`. The script defaults to `--allocator_warmup auto` (which disables warmup on MPS). You can also force it off with `--allocator_warmup off`.
- `--keyword_refine` adds an extra generation call before the ponder step (slower, but often produces better keywords).
