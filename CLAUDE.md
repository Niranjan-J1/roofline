# roofline

A CLI tool that analyzes a HuggingFace model config for researchers, engineers and
developers: what the model is, and what to watch for when testing or post-training it.
Memory estimates and a hardware fit verdict are optional extras.

## What it does

Input: a HuggingFace model (repo name, local path, or a `config.json`). Only `config.json`
is read or downloaded, never weights.

| command                    | output                                                        |
|----------------------------|---------------------------------------------------------------|
| `roofline <model>`         | short model analysis, about one screen (the main feature)     |
| `roofline <model> --full`  | all details                                                   |
| `roofline <model> --memory`| + memory totals (`--batch`/`--seq`/`--lora-rank` imply it)    |
| `roofline <model> --verify`| + comparison with HuggingFace's parameter count (online)      |
| `roofline <model> --fit`   | + memory and detected hardware (fit verdicts not built yet)   |

**Keep the default short** (about 15 lines for a typical model). It shows key facts,
every warning, assumptions that affect the numbers, and "worth checking" items only when
something about this model calls for them. No generic advice. Never drop a warning to
meet the length target. Everything else goes under `--full`: parameter breakdown and
derived figures, all testing and post-training notes, other config settings as written,
and what cannot be determined from `config.json`. `--memory --full` shows the memory
breakdown and assumptions.

Rules for analysis items: each comes from a config value or a calculation from one. Never
guess; say when something is not stated. A setting not found is not proof of absence.
Config values are untrusted text and must be sanitized before printing.

## Config key mapping

| config.json key          | symbol | fallback if missing      |
|--------------------------|--------|--------------------------|
| num_hidden_layers        | L      | —                        |
| hidden_size              | D      | —                        |
| intermediate_size        | F      | —                        |
| num_attention_heads      | N      | —                        |
| num_key_value_heads      | K      | defaults to N (i.e. MHA) |
| head_dim                 | H      | defaults to D / N        |
| vocab_size               | V      | —                        |
| tie_word_embeddings      | —      | false → 2 vocab matrices |

If a required key is missing, fail with a clear message naming the key — never silently guess.

GPT-2 (`model_type: gpt2`) uses `n_layer`, `n_embd`, `n_head`, `n_inner`, `n_positions`.
Its family defaults (F = 4·D when `n_inner` is unset; shared weights when unstated) are
labeled in output. For other models with `tie_word_embeddings` unstated, assume separate,
and show both totals, saying which one memory uses.

## Parameter formulas (per the "How to Scale Your Model" scaling book)

- **FFW params**    = 3 · D · F · L          (SwiGLU: gate, up, down)
- **Attention params** = L · 2 · D · H · (N + K)   ← GQA-aware
- **Vocab params**   = (1 if tied else 2) · V · D
- **Total**       = FFW + Attention + Vocab

### Family-specific layouts (chosen by `model_type`, never by activation name)

- **Two-matrix FFW** (`phi`, `gpt_neox`, `gpt2`): 2 · D · F · L. All others use three.
- **GPT-2** also counts learned position embeddings: n_positions · D.
- **Mixture of experts** (`mixtral`, `olmoe`, `qwen2_moe`, `qwen3_moe`, `gpt_oss`):
  per expert layer, experts · 3·D·F_expert + router D·experts, plus for `qwen2_moe` a
  shared expert 3·D·F_shared + gate D. Qwen layers can be dense (`decoder_sparse_step`,
  `mlp_only_layers`). Show total and active parameters. Memory weights count all
  experts; activations use only active ones.
- **Unsupported** (compressed/latent attention such as DeepSeek, and any other MoE
  layout): show basic facts, and withhold parameter, weight, compute and memory figures.
  Never show totals known to be wrong.

**Adding a family** requires a fixture in `tests/fixtures` proving the layout: public
`config.json` plus a reference count from weight-file headers, with adjustments listed.
Totals must be within 0.1%. Families without one are labeled "not checked".

`--verify` compares with HuggingFace's count. That count can include buffers, packed or
quantized data, or a second copy of tied weights, so it never replaces our figure.

### IMPORTANT: attention must be GQA-aware
Use `2 · D · H · (N + K)` per layer, NOT `4 · D²`.
The `4D²` shortcut only holds when K = N (plain MHA). Modern models use grouped-query
attention where K < N, and `4D²` would overcount attention badly.

**Correctness anchor — LLaMA-3-70B** (L=80, D=8192, F=28672, N=64, K=8, H=128, V=128256):
- FFW ≈ 56.3e9, Attention ≈ 12.0e9, Vocab ≈ 2.1e9, Total ≈ 70.4e9
- The 70.4e9 total is the tripwire: if attention comes out ~21e9, the GQA formula is wrong.

## Memory scenarios

Bytes per parameter:
- **Inference (bf16)**: 2 bytes/param, plus the KV cache (4 · L · K · H · batch · seq bytes),
  included in the total and shown separately.
- **LoRA finetune**: base weights (2 bytes/param) + 12 bytes per adapter param + activations.
  Adapters on q, k, v, o only.
- **Full training (Adam)**: 12 bytes/param (2 bf16 weights + 2 bf16 gradients + 4 fp32 first
  moment + 4 fp32 second moment), PLUS activation memory.
- **Activations** (rough, unmeasured estimate): 2 · batch · seq · L · (5D + 3F) bytes, with no
  gradient checkpointing. Dominates at large batch.

Always flag which component dominates, and label time/sharding/MFU outputs as idealized
roofline estimates, not guarantees.

## Environment detection (optional, under --fit; no user input required)

- GPUs present + VRAM: pynvml, falling back to parsing `nvidia-smi`.
- Total system RAM: psutil.
- Slurm allocation: read SLURM_* env vars (SLURM_JOB_ID, SLURM_NNODES, SLURM_GPUS_ON_NODE,
  SLURM_NTASKS, etc.).
- No CUDA devices → report CPU-only clearly. Distinguish "none found" from "check failed".
- Respect CUDA_VISIBLE_DEVICES; if it can't be matched reliably (e.g. MIG), say so.
- Direct detection sees one node. A multi-node Slurm total is labeled as assuming
  identical nodes. Mixed GPUs: use the smallest VRAM.
- Runs only under --fit (hardware flags imply it). Detection never crashes the analysis.
- If a value can't be detected, say so and state the assumption; allow override via flags.

## Verdict (optional, under --fit)

For each of {inference, LoRA, full training}: does it fit on the detected hardware?
If not, how many devices would it need, and what sharding strategy (e.g. FSDP / ZeRO-3
across available GPUs)? Print a clear ✓/✗ per scenario with one line of reasoning.

"Fits" (weights) is necessary, not sufficient — bias full-training verdicts toward
"this is tight" rather than a confident green.

## Flags

- `--gpus N`     override GPU count (total across all nodes)
- `--vram GB`    override per-GPU VRAM
- `--ram GB`     override system RAM per node
                 (1 GB = 1e9 bytes everywhere)
- `--json`       machine-readable output
- `--memory`     add memory estimates
- `--full`       show all details
- `--verify`     compare the total with HuggingFace's count
- `--fit`        add memory, hardware detection and (later) fit verdicts
- (defaults should let it run with just a model name)

## How to work in this repo

- **Plan before coding.** For any non-trivial change, propose a plan and wait for approval.
- **Work in buckets, one at a time.** Suggested order: (1) scaffold + CLI skeleton,
  (2) config parsing + param math, (3) memory scenarios, (4) model insights,
  (5) environment detection, (6) verdict + recommendations, (7) --json + polish.
  After each bucket, run the relevant check, show the result, and stop for go-ahead
  before the next.
- **Commit and push at each completed bucket (milestone).** Keep commits scoped to one bucket.
- Verify against the LLaMA-3-70B anchor whenever the param math changes.

## Git conventions

- Commits are authored by the repo owner only.
- **Never** add a "Co-Authored-By: Claude" trailer to any commit.
- **Never** add a "Generated with Claude Code" line (or similar) to commit messages.
- Keep commit messages plain, imperative, and scoped to one bucket/milestone.
- Do not change git config or the committer identity; use the repo owner's existing identity.