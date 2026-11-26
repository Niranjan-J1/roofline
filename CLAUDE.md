# roofline

A CLI tool that analyzes a HuggingFace model config and prints a "fit verdict" for the
machine it's running on: given a model, can this hardware load it, finetune it, or train it,
and if not, what would it take?

## What it does

Input: a HuggingFace model (repo name, local path, or a `config.json`).
Output: a model analysis + an environment-aware fit verdict.

Two halves:

1. **Model analysis** (from the config).
2. **Environment detection** (what hardware the command is running on).

Then it compares the two and prints a per-scenario ✓/✗ verdict with one line of reasoning each.

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

## Parameter formulas (per the "How to Scale Your Model" scaling book)

- **FFW params**    = 3 · D · F · L          (SwiGLU: gate, up, down)
- **Attention params** = L · 2 · D · H · (N + K)   ← GQA-aware
- **Vocab params**   = (1 if tied else 2) · V · D
- **Total**       = FFW + Attention + Vocab

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

## Environment detection (no user input required)

- GPUs present + VRAM: pynvml, falling back to parsing `nvidia-smi`.
- Total system RAM: psutil.
- Slurm allocation: read SLURM_* env vars (SLURM_JOB_ID, SLURM_NNODES, SLURM_GPUS_ON_NODE,
  SLURM_NTASKS, etc.).
- No CUDA devices → report CPU-only clearly.
- If a value can't be detected, say so and state the assumption; allow override via flags.

## Verdict

For each of {inference, LoRA, full training}: does it fit on the detected hardware?
If not, how many devices would it need, and what sharding strategy (e.g. FSDP / ZeRO-3
across available GPUs)? Print a clear ✓/✗ per scenario with one line of reasoning.

"Fits" (weights) is necessary, not sufficient — bias full-training verdicts toward
"this is tight" rather than a confident green.

## Flags

- `--gpus N`     override detected GPU count
- `--vram GB`    override detected per-GPU VRAM
- `--ram GB`     override detected system RAM
- `--json`       machine-readable output
- (defaults should let it run with just a model name)

## How to work in this repo

- **Plan before coding.** For any non-trivial change, propose a plan and wait for approval.
- **Work in buckets, one at a time.** Suggested order: (1) scaffold + CLI skeleton,
  (2) config parsing + param math, (3) memory scenarios, (4) environment detection,
  (5) verdict + recommendations, (6) --json + polish. After each bucket, run the relevant
  check, show the result, and stop for go-ahead before the next.
- **Commit and push at each completed bucket (milestone).** Keep commits scoped to one bucket.
- Verify against the LLaMA-3-70B anchor whenever the param math changes.

## Git conventions

- Commits are authored by the repo owner only.
- **Never** add a "Co-Authored-By: Claude" trailer to any commit.
- **Never** add a "Generated with Claude Code" line (or similar) to commit messages.
- Keep commit messages plain, imperative, and scoped to one bucket/milestone.
- Do not change git config or the committer identity; use the repo owner's existing identity.