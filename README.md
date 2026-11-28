# roofline

A quick check of a HuggingFace model before you spend time or hardware on it: what the
model is, what to watch for when testing or post-training it, and roughly how much
memory it needs.

Only `config.json` is read. Model weights are never downloaded or loaded.

## Install

Needs Python 3.10 or newer.

```bash
git clone https://github.com/Niranjan-J1/roofline.git
cd roofline
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
```

For NVIDIA GPU detection through the NVIDIA Python library, use `pip install -e ".[gpu]"`.
Without it, `nvidia-smi` is used if it is available.

## Use

```bash
roofline Qwen/Qwen2.5-7B
```

```text
Qwen/Qwen2.5-7B   Qwen2ForCausalLM, bfloat16   (config: HF cache)

  Params     7.62B (FFW 75%, attention 11%, vocab 14%)
  Shape      28 layers, hidden 3,584, FFW 18,944
  Attention  grouped-query: 28 query / 4 KV heads, head size 128
  Context    131,072 position limit
  Weights    bf16 15.2 GB | int8 7.6 GB | 4-bit 3.8 GB

Worth checking
  [train] Config lists no pad ID: check the tokenizer's. Start = end ID.
  [test]  KV cache at full context: 7.52 GB per sequence (bf16).

Estimates from config.json. More: --full   Memory: --memory
```

The model can be a HuggingFace repo name, a local model folder or a path to a
`config.json`. For a repo name, your HuggingFace cache is checked first; otherwise only
`config.json` is downloaded. Gated models need access and `hf auth login`.

| Command | Shows |
|---|---|
| `roofline <model>` | Short analysis: key facts, warnings and things worth checking |
| `roofline <model> --full` | Every detail, including all other config settings |
| `roofline <model> --memory` | Adds memory totals for inference, LoRA and full training |
| `roofline <model> --memory --full` | Adds the memory breakdown and assumptions |
| `roofline <model> --fit` | Adds memory totals and the hardware on this machine |
| `roofline <model> --verify` | Compares the total with HuggingFace's count (online) |

### Workload settings

`--batch B` (default 1), `--seq T` (default 4096) and `--lora-rank R` (default 16).
Passing any of them turns on `--memory`.

### Hardware settings

`--gpus N` (total across all nodes), `--vram GB` (per GPU) and `--ram GB` (per node)
replace detected values and turn on `--fit`. Use them to describe hardware you don't
have. **`--fit` does not yet say whether a model fits.** It reports memory needs and
hardware side by side.

## How the numbers are worked out

1 GB is 1e9 bytes, so a card sold as "24 GB" has about 25.8 GB.

With `L` layers, hidden size `D`, feed-forward width `F`, `N` attention heads, `K`
key/value heads, head size `H` and vocabulary `V`:

| Part | Parameters |
|---|---|
| Feed-forward | `3 * D * F * L` (`2 * D * F * L` for the `phi`, `gpt_neox` and `gpt2` model types) |
| Attention | `L * 2 * D * H * (N + K)` |
| Vocabulary | `V * D`, doubled if input and output weights are separate |
| Positions | `positions * D`, counted for GPT-2 only |

Bias and normalization weights are not counted. Layouts are chosen by `model_type`.

If a config doesn't say whether input and output weights are shared, `roofline` shows
both totals and uses the larger one. GPT-2 is the exception: its family default
(shared) is used and labelled.

### Mixture-of-experts models

Each expert layer has `experts * 3 * D * F_expert` parameters plus a `D * experts`
router. Qwen2-MoE adds a shared expert and its gate, and Qwen MoE models can mix dense
and expert layers. Two figures are shown: **total** parameters, which must all be held
in memory, and **active** parameters, used for each token.

| Memory | Estimate (`B` batch, `T` sequence length) |
|---|---|
| Inference | 2 bytes per parameter + KV cache `4 * L * K * H * B * T` |
| LoRA | 2 bytes per parameter + 12 per adapter parameter + activations |
| Full training | 12 bytes per parameter + activations |
| Activations | `2 * B * T * L * (5D + 3F)` |

LoRA adapters are assumed to be on the q, k, v and o projections. For
mixture-of-experts models, activations count only the active experts' width.

## Checked models

Parameter totals are within 0.1% of reference counts for these models. The references
come from weight-file headers, and any adjustments (such as stored masks that aren't
parameters, or packed weights) are recorded in [`tests/fixtures`](tests/fixtures).

| Layout | Checked on |
|---|---|
| Three-matrix feed-forward | Qwen2.5-0.5B, Qwen3-0.6B, SmolLM2-135M, Phi-3-mini-4k, OLMo-2-7B, Mistral-7B |
| Two-matrix feed-forward | Phi-2, Pythia-160M, GPT-2 |
| Mixture of experts | Mixtral-8x7B, OLMoE-1B-7B, Qwen1.5-MoE-A2.7B, Qwen3-30B-A3B, gpt-oss-20b |

Passing on one model does not guarantee every model in its family. A model type that
hasn't been checked gets standard formulas, and the output says so.

**Not supported:** models with compressed (latent) attention, such as DeepSeek-V3, and
other mixture-of-experts layouts. `roofline` shows their basic facts and withholds
parameter, weight, compute and memory figures.

### About `--verify`

HuggingFace's count describes what the weight files store. That can include masks and
other buffers, packed or quantized data, or a second copy of shared weights, so a
difference doesn't always mean either number is wrong. `--verify` says when it can't
make a reliable comparison, and never replaces `roofline`'s figure.

## Limitations

- **Memory figures are estimates and have not been compared with real runs.** They leave
  out framework overhead and temporary buffers. The activation formula is the least
  certain, and assumes no gradient checkpointing.
- The activation formula uses `3F` for every model, including two-matrix ones.
- Memory estimates for quantized models do not describe their quantized weights.
- Some model families use other config names or structures and stop with an error
  (Falcon is one).
- Multimodal models are only partly covered: non-text parts are not counted.
- Tokenizer files are not read. A pad ID missing from `config.json` may still be set by
  the tokenizer.
- A setting that `roofline` doesn't find is not proof that the model lacks it.
- GPU detection is NVIDIA only. Real NVIDIA and Slurm detection have only been tested
  with simulated data.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests run offline.
