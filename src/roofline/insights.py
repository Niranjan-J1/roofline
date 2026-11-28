"""Model analysis: parameter details and things to check, from the config alone.

Every item is triggered by a config value or calculated from one. A setting that
is not found is not proof that the model lacks the feature.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from roofline.config import CHECKED_FAMILIES, ModelConfig, dimension_source
from roofline.memory import kv_cache_bytes, lora_params
from roofline.params import ParamCounts

# Keys already shown in the dimensions line.
_CORE_KEYS = (
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "tie_word_embeddings",
    "text_config",
    "n_layer",
    "n_embd",
    "n_head",
)

# GPT-2 names for settings the analysis reads.
_GPT2_ALIASES = {
    "max_position_embeddings": "n_positions",
    "attention_dropout": "attn_pdrop",
    "hidden_act": "activation_function",
}

# Activations that commonly go with the three-matrix gated FFW we assume.
_EXPECTED_ACTIVATIONS = ("silu", "swish")

STORAGE_BYTES = (("fp32", 4.0), ("bf16 / fp16", 2.0), ("int8", 1.0), ("4-bit", 0.5))

UNAVAILABLE = (
    "chat template and the tokenizer's special-token and padding settings",
    "whether this is a base or instruction-tuned model",
    "module names to target with LoRA",
    "training data, quality or benchmark results, and license",
)


@dataclass(frozen=True)
class Item:
    title: str
    text: str
    short: str | None = None  # set only if this belongs in the short view


@dataclass(frozen=True)
class Analysis:
    overview: tuple[Item, ...]
    warnings: tuple[str, ...]
    parameter_details: tuple[Item, ...]
    testing: tuple[Item, ...]
    post_training: tuple[Item, ...]
    other_settings: tuple[tuple[str, str], ...]
    architecture: str
    dtype: str | None
    context: int | None


def safe_text(value: object, limit: int = 70) -> str:
    """Render an untrusted config value as short, printable ASCII."""
    if isinstance(value, str) and value.isascii() and value.isprintable():
        text = value
    else:
        text = json.dumps(value)  # escapes control and non-ASCII characters
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _positive(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None


def _millions(n: float) -> str:
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.2f}M"


def _sliding_window(view: dict, layers: int) -> tuple[str, str] | None:
    """Return (state, description), or None if the config says nothing."""
    layer_types = view.get("layer_types")
    if isinstance(layer_types, list) and "sliding_attention" in layer_types:
        count = layer_types.count("sliding_attention")
        return "on", f"listed for {count} of {layers} layers"
    window = _positive(view.get("sliding_window"))
    enabled = view.get("use_sliding_window")
    if enabled is False:
        unused = " The window size is listed but unused." if window else ""
        return "off", f"off.{unused}"
    if window and enabled:
        return "on", (
            f"window of {window:,} tokens; check whether it applies to all layers"
        )
    if window:
        return "listed", (
            f"a window of {window:,} tokens is listed, with no on/off setting; check "
            "whether your code applies it"
        )
    if enabled:
        return "on", "enabled, but no window size was found"
    return None


def _window_may_apply(
    view: dict, window: tuple[str, str] | None, limit: int | None
) -> bool:
    """False if there is no window, it is off, or it can never be reached."""
    if not window or window[0] == "off":
        return False
    if window[0] == "listed":
        size = _positive(view.get("sliding_window"))
        return not (size and limit and size >= limit)
    return True


def _rope_scaling(view: dict) -> str | None:
    """Name of a non-default position-scaling method, if the config lists one."""
    for key in ("rope_scaling", "rope_parameters"):
        block = view.get(key)
        if not isinstance(block, dict):
            continue
        # rope_parameters may be nested once, per layer type.
        blocks = [block, *(v for v in block.values() if isinstance(v, dict))]
        for b in blocks:
            kind = b.get("rope_type") or b.get("type")
            if kind and kind != "default":
                return safe_text(kind)
    return None


def _overview(view: dict, c: ModelConfig, counts: ParamCounts | None) -> list[Item]:
    items = []
    arch = view.get("architectures")
    if isinstance(arch, list) and arch:
        name = ", ".join(safe_text(a) for a in arch)
    else:
        name = "not stated"
    family = safe_text(view["model_type"]) if view.get("model_type") else "not stated"
    items.append(Item("Architecture", f"{name} (model type: {family})"))
    shape = f"{c.L} layers, hidden size {c.D:,}"
    if counts is None:
        items.append(Item("Size", f"{shape} (parameter count withheld)"))
    else:
        total = f"{_millions(counts.total)} parameters (formula estimate)"
        items.append(Item("Size", f"{total}, {shape}"))
        if c.K == c.N:
            attention = f"multi-head, {c.N} heads of size {c.H}"
        else:
            kind = "multi-query" if c.K == 1 else "grouped-query"
            attention = (
                f"{kind}, {c.N} query heads of size {c.H} share {c.K} key/value heads"
            )
        items.append(Item("Attention", attention))
        if c.moe:
            m = c.moe
            forward = (
                f"mixture of experts: {m.experts} experts of width {m.width:,}, "
                f"{m.active} active per token, in {m.sparse_layers} of {c.L} layers"
            )
            if m.shared_width:
                forward += f", plus a shared expert of width {m.shared_width:,}"
        else:
            forward = f"{c.F:,} wide ({c.F / c.D:.2f}x hidden size)"
        items.append(Item("Feed-forward", forward))
    dtype = view.get("torch_dtype") or view.get("dtype")
    items.append(
        Item("Saved precision", safe_text(dtype) if dtype else "not stated")
    )
    limit = _positive(view.get("max_position_embeddings"))
    items.append(
        Item("Position limit", f"{limit:,} tokens" if limit else "not found")
    )
    return items


def _warnings(view: dict, raw: dict, c: ModelConfig) -> list[str]:
    out = []
    if c.unsupported:
        out.append(
            f"{c.unsupported}. roofline cannot count this layout, so parameter, "
            "weight, compute and memory figures are withheld."
        )
    if view.get("auto_map") or raw.get("auto_map"):
        out.append(
            "The config points to custom model code. Loading it may need "
            "trust_remote_code, which runs code from the model repo. Review it first."
        )
    if view.get("quantization_config") or raw.get("quantization_config"):
        out.append(
            "The model is already quantized (see quantization_config). Any storage "
            "or memory figures shown do not describe these weights, and finetuning "
            "options depend on the method."
        )
    extra = [k for k in ("vision_config", "audio_config") if raw.get(k) or view.get(k)]
    if extra:
        out.append(
            f"Multimodal config ({', '.join(extra)}): parameter counts do not "
            "include the non-text parts."
        )
    if view.get("is_encoder_decoder") or raw.get("is_encoder_decoder"):
        out.append(
            "Encoder-decoder model: the formulas count only one stack of layers."
        )
    act = view.get("hidden_act") or view.get("hidden_activation")
    # For checked families the FFW layout is known, whatever the activation.
    checked = c.unsupported or c.family in CHECKED_FAMILIES
    if act and act not in _EXPECTED_ACTIVATIONS and not checked:
        out.append(
            f"Activation is {safe_text(act)}. The FFW formula assumes three matrices "
            "per layer (gated, as in SwiGLU); check that this model matches."
        )
    return out


def _parameter_details(c: ModelConfig, counts: ParamCounts) -> list[Item]:
    mixed = c.moe is not None and c.moe.sparse_layers < c.L
    items = [
        Item(
            "Per layer",
            f"{(counts.ffw + counts.attention) // c.L:,} "
            f"(FFW {counts.ffw // c.L:,}, attention {counts.attention // c.L:,})"
            + (", averaged over dense and expert layers" if mixed else ""),
        )
    ]
    if c.ffw_matrices == 2:
        items.append(
            Item(
                "FFW layout",
                f"two matrices (2 * D * F per layer), as in the {c.family} family",
            )
        )
    if c.moe:
        items.append(
            Item(
                "Experts",
                f"{_millions(counts.total)} parameters in total (all held in memory), "
                f"{_millions(counts.active_total)} active per token; includes routers",
            )
        )
    if c.positions:
        items.append(
            Item("Positions", f"{c.positions:,} learned position embeddings")
        )
    matrix = c.V * c.D
    items.append(
        Item(
            "Vocabulary",
            f"one shared {c.V:,} x {c.D:,} matrix"
            if c.tied
            else f"input and output matrices, {matrix:,} each",
        )
    )
    if c.K < c.N:
        full = c.L * 2 * c.D * c.H * (2 * c.N)
        items.append(
            Item(
                "Shared K/V heads",
                f"attention has {_millions(counts.attention)} parameters; with "
                f"{c.N} key/value heads it would have {_millions(full)}",
            )
        )
    storage = ", ".join(
        f"{name} {counts.total * size / 1e9:,.2f} GB" for name, size in STORAGE_BYTES
    )
    items.append(
        Item(
            "Weight storage",
            f"{storage} (weights only, 1 GB = 1e9 bytes; quantization overhead and "
            "layers kept at higher precision are not counted)",
        )
    )
    # Embeddings are lookups, so only the output head multiplies. With experts,
    # only the active ones do work.
    matmul = counts.ffw_active + counts.attention + c.V * c.D
    items.append(
        Item(
            "Compute per token",
            f"about {2 * matmul / 1e9:,.2f} GFLOPs forward and "
            f"{6 * matmul / 1e9:,.2f} GFLOPs forward + backward (rough, matrix "
            "weights only; attention-score work, which grows with sequence "
            "length, is not counted)",
        )
    )
    items.append(
        Item("Not counted", "bias and normalization parameters in the totals above")
    )
    return items


def _testing(view: dict, c: ModelConfig) -> list[Item]:
    items = []
    window = _sliding_window(view, c.L)
    limit = _positive(view.get("max_position_embeddings"))
    per_token = kv_cache_bytes(c, 1, 1)
    text = f"The KV cache is about {per_token:,} bytes per token at bf16"
    if limit:
        text += (
            f", or {per_token * limit / 1e9:,.2f} GB per sequence at the full "
            f"{limit:,}-token position limit"
        )
    text += "."
    limited = _window_may_apply(view, window, limit)
    if limited:
        text += (
            " This assumes every token is cached; sliding-window layers may keep fewer."
        )
    text += (
        " Test quality at your target lengths; a position limit does not show how "
        "well the model performs there."
    )
    short = None
    if limit:
        bound = "up to " if limited else ""
        short = (
            f"KV cache at full context: {bound}{per_token * limit / 1e9:,.2f} GB "
            "per sequence (bf16)."
        )
    if not c.unsupported:
        items.append(Item("Long inputs", text, short))
    if window:
        advice = (
            " Include inputs longer than the window in your tests." if limited else ""
        )
        short = None
        if limited and window[0] == "on":
            short = "Sliding-window attention is on. Test inputs longer than the window."
        elif limited:
            size = _positive(view.get("sliding_window"))
            short = f"A {size:,}-token sliding window is listed: check it applies."
        items.append(
            Item("Sliding-window attention", f"{window[1].rstrip('.')}.{advice}", short)
        )
    scaling = _rope_scaling(view)
    if scaling:
        items.append(
            Item(
                "Position scaling",
                f"The config uses '{scaling}' position (RoPE) scaling. Check that "
                "your inference library supports it, and test long inputs.",
                f"'{scaling}' RoPE scaling. Check library support; test long inputs.",
            )
        )
    dtype = view.get("torch_dtype") or view.get("dtype")
    saved = f"The config lists {safe_text(dtype)}." if dtype else "No data type is listed."
    items.append(
        Item(
            "Precision",
            f"{saved} Record which data type you test with, and compare results "
            "if you change it.",
        )
    )
    dropout = view.get("attention_dropout")
    if isinstance(dropout, (int, float)) and dropout > 0:
        items.append(
            Item(
                "Dropout",
                f"attention_dropout is {dropout}. Make sure the model is in "
                "evaluation mode when testing.",
                f"attention_dropout is {dropout}. Test in evaluation mode.",
            )
        )
    return items


def _token_item(view: dict) -> Item:
    bos, eos, pad = (view.get(f"{k}_token_id") for k in ("bos", "eos", "pad"))
    found = [
        f"{name} {safe_text(value)}"
        for name, value in (("start", bos), ("end", eos), ("padding", pad))
        if value is not None
    ]
    parts = [
        f"Token IDs in config: {', '.join(found)}."
        if found
        else "No start, end or padding token IDs were found."
    ]
    if bos is not None and bos == eos:
        parts.append("The start and end IDs are the same.")
    if isinstance(eos, list) and len(eos) > 1:
        parts.append("There are several end IDs; check that generation stops on each.")
    if pad is None:
        parts.append("The config lists no pad ID; the tokenizer may define one.")
    elif pad == eos:
        parts.append("Padding and end share an ID.")
    parts.append(
        "Check these against the tokenizer, and make sure masking padding does not "
        "also hide the real end token from the training loss."
    )
    issues = []
    if not found:
        issues.append("Config lists no token IDs: check the tokenizer's.")
    elif pad is None:
        issues.append("Config lists no pad ID: check the tokenizer's.")
    elif pad == eos:
        issues.append("Pad = end ID: check the loss mask.")
    if bos is not None and bos == eos:
        issues.append("Start = end ID.")
    if isinstance(eos, list) and len(eos) > 1:
        issues.append("Several end IDs: check generation stops on each.")
    return Item("Special tokens", " ".join(parts), " ".join(issues) or None)


def _post_training(
    view: dict, c: ModelConfig, counts: ParamCounts | None, lora_rank: int
) -> list[Item]:
    items = [_token_item(view)]
    items.append(
        Item(
            "Vocabulary size",
            f"The model has {c.V:,} embedding slots, which may differ from the "
            "tokenizer's token count. Check both before adding tokens or resizing.",
        )
    )
    if c.tied:
        embeddings = (
            "Input and output weights are shared (tied). Training or resizing one "
            "affects both; check the tie survives saving and merging."
        )
    elif c.tie_assumed:
        embeddings = (
            "The config does not say whether input and output weights are shared. "
            "Untied is assumed; check the loaded model."
        )
    else:
        embeddings = "Input and output weights are separate (untied)."
    short = "Input/output weights are tied. Check the tie survives save/merge."
    items.append(Item("Embeddings", embeddings, short if c.tied else None))
    if counts is not None:
        adapters = lora_params(c, lora_rank)
        items.append(
            Item(
                "LoRA size",
                f"Rank {lora_rank} adapters on q, k, v and o would add "
                f"{_millions(adapters)} parameters "
                f"({100 * adapters / counts.total:.2f}% of the base model).",
            )
        )
    if view.get("use_cache") is True:
        items.append(
            Item(
                "KV cache setting",
                "use_cache is on. Training does not normally need it; consider "
                "turning it off, especially with gradient checkpointing.",
            )
        )
    return items


def analyze(
    raw: dict, c: ModelConfig, counts: ParamCounts | None, lora_rank: int = 16
) -> Analysis:
    """`counts` is None when the model's layout cannot be counted."""
    view = dimension_source(raw)
    if view is not raw:
        view = {**raw, **view}
    other = tuple(
        (safe_text(key, 40), safe_text(view[key]))
        for key in sorted(view)
        if key not in _CORE_KEYS
    )
    if c.family == "gpt2":
        aliases = {new: view[old] for new, old in _GPT2_ALIASES.items() if old in view}
        view = {**aliases, **view}
    arch = view.get("architectures")
    dtype = view.get("torch_dtype") or view.get("dtype")
    return Analysis(
        overview=tuple(_overview(view, c, counts)),
        warnings=tuple(_warnings(view, raw, c)),
        parameter_details=(
            tuple(_parameter_details(c, counts)) if counts is not None else ()
        ),
        testing=tuple(_testing(view, c)),
        post_training=tuple(_post_training(view, c, counts, lora_rank)),
        other_settings=other,
        architecture=(
            ", ".join(safe_text(a) for a in arch)
            if isinstance(arch, list) and arch
            else "architecture not stated"
        ),
        dtype=safe_text(dtype) if dtype else None,
        context=_positive(view.get("max_position_embeddings")),
    )
