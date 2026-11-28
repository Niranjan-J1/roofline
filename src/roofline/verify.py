"""Compare our total with HuggingFace's count for a repo (the --verify flag).

HuggingFace counts what is stored in the weight files, which is not always the
number of parameters: files can hold buffers, packed quantized data, or a second
copy of shared weights. So this reports a comparison and never replaces our figure.
"""

from __future__ import annotations

from huggingface_hub import HfApi

from roofline.config import ModelConfig
from roofline.params import ParamCounts

AGREEMENT = 0.005  # our formulas skip bias and normalization weights
_PLAIN_DTYPES = {"F64", "F32", "F16", "BF16"}
_LOCAL_SOURCES = ("local file", "local directory")


def compare(
    ours: int, theirs: int, dtypes: set[str], quantized: bool, c: ModelConfig
) -> str:
    """Describe how our count relates to HuggingFace's."""
    odd = sorted(dtypes - _PLAIN_DTYPES)
    caveat = None
    if odd:
        names = ", ".join(d for d in odd if d.isascii() and d.isprintable())
        caveat = (
            f"the files include {names} data, which may be buffers or quantized or "
            "packed weights"
        )
    elif quantized:
        caveat = "the model is quantized"

    gap = ours - theirs
    if abs(gap) <= AGREEMENT * theirs:
        if caveat:
            return f"Close, but not a reliable check because {caveat}."
        return "Agrees: the difference is within 0.5%."
    shared_copy = c.V * c.D
    if c.tied and abs(theirs - ours - shared_copy) <= AGREEMENT * ours:
        return (
            "The gap is about the size of the shared vocabulary weights, which is "
            "consistent with the files storing them twice (not extra parameters)."
        )
    masks = c.L * c.positions**2
    if c.family == "gpt2" and abs(theirs - ours - masks) <= AGREEMENT * ours:
        return (
            "The gap is about the size of GPT-2's attention masks, which is "
            "consistent with the files storing them (not parameters)."
        )
    if caveat:
        return f"Not a reliable comparison because {caveat}."
    return (
        "Differs by more than 0.5%. The formulas may not fit this model, or the "
        "files may hold extra data. Treat the totals with caution."
    )


def verify_count(
    model: str,
    source: str,
    raw: dict,
    c: ModelConfig,
    counts: ParamCounts | None,
) -> tuple[str, ...]:
    """Return the lines to show. Never raises."""
    if source in _LOCAL_SOURCES:
        return ("Not run: --verify needs a HuggingFace repo name, not a local path.",)
    if counts is None:
        return ("Not compared: roofline withholds totals for this model.",)
    try:
        info = HfApi().model_info(model, timeout=15).safetensors
        if info is None:
            return ("HuggingFace has no parameter count for this repo.",)
        theirs = int(info.total)
        dtypes = {str(d) for d in (info.parameters or {})}
    except Exception as e:  # offline, gated, rate limited, API changes
        return (f"Could not look up the count ({type(e).__name__}).",)
    if theirs <= 0:
        return ("HuggingFace has no parameter count for this repo.",)
    percent = 100 * (counts.total - theirs) / theirs
    quantized = bool(raw.get("quantization_config"))
    return (
        f"HuggingFace {theirs:,}; roofline {counts.total:,} ({percent:+.2f}%).",
        compare(counts.total, theirs, dtypes, quantized, c),
    )
