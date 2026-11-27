"""Human-readable output (JSON output comes later)."""

from __future__ import annotations

import textwrap
from pathlib import Path

from roofline import __version__
from roofline.config import ModelConfig
from roofline.insights import UNAVAILABLE, Analysis, Item
from roofline.memory import ASSUMPTIONS, GB, Scenario
from roofline.params import ParamCounts

WIDTH = 88
SHORT_WIDTH = 80

# Config assumptions that affect the numbers or their scope, so always shown.
_IMPORTANT_NOTES = ("num_key_value_heads missing", "tie_word_embeddings missing", "text_config")


def _wrap(text: str, first: str, rest: str, width: int = WIDTH) -> str:
    return textwrap.fill(
        text,
        width=width,
        initial_indent=first,
        subsequent_indent=rest,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _labelled(items: tuple[Item, ...]) -> list[str]:
    return [_wrap(i.text, f"  {i.title + ':':<19}", " " * 21) for i in items]


def _checks(items: tuple[Item, ...]) -> list[str]:
    return [_wrap(f"{i.title}: {i.text}", "  - ", "    ") for i in items]


def render_short(
    model: str, source: str, c: ModelConfig, counts: ParamCounts, a: Analysis
) -> str:
    """The default view: about one screen. Warnings are never dropped."""

    def wrap(text: str, first: str, rest: str) -> str:
        return _wrap(text, first, rest, SHORT_WIDTH)

    kind = f"{a.architecture}, {a.dtype}" if a.dtype else a.architecture
    total = counts.total
    params = (
        f"{total / 1e9:.2f}B" if total >= 1e9 else f"{total / 1e6:.2f}M"
    )
    split = (
        f"FFW {100 * counts.ffw / total:.0f}%, "
        f"attention {100 * counts.attention / total:.0f}%, "
        f"vocab {100 * counts.vocab / total:.0f}%"
    )
    if c.K == c.N:
        attention = f"multi-head: {c.N} heads, head size {c.H}"
    else:
        name = "multi-query" if c.K == 1 else "grouped-query"
        attention = f"{name}: {c.N} query / {c.K} KV heads, head size {c.H}"
    storage = " | ".join(
        f"{label} {total * size / 1e9:,.1f} GB"
        for label, size in (("bf16", 2), ("int8", 1), ("4-bit", 0.5))
    )
    facts = (
        ("Params", f"{params} ({split})"),
        ("Shape", f"{c.L} layers, hidden {c.D:,}, FFW {c.F:,}"),
        ("Attention", attention),
        ("Context", f"{a.context:,} position limit" if a.context else "not found"),
        ("Weights", storage),
    )

    lines = [wrap(f"{model}   {kind}   (config: {source})", "", "  "), ""]
    lines += [wrap(text, f"  {label:<11}", " " * 13) for label, text in facts]

    if a.warnings:
        lines += ["", "Warnings"]
        lines += [wrap(w, "  ! ", "    ") for w in a.warnings]
    notes = [n for n in c.notes if any(key in n for key in _IMPORTANT_NOTES)]
    if notes:
        lines += ["", "Assumed"]
        lines += [wrap(n, "  ? ", "    ") for n in notes]

    checks = [
        (tag, item.short)
        for tag, items in (("[train]", a.post_training), ("[test]", a.testing))
        for item in items
        if item.short
    ]
    if checks:
        lines += ["", "Worth checking"]
        lines += [wrap(text, f"  {tag:<8}", " " * 10) for tag, text in checks]
    return "\n".join(lines)


def render_short_memory(
    scenarios: tuple[Scenario, ...], batch: int, seq: int, lora_rank: int
) -> str:
    lines = [f"Memory (batch={batch}, seq={seq}, LoRA rank={lora_rank}; 1 GB = 1e9 bytes)"]
    lines += [
        f"  {s.label:<22} {s.total / GB:>9,.2f} GB   largest: {s.dominant.name}"
        for s in scenarios
    ]
    lines.append(
        _wrap(
            "Rough: no overhead or checkpointing; activations unmeasured; LoRA on "
            "q/k/v/o.",
            "  ",
            "  ",
            SHORT_WIDTH,
        )
    )
    return "\n".join(lines)


def render_analysis(
    model: str,
    source: str,
    path: Path,
    c: ModelConfig,
    counts: ParamCounts,
    a: Analysis,
) -> str:
    lines = [
        f"roofline {__version__}",
        f"  model:  {model}",
        f"  config: {source} ({path})",
        "",
        "Overview",
        *_labelled(a.overview),
    ]
    if a.warnings:
        lines += ["", "Warnings"]
        lines += [_wrap(w, "  ! ", "    ") for w in a.warnings]
    lines += [
        "",
        "Dimensions",
        f"  L={c.L}  D={c.D}  F={c.F}  N={c.N}  K={c.K}  H={c.H}  V={c.V}",
        f"  tied embeddings: {'yes' if c.tied else 'no'}",
    ]
    lines += [
        _wrap(n, "  note: ", "        ")
        for n in c.notes
        if not n.startswith("WARNING")
    ]
    lines += ["", "Parameters"]
    for label, n in (
        ("FFW", counts.ffw),
        ("Attention", counts.attention),
        ("Vocab", counts.vocab),
        ("Total", counts.total),
    ):
        share = 100 * n / counts.total
        lines.append(f"  {label:<10} {n:>18,}  {n / 1e9:8.2f}e9  {share:5.1f}%")
    lines += ["", *_labelled(a.parameter_details)]
    lines += ["", "For testing", *_checks(a.testing)]
    lines += ["", "For post-training", *_checks(a.post_training)]
    if a.other_settings:
        lines += ["", "Other config settings (as written; not all are interpreted)"]
        lines += [f"  {key:<28} {value}" for key, value in a.other_settings]
    lines += ["", "Not determined from config.json"]
    lines += [f"  - {line}" for line in UNAVAILABLE]
    return "\n".join(lines)


def render_memory(
    scenarios: tuple[Scenario, ...], batch: int, seq: int, lora_rank: int
) -> str:
    lines = [
        f"Memory estimates (batch={batch}, seq={seq}, lora_rank={lora_rank}; "
        "1 GB = 1e9 bytes)"
    ]
    for s in scenarios:
        lines.append(f"  {s.label:<30} {s.total / GB:>10,.2f} GB")
        for part in s.components:
            marker = "  <- dominates" if part == s.dominant else ""
            lines.append(f"    {part.name:<28} {part.bytes / GB:>10,.2f} GB{marker}")
    lines += ["", "Memory assumptions"]
    lines += [_wrap(a, "  - ", "    ") for a in ASSUMPTIONS]
    return "\n".join(lines)
