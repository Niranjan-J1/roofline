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


def _wrap(text: str, first: str, rest: str) -> str:
    return textwrap.fill(
        text,
        width=WIDTH,
        initial_indent=first,
        subsequent_indent=rest,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _labelled(items: tuple[Item, ...]) -> list[str]:
    return [_wrap(i.text, f"  {i.title + ':':<19}", " " * 21) for i in items]


def _checks(items: tuple[Item, ...]) -> list[str]:
    return [_wrap(f"{i.title}: {i.text}", "  - ", "    ") for i in items]


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
