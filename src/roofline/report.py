"""Human-readable output (JSON output comes later)."""

from __future__ import annotations

import textwrap
from pathlib import Path

from roofline import __version__
from roofline.config import ModelConfig
from roofline.env import SCOPE_NOTE, Hardware, Slurm
from roofline.insights import UNAVAILABLE, Analysis, Item, safe_text
from roofline.memory import ASSUMPTIONS, GB, Scenario
from roofline.params import ParamCounts

WIDTH = 88
SHORT_WIDTH = 80

WITHHELD_MEMORY = (
    "Memory estimates withheld: roofline cannot count this model's parameters."
)

# Config assumptions that affect the numbers or their scope, so always shown.
_IMPORTANT_NOTES = (
    "num_key_value_heads missing",
    "tie_word_embeddings",
    "family default",
    "has not been checked",
    "text_config",
)


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


def _count(n: int) -> str:
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.2f}M"


def _notes(c: ModelConfig, counts: ParamCounts | None) -> list[str]:
    """Config notes; an unstated shared-weight setting shows both totals."""
    if not (c.tie_assumed and counts):
        return list(c.notes)
    both = (
        f"tie_word_embeddings missing: {_count(counts.total)} parameters if separate "
        f"(used here, including memory), {_count(counts.total - c.V * c.D)} if shared"
    )
    return [both if "tie_word_embeddings" in n else n for n in c.notes]


def render_short(
    model: str, source: str, c: ModelConfig, counts: ParamCounts | None, a: Analysis
) -> str:
    """The default view: about one screen. Warnings are never dropped."""

    def wrap(text: str, first: str, rest: str) -> str:
        return _wrap(text, first, rest, SHORT_WIDTH)

    kind = f"{a.architecture}, {a.dtype}" if a.dtype else a.architecture
    context = (
        "Context",
        f"{a.context:,} position limit" if a.context else "not found",
    )
    if counts is None:
        facts = [
            ("Params", "withheld (see warnings)"),
            ("Shape", f"{c.L} layers, hidden {c.D:,}, {c.N} attention heads"),
            ("Vocab", f"{c.V:,} slots"),
            context,
        ]
    else:
        total = counts.total
        parts = [
            ("FFW", counts.ffw),
            ("attention", counts.attention),
            ("vocab", counts.vocab),
        ]
        if counts.position:
            parts.append(("positions", counts.position))
        split = ", ".join(f"{name} {100 * n / total:.0f}%" for name, n in parts)
        if c.K == c.N:
            attention = f"multi-head: {c.N} heads, head size {c.H}"
        else:
            name = "multi-query" if c.K == 1 else "grouped-query"
            attention = f"{name}: {c.N} query / {c.K} KV heads, head size {c.H}"
        storage = " | ".join(
            f"{label} {total * size / 1e9:,.1f} GB"
            for label, size in (("bf16", 2), ("int8", 1), ("4-bit", 0.5))
        )
        width = f"expert FFW {c.moe.width:,}" if c.moe else f"FFW {c.F:,}"
        facts = [
            ("Params", f"{_count(total)} ({split})"),
            ("Shape", f"{c.L} layers, hidden {c.D:,}, {width}"),
            ("Attention", attention),
            context,
            ("Weights", storage),
        ]
        if c.moe:
            experts = (
                f"{c.moe.experts} per layer, {c.moe.active} active; "
                f"{_count(counts.active_total)} parameters active per token"
            )
            if c.moe.sparse_layers < c.L:
                experts += f"; {c.moe.sparse_layers} of {c.L} layers use experts"
            facts.insert(1, ("Experts", experts))

    header = f"{model}   {kind}   (config: {source})"
    if len(header) > SHORT_WIDTH:
        head = [wrap(f"{model}   {kind}", "", "  "), f"  (config: {source})"]
    else:
        head = [header]
    lines = [*head, ""]
    lines += [wrap(text, f"  {label:<11}", " " * 13) for label, text in facts]

    if a.warnings:
        lines += ["", "Warnings"]
        lines += [wrap(w, "  ! ", "    ") for w in a.warnings]
    notes = [
        n for n in _notes(c, counts) if any(key in n for key in _IMPORTANT_NOTES)
    ]
    if counts is None:  # head and FFW assumptions have nothing to apply to
        notes = [n for n in notes if "text_config" in n]
    # Defaults with the same prefix share one entry to keep the view short.
    merged: dict[str, list[str]] = {}
    for note in notes:
        prefix, found, value = note.partition("family default: ")
        merged.setdefault(prefix + found if found else note, []).append(value)
    notes = [
        key.replace("default: ", "defaults: " if len(values) > 1 else "default: ")
        + "; ".join(values)
        for key, values in merged.items()
    ]
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
    counts: ParamCounts | None,
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
    dimensions = f"  L={c.L}  D={c.D}  F={c.F}  N={c.N}  K={c.K}  H={c.H}  V={c.V}"
    if counts is None:
        dimensions = f"  L={c.L}  D={c.D}  N={c.N}  V={c.V}"
    lines += [
        "",
        "Dimensions",
        dimensions,
        f"  tied embeddings: {'yes' if c.tied else 'no'}",
    ]
    # With figures withheld, head and FFW assumptions have nothing to apply to.
    notes = _notes(c, counts)
    if counts is None:
        notes = [n for n in notes if "text_config" in n or "tie_word" in n]
    lines += [_wrap(n, "  note: ", "        ") for n in notes]
    if counts is not None:
        lines += ["", "Parameters"]
        rows = [
            ("FFW", counts.ffw),
            ("Attention", counts.attention),
            ("Vocab", counts.vocab),
        ]
        if counts.position:
            rows.append(("Positions", counts.position))
        rows.append(("Total", counts.total))
        for label, n in rows:
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


def _gb(value: int) -> str:
    return f"{value / GB:,.2f} GB"


def _gpu_summary(h: Hardware) -> str:
    if h.gpu_count is None:
        return "count unknown (see warnings); pass --gpus to set it"
    if h.gpu_count == 0:
        if h.gpu_count_source == "override":
            return "0 (override); CPU-only"
        if h.probe.status == "unavailable":
            return (
                f"no NVIDIA GPU detected ({safe_text(h.probe.detail)}); "
                "assuming CPU-only"
            )
        if h.probe.gpus:
            return "none visible (hidden by CUDA_VISIBLE_DEVICES); CPU-only"
        return f"no NVIDIA GPUs found by {h.probe.detail}; CPU-only"
    text = f"{h.gpu_count} total"
    if h.gpu_count_source == "override":
        text += " (override)"
    else:
        names = sorted({g.name for g in h.visible})
        text += f": {safe_text(names[0]) if len(names) == 1 else 'mixed NVIDIA GPUs'}"
    if h.vram_bytes is None:
        return f"{text}, memory unknown"
    text += f", {_gb(h.vram_bytes)} each ({h.vram_source})"
    free = [g.free_bytes for g in h.visible if g.free_bytes is not None]
    if free and h.vram_source == "detected":
        text += f", lowest free {_gb(min(free))}"
    return text


def _ram_summary(h: Hardware) -> str:
    if h.ram_total_bytes is None:
        return "unknown"
    if h.ram_source == "override":
        return f"{_gb(h.ram_total_bytes)} per node (override)"
    text = f"{_gb(h.ram_total_bytes)} total"
    if h.ram_available_bytes is not None:
        text += f", {_gb(h.ram_available_bytes)} available"
    return text


def _slurm_summary(s: Slurm) -> str:
    def known(value: int | None) -> str:
        return "unknown" if value is None else str(value)

    text = (
        f"job {safe_text(s.job_id, 20)}: nodes {known(s.nodes)}, "
        f"tasks {known(s.tasks)}, GPUs on this node {known(s.gpus_on_node)}"
    )
    if s.mem_per_node_bytes is not None:
        text += f", memory limit {_gb(s.mem_per_node_bytes)} per node"
    return text


def render_hardware(h: Hardware, full: bool) -> str:
    def row(label: str, text: str) -> str:
        return _wrap(text, f"  {label:<7}", " " * 9, SHORT_WIDTH)

    lines = [
        "Hardware (this machine)",
        row("GPUs", _gpu_summary(h)),
        row("RAM", _ram_summary(h)),
    ]
    if h.slurm:
        lines.append(row("Slurm", _slurm_summary(h.slurm)))
    lines += [_wrap(w, "  ! ", "    ", SHORT_WIDTH) for w in h.warnings]
    if full:
        lines += ["", "Hardware details"]
        check = f"{h.probe.status}: {safe_text(h.probe.detail, 200)}"
        lines.append(row("Check", check))
        for g in h.probe.gpus:
            total = _gb(g.total_bytes) if g.total_bytes is not None else "unknown"
            free = _gb(g.free_bytes) if g.free_bytes is not None else "unknown"
            seen = "visible" if g in h.visible else "not confirmed visible"
            mig = ", MIG on" if g.mig else ""
            text = f"{safe_text(g.name)}: {total} total, {free} free, {seen}{mig}"
            lines.append(row(f"GPU {g.index}", text))
        if h.cuda_visible_devices is not None:
            lines.append(f"  CUDA_VISIBLE_DEVICES={safe_text(h.cuda_visible_devices)}")
        if h.slurm:
            lines += [f"  {k}={safe_text(v)}" for k, v in h.slurm.variables]
        else:
            lines.append(row("Slurm", "no job detected"))
        lines.append(row("RAM", f"source: {h.ram_source or 'not detected'}"))
        lines.append(_wrap(SCOPE_NOTE, "  ", "  ", SHORT_WIDTH))
    lines.append("Fit verdicts are not available yet.")
    return "\n".join(lines)


def render_verify(lines: tuple[str, ...]) -> str:
    return "\n".join(
        ["HuggingFace check (--verify)"]
        + [_wrap(line, "  ", "  ", SHORT_WIDTH) for line in lines]
    )
