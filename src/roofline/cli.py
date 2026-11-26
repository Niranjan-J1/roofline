"""Command-line entry point for roofline."""

from __future__ import annotations

import argparse
import sys

from roofline import __version__
from roofline.config import ConfigError, find_config, parse_config, read_config
from roofline.memory import ASSUMPTIONS, GB, memory_scenarios
from roofline.params import count_params


def _nonneg_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return n


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return n


def _positive_float(value: str) -> float:
    x = float(value)
    if x <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return x


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="roofline",
        description=(
            "Analyze a HuggingFace model config and print a fit verdict "
            "(inference / LoRA / full training) for this machine."
        ),
    )
    p.add_argument(
        "model",
        help="HuggingFace repo id (e.g. Qwen/Qwen2.5-0.5B), a local model directory, "
        "or a path to a config.json",
    )

    hw = p.add_argument_group("hardware overrides (default: auto-detect)")
    hw.add_argument("--gpus", type=_nonneg_int, metavar="N", help="override detected GPU count")
    hw.add_argument("--vram", type=_positive_float, metavar="GB", help="override detected per-GPU VRAM in GB")
    hw.add_argument("--ram", type=_positive_float, metavar="GB", help="override detected system RAM in GB")

    wl = p.add_argument_group("workload assumptions")
    wl.add_argument("--batch", type=_positive_int, default=1, metavar="B", help="batch size (default: 1)")
    wl.add_argument("--seq", type=_positive_int, default=4096, metavar="T", help="sequence length (default: 4096)")
    wl.add_argument("--lora-rank", type=_positive_int, default=16, metavar="R", help="LoRA rank (default: 16)")

    p.add_argument("--json", action="store_true", help="machine-readable JSON output")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        path, source = find_config(args.model)
        cfg = parse_config(read_config(path))
    except ConfigError as e:
        print(f"roofline: error: {e}", file=sys.stderr)
        return 1
    counts = count_params(cfg)

    print(f"roofline {__version__}")
    print(f"  model:  {args.model}")
    print(f"  config: {source} ({path})")
    print()
    print("Model config")
    print(f"  L={cfg.L}  D={cfg.D}  F={cfg.F}  N={cfg.N}  K={cfg.K}  H={cfg.H}  V={cfg.V}")
    print(f"  tied embeddings: {'yes' if cfg.tied else 'no'}")
    for note in cfg.notes:
        print(f"  note: {note}")
    print()
    print("Parameters")
    for label, n in (
        ("FFW", counts.ffw),
        ("Attention", counts.attention),
        ("Vocab", counts.vocab),
        ("Total", counts.total),
    ):
        share = 100 * n / counts.total
        print(f"  {label:<10} {n:>18,}  {n / 1e9:8.2f}e9  {share:5.1f}%")
    print()
    print(
        f"Memory estimates (batch={args.batch}, seq={args.seq}, "
        f"lora_rank={args.lora_rank}; 1 GB = 1e9 bytes)"
    )
    for s in memory_scenarios(cfg, args.batch, args.seq, args.lora_rank):
        print(f"  {s.label:<30} {s.total / GB:>10,.2f} GB")
        for part in s.components:
            marker = "  <- dominates" if part == s.dominant else ""
            print(f"    {part.name:<28} {part.bytes / GB:>10,.2f} GB{marker}")
    print()
    print("Assumptions")
    for line in ASSUMPTIONS:
        print(f"  - {line}")
    print()
    for stage in ("environment detection", "verdict"):
        print(f"  [{stage}] not implemented yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
