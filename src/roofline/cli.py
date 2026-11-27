"""Command-line entry point for roofline."""

from __future__ import annotations

import argparse
import sys

from roofline import __version__
from roofline.config import ConfigError, find_config, parse_config, read_config
from roofline.insights import analyze
from roofline.memory import memory_scenarios
from roofline.params import count_params
from roofline.report import (
    render_analysis,
    render_memory,
    render_short,
    render_short_memory,
)

DEFAULT_BATCH, DEFAULT_SEQ, DEFAULT_LORA_RANK = 1, 4096, 16


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
            "Analyze a HuggingFace model config: architecture, parameters, and "
            "things to check for testing and post-training."
        ),
    )
    p.add_argument(
        "model",
        help="HuggingFace repo id (e.g. Qwen/Qwen2.5-0.5B), a local model directory, "
        "or a path to a config.json",
    )

    p.add_argument(
        "--full",
        action="store_true",
        help="show all details (with --memory, the full memory breakdown)",
    )

    mem = p.add_argument_group("memory estimates (optional)")
    mem.add_argument("--memory", action="store_true", help="add memory estimates")
    # Defaults stay None so that passing any of these turns on --memory.
    mem.add_argument(
        "--batch",
        type=_positive_int,
        metavar="B",
        help=f"batch size (default: {DEFAULT_BATCH}); implies --memory",
    )
    mem.add_argument(
        "--seq",
        type=_positive_int,
        metavar="T",
        help=f"sequence length (default: {DEFAULT_SEQ}); implies --memory",
    )
    mem.add_argument(
        "--lora-rank",
        type=_positive_int,
        metavar="R",
        help=f"LoRA rank (default: {DEFAULT_LORA_RANK}); implies --memory",
    )

    hw = p.add_argument_group("hardware fit (not available yet)")
    hw.add_argument("--fit", action="store_true", help="check the model against this machine")
    hw.add_argument("--gpus",type=_nonneg_int, metavar="N", help="override detected GPU count")
    hw.add_argument("--vram", type=_positive_float, metavar="GB", help="override detected per-GPU VRAM in GB")
    hw.add_argument("--ram", type=_positive_float, metavar="GB", help="override detected system RAM in GB")

    p.add_argument("--json", action="store_true", help="JSON output (not available yet)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        path, source = find_config(args.model)
        raw = read_config(path)
        cfg = parse_config(raw)
    except ConfigError as e:
        print(f"roofline: error: {e}", file=sys.stderr)
        return 1
    counts = count_params(cfg)

    workload = (args.batch, args.seq, args.lora_rank)
    show_memory = args.memory or any(v is not None for v in workload)
    batch = args.batch or DEFAULT_BATCH
    seq = args.seq or DEFAULT_SEQ
    rank = args.lora_rank or DEFAULT_LORA_RANK

    analysis = analyze(raw, cfg, counts, rank)
    if args.full:
        print(render_analysis(args.model, source, path, cfg, counts, analysis))
    else:
        print(render_short(args.model, source, cfg, counts, analysis))
    if show_memory:
        render = render_memory if args.full else render_short_memory
        print()
        print(render(memory_scenarios(cfg, batch, seq, rank), batch, seq, rank))
    if not args.full:
        more = "--full" if show_memory else "--full   Memory: --memory"
        print(f"\nEstimates from config.json. More: {more}")

    hardware = (args.gpus, args.vram, args.ram)
    if args.fit or any(v is not None for v in hardware):
        print("roofline: note: hardware fit is not available yet", file=sys.stderr)
    if args.json:
        print("roofline: note: --json is not available yet", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
