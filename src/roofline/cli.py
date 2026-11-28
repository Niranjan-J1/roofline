"""Command-line entry point for roofline."""

from __future__ import annotations

import argparse
import math
import sys

from roofline import __version__
from roofline.config import ConfigError, find_config, parse_config, read_config
from roofline.env import detect_hardware
from roofline.insights import analyze
from roofline.memory import memory_scenarios
from roofline.params import count_params
from roofline.report import (
    WITHHELD_MEMORY,
    render_analysis,
    render_hardware,
    render_memory,
    render_short,
    render_short_memory,
    render_verify,
)
from roofline.verify import verify_count

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
    if not (math.isfinite(x) and x > 0):  # also rejects nan and inf
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

    hw = p.add_argument_group(
        "hardware (optional; 1 GB = 1e9 bytes, so a '24 GB' card is about 25.8)"
    )
    hw.add_argument(
        "--fit",
        action="store_true",
        help="show memory estimates and this machine's hardware (no verdicts yet)",
    )
    hw.add_argument(
        "--gpus",
        type=_nonneg_int,
        metavar="N",
        help="total GPUs across all nodes; implies --fit",
    )
    hw.add_argument(
        "--vram",
        type=_positive_float,
        metavar="GB",
        help="memory per GPU; implies --fit",
    )
    hw.add_argument(
        "--ram",
        type=_positive_float,
        metavar="GB",
        help="system RAM per node; implies --fit",
    )

    p.add_argument(
        "--verify",
        action="store_true",
        help="compare the total with HuggingFace's count (extra online lookup)",
    )
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
    counts = None if cfg.unsupported else count_params(cfg)

    workload = (args.batch, args.seq, args.lora_rank)
    overrides = (args.gpus, args.vram, args.ram)
    show_hardware = args.fit or any(v is not None for v in overrides)
    show_memory = (
        args.memory or show_hardware or any(v is not None for v in workload)
    )
    batch = args.batch or DEFAULT_BATCH
    seq = args.seq or DEFAULT_SEQ
    rank = args.lora_rank or DEFAULT_LORA_RANK

    analysis = analyze(raw, cfg, counts, rank)
    if args.full:
        print(render_analysis(args.model, source, path, cfg, counts, analysis))
    else:
        print(render_short(args.model, source, cfg, counts, analysis))
    if show_memory and counts is None:
        print(f"\n{WITHHELD_MEMORY}")
    elif show_memory:
        render = render_memory if args.full else render_short_memory
        print()
        print(render(memory_scenarios(cfg, batch, seq, rank), batch, seq, rank))
    if args.verify:
        print()
        print(render_verify(verify_count(args.model, source, raw, cfg, counts)))
    if show_hardware:
        hardware = detect_hardware(args.gpus, args.vram, args.ram)
        print()
        print(render_hardware(hardware, args.full))
    if not args.full:
        more = "--full" if show_memory else "--full   Memory: --memory"
        print(f"\nEstimates from config.json. More: {more}")

    if args.json:
        print("roofline: note: --json is not available yet", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
