import json

import pytest

from roofline.cli import build_parser, main
from test_params import LLAMA_70B


def test_defaults():
    args = build_parser().parse_args(["some/model"])
    assert args.model == "some/model"
    assert args.gpus is None and args.vram is None and args.ram is None
    assert args.batch is None and args.seq is None and args.lora_rank is None
    assert args.memory is False and args.fit is False
    assert args.json is False


def test_overrides():
    args = build_parser().parse_args(
        ["m", "--gpus", "8", "--vram", "80", "--ram", "512", "--json"]
    )
    assert (args.gpus, args.vram, args.ram, args.json) == (8, 80.0, 512.0, True)


def test_rejects_bad_vram():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["m", "--vram", "0"])


@pytest.mark.parametrize("flag", ["--batch", "--seq", "--lora-rank"])
def test_rejects_zero_workload(flag):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["m", flag, "0"])


def test_allows_zero_gpus():
    assert build_parser().parse_args(["m", "--gpus", "0"]).gpus == 0


def test_main_prints_breakdown(tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(LLAMA_70B))
    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "local file" in out
    assert "70.55e9" in out


def run_main(tmp_path, capsys, *flags):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(LLAMA_70B))
    assert main([str(path), *flags]) == 0
    return capsys.readouterr()


def test_default_output_is_analysis_only(tmp_path, capsys):
    out = run_main(tmp_path, capsys).out
    for heading in ("Overview", "Parameters", "For testing", "For post-training"):
        assert heading in out
    assert "Memory estimates" not in out


@pytest.mark.parametrize(
    "flags", [["--memory"], ["--batch", "1"], ["--seq", "64"], ["--lora-rank", "8"]]
)
def test_memory_flags_add_estimates(tmp_path, capsys, flags):
    assert "Memory estimates" in run_main(tmp_path, capsys, *flags).out


@pytest.mark.parametrize("flags", [["--fit"], ["--gpus", "0"], ["--ram", "8"]])
def test_fit_flags_say_not_available(tmp_path, capsys, flags):
    assert "not available yet" in run_main(tmp_path, capsys, *flags).err


def test_main_reports_config_errors(tmp_path, capsys):
    assert main([str(tmp_path / "nope.json")]) == 1
    assert "config file not found" in capsys.readouterr().err
