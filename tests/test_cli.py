import json

import pytest

from roofline.cli import build_parser, main
from test_params import LLAMA_70B


def test_defaults():
    args = build_parser().parse_args(["some/model"])
    assert args.model == "some/model"
    assert args.gpus is None and args.vram is None and args.ram is None
    assert args.batch == 1 and args.seq == 4096 and args.lora_rank == 16
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


def test_main_reports_config_errors(tmp_path, capsys):
    assert main([str(tmp_path / "nope.json")]) == 1
    assert "config file not found" in capsys.readouterr().err
