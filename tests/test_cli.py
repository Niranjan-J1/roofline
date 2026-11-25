import pytest

from roofline.cli import build_parser, main


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


def test_main_runs(capsys):
    assert main(["some/model"]) == 0
    assert "some/model" in capsys.readouterr().out
