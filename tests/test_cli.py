import json

import pytest

from roofline import env
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


def run_main(tmp_path, capsys, *flags, **changes):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**LLAMA_70B, **changes}))
    assert main([str(path), *flags]) == 0
    return capsys.readouterr()


def test_full_view_has_all_sections(tmp_path, capsys):
    out = run_main(tmp_path, capsys, "--full").out
    for heading in ("Overview", "Parameters", "For testing", "For post-training"):
        assert heading in out
    assert "70.55e9" in out and "local file" in out
    assert "Memory estimates" not in out


def test_default_view_is_short(tmp_path, capsys):
    out = run_main(tmp_path, capsys).out
    assert "70.55B" in out and "local file" in out
    for heading in ("Overview", "For testing", "Other config settings", "Memory ("):
        assert heading not in out
    # The first line holds a long temp path, so count from the facts onward.
    assert len(out[out.index("  Params") :].splitlines()) <= 14


@pytest.mark.parametrize(
    "flags", [["--memory"], ["--batch", "1"], ["--seq", "64"], ["--lora-rank", "8"]]
)
def test_memory_flags_add_short_estimates(tmp_path, capsys, flags):
    out = run_main(tmp_path, capsys, *flags).out
    assert "Memory (batch=" in out and "largest:" in out
    assert "activations unmeasured" in out
    assert "adapters + optimizer" not in out  # breakdown is --full only


def test_full_memory_shows_breakdown(tmp_path, capsys):
    out = run_main(tmp_path, capsys, "--full", "--memory").out
    assert "Memory estimates" in out and "frozen weights" in out
    assert "Memory assumptions" in out


def test_short_view_keeps_every_warning(tmp_path, capsys):
    out = run_main(
        tmp_path,
        capsys,
        auto_map={"AutoModel": "x.Y"},
        quantization_config={"bits": 4},
        vision_config={"depth": 2},
        hidden_act="gelu",
    ).out
    text = " ".join(out.split())
    for expected in (
        "custom model code",
        "already quantized",
        "Multimodal",
        "Activation is gelu",
    ):
        assert expected in text


def test_short_view_shows_important_assumptions(tmp_path, capsys):
    # LLAMA_70B states K and tying, so null them to force both guesses.
    out = run_main(
        tmp_path, capsys, num_key_value_heads=None, tie_word_embeddings=None
    ).out
    assert "Assumed" in out
    assert "num_key_value_heads missing" in out
    assert "tie_word_embeddings missing" in out
    assert "head_dim missing" not in out


@pytest.mark.parametrize(
    "flags", [["--fit"], ["--gpus", "0"], ["--vram", "80"], ["--ram", "8"]]
)
def test_hardware_flags_add_memory_and_hardware(tmp_path, capsys, monkeypatch, flags):
    monkeypatch.setattr(env, "probe_gpus", lambda: env.Probe("unavailable", "test"))
    out = run_main(tmp_path, capsys, *flags).out
    assert "Memory (batch=" in out and "Hardware (this machine)" in out
    assert "Fit verdicts are not available yet." in out


@pytest.mark.parametrize("flags", [[], ["--full"], ["--memory"]])
def test_hardware_is_not_detected_unless_asked(tmp_path, capsys, monkeypatch, flags):
    def fail():
        raise AssertionError("hardware detection ran")

    monkeypatch.setattr(env, "probe_gpus", fail)
    assert "Hardware (this machine)" not in run_main(tmp_path, capsys, *flags).out


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_rejects_non_finite_memory(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["m", "--ram", value])


def test_main_reports_config_errors(tmp_path, capsys):
    assert main([str(tmp_path / "nope.json")]) == 1
    assert "config file not found" in capsys.readouterr().err
