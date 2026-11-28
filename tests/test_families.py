"""Family-specific layouts: experts, two-matrix FFW, GPT-2, unsupported models."""

import json
from pathlib import Path

import pytest

from roofline.cli import main
from roofline.config import ConfigError, parse_config
from roofline.memory import activation_bytes, memory_scenarios
from roofline.params import count_params
from test_params import LLAMA_70B

FIXTURES = Path(__file__).parent / "fixtures"


def config(name):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))["config"]


def run_main(tmp_path, capsys, raw, *flags):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    assert main([str(path), *flags]) == 0
    return " ".join(capsys.readouterr().out.split())


# --- mixture of experts ------------------------------------------------------


@pytest.mark.parametrize(
    "name, experts, active, shared",
    [
        ("mixtral-8x7b-v0_1", 8, 2, 0),
        ("olmoe-1b-7b-0924", 64, 8, 0),
        ("qwen1_5-moe-a2_7b", 60, 4, 5632),
        ("qwen3-30b-a3b", 128, 8, 0),
        ("gpt-oss-20b", 32, 4, 0),
    ],
)
def test_expert_layouts(name, experts, active, shared):
    c = parse_config(config(name))
    assert (c.moe.experts, c.moe.active, c.moe.shared_width) == (experts, active, shared)
    counts = count_params(c)
    assert counts.active_total < counts.total


def test_mixtral_ffw_by_hand():
    counts = count_params(parse_config(config("mixtral-8x7b-v0_1")))
    expert, router = 3 * 4096 * 14336, 4096 * 8
    assert counts.ffw == 32 * (8 * expert + router)
    assert counts.ffw_active == 32 * (2 * expert + router)


def test_mixed_dense_and_expert_layers():
    raw = {**config("qwen3-30b-a3b"), "decoder_sparse_step": 2, "mlp_only_layers": [1]}
    c = parse_config(raw)
    assert c.moe.sparse_layers == 23  # every second of 48 layers, minus layer 1
    expert = 3 * c.D * c.moe.width
    assert count_params(c).ffw == (48 - 23) * 3 * c.D * c.F + 23 * (
        128 * expert + c.D * 128
    )


def test_activations_use_active_experts_and_weights_use_all():
    c = parse_config(config("mixtral-8x7b-v0_1"))
    assert activation_bytes(c, 1, 1) == 2 * 32 * (5 * 4096 + 3 * 2 * 14336)
    inference = memory_scenarios(c, 1, 1, 16)[0]
    assert inference.components[0].bytes == 2 * count_params(c).total


@pytest.mark.parametrize(
    "change, key",
    [
        ({"num_experts_per_tok": None}, "num_experts_per_tok"),
        ({"num_local_experts": None}, "num_local_experts"),
        ({"num_experts_per_tok": 99}, "exceeds"),
    ],
)
def test_expert_settings_are_never_guessed(change, key):
    with pytest.raises(ConfigError, match=key):
        parse_config({**config("mixtral-8x7b-v0_1"), **change})


# --- two-matrix families and GPT-2 -------------------------------------------


@pytest.mark.parametrize("name", ["phi-2", "pythia-160m", "gpt2"])
def test_two_matrix_families(name):
    c = parse_config(config(name))
    assert c.ffw_matrices == 2
    assert count_params(c).ffw == 2 * c.D * c.F * c.L


def test_family_is_chosen_by_model_type_not_activation():
    raw = {**LLAMA_70B, "model_type": "gemma2", "hidden_act": "gelu_pytorch_tanh"}
    assert parse_config(raw).ffw_matrices == 3


def test_gpt2_names_and_labelled_defaults():
    c = parse_config(config("gpt2"))
    assert (c.L, c.D, c.F, c.N, c.K, c.positions) == (12, 768, 3072, 12, 12, 1024)
    assert c.tied and not c.tie_assumed
    assert sum("family default" in n for n in c.notes) >= 2
    assert count_params(c).position == 1024 * 768


def test_gpt2_missing_key_is_named_in_gpt2_terms():
    raw = {k: v for k, v in config("gpt2").items() if k != "n_layer"}
    with pytest.raises(ConfigError, match="n_layer"):
        parse_config(raw)


def test_gpt2_output(tmp_path, capsys):
    out = run_main(tmp_path, capsys, config("gpt2"))
    assert "1,024 position limit" in out and "family default" in out
    assert "attention_dropout is 0.1" in out and "positions 1%" in out
    assert "Activation is" not in out


# --- missing shared-weight setting -------------------------------------------


@pytest.mark.parametrize("flags", [[], ["--full"]])
def test_unstated_tie_shows_both_totals(tmp_path, capsys, flags):
    raw = {k: v for k, v in LLAMA_70B.items() if k != "tie_word_embeddings"}
    out = run_main(tmp_path, capsys, raw, *flags)
    assert "70.55B parameters if separate (used here, including memory)" in out
    assert "69.50B if shared" in out


# --- unsupported layouts -----------------------------------------------------

DEEPSEEK = config("unsupported/deepseek-v3")
UNKNOWN_MOE = {**LLAMA_70B, "model_type": "newthing_moe", "num_experts": 8}


@pytest.mark.parametrize("raw", [DEEPSEEK, UNKNOWN_MOE], ids=["deepseek", "unknown"])
@pytest.mark.parametrize(
    "flags", [[], ["--full"], ["--memory"], ["--full", "--memory"], ["--seq", "8"]]
)
def test_unsupported_models_show_no_derived_figures(tmp_path, capsys, raw, flags):
    out = run_main(tmp_path, capsys, raw, *flags)
    assert "figures are withheld" in out
    for figure in (" GB", "GFLOPs", "B parameters", "M parameters", "FFW "):
        assert figure not in out
    if "--memory" in flags or "--seq" in flags:
        assert "Memory estimates withheld" in out


def test_unsupported_model_cannot_be_counted():
    with pytest.raises(ValueError):
        count_params(parse_config(DEEPSEEK))


def test_compressed_attention_is_unsupported_without_experts():
    c = parse_config({**LLAMA_70B, "kv_lora_rank": 128})
    assert "compressed" in c.unsupported


# --- unchecked families ------------------------------------------------------


def test_unchecked_family_is_labelled(tmp_path, capsys):
    out = run_main(tmp_path, capsys, {**LLAMA_70B, "model_type": "newfamily"})
    assert "'newfamily' model type has not been checked" in out
    checked = run_main(tmp_path, capsys, {**LLAMA_70B, "model_type": "llama"})
    assert "has not been checked" not in checked


def test_hostile_model_type_is_not_echoed_raw(tmp_path, capsys):
    out = run_main(tmp_path, capsys, {**LLAMA_70B, "model_type": "\x1b[31mbad"})
    assert "\x1b" not in out


@pytest.mark.parametrize(
    "name",
    ["qwen2_5-0_5b", "mistral-7b-v0_1", "mixtral-8x7b-v0_1", "phi-2", "gpt2"],
)
def test_short_view_stays_short_for_real_models(tmp_path, capsys, name):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config(name)))
    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    # Skip the header, which holds a long temporary path.
    assert len(out[out.index("  Params") :].splitlines()) <= 18
