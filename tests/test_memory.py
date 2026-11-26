import pytest

from roofline.config import parse_config
from roofline.memory import lora_params, memory_scenarios
from test_params import LLAMA_70B

QWEN_7B = parse_config(
    {
        "num_hidden_layers": 28,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "vocab_size": 152064,
        "tie_word_embeddings": False,
    }
)


def run(cfg=QWEN_7B, batch=1, seq=4096, rank=16):
    return {s.key: s for s in memory_scenarios(cfg, batch, seq, rank)}


def parts(scenario):
    return {c.name: c.bytes for c in scenario.components}


def test_qwen_7b_inference():
    s = run()["inference"]
    assert parts(s) == {"weights": 15_230_566_400, "KV cache": 234_881_024}
    assert s.total == 15_465_447_424


def test_qwen_7b_lora():
    assert lora_params(QWEN_7B, 16) == 10_092_544
    s = run()["lora"]
    assert parts(s) == {
        "frozen weights": 15_230_566_400,
        "adapters + optimizer": 121_110_528,
        "activations": 17_146_314_752,
    }
    assert s.total == 32_497_991_680


def test_qwen_7b_full_training_counts_12_bytes_per_param():
    s = run()["full_training"]
    assert s.total == 12 * 7_615_283_200 + 17_146_314_752


def test_llama_70b_weights():
    s = run(parse_config(LLAMA_70B))
    assert parts(s["inference"])["weights"] == 2 * 70_552_387_584
    full = parts(s["full_training"])
    assert full["weights"] + full["gradients + optimizer"] == 12 * 70_552_387_584


@pytest.mark.parametrize("kwargs", [{"batch": 8}, {"seq": 8 * 4096}])
def test_workload_scales_activations_and_kv_cache_only(kwargs):
    base, big = run(), run(**kwargs)
    assert parts(big["lora"])["activations"] == 8 * parts(base["lora"])["activations"]
    assert parts(big["inference"])["KV cache"] == 8 * parts(base["inference"])["KV cache"]
    assert parts(big["inference"])["weights"] == parts(base["inference"])["weights"]


def test_rank_scales_only_the_adapters():
    base, big = parts(run()["lora"]), parts(run(rank=32)["lora"])
    assert big["adapters + optimizer"] == 2 * base["adapters + optimizer"]
    assert big["activations"] == base["activations"]


def test_dominant_component_follows_workload():
    assert run()["full_training"].dominant.name == "gradients + optimizer"
    assert run(batch=8)["full_training"].dominant.name == "activations"
    assert run()["inference"].dominant.name == "weights"
    assert run(batch=32, seq=32768)["inference"].dominant.name == "KV cache"
