import pytest

from roofline.config import parse_config
from roofline.params import count_params

LLAMA_70B = {
    "num_hidden_layers": 80,
    "hidden_size": 8192,
    "intermediate_size": 28672,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "tie_word_embeddings": False,
}


def test_llama_70b_anchor():
    p = count_params(parse_config(LLAMA_70B))
    assert p.ffw == pytest.approx(56.3e9, rel=0.005)
    assert p.attention == pytest.approx(12.0e9, rel=0.01)  # ~21e9 means GQA is broken
    assert p.vocab == pytest.approx(2.1e9, rel=0.01)
    assert p.total == pytest.approx(70.4e9, rel=0.005)  # doc figure sums truncated parts
    assert (p.ffw, p.attention, p.vocab) == (56_371_445_760, 12_079_595_520, 2_101_346_304)
    assert p.total == 70_552_387_584


def test_tied_embeddings_count_vocab_once():
    c = parse_config({**LLAMA_70B, "tie_word_embeddings": True})
    assert count_params(c).vocab == 128256 * 8192


def test_fallbacks_are_applied_and_noted():
    raw = dict(LLAMA_70B)
    del raw["num_key_value_heads"], raw["tie_word_embeddings"]
    c = parse_config(raw)
    assert (c.K, c.H, c.tied) == (64, 128, False)
    for expected in ("num_key_value_heads", "head_dim", "tie_word_embeddings"):
        assert any(expected in note for note in c.notes)
    assert c.tie_assumed


def test_explicit_head_dim_wins():
    assert parse_config({**LLAMA_70B, "head_dim": 256}).H == 256
