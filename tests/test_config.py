import json

import pytest
from huggingface_hub import errors

from roofline import config
from roofline.config import ConfigError, find_config, parse_config, read_config
from test_params import LLAMA_70B

REQUIRED = [
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "vocab_size",
]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if a test reaches the cache or Hub without faking it."""

    def boom(*args, **kwargs):
        raise AssertionError("unexpected HF lookup")

    monkeypatch.setattr(config, "try_to_load_from_cache", boom)
    monkeypatch.setattr(config, "hf_hub_download", boom)


@pytest.mark.parametrize("key", REQUIRED)
def test_missing_required_key_is_named(key):
    raw = {k: v for k, v in LLAMA_70B.items() if k != key}
    with pytest.raises(ConfigError, match=key):
        parse_config(raw)


@pytest.mark.parametrize("bad", [0, -1, "80", 1.5, True])
def test_bad_values_are_rejected(bad):
    with pytest.raises(ConfigError, match="num_hidden_layers"):
        parse_config({**LLAMA_70B, "num_hidden_layers": bad})


def test_indivisible_head_dim_is_not_guessed():
    with pytest.raises(ConfigError, match="head_dim"):
        parse_config({**LLAMA_70B, "num_attention_heads": 60})


def test_nested_text_config():
    c = parse_config({"model_type": "vlm", "text_config": LLAMA_70B})
    assert c.L == 80
    assert any("text_config" in n for n in c.notes)


def test_moe_warning():
    c = parse_config({**LLAMA_70B, "num_local_experts": 8})
    assert any("mixture-of-experts" in n for n in c.notes)


def test_json_file_and_directory(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(LLAMA_70B))
    assert find_config(str(path)) == (path, "local file")
    assert find_config(str(tmp_path)) == (path, "local directory")
    assert read_config(path) == LLAMA_70B


def test_directory_without_config(tmp_path):
    with pytest.raises(ConfigError, match="no config.json"):
        find_config(str(tmp_path))


def test_invalid_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json")
    with pytest.raises(ConfigError, match="could not read"):
        read_config(path)


def test_cache_hit_skips_the_network(monkeypatch):
    # hf_hub_download stays faked to fail, so a download attempt would be caught.
    monkeypatch.setattr(config, "try_to_load_from_cache", lambda *a: "cached/config.json")
    path, source = find_config("org/model")
    assert source == "HF cache" and path.name == "config.json"


@pytest.mark.parametrize("cached", [None, object()])
def test_cache_miss_downloads_only_the_config(monkeypatch, cached):
    calls = []
    monkeypatch.setattr(config, "try_to_load_from_cache", lambda *a: cached)
    monkeypatch.setattr(
        config, "hf_hub_download", lambda *a: calls.append(a) or "dl/config.json"
    )
    assert find_config("org/model")[1] == "downloaded from HF Hub"
    assert calls == [("org/model", "config.json")]


def _fake(cls):
    # Skip __init__: the HTTP error constructors differ between hub versions.
    return cls.__new__(cls)


@pytest.mark.parametrize(
    "cls, message",
    [
        (errors.GatedRepoError, "gated"),
        (errors.RepositoryNotFoundError, "was not found"),
        (errors.RemoteEntryNotFoundError, "has no config.json"),
        (errors.LocalEntryNotFoundError, "could not be reached"),
    ],
)
def test_hub_errors_are_explained(monkeypatch, cls, message):
    def fail(*args):
        raise _fake(cls)

    monkeypatch.setattr(config, "try_to_load_from_cache", lambda *a: None)
    monkeypatch.setattr(config, "hf_hub_download", fail)
    with pytest.raises(ConfigError, match=message):
        find_config("org/model")
