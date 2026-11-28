import json
from types import SimpleNamespace

import pytest

from roofline import verify
from roofline.cli import main
from roofline.config import parse_config
from roofline.params import count_params
from roofline.verify import compare, verify_count
from test_params import LLAMA_70B

CONFIG = parse_config(LLAMA_70B)
COUNTS = count_params(CONFIG)
OURS = COUNTS.total


def fake_hub(monkeypatch, info=None, error=None):
    calls = []

    class Api:
        def model_info(self, repo, **kwargs):
            calls.append(repo)
            if error:
                raise error
            return SimpleNamespace(safetensors=info)

    monkeypatch.setattr(verify, "HfApi", Api)
    return calls


def check(source="HF cache", counts=COUNTS, raw=LLAMA_70B):
    return " ".join(verify_count("org/model", source, raw, CONFIG, counts))


def info(total, **dtypes):
    return SimpleNamespace(total=total, parameters=dtypes or {"BF16": total})


def test_agreement(monkeypatch):
    fake_hub(monkeypatch, info(OURS + 1000))
    assert "Agrees" in check()


@pytest.mark.parametrize(
    "dtypes, quantized, expected",
    [
        ({"BF16", "U8"}, False, "not a reliable check because"),
        ({"BF16"}, True, "not a reliable check because"),
    ],
)
def test_close_but_unreliable(dtypes, quantized, expected):
    assert expected in compare(OURS, OURS, dtypes, quantized, CONFIG)


def test_unreliable_mismatch():
    assert "Not a reliable" in compare(OURS, OURS * 2, {"F8_E4M3"}, False, CONFIG)


def test_plain_mismatch():
    assert "caution" in compare(OURS, OURS * 2, {"BF16"}, False, CONFIG)


def test_second_copy_of_shared_weights():
    tied = parse_config({**LLAMA_70B, "tie_word_embeddings": True})
    ours = count_params(tied).total
    text = compare(ours, ours + tied.V * tied.D, {"BF16"}, False, tied)
    assert "storing them twice" in text


@pytest.mark.parametrize("value", [None, info(0)])
def test_no_count(monkeypatch, value):
    fake_hub(monkeypatch, value)
    assert "has no parameter count" in check()


def test_lookup_failure_does_not_crash(monkeypatch):
    fake_hub(monkeypatch, error=ConnectionError("offline"))
    assert "Could not look up the count (ConnectionError)" in check()


@pytest.mark.parametrize(
    "kwargs", [{"source": "local file"}, {"source": "local directory"}, {"counts": None}]
)
def test_no_lookup_when_not_comparable(monkeypatch, kwargs):
    calls = fake_hub(monkeypatch, info(OURS))
    assert "Not" in check(**kwargs) and calls == []


@pytest.mark.parametrize("flags, expected", [([], False), (["--verify"], True)])
def test_cli_looks_up_only_with_verify(tmp_path, capsys, monkeypatch, flags, expected):
    calls = fake_hub(monkeypatch, info(OURS))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(LLAMA_70B))
    assert main([str(path), *flags]) == 0
    assert ("HuggingFace check" in capsys.readouterr().out) is expected
    assert calls == []  # a local path is never looked up


def test_gpt2_stored_masks():
    gpt2 = parse_config(
        {
            "model_type": "gpt2",
            "n_layer": 12,
            "n_embd": 768,
            "n_head": 12,
            "n_positions": 1024,
            "vocab_size": 50257,
        }
    )
    text = compare(124_318_464, 137_022_720, {"F32"}, False, gpt2)
    assert "attention masks" in text
    assert "caution" in compare(124_318_464, 200_000_000, {"F32"}, False, gpt2)
