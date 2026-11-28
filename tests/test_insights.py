from roofline.config import parse_config
from roofline.insights import analyze
from roofline.params import count_params
from test_params import LLAMA_70B

QWEN_7B = {
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    "num_hidden_layers": 28,
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "vocab_size": 152064,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "max_position_embeddings": 131072,
    "sliding_window": 131072,
    "use_sliding_window": False,
    "bos_token_id": 151643,
    "eos_token_id": 151643,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "attention_dropout": 0.0,
}


def run(raw=QWEN_7B, **changes):
    raw = {**raw, **changes}
    cfg = parse_config(raw)
    return analyze(raw, cfg, count_params(cfg))


def texts(items):
    return {i.title: i.text for i in items}


def test_qwen_has_no_warnings_and_expected_figures():
    a = run()
    assert a.warnings == ()
    testing = texts(a.testing)
    assert "57,344 bytes per token" in testing["Long inputs"]
    assert "7.52 GB per sequence" in testing["Long inputs"]
    assert testing["Sliding-window attention"].startswith("off.")
    assert "Position scaling" not in testing and "Dropout" not in testing
    details = texts(a.parameter_details)
    assert "822.08M" in details["Shared K/V heads"]
    assert "1.44B" in details["Shared K/V heads"]
    assert "14.14 GFLOPs forward" in details["Compute per token"]
    assert "fp32 30.46 GB" in details["Weight storage"]
    training = texts(a.post_training)
    assert "start and end IDs are the same" in training["Special tokens"]
    assert "10.09M" in training["LoRA size"]
    assert "KV cache setting" in training


def test_minimal_config_does_not_invent_settings():
    a = run(LLAMA_70B)
    overview = texts(a.overview)
    assert overview["Position limit"] == "not found"
    assert overview["Saved precision"] == "not stated"
    assert "Sliding-window attention" not in texts(a.testing)
    assert "KV cache setting" not in texts(a.post_training)
    assert "No start, end or padding" in texts(a.post_training)["Special tokens"]


def test_each_warning_is_triggered_by_its_setting():
    cases = {
        "custom model code": {"auto_map": {"AutoModel": "x.Y"}},
        "already quantized": {"quantization_config": {"bits": 4}},
        "Multimodal": {"vision_config": {"depth": 2}},
        "Encoder-decoder": {"is_encoder_decoder": True},
        "Activation is gelu": {"hidden_act": "gelu", "model_type": "other"},
    }
    for expected, change in cases.items():
        warnings = run(**change).warnings
        assert len(warnings) == 1 and expected in warnings[0], expected


def test_active_sliding_window():
    testing = texts(run(use_sliding_window=True).testing)
    assert "window of 131,072" in testing["Sliding-window attention"]
    assert "may keep fewer" in testing["Long inputs"]
    layered = run(layer_types=["sliding_attention", "full_attention"])
    assert "1 of 28" in texts(layered.testing)["Sliding-window attention"]


def test_position_scaling_and_dropout():
    changed = run(rope_scaling={"rope_type": "yarn"}, attention_dropout=0.1)
    testing = texts(changed.testing)
    assert "yarn" in testing["Position scaling"] and "Dropout" in testing
    nested = run(rope_parameters={"full_attention": {"rope_type": "default"}})
    assert "Position scaling" not in texts(nested.testing)


def test_token_cases():
    changed = run(eos_token_id=[1, 2], pad_token_id=0)
    text = texts(changed.post_training)["Special tokens"]
    assert "several end IDs" in text and "No padding" not in text


def test_assumed_untied_is_labelled():
    raw = {k: v for k, v in QWEN_7B.items() if k != "tie_word_embeddings"}
    assert "Untied is assumed" in texts(run(raw).post_training)["Embeddings"]


def test_nested_text_config_settings_are_used():
    a = run({"model_type": "vlm", "vision_config": {"x": 1}, "text_config": QWEN_7B})
    assert texts(a.overview)["Position limit"] == "131,072 tokens"
    assert any("Multimodal" in w for w in a.warnings)


def test_other_settings_are_listed_and_sanitized():
    a = run(rope_theta=1000000.0, nasty="\x1b[31mred", long="x" * 500)
    other = dict(a.other_settings)
    assert other["rope_theta"] == "1000000.0"
    assert "\x1b" not in other["nasty"] and len(other["long"]) == 70
    assert "hidden_size" not in other
    assert "\x1b" not in texts(run(model_type="\x1b]0;x").overview)["Architecture"]


def shorts(items):
    return {i.title: i.short for i in items if i.short}


def test_short_checks_are_only_model_specific():
    a = run()
    assert set(shorts(a.testing)) == {"Long inputs"}
    assert set(shorts(a.post_training)) == {"Special tokens"}


def test_short_checks_follow_their_settings():
    clean = run(pad_token_id=0, eos_token_id=2, bos_token_id=1)
    assert shorts(clean.post_training) == {}
    assert "Embeddings" in shorts(run(tie_word_embeddings=True).post_training)
    changed = run(
        use_sliding_window=True, rope_scaling={"type": "yarn"}, attention_dropout=0.1
    )
    testing = shorts(changed.testing)
    assert set(testing) == {
        "Long inputs",
        "Sliding-window attention",
        "Position scaling",
        "Dropout",
    }
    assert "up to" in testing["Long inputs"]


def test_listed_window_is_not_reported_as_on():
    raw = {k: v for k, v in QWEN_7B.items() if k != "use_sliding_window"}
    listed = shorts(run(raw, sliding_window=2047).testing)
    assert "2,047-token sliding window is listed" in listed["Sliding-window attention"]
    assert "is on" not in listed["Sliding-window attention"]
    assert "up to" in listed["Long inputs"]
    # A window as large as the position limit can never take effect.
    unreachable = shorts(run(raw).testing)
    assert "Sliding-window attention" not in unreachable
    assert "up to" not in unreachable["Long inputs"]
