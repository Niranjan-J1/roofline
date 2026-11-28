"""Load a HuggingFace config and map it to L/D/F/N/K/H/V."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Hide the Windows symlink notice, which is noise for a single small file.
# Must be set before huggingface_hub is imported. Errors are unaffected.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import hf_hub_download, try_to_load_from_cache
from huggingface_hub.errors import (
    GatedRepoError,
    HFValidationError,
    LocalEntryNotFoundError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
)

CONFIG_NAME = "config.json"

# The family rules below were each checked against a real model's tensor shapes
# (see tests/fixtures). Add a family only with a fixture that proves its layout.

# Starts every note about a value that comes from a family's own convention.
FAMILY_DEFAULT = "family default"
_FAMILY_NAMES = {"gpt2": "GPT-2", "gpt_neox": "GPT-NeoX"}

# Families whose FFW has two matrices (not gated). All others are assumed to have three.
TWO_MATRIX_FAMILIES = ("phi", "gpt_neox", "gpt2")

# Mixture-of-experts layouts: (expert count key, expert width key, shared width key)
MOE_LAYOUTS = {
    "mixtral": ("num_local_experts", "intermediate_size", None),
    "olmoe": ("num_experts", "intermediate_size", None),
    "qwen2_moe": (
        "num_experts",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
    ),
    "qwen3_moe": ("num_experts", "moe_intermediate_size", None),
    "gpt_oss": ("num_local_experts", "intermediate_size", None),
}
# Families that can mix dense and expert layers.
_MIXED_LAYER_FAMILIES = ("qwen2_moe", "qwen3_moe")

CHECKED_FAMILIES = (
    "llama",
    "mistral",
    "qwen2",
    "qwen3",
    "phi3",
    "olmo2",
    *TWO_MATRIX_FAMILIES,
    *MOE_LAYOUTS,
)

# Any of these suggests a mixture-of-experts model.
_MOE_SIGNALS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "n_shared_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "moe_num_experts",
    "moe_config",
    "ffn_config",
)


class ConfigError(Exception):
    """A config could not be found, read, or understood."""


@dataclass(frozen=True)
class MoE:
    experts: int  # per expert layer
    active: int  # experts used for each token
    width: int  # FFW width of one expert
    shared_width: int  # FFW width of the always-used shared expert; 0 if none
    sparse_layers: int  # layers that use experts; the rest are dense


@dataclass(frozen=True)
class ModelConfig:
    L: int
    D: int
    F: int
    N: int
    K: int
    H: int
    V: int
    tied: bool
    notes: tuple[str, ...] = ()  # assumptions to show the user
    family: str | None = None  # model_type
    ffw_matrices: int = 3
    moe: MoE | None = None
    positions: int = 0  # learned position embeddings (counted for GPT-2)
    tie_assumed: bool = False
    unsupported: str | None = None  # why parameters cannot be counted


def find_config(model: str) -> tuple[Path, str]:
    """Locate config.json for `model`; return (path, where it came from).

    Order: .json file, local directory, HuggingFace cache, Hub download.
    Only config.json is ever downloaded, never weights.
    """
    path = Path(model).expanduser()
    is_json = path.suffix.lower() == ".json"

    if path.is_file():
        if not is_json:
            raise ConfigError(f"{model} is a file but not a .json config")
        return path, "local file"
    if path.is_dir():
        candidate = path / CONFIG_NAME
        if not candidate.is_file():
            raise ConfigError(f"no {CONFIG_NAME} in local directory {path}")
        return candidate, "local directory"
    if is_json:
        raise ConfigError(f"config file not found: {model}")

    try:
        cached = try_to_load_from_cache(model, CONFIG_NAME)
        # A non-string result means "not cached" (None or a no-exist marker).
        if isinstance(cached, str):
            return Path(cached), "HF cache"
        return Path(hf_hub_download(model, CONFIG_NAME)), "downloaded from HF Hub"
    except HFValidationError:
        raise ConfigError(
            f"'{model}' is not an existing local path or a valid HuggingFace repo id"
        ) from None
    # GatedRepoError subclasses RepositoryNotFoundError, so it must come first.
    except GatedRepoError:
        raise ConfigError(
            f"'{model}' is gated and not in the local HF cache. Request access on its "
            "HuggingFace page and run `huggingface-cli login` (or `hf auth login`), "
            "or pass a local path."
        ) from None
    except RepositoryNotFoundError:
        raise ConfigError(
            f"'{model}' was not found as a local path, in the HF cache, or on the Hub. "
            "Check the spelling. If the repo is private, run `huggingface-cli login` "
            "(or `hf auth login`)."
        ) from None
    except RemoteEntryNotFoundError:
        raise ConfigError(f"Hub repo '{model}' has no {CONFIG_NAME}") from None
    except LocalEntryNotFoundError:
        raise ConfigError(
            f"'{model}' is not a local path or in the HF cache, and the Hub could not "
            "be reached. Check your connection (or HF_HUB_OFFLINE), or pass a local path."
        ) from None
    except OSError as e:
        raise ConfigError(f"could not fetch {CONFIG_NAME} for '{model}': {e}") from None


def read_config(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ConfigError(f"could not read {path}: {e}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} does not contain a JSON object")
    return raw


def _positive_int(cfg: dict, key: str) -> int:
    value = cfg.get(key)
    if value is None:
        raise ConfigError(f"config is missing required key '{key}'")
    if type(value) is not int or value < 1:
        raise ConfigError(f"config key '{key}' must be a positive integer, got {value!r}")
    return value


def dimension_source(raw: dict) -> dict:
    """The dict holding the model dimensions: top level, or a nested text_config."""
    nested = raw.get("text_config")
    if "num_hidden_layers" not in raw and isinstance(nested, dict):
        return nested
    return raw


def _unsupported_reason(cfg: dict, family: str | None) -> str | None:
    if cfg.get("kv_lora_rank") or cfg.get("q_lora_rank"):
        return "This model uses compressed (latent) attention, as in DeepSeek"
    if family in MOE_LAYOUTS:
        return None
    if any(cfg.get(key) for key in _MOE_SIGNALS) or "moe" in (family or ""):
        return "This is a mixture-of-experts layout that has not been checked"
    return None


def _parse_moe(cfg: dict, family: str, layers: int, notes: list[str]) -> MoE:
    count_key, width_key, shared_key = MOE_LAYOUTS[family]
    experts = _positive_int(cfg, count_key)
    active_key = "num_experts_per_tok"
    if cfg.get(active_key) is None and cfg.get("experts_per_token") is not None:
        active_key = "experts_per_token"
    active = _positive_int(cfg, active_key)
    if active > experts:
        raise ConfigError(f"config key '{active_key}' exceeds '{count_key}'")
    width = _positive_int(cfg, width_key)
    shared = 0
    if shared_key and cfg.get(shared_key) != 0:
        shared = _positive_int(cfg, shared_key)

    sparse = layers
    if family in _MIXED_LAYER_FAMILIES:
        step = 1
        if cfg.get("decoder_sparse_step") is None:
            notes.append("decoder_sparse_step missing, family default 1 used")
        else:
            step = _positive_int(cfg, "decoder_sparse_step")
        dense_only = cfg.get("mlp_only_layers") or []
        if not isinstance(dense_only, list):
            raise ConfigError("config key 'mlp_only_layers' must be a list")
        sparse = sum(
            1 for i in range(layers) if i not in dense_only and (i + 1) % step == 0
        )
    return MoE(experts, active, width, shared, sparse)


def parse_config(raw: dict) -> ModelConfig:
    """Map a raw config dict to the symbols used by the parameter formulas."""
    notes: list[str] = []
    cfg = dimension_source(raw)
    if cfg is not raw:
        notes.append("model dimensions read from nested text_config (text model only)")
    family = cfg.get("model_type") or raw.get("model_type")
    family = family if isinstance(family, str) else None
    gpt2 = family == "gpt2"  # uses its own key names

    L = _positive_int(cfg, "n_layer" if gpt2 else "num_hidden_layers")
    D = _positive_int(cfg, "n_embd" if gpt2 else "hidden_size")
    if not gpt2:
        F = _positive_int(cfg, "intermediate_size")
    elif cfg.get("n_inner") is None:
        F = 4 * D
        notes.append(f"GPT-2 {FAMILY_DEFAULT}: F = 4 * D = {F}")
    else:
        F = _positive_int(cfg, "n_inner")
    N = _positive_int(cfg, "n_head" if gpt2 else "num_attention_heads")
    V = _positive_int(cfg, "vocab_size")
    positions = _positive_int(cfg, "n_positions") if gpt2 else 0

    if cfg.get("num_key_value_heads") is not None:
        K = _positive_int(cfg, "num_key_value_heads")
    else:
        K = N
        if family in _FAMILY_NAMES:
            notes.append(f"{_FAMILY_NAMES[family]} {FAMILY_DEFAULT}: K = N = {N}")
        else:
            notes.append(f"num_key_value_heads missing, assumed K = N = {N}")

    if cfg.get("head_dim") is None:
        if D % N:
            raise ConfigError(
                f"config is missing 'head_dim' and hidden size {D} is not divisible "
                f"by the {N} attention heads"
            )
        H = D // N
        notes.append(f"head_dim missing, assumed H = D / N = {H}")
    else:
        H = _positive_int(cfg, "head_dim")

    tied = cfg.get("tie_word_embeddings", raw.get("tie_word_embeddings"))
    tie_assumed = False
    if tied is None and gpt2:
        tied = True
        notes.append(f"GPT-2 {FAMILY_DEFAULT}: shared input/output weights")
    elif tied is None:
        tied, tie_assumed = False, True
        notes.append("tie_word_embeddings missing, assumed false (2 vocab matrices)")

    unsupported = _unsupported_reason(cfg, family)
    moe = None
    if not unsupported and family in MOE_LAYOUTS:
        moe = _parse_moe(cfg, family, L, notes)
    if not unsupported and family not in CHECKED_FAMILIES:
        printable = family and family.isascii() and family.isprintable()
        named = f"'{family[:40]}'" if printable else "this config's"
        notes.append(
            f"{named} model type has not been checked against real weights; "
            "standard formulas used"
        )

    return ModelConfig(
        L,
        D,
        F,
        N,
        K,
        H,
        V,
        bool(tied),
        tuple(notes),
        family=family,
        ffw_matrices=2 if family in TWO_MATRIX_FAMILIES else 3,
        moe=moe,
        positions=positions,
        tie_assumed=tie_assumed,
        unsupported=unsupported,
    )
