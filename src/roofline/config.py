"""Load a HuggingFace config and map it to L/D/F/N/K/H/V (bucket 2)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, try_to_load_from_cache
from huggingface_hub.errors import (
    GatedRepoError,
    HFValidationError,
    LocalEntryNotFoundError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
)

CONFIG_NAME = "config.json"

# Keys that signal a mixture-of-experts model.
_MOE_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")


class ConfigError(Exception):
    """A config could not be found, read, or understood."""


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
    notes: tuple[str, ...] = ()  # assumptions and warnings to show the user


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


def parse_config(raw: dict) -> ModelConfig:
    """Map a raw config dict to the symbols used by the parameter formulas."""
    notes: list[str] = []
    cfg = raw

    nested = raw.get("text_config")
    if "num_hidden_layers" not in raw and isinstance(nested, dict):
        cfg = nested
        notes.append("model dimensions read from nested text_config (text model only)")

    L = _positive_int(cfg, "num_hidden_layers")
    D = _positive_int(cfg, "hidden_size")
    F = _positive_int(cfg, "intermediate_size")
    N = _positive_int(cfg, "num_attention_heads")
    V = _positive_int(cfg, "vocab_size")

    if cfg.get("num_key_value_heads") is None:
        K = N
        notes.append(f"num_key_value_heads missing, assumed K = N = {N}")
    else:
        K = _positive_int(cfg, "num_key_value_heads")

    if cfg.get("head_dim") is None:
        if D % N:
            raise ConfigError(
                f"config is missing 'head_dim' and hidden_size {D} is not divisible "
                f"by num_attention_heads {N}"
            )
        H = D // N
        notes.append(f"head_dim missing, assumed H = D / N = {H}")
    else:
        H = _positive_int(cfg, "head_dim")

    tied = cfg.get("tie_word_embeddings", raw.get("tie_word_embeddings"))
    if tied is None:
        tied = False
        notes.append("tie_word_embeddings missing, assumed false (2 vocab matrices)")

    if any(cfg.get(key) for key in _MOE_KEYS):
        notes.append(
            "WARNING: this looks like a mixture-of-experts model; the dense FFW "
            "formula does not count experts, so these totals are not reliable"
        )

    return ModelConfig(L, D, F, N, K, H, V, bool(tied), tuple(notes))
