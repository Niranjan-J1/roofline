"""Memory scenarios: inference, LoRA finetune, full training.

All results are idealized estimates in bytes. Framework overhead, temporary
buffers and fragmentation are not counted, so real usage is higher.
"""

from __future__ import annotations

from dataclasses import dataclass

from roofline.config import ModelConfig
from roofline.params import count_params

GB = 1e9

WEIGHT_BYTES = 2  # bf16
# 2 bf16 weight + 2 bf16 gradient + 4 fp32 Adam m + 4 fp32 Adam v
TRAINED_PARAM_BYTES = 12
ACTIVATION_BYTES = 2  # bf16
KV_BYTES = 2  # bf16

ASSUMPTIONS = (
    "Idealized estimates: framework overhead and temporary buffers are not counted.",
    "Activations are a rough, unmeasured estimate: 2*B*T*L*(5D+3F) bytes, with no "
    "gradient checkpointing and no stored attention matrix.",
    "Trained parameters use 12 bytes each (weight, gradient, two Adam moments).",
    "LoRA adapters are on the four attention projections (q, k, v, o) only.",
)


@dataclass(frozen=True)
class Component:
    name: str
    bytes: int


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    components: tuple[Component, ...]

    @property
    def total(self) -> int:
        return sum(c.bytes for c in self.components)

    @property
    def dominant(self) -> Component:
        return max(self.components, key=lambda c: c.bytes)


def activation_bytes(c: ModelConfig, batch: int, seq: int) -> int:
    return ACTIVATION_BYTES * batch * seq * c.L * (5 * c.D + 3 * c.F)


def kv_cache_bytes(c: ModelConfig, batch: int, seq: int) -> int:
    # Keys and values: one K*H vector each, per layer, per token.
    return 2 * KV_BYTES * c.L * c.K * c.H * batch * seq


def lora_params(c: ModelConfig, rank: int) -> int:
    # Each adapter adds rank * (input width + output width):
    #   q: D + N*H,  k: D + K*H,  v: D + K*H,  o: N*H + D
    return rank * c.L * (4 * c.D + 2 * c.H * (c.N + c.K))


def memory_scenarios(
    c: ModelConfig, batch: int, seq: int, lora_rank: int
) -> tuple[Scenario, ...]:
    total_params = count_params(c).total
    weights = WEIGHT_BYTES * total_params
    activations = Component("activations", activation_bytes(c, batch, seq))

    inference = Scenario(
        "inference",
        "Inference (bf16)",
        (
            Component("weights", weights),
            Component("KV cache", kv_cache_bytes(c, batch, seq)),
        ),
    )
    lora = Scenario(
        "lora",
        "LoRA finetune",
        (
            Component("frozen weights", weights),
            Component(
                "adapters + optimizer",
                TRAINED_PARAM_BYTES * lora_params(c, lora_rank),
            ),
            activations,
        ),
    )
    full = Scenario(
        "full_training",
        "Full training (Adam)",
        (
            Component("weights", weights),
            Component(
                "gradients + optimizer",
                (TRAINED_PARAM_BYTES - WEIGHT_BYTES) * total_params,
            ),
            activations,
        ),
    )
    return inference, lora, full
