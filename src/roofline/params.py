"""Parameter counts: FFW, attention (GQA-aware), vocab (bucket 2)."""

from __future__ import annotations

from dataclasses import dataclass

from roofline.config import ModelConfig


@dataclass(frozen=True)
class ParamCounts:
    ffw: int
    attention: int
    vocab: int

    @property
    def total(self) -> int:
        return self.ffw + self.attention + self.vocab


def count_params(c: ModelConfig) -> ParamCounts:
    return ParamCounts(
        ffw=3 * c.D * c.F * c.L,  # SwiGLU: gate, up, down
        # GQA-aware: Q and O scale with N, K and V with K. Not 4*D^2.
        attention=c.L * 2 * c.D * c.H * (c.N + c.K),
        vocab=(1 if c.tied else 2) * c.V * c.D,
    )
