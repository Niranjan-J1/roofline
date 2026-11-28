"""Parameter counts: FFW, attention (GQA-aware), vocab, and experts."""

from __future__ import annotations

from dataclasses import dataclass

from roofline.config import ModelConfig


@dataclass(frozen=True)
class ParamCounts:
    ffw: int  # includes every expert and any router
    attention: int
    vocab: int
    ffw_active: int  # FFW used for each token; equals ffw unless there are experts
    position: int = 0  # learned position embeddings

    @property
    def total(self) -> int:
        return self.ffw + self.attention + self.vocab + self.position

    @property
    def active_total(self) -> int:
        return self.total - self.ffw + self.ffw_active


def _ffw(c: ModelConfig) -> tuple[int, int]:
    """Return (all FFW parameters, FFW parameters active per token)."""
    # Three matrices when gated (SwiGLU: gate, up, down), otherwise two.
    dense_layer = c.ffw_matrices * c.D * c.F
    if c.moe is None:
        return dense_layer * c.L, dense_layer * c.L
    m = c.moe
    expert = 3 * c.D * m.width
    shared = 3 * c.D * m.shared_width + (c.D if m.shared_width else 0)  # + its gate
    router = c.D * m.experts
    dense = (c.L - m.sparse_layers) * dense_layer
    return (
        dense + m.sparse_layers * (m.experts * expert + shared + router),
        dense + m.sparse_layers * (m.active * expert + shared + router),
    )


def count_params(c: ModelConfig) -> ParamCounts:
    if c.unsupported:
        raise ValueError(c.unsupported)
    ffw, ffw_active = _ffw(c)
    return ParamCounts(
        ffw=ffw,
        # GQA-aware: Q and O scale with N, K and V with K. Not 4*D^2.
        attention=c.L * 2 * c.D * c.H * (c.N + c.K),
        vocab=(1 if c.tied else 2) * c.V * c.D,
        ffw_active=ffw_active,
        position=c.positions * c.D,
    )
