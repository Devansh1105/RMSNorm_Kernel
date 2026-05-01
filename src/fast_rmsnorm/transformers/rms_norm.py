"""nn.Module wrapper."""
from __future__ import annotations

import torch
import torch.nn as nn

from fast_rmsnorm.transformers.functional import rms_norm


class FastRMSNorm(nn.Module):
    """Drop-in RMSNorm.

    Matches HuggingFace ``LlamaRMSNorm`` / ``GemmaRMSNorm`` semantics depending
    on ``casting_mode`` and ``offset``.

    Set ``offset=1.0, casting_mode='gemma'`` to mimic ``GemmaRMSNorm`` (where
    weight is initialized to 0 and `(1 + weight)` is used).

    Module-level knobs that map to kernel args:
        mode:         'train' | 'infer' | 'auto'
        cache_rstd:   pass False alongside activation checkpointing
        in_place:     reuse dY storage for dX during backward

    The ``_gamma_folded`` flag is flipped True by
    ``fast_rmsnorm.transformers.folding.fold_rmsnorm_gamma_into_next_linear``
    once γ has been pre-multiplied into the next Linear's weight; this module
    then runs as a pure normalize (no weight, no offset).
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        offset: float = 0.0,
        casting_mode: str = "llama",
        in_place: bool = True,
        elementwise_affine: bool = True,
        mode: str = "auto",
        cache_rstd: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.offset = offset
        self.casting_mode = casting_mode
        self.in_place = in_place
        self.elementwise_affine = elementwise_affine
        self.mode = mode
        self.cache_rstd = cache_rstd
        self._gamma_folded = False

        if elementwise_affine:
            init = 0.0 if offset == 1.0 else 1.0  # Gemma init=0, Llama init=1
            self.weight = nn.Parameter(torch.full((hidden_size,), init))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._gamma_folded:
            return rms_norm(
                x, None, self.eps,
                offset=0.0, casting_mode=self.casting_mode, in_place=self.in_place,
                mode=self.mode, cache_rstd=self.cache_rstd,
            )
        return rms_norm(
            x, self.weight, self.eps,
            offset=self.offset, casting_mode=self.casting_mode, in_place=self.in_place,
            mode=self.mode, cache_rstd=self.cache_rstd,
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, eps={self.eps}, offset={self.offset}, "
            f"casting_mode={self.casting_mode!r}, mode={self.mode!r}, "
            f"cache_rstd={self.cache_rstd}, gamma_folded={self._gamma_folded}"
        )
