"""Functional API."""
from __future__ import annotations

import torch

from fast_rmsnorm.ops.rms_norm import FastRMSNormFunction


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float = 1e-6,
    *,
    offset: float = 0.0,
    casting_mode: str = "llama",
    in_place: bool = True,
    row_mode: bool | None = None,
    mode: str = "auto",
    cache_rstd: bool = True,
) -> torch.Tensor:
    """Compute RMSNorm.

    Args:
        x: input, shape (..., H)
        weight: scale, shape (H,) — or None for non-affine
        eps: epsilon inside the sqrt
        offset: bias added to weight (Gemma: 1.0, Llama: 0.0)
        casting_mode: 'llama' | 'gemma' | 'none' (Liger's three-mode design)
        in_place: reuse dY storage for dX in backward (saves memory)
        row_mode: force row-per-program kernel; None lets the dispatcher choose
        mode: 'train' uses autotuned kernels, 'infer' uses the heuristic ladder,
              'auto' picks by ``requires_grad``
        cache_rstd: if False, fwd skips rstd store and bwd recomputes (use with
                    activation checkpointing)
    """
    return FastRMSNormFunction.apply(
        x, weight, eps, offset, casting_mode, in_place, row_mode, mode, cache_rstd,
    )
