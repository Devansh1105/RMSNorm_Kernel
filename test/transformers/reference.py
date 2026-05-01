"""Plain PyTorch RMSNorm references for correctness checks."""
from __future__ import annotations

import torch


def rms_norm_llama(x: torch.Tensor, w: torch.Tensor | None, eps: float, offset: float = 0.0) -> torch.Tensor:
    """Llama-style: rstd in fp32, multiply by W in input dtype.

    Mirrors HF's LlamaRMSNorm: ``(x * rsqrt(mean(x^2) + eps)).to(input_dtype) * (offset + w)``.
    """
    in_dtype = x.dtype
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(var + eps)
    out = x32.to(in_dtype)
    if w is not None:
        out = out * (offset + w)
    return out


def rms_norm_gemma(x: torch.Tensor, w: torch.Tensor | None, eps: float, offset: float = 1.0) -> torch.Tensor:
    """Gemma-style: full fp32 compute, cast back at the end."""
    in_dtype = x.dtype
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(var + eps)
    if w is not None:
        x32 = x32 * (offset + w.to(torch.float32))
    return x32.to(in_dtype)


def rms_norm_none(x: torch.Tensor, w: torch.Tensor | None, eps: float, offset: float = 0.0) -> torch.Tensor:
    """No-cast: everything in input dtype."""
    var = x.pow(2).mean(-1, keepdim=True)
    out = x * torch.rsqrt(var + eps)
    if w is not None:
        out = out * (offset + w)
    return out


REF = {"llama": rms_norm_llama, "gemma": rms_norm_gemma, "none": rms_norm_none}
