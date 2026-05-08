"""Plain PyTorch RMSNorm references."""
from __future__ import annotations


def rms_norm_llama(torch, x, weight, eps: float, offset: float = 0.0):
    input_dtype = x.dtype
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(-1, keepdim=True)
    out = x32 * torch.rsqrt(variance + eps)
    out = out.to(input_dtype)
    if weight is not None:
        out = out * (offset + weight)
    return out


def rms_norm_gemma(torch, x, weight, eps: float, offset: float = 1.0):
    input_dtype = x.dtype
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(-1, keepdim=True)
    out = x32 * torch.rsqrt(variance + eps)
    if weight is not None:
        out = out * (offset + weight.to(torch.float32))
    return out.to(input_dtype)


def rms_norm_none(torch, x, weight, eps: float, offset: float = 0.0):
    variance = x.pow(2).mean(-1, keepdim=True)
    out = x * torch.rsqrt(variance + eps)
    if weight is not None:
        out = out * (offset + weight)
    return out


REFERENCES = {
    "llama": ("PyTorch llama formula", rms_norm_llama),
    "gemma": ("PyTorch gemma formula", rms_norm_gemma),
    "none": ("PyTorch no-cast formula", rms_norm_none),
}
