"""Internal utilities: heuristics, dtype mapping, contiguous helper, eps strategy.

Lifted from Liger-Kernel/Unsloth (Apache 2.0). Trimmed to what we use.
"""
from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl


def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe(a) for a in args]
        kwargs = {k: maybe(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


def calculate_settings(n: int):
    """Liger-style heuristic. Used in ``mode='infer'`` to skip autotune cold-start."""
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"feature dim n={n} exceeds MAX_FUSED_SIZE={MAX_FUSED_SIZE} — streaming fallback not yet implemented (v1.5)."
        )
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 16 if torch.version.hip is not None else 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


torch_to_triton_dtype = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


CASTING_NONE: tl.constexpr = tl.constexpr(-1)
CASTING_LLAMA: tl.constexpr = tl.constexpr(0)
CASTING_GEMMA: tl.constexpr = tl.constexpr(1)

_str_to_casting_mode = {
    "llama": CASTING_LLAMA.value,
    "gemma": CASTING_GEMMA.value,
    "none": CASTING_NONE.value,
}


def resolve_casting_mode(casting_mode):
    if isinstance(casting_mode, int):
        assert casting_mode in _str_to_casting_mode.values(), f"Invalid casting mode: {casting_mode}"
        return casting_mode
    assert casting_mode in _str_to_casting_mode, f"Invalid casting mode: {casting_mode}"
    return _str_to_casting_mode[casting_mode]


# E2: dgamma reduction strategy.
REDUCE_STRATEGY_SCRATCH: tl.constexpr = tl.constexpr(0)  # Liger-style 2-stage scratch buffer
REDUCE_STRATEGY_ATOMIC: tl.constexpr = tl.constexpr(1)   # tl.atomic_add into a single dW buffer


def dtype_id(dtype: torch.dtype) -> int:
    """Stable int per dtype — used in autotune key so configs differ by dtype."""
    return {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[dtype]


def pick_reduce_strategy(sm_count: int, n_cols: int, device: torch.device) -> int:
    """E2 heuristic: ATOMIC when per-block fp32 partials fit in L2, else SCRATCH.

    Working-set proxy = sm_count * n_cols * 4 bytes.
    L2 sizes: ~50 MB H100, ~40 MB A100, ~6 MB T4. Read at runtime.
    The 0.5x leaves room for X, dY, RSTD in the same cache.
    """
    if device.type != "cuda":
        return REDUCE_STRATEGY_SCRATCH.value
    props = torch.cuda.get_device_properties(device)
    # PyTorch renamed l2_cache_size -> L2_cache_size at some point. Try both.
    l2 = getattr(props, "L2_cache_size", None) or getattr(props, "l2_cache_size", None)
    if l2 is None:
        return REDUCE_STRATEGY_SCRATCH.value  # safe default if we can't introspect
    working_set = sm_count * n_cols * 4
    return REDUCE_STRATEGY_ATOMIC.value if working_set < 0.5 * l2 else REDUCE_STRATEGY_SCRATCH.value
