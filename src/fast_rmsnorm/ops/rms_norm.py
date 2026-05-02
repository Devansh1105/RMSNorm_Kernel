"""RMSNorm Triton kernels + dispatch + autograd Function (v1).

Built on Liger-Kernel's rms_norm.py (Apache 2.0). The math and dispatch
structure are Liger; the deltas are:

  E6  Pin eps to fp32 at kernel entry — defensive against AMD/HIP type
      promotion bugs in CASTING_NONE. No-op on NVIDIA.

  E1  Per-kernel @triton.autotune wrappers exposed as ``*_at`` variants.
      BLOCK_SIZE is forced to next_power_of_2(N) at launch (it must be ≥ N
      for the row-per-program reduction), so autotune only varies num_warps
      and num_stages. Key includes BLOCK_SIZE, DTYPE_ID so different
      shapes/dtypes get separately tuned configs. Heuristic-only entries are
      kept for inference, where cold-start cost matters more than the optimum.

  E2  REDUCE_STRATEGY: tl.constexpr branches the dgamma accumulation:
        - SCRATCH (Liger): write per-block fp32 partials to a [num_blocks, N]
          buffer; a separate sum() finishes the job.
        - ATOMIC: each block does tl.atomic_add into a single [N] fp32 buffer.
      Wrapper picks based on whether per-block working set fits in L2.

  I3  STORE_RSTD constexpr in fwd, RECOMPUTE_RSTD constexpr in bwd, plus
      eps argument carried into bwd. When STORE_RSTD=False / RECOMPUTE_RSTD
      =True, RSTD is not persisted — backward recomputes from X. Used with
      activation checkpointing.

Math (per row x in R^N):
    m    = (1/N) sum(x_i^2)
    s    = 1 / sqrt(m + eps)
    y_i  = (offset + gamma_i) * x_i * s

Backward (g = dL/dy):
    u    = g * (offset + gamma)
    c    = (1/N) sum(x_i * u_i)
    dx_i = s * u_i - s^3 * x_i * c
    dg_i = sum_rows g_i * x_i * s
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from fast_rmsnorm.ops.utils import (
    CASTING_GEMMA,
    CASTING_LLAMA,
    CASTING_NONE,
    REDUCE_STRATEGY_ATOMIC,
    REDUCE_STRATEGY_SCRATCH,
    calculate_settings,
    dtype_id,
    ensure_contiguous,
    pick_reduce_strategy,
    resolve_casting_mode,
    torch_to_triton_dtype,
)


# rsqrt import path varies across Triton versions and backends.
try:
    from triton.language.extra.libdevice import rsqrt  # triton >= 3.0
except ModuleNotFoundError:
    try:
        from triton.language.extra.cuda.libdevice import rsqrt  # NGC containers
    except ModuleNotFoundError:
        from triton.language.math import rsqrt  # older fallback


# DTensor support (tensor-parallel). The submodule needs an explicit import —
# `torch.distributed.tensor` is not auto-loaded by `import torch.distributed`.
# Some PyTorch builds also lack the submodule entirely; guard for both.
try:
    from torch.distributed.tensor import DTensor as _DTensor  # type: ignore
except (ImportError, AttributeError):
    _DTensor = None


def _is_dtensor(x):
    return _DTensor is not None and isinstance(x, _DTensor)


# ---------------------------------------------------------------------------
# Autotune configurations
# ---------------------------------------------------------------------------
# Forward kernel is memory-bound (1 X read + 1 W read + 1 Y write per element,
# ~3 flops compute). num_stages mostly affects load prefetching; we expect
# small wins. num_warps drives HBM saturation — needs enough in-flight loads
# to hide ~500-cycle DRAM latency. Six configs cover BLOCK_SIZE ∈ [256, 65536]
# across A100/H100/T4.

_FORWARD_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=1),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=16, num_stages=2),
    triton.Config({}, num_warps=16, num_stages=3),
    triton.Config({}, num_warps=32, num_stages=2),
]

# Backward has more loads (dY, X, RSTD, W) — more headroom for num_stages>=2.
_BACKWARD_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=3),
    triton.Config({}, num_warps=16, num_stages=2),
    triton.Config({}, num_warps=16, num_stages=3),
    triton.Config({}, num_warps=32, num_stages=2),
]

# Block-impl forward and backward additionally tune BLOCK_ROW (Liger hard-codes
# 16). Useful for QK-norm (small N, huge M) where Liger leaves perf on the table.
# Forward needs num_stages ∈ {1, 2} (less compute to hide loads behind);
# backward goes up to {2, 3} since it has more loads to overlap.
_BLOCK_FORWARD_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_ROW": br}, num_warps=nw, num_stages=ns)
    for br in (4, 16, 32, 64)
    for nw in (4, 8, 16)
    for ns in (1, 2)
]

_BLOCK_BACKWARD_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_ROW": br}, num_warps=nw, num_stages=ns)
    for br in (4, 16, 32, 64)
    for nw in (4, 8, 16)
    for ns in (2, 3)
]


# ---------------------------------------------------------------------------
# Forward kernels
# ---------------------------------------------------------------------------


@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr, Y_row_stride,
    X_ptr, X_row_stride,
    W_ptr, W_row_stride,
    RSTD_ptr, RSTD_row_stride,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    STORE_RSTD: tl.constexpr,
    DTYPE_ID: tl.constexpr,  # autotune key
    BLOCK_SIZE: tl.constexpr,
):
    """Row-per-program forward. One program processes one row of length n_cols.

    Tiles the entire row in one tile (BLOCK_SIZE >= n_cols), so the sum-of-squares
    is a single tl.sum across the BLOCK_SIZE-wide vector.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    y_base = Y_ptr + row_idx * Y_row_stride
    x_base = X_ptr + row_idx * X_row_stride

    X_row = tl.load(x_base + col_offsets, mask=mask, other=0)
    X_row_dtype = X_row.dtype
    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    # E6: pin eps/offset to fp32 at entry. No-op on NVIDIA where they already
    # specialize to fp32; defensive against HIP/AMD type-promotion edge cases.
    eps = eps.to(tl.float32)
    offset = offset.to(tl.float32)

    # Casting modes 
    if casting_mode == CASTING_LLAMA:
        # Llama: rstd computed in fp32; multiply by W done in input dtype.
        X_row = X_row.to(tl.float32)
    if casting_mode == CASTING_GEMMA:
        # Gemma: full fp32 compute, single cast to input dtype at end.
        if elementwise_affine:
            W_row = W_row.to(tl.float32)
        X_row = X_row.to(tl.float32)
    if casting_mode == CASTING_NONE:
        # NONE: everything stays in input dtype — eps/offset get demoted now.
        eps = eps.to(X_row_dtype)
        offset = offset.to(X_row_dtype)

    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)

    # I3: store rstd only when caller asked us to. When STORE_RSTD=False,
    # backward will recompute from X.
    if STORE_RSTD:
        rstd_base = RSTD_ptr + row_idx * RSTD_row_stride
        tl.store(rstd_base, rstd)

    X_row = X_row * rstd

    if casting_mode == CASTING_LLAMA:
        X_row = X_row.to(X_row_dtype)

    if elementwise_affine:
        Y_row = X_row * (offset + W_row)
    else:
        Y_row = X_row

    if casting_mode == CASTING_GEMMA:
        Y_row = Y_row.to(X_row_dtype)

    tl.store(y_base + col_offsets, Y_row, mask=mask)


@triton.jit
def _block_rms_norm_forward_kernel(
    Y_ptr, Y_row_stride,
    X_ptr, X_row_stride,
    W_ptr, W_row_stride,
    RSTD_ptr, RSTD_row_stride,
    n_rows, n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    STORE_RSTD: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    """Block forward: BLOCK_ROW rows per program. Used for small N + huge M
    (e.g. Qwen3 QK-norm: N=128, M=131072) where the row-per-program kernel
    pays too much dispatch overhead per actual work item."""
    row_idx = tl.program_id(0) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = row_idx < n_rows
    col_mask = col_offsets < n_cols

    X_row = tl.load(
        X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0,
    )
    X_row_dtype = X_row.dtype
    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0)

    # E6
    eps = eps.to(tl.float32)
    offset = offset.to(tl.float32)

    if casting_mode == CASTING_LLAMA:
        X_row = X_row.to(tl.float32)
    if casting_mode == CASTING_GEMMA:
        if elementwise_affine:
            W_row = W_row.to(tl.float32)
        X_row = X_row.to(tl.float32)
    if casting_mode == CASTING_NONE:
        eps = eps.to(X_row_dtype)
        offset = offset.to(X_row_dtype)

    mean_square = tl.sum(X_row * X_row, axis=1) / n_cols
    rstd = rsqrt(mean_square + eps)

    if STORE_RSTD:
        tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd, row_mask)

    X_row = X_row * rstd[:, None]

    if casting_mode == CASTING_LLAMA:
        X_row = X_row.to(X_row_dtype)

    if elementwise_affine:
        Y_row = X_row * (offset + W_row)[None, :]
    else:
        Y_row = X_row

    if casting_mode == CASTING_GEMMA:
        Y_row = Y_row.to(X_row_dtype)

    tl.store(
        Y_ptr + row_idx[:, None] * Y_row_stride + col_offsets[None, :],
        Y_row,
        mask=row_mask[:, None] & col_mask[None, :],
    )


# ---------------------------------------------------------------------------
# Backward kernels
# ---------------------------------------------------------------------------


@triton.jit
def _rms_norm_backward_kernel(
    dY_ptr, dY_row_stride,
    dX_ptr, dX_row_stride,
    X_ptr, X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr, W_row_stride,
    RSTD_ptr, RSTD_row_stride,
    dW_ptr, dW_row_stride,
    n_rows, n_cols,
    eps,                             # I3: needed when RECOMPUTE_RSTD=True
    offset,
    rows_per_program,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    REDUCE_STRATEGY: tl.constexpr,   # E2: ATOMIC vs SCRATCH for dgamma
    RECOMPUTE_RSTD: tl.constexpr,    # I3: recompute rstd from X if True
    DTYPE_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-per-program backward. Each program owns ``rows_per_program``
    consecutive rows; computes per-row dx and accumulates per-block dW partials.

    dx formula:
        u    = dy * (offset + γ)        (in fp32 for Llama/Gemma modes)
        c    = (1/N) Σ x_i * u_i
        dx_i = rstd * (u_i - rstd^2 * x_i * c)

    dgamma per row contribution: dy_i * (x_i * rstd). Summed across rows here,
    then either written to scratch (axis=0 sum across blocks happens in the
    Python wrapper) or atomic_add'd directly into the global dW vector.
    """
    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # E6
    eps = eps.to(tl.float32)

    if elementwise_affine:
        # Per-block dW partial. Stays fp32 throughout.
        dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
        # Bake the offset into W_row once — saves an add per inner iteration.
        W_row = W_row + offset

    for row_idx in range(row_start, row_end):
        dy_base = dY_ptr + row_idx * dY_row_stride
        dx_base = dX_ptr + row_idx * dX_row_stride
        x_base = X_ptr + row_idx * X_row_stride

        dY_row = tl.load(dy_base + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(x_base + col_offsets, mask=mask, other=0.0)

        # I3: load cached rstd or recompute it on the fly.
        if RECOMPUTE_RSTD:
            X_for_rstd = X_row.to(tl.float32)
            mean_square = tl.sum(X_for_rstd * X_for_rstd, axis=0) / n_cols
            rstd_row = rsqrt(mean_square + eps)
        else:
            rstd_base = RSTD_ptr + row_idx * RSTD_row_stride
            rstd_row = tl.load(rstd_base)

        X_row = X_row.to(tl.float32)

        # m = u in the docstring math: dy * (offset + γ).
        # Casting branches differ only in *when* we promote dY to fp32 — same
        # algebra otherwise. Llama promotes after the dY*W multiply; Gemma
        # promotes dY first; NONE leaves dY in input dtype.
        if casting_mode == CASTING_LLAMA:
            if elementwise_affine:
                m = (dY_row * W_row).to(tl.float32)
            else:
                m = dY_row.to(tl.float32)
        elif casting_mode == CASTING_GEMMA:
            dY_row = dY_row.to(tl.float32)
            if elementwise_affine:
                m = dY_row * W_row
            else:
                m = dY_row
        else:  # CASTING_NONE
            if elementwise_affine:
                m = dY_row * W_row
            else:
                m = dY_row

        # dx = rstd * m - rstd^3 * (1/N) * sum(m*x) * x
        dX_row = rstd_row * m
        dX_row += rstd_row * (
            -(1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0) * X_row
        )

        if elementwise_affine:
            # dgamma row contribution. For LLAMA we cast (X*rstd) back to input
            # dtype before multiplying by dY (already in input dtype) so the
            # multiply matches what fwd's output would have been.
            if casting_mode == CASTING_LLAMA:
                dW_row += dY_row * (X_row * rstd_row).to(X_dtype)
            else:
                dW_row += dY_row * (X_row * rstd_row)

        tl.store(dx_base + col_offsets, dX_row.to(X_dtype), mask=mask)

    # E2: write the per-block dgamma partial.
    if elementwise_affine:
        if REDUCE_STRATEGY == REDUCE_STRATEGY_SCRATCH:
            # Scratch: dW_ptr is [num_blocks, N]; each block writes its slot.
            # Python wrapper finishes with .sum(dim=0).
            tl.store(dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row, mask=mask)
        else:  # REDUCE_STRATEGY_ATOMIC
            # Atomic: dW_ptr is just [N] (zero-initialized); blocks atomic_add.
            # Wins when num_blocks * N * 4 bytes fits in L2 — atomics coalesce
            # in cache and we save the second-stage reduction launch.
            tl.atomic_add(dW_ptr + col_offsets, dW_row, mask=mask)


@triton.jit
def _block_rms_norm_backward_kernel(
    dY_ptr, dY_row_stride,
    dX_ptr, dX_row_stride,
    X_ptr, X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr, W_row_stride,
    RSTD_ptr, RSTD_row_stride,
    dW_ptr, dW_row_stride,
    n_rows, n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    REDUCE_STRATEGY: tl.constexpr,
    RECOMPUTE_RSTD: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    """Block backward: persistent grid (one program per SM), each program loops
    over rows in BLOCK_ROW chunks. Used for small N + huge M shapes."""
    pid = tl.program_id(0).cast(tl.int64)
    NUM_SMS = tl.num_programs(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols

    eps = eps.to(tl.float32)

    if elementwise_affine:
        dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0)
        W_row = W_row + offset

    for start in range(pid * BLOCK_ROW, n_rows, NUM_SMS * BLOCK_ROW):
        row_idx = start + tl.arange(0, BLOCK_ROW)
        row_mask = row_idx < n_rows
        dY_row = tl.load(
            dY_ptr + row_idx[:, None] * dY_row_stride + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        X_row = tl.load(
            X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )

        if RECOMPUTE_RSTD:
            X_for_rstd = X_row.to(tl.float32)
            mean_square = tl.sum(X_for_rstd * X_for_rstd, axis=1) / n_cols
            rstd_row = rsqrt(mean_square + eps)
        else:
            rstd_row = tl.load(RSTD_ptr + row_idx * RSTD_row_stride, row_mask)

        X_row = X_row.to(tl.float32)

        if casting_mode == CASTING_LLAMA:
            if elementwise_affine:
                m = (dY_row * W_row[None, :]).to(tl.float32)
            else:
                m = dY_row.to(tl.float32)
        elif casting_mode == CASTING_GEMMA:
            dY_row = dY_row.to(tl.float32)
            if elementwise_affine:
                m = dY_row * W_row[None, :]
            else:
                m = dY_row
        else:
            if elementwise_affine:
                m = dY_row * W_row[None, :]
            else:
                m = dY_row

        dX_row = rstd_row[:, None] * m
        dX_row += (rstd_row[:, None]) * (
            -(1 / n_cols) * (rstd_row * rstd_row * tl.sum(m * X_row, axis=1))[:, None] * X_row
        )

        if elementwise_affine:
            if casting_mode == CASTING_LLAMA:
                dW_row += tl.sum((dY_row * (X_row * rstd_row[:, None]).to(X_dtype)).to(tl.float32), 0)
            else:
                dW_row += tl.sum(dY_row * (X_row * rstd_row[:, None]), 0)

        tl.store(
            dX_ptr + row_idx[:, None] * dX_row_stride + col_offsets[None, :],
            dX_row,
            mask=row_mask[:, None] & col_mask[None, :],
        )

    if elementwise_affine:
        if REDUCE_STRATEGY == REDUCE_STRATEGY_SCRATCH:
            tl.store(dW_ptr + pid * dW_row_stride + col_offsets, dW_row, mask=col_mask)
        else:  # ATOMIC
            tl.atomic_add(dW_ptr + col_offsets, dW_row, mask=col_mask)


# ---------------------------------------------------------------------------
# Autotune-wrapped variants
# ---------------------------------------------------------------------------
# Same kernel body, wrapped at runtime so we can ship both an autotuned variant
# (for training, where cold-start is amortized) and a heuristic variant (for
# inference, where dispatch latency matters more than the perf optimum).

_rms_norm_forward_kernel_at = triton.autotune(
    configs=_FORWARD_AUTOTUNE_CONFIGS,
    key=["BLOCK_SIZE", "DTYPE_ID"],
)(_rms_norm_forward_kernel)

_block_rms_norm_forward_kernel_at = triton.autotune(
    configs=_BLOCK_FORWARD_AUTOTUNE_CONFIGS,
    key=["BLOCK_SIZE", "DTYPE_ID"],
)(_block_rms_norm_forward_kernel)

_rms_norm_backward_kernel_at = triton.autotune(
    configs=_BACKWARD_AUTOTUNE_CONFIGS,
    key=["BLOCK_SIZE", "DTYPE_ID"],
)(_rms_norm_backward_kernel)

# Block backward gets BLOCK_ROW autotuned too — this is where QK-norm wins live.
_block_rms_norm_backward_kernel_at = triton.autotune(
    configs=_BLOCK_BACKWARD_AUTOTUNE_CONFIGS,
    key=["BLOCK_SIZE", "DTYPE_ID"],
)(_block_rms_norm_backward_kernel)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _use_block_impl(BLOCK_SIZE: int, n_rows: int, row_mode: bool | None) -> bool:
    """Liger's dispatch heuristic — block-impl for small N + many rows.

    Block-impl wins when one row of work is too cheap to justify a full
    program launch (small N) and there are enough rows to fill the SMs (large
    M). The threshold (BLOCK_SIZE<=256, M>=32K) is Liger's; we leave it.
    """
    if row_mode:
        return False
    return not (BLOCK_SIZE > 256 or n_rows < 4096 * 8)


def _resolve_mode(mode: str, requires_grad: bool) -> str:
    """auto -> infer if no grad needed (typical inference), else train."""
    if mode == "auto":
        return "train" if requires_grad else "infer"
    if mode in ("train", "infer"):
        return mode
    raise ValueError(f"mode must be one of 'train'|'infer'|'auto'; got {mode!r}")


def rms_norm_forward(X, W, eps, offset, casting_mode, *, mode="auto", row_mode=None, cache_rstd=True):
    casting_mode = resolve_casting_mode(casting_mode)
    eps = float(eps)  # Triton scalar — float keeps inference of fp32 stable

    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps_heuristic = calculate_settings(n_cols)

    rg = X.requires_grad or (W is not None and W.requires_grad)
    resolved_mode = _resolve_mode(mode, rg)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)

    # I3: when not caching rstd, allocate a 1-element placeholder so the kernel
    # arg list is unchanged. The kernel branches on STORE_RSTD constexpr and
    # never touches the placeholder when STORE_RSTD=False.
    if cache_rstd:
        rstd_dtype = torch.float32 if casting_mode in (CASTING_LLAMA.value, CASTING_GEMMA.value) else X.dtype
        RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)
        rstd_stride = RSTD.stride(0)
    else:
        RSTD = torch.empty(1, dtype=torch.float32, device=X.device)
        rstd_stride = 0

    elementwise_affine = W is not None
    if elementwise_affine:
        assert X.shape[1] == W.shape[0], "Hidden-dim mismatch between X and W"

    use_block = _use_block_impl(BLOCK_SIZE, n_rows, row_mode)
    dtype_key = dtype_id(X.dtype)

    if not use_block:
        kernel = _rms_norm_forward_kernel_at if resolved_mode == "train" else _rms_norm_forward_kernel
        launch_kwargs = dict(
            elementwise_affine=elementwise_affine,
            STORE_RSTD=cache_rstd,
            DTYPE_ID=dtype_key,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        if resolved_mode == "infer":
            launch_kwargs["num_warps"] = num_warps_heuristic
        kernel[(n_rows,)](
            Y, Y.stride(0),
            X, X.stride(0),
            W, W.stride(0) if elementwise_affine else 0,
            RSTD, rstd_stride,
            n_cols, eps, offset, casting_mode,
            **launch_kwargs,
        )
    else:
        # Block-impl forward. In train mode, BLOCK_ROW is autotuned and the grid
        # depends on the chosen BLOCK_ROW — pass a callable grid that reads
        # BLOCK_ROW from the autotune meta. In infer mode, BLOCK_ROW=16 (Liger's
        # heuristic) is fixed and the grid is a tuple.
        if resolved_mode == "train":
            _block_rms_norm_forward_kernel_at[
                lambda meta: (triton.cdiv(n_rows, meta["BLOCK_ROW"]),)
            ](
                Y, Y.stride(0),
                X, X.stride(0),
                W, W.stride(0) if elementwise_affine else 0,
                RSTD, rstd_stride,
                n_rows, n_cols, eps, offset, casting_mode,
                elementwise_affine=elementwise_affine,
                STORE_RSTD=cache_rstd,
                DTYPE_ID=dtype_key,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:  # infer
            BLOCK_ROW = 16
            _block_rms_norm_forward_kernel[(triton.cdiv(n_rows, BLOCK_ROW),)](
                Y, Y.stride(0),
                X, X.stride(0),
                W, W.stride(0) if elementwise_affine else 0,
                RSTD, rstd_stride,
                n_rows, n_cols, eps, offset, casting_mode,
                elementwise_affine=elementwise_affine,
                STORE_RSTD=cache_rstd,
                DTYPE_ID=dtype_key,
                BLOCK_SIZE=BLOCK_SIZE,
                BLOCK_ROW=BLOCK_ROW,
                num_warps=num_warps_heuristic,
            )

    return Y.view(*shape), X, RSTD, BLOCK_SIZE, num_warps_heuristic, casting_mode, resolved_mode


def rms_norm_backward(
    dY, X, W, RSTD, eps, offset, casting_mode, BLOCK_SIZE, num_warps, in_place,
    *, mode="train", row_mode=None, cache_rstd=True,
):
    eps = float(eps)
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    sm_count = 1
    if X.device.type == "cuda":
        sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count

    if n_cols > BLOCK_SIZE:
        raise RuntimeError("Feature dim exceeds BLOCK_SIZE; streaming fallback is v1.5 work.")

    elementwise_affine = W is not None

    # E2: pick reduction strategy. ATOMIC for small working set (fits in L2),
    # SCRATCH for large (would spill). The strategy is a constexpr in the
    # kernel — both paths compile, so this is a runtime data choice only.
    if elementwise_affine:
        reduce_strategy = pick_reduce_strategy(sm_count, n_cols, X.device)
        if reduce_strategy == REDUCE_STRATEGY_ATOMIC.value:
            _dW = torch.zeros(n_cols, dtype=torch.float32, device=W.device)
            dW_row_stride = 0
        else:
            _dW = torch.empty((sm_count, n_cols), dtype=torch.float32, device=W.device)
            dW_row_stride = _dW.stride(0)
    else:
        _dW = None
        dW_row_stride = 0
        reduce_strategy = REDUCE_STRATEGY_SCRATCH.value  # unused

    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count,)
    dX = dY if in_place else torch.zeros_like(dY)
    use_block = _use_block_impl(BLOCK_SIZE, n_rows, row_mode)
    dtype_key = dtype_id(X.dtype)

    if not use_block:
        kernel = _rms_norm_backward_kernel_at if mode == "train" else _rms_norm_backward_kernel
        launch_kwargs = dict(
            elementwise_affine=elementwise_affine,
            REDUCE_STRATEGY=reduce_strategy,
            RECOMPUTE_RSTD=not cache_rstd,
            DTYPE_ID=dtype_key,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        if mode == "infer":
            launch_kwargs["num_warps"] = num_warps
        kernel[grid](
            dY, dY.stride(0),
            dX, dX.stride(0),
            X, X.stride(0),
            torch_to_triton_dtype[X.dtype],
            W, W.stride(0) if elementwise_affine else 0,
            RSTD, RSTD.stride(0) if cache_rstd else 0,
            _dW, dW_row_stride,
            n_rows, n_cols,
            eps, offset, rows_per_program, casting_mode,
            **launch_kwargs,
        )
    else:
        kernel = _block_rms_norm_backward_kernel_at if mode == "train" else _block_rms_norm_backward_kernel
        launch_kwargs = dict(
            elementwise_affine=elementwise_affine,
            REDUCE_STRATEGY=reduce_strategy,
            RECOMPUTE_RSTD=not cache_rstd,
            DTYPE_ID=dtype_key,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        if mode == "infer":
            launch_kwargs["num_warps"] = num_warps
            launch_kwargs["BLOCK_ROW"] = 16
        kernel[grid](
            dY, dY.stride(0),
            dX, dX.stride(0),
            X, X.stride(0),
            torch_to_triton_dtype[X.dtype],
            W, W.stride(0) if elementwise_affine else 0,
            RSTD, RSTD.stride(0) if cache_rstd else 0,
            _dW, dW_row_stride,
            n_rows, n_cols,
            eps, offset, casting_mode,
            **launch_kwargs,
        )

    dX = dX.view(*shape)

    if elementwise_affine:
        if reduce_strategy == REDUCE_STRATEGY_ATOMIC.value:
            dW = _dW.to(W.dtype)
        else:
            dW = _dW.sum(dim=0).to(W.dtype)
    else:
        dW = None
    return dX, dW


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------


class FastRMSNormFunction(torch.autograd.Function):
    """Autograd RMSNorm with v1 deltas exposed via kwargs.

    Args (forward):
        X: input, shape (..., H)
        W: weight, shape (H,) or None for non-affine
        eps: epsilon scalar
        offset: added to W before multiply (Gemma uses 1.0)
        casting_mode: 'llama' | 'gemma' | 'none'
        in_place: reuse dY storage for dX in backward
        row_mode: force row-per-program path; None lets dispatcher choose
        mode: 'train' | 'infer' | 'auto' (auto picks by requires_grad)
        cache_rstd: if False, fwd skips rstd store and bwd recomputes from X
    """

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx, X, W, eps, offset=0.0, casting_mode="llama", in_place=True,
        row_mode=None, mode="auto", cache_rstd=True,
    ):
        if _is_dtensor(X):
            X = X.full_tensor()

        Y, X_flat, RSTD, BLOCK_SIZE, num_warps, casting_mode_int, resolved_mode = rms_norm_forward(
            X, W, eps, offset, casting_mode, mode=mode, row_mode=row_mode, cache_rstd=cache_rstd,
        )
        ctx.eps = float(eps)
        ctx.offset = offset
        ctx.casting_mode = casting_mode_int
        ctx.in_place = in_place
        ctx.row_mode = row_mode
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.elementwise_affine = W is not None
        ctx.mode = resolved_mode
        ctx.cache_rstd = cache_rstd
        if W is not None:
            ctx.save_for_backward(X_flat, W, RSTD)
        else:
            ctx.save_for_backward(X_flat, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        if ctx.elementwise_affine:
            X, W, RSTD = ctx.saved_tensors
        else:
            X, RSTD = ctx.saved_tensors
            W = None

        if _is_dtensor(dY):
            dY = dY.full_tensor()

        # Backward never benefits from 'auto' — if we're here, requires_grad was True.
        bwd_mode = "train" if ctx.mode == "train" else "infer"
        dX, dW = rms_norm_backward(
            dY, X, W, RSTD, ctx.eps, ctx.offset, ctx.casting_mode,
            ctx.BLOCK_SIZE, ctx.num_warps, ctx.in_place,
            mode=bwd_mode, row_mode=ctx.row_mode, cache_rstd=ctx.cache_rstd,
        )
        return dX, dW, None, None, None, None, None, None, None
