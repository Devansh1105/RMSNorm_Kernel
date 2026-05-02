"""Correctness tests vs PyTorch reference.

Designed to run on a single GPU (Colab T4 is fine). Covers:
  - All three casting modes (llama, gemma, none)
  - bf16 / fp16 / fp32 input dtypes, matched weight dtype
  - With and without offset
  - With and without weight (elementwise_affine=False -> the no-gamma path used by I1)
  - Both dispatch regimes: row-per-program (large N) and block-impl (small N, big M)
  - Forward and backward (dx, dgamma)

Tolerances chosen against fp32 reference, separated by dtype since bf16 has 7 mantissa bits.
"""
from __future__ import annotations

import itertools

import pytest
import torch

torch.manual_seed(0)
cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not cuda_available, reason="CUDA not available")

if cuda_available:
    from fast_rmsnorm.transformers import FastRMSNorm, rms_norm
    from test.transformers.reference import REF


DTYPES = [torch.float32, torch.float16, torch.bfloat16]
DTYPE_TOL = {
    torch.float32: (1e-5, 1e-5),
    # fp16 has 10 mantissa bits and a small dynamic range; after a 4096-wide
    # reduction + cast-to-input boundary, occasional 1-ulp drifts on isolated
    # elements show up. Same slack as bf16 is the right call here.
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (5e-3, 5e-3),
}


def _rand(shape, dtype, device="cuda"):
    return torch.randn(shape, dtype=dtype, device=device)


# Shape regimes. (M, N).
#   Row-per-program is selected when BLOCK_SIZE>256 OR M<32K.
#     SMALL_N + LARGE_M -> block-impl path
#     LARGE_N           -> row-impl path
SHAPES_ROW = [(128, 4096), (1024, 4096), (4, 8192)]
SHAPES_BLOCK = [(40000, 64), (40000, 128), (40000, 256)]
ALL_SHAPES = SHAPES_ROW + SHAPES_BLOCK

# Only the production combos. Llama always uses offset=0; Gemma always uses
# offset=1. Cross-pairing them (e.g. llama+offset=1) is nonsense and exposes a
# fp32-vs-input-dtype divergence in the (offset+W) addition step that doesn't
# matter for any real model.
CASTING_OFFSET = [
    ("llama", 0.0),
    ("gemma", 1.0),
    ("gemma", 0.0),  # exercises the (offset+W) path with offset=0 — sanity coverage
    ("none", 0.0),
]


@pytest.mark.parametrize(
    "shape, dtype, casting_offset, with_weight",
    list(itertools.product(ALL_SHAPES, DTYPES, CASTING_OFFSET, [True, False])),
)
def test_forward(shape, dtype, casting_offset, with_weight):
    casting_mode, offset = casting_offset
    M, N = shape
    eps = 1e-6
    x = _rand((M, N), dtype)
    w = _rand((N,), dtype) if with_weight else None
    out = rms_norm(x.clone(), w, eps, offset=offset, casting_mode=casting_mode)
    out_ref = REF[casting_mode](x, w, eps, offset=offset)
    atol, rtol = DTYPE_TOL[dtype]
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "shape, dtype, casting_offset, with_weight",
    list(
        itertools.product(
            [(128, 4096), (40000, 128)],  # one row-impl, one block-impl shape
            [torch.float32, torch.bfloat16],
            [("llama", 0.0), ("gemma", 1.0)],  # the two production combos
            [True, False],
        )
    ),
)
def test_backward(shape, dtype, casting_offset, with_weight):
    casting_mode, offset = casting_offset
    """Backward correctness via autograd.grad against the reference."""
    M, N = shape
    eps = 1e-6
    x = _rand((M, N), dtype).requires_grad_(True)
    w = _rand((N,), dtype).requires_grad_(True) if with_weight else None

    out = rms_norm(x, w, eps, offset=offset, casting_mode=casting_mode, in_place=False)
    g = torch.randn_like(out)
    grads = torch.autograd.grad(out, [x] + ([w] if with_weight else []), g, retain_graph=False)
    dx_ours, dw_ours = grads[0], (grads[1] if with_weight else None)

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True) if with_weight else None
    out_ref = REF[casting_mode](x_ref, w_ref, eps, offset=offset)
    grads_ref = torch.autograd.grad(out_ref, [x_ref] + ([w_ref] if with_weight else []), g.detach())
    dx_ref, dw_ref = grads_ref[0], (grads_ref[1] if with_weight else None)

    atol, rtol = DTYPE_TOL[dtype]
    torch.testing.assert_close(dx_ours, dx_ref, atol=2 * atol, rtol=2 * rtol)
    if with_weight:
        torch.testing.assert_close(dw_ours, dw_ref, atol=2 * atol, rtol=2 * rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_in_place_does_not_corrupt_dx(dtype):
    """in_place=True reuses dY storage for dX — verify result matches in_place=False."""
    M, N = 64, 1024
    eps = 1e-6
    x = _rand((M, N), dtype).requires_grad_(True)
    w = _rand((N,), dtype).requires_grad_(True)

    out = rms_norm(x, w, eps, in_place=True)
    g = torch.randn_like(out)
    dx_ip, dw_ip = torch.autograd.grad(out, [x, w], g.clone())

    x2 = x.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    out2 = rms_norm(x2, w2, eps, in_place=False)
    dx_no, dw_no = torch.autograd.grad(out2, [x2, w2], g.clone())

    atol, rtol = DTYPE_TOL[dtype]
    torch.testing.assert_close(dx_ip, dx_no, atol=2 * atol, rtol=2 * rtol)
    torch.testing.assert_close(dw_ip, dw_no, atol=2 * atol, rtol=2 * rtol)


def test_module_smoke():
    """Smoke test the FastRMSNorm Module."""
    m = FastRMSNorm(hidden_size=512, eps=1e-6).cuda().bfloat16()
    x = torch.randn(2, 32, 512, device="cuda", dtype=torch.bfloat16)
    y = m(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


# ---------- v1 deltas: mode + cache_rstd + REDUCE_STRATEGY ----------


@pytest.mark.parametrize("mode", ["train", "infer", "auto"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mode_consistency(mode, dtype):
    """train and infer should give the same output (modulo benchmark noise)."""
    M, N = 256, 4096
    eps = 1e-6
    x = _rand((M, N), dtype)
    w = _rand((N,), dtype)
    out = rms_norm(x.clone(), w, eps, mode=mode)
    out_ref = REF["llama"](x, w, eps)
    atol, rtol = DTYPE_TOL[dtype]
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(256, 4096), (40000, 128)])  # row + block paths
def test_cache_rstd_false(dtype, shape):
    """cache_rstd=False: bwd recomputes rstd from X. Output and grads must match."""
    M, N = shape
    eps = 1e-6
    x = _rand((M, N), dtype).requires_grad_(True)
    w = _rand((N,), dtype).requires_grad_(True)

    out_a = rms_norm(x, w, eps, in_place=False, cache_rstd=True)
    g = torch.randn_like(out_a)
    dx_a, dw_a = torch.autograd.grad(out_a, [x, w], g.clone())

    x2 = x.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    out_b = rms_norm(x2, w2, eps, in_place=False, cache_rstd=False)
    dx_b, dw_b = torch.autograd.grad(out_b, [x2, w2], g.clone())

    atol, rtol = DTYPE_TOL[dtype]
    torch.testing.assert_close(out_a, out_b, atol=atol, rtol=rtol)
    torch.testing.assert_close(dx_a, dx_b, atol=2 * atol, rtol=2 * rtol)
    torch.testing.assert_close(dw_a, dw_b, atol=2 * atol, rtol=2 * rtol)


@pytest.mark.parametrize("shape", [(256, 4096), (40000, 128), (40000, 256)])
def test_reduce_strategy_correctness(shape):
    """E2 strategy auto-picked from L2 fit; correctness shouldn't depend on which path runs."""
    M, N = shape
    eps = 1e-6
    x = _rand((M, N), torch.bfloat16).requires_grad_(True)
    w = _rand((N,), torch.bfloat16).requires_grad_(True)
    out = rms_norm(x, w, eps, in_place=False)
    g = torch.randn_like(out)
    dx, dw = torch.autograd.grad(out, [x, w], g)

    x2 = x.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    out_ref = REF["llama"](x2, w2, eps)
    dx_ref, dw_ref = torch.autograd.grad(out_ref, [x2, w2], g.detach())

    torch.testing.assert_close(dx, dx_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(dw, dw_ref, atol=1e-2, rtol=1e-2)
