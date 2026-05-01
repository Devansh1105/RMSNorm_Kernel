"""Forward + backward benchmark vs Liger.

Run on Colab T4 for sanity, A100/H100 for real numbers.

  python -m benchmark.scripts.benchmark_rms_norm

For each shape, prints:
  - fwd ms (ours, infer mode)
  - fwd ms (ours, train mode)
  - fwd ms (Liger)
  - bwd ms (ours, train mode)
  - bwd ms (Liger)
  - achieved HBM bandwidth GB/s for each (fwd reads X+W and writes Y)

Shapes mirror the v1 plan:
  - Llama-3 pre-block:    M ∈ {1024, 8192, 65536}, N=4096
  - Qwen3 QK-norm:        M ∈ {65536, 262144}, N=128
  - Gemma offset variant: same shapes, casting_mode='gemma', offset=1.0
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable

import torch
import triton

from fast_rmsnorm.transformers import rms_norm as fast_rms_norm

try:
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction

    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False


def _peak_hbm_bw_gbs(device: torch.device) -> float:
    """Return advertised peak HBM bandwidth in GB/s for the device.

    Best-effort: based on device name. Returns 0.0 if unknown — utilization
    column will be blank. Hand-tabulated rather than read from a bandwidth
    benchmark to keep this script dependency-free.
    """
    name = torch.cuda.get_device_name(device)
    table = {
        # Datacenter
        "A100": 1555.0, "A100-SXM4-40GB": 1555.0, "A100-SXM4-80GB": 2039.0,
        "H100": 3350.0, "H100 PCIe": 2000.0, "H200": 4800.0,
        "L40": 864.0, "L4": 300.0,
        # Older
        "V100": 900.0, "T4": 320.0, "P100": 732.0,
        # Consumer
        "RTX 4090": 1008.0, "RTX 3090": 936.0, "RTX 3080": 760.0,
    }
    for k, v in table.items():
        if k in name:
            return v
    return 0.0


@dataclass
class Result:
    label: str
    shape: tuple
    dtype: torch.dtype
    fwd_ms: dict[str, float]
    bwd_ms: dict[str, float]
    bytes_moved_fwd: int


def _bytes_for_fwd(M: int, N: int, dtype: torch.dtype, has_w: bool) -> int:
    """Theoretical HBM traffic for one fwd: read X (M*N), read W (N), write Y (M*N).
    rstd write is M scalars (negligible). Used to compute achieved bandwidth.
    """
    bs = torch.tensor([], dtype=dtype).element_size()
    return (2 * M * N + (N if has_w else 0)) * bs


def _bench(fn: Callable, warmup: int = 25, rep: int = 100) -> float:
    """Wraps triton.testing.do_bench with our defaults."""
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


def run_one(label: str, M: int, N: int, dtype: torch.dtype, casting_mode: str, offset: float) -> Result:
    device = torch.device("cuda")
    x = torch.randn(M, N, dtype=dtype, device=device, requires_grad=True)
    w = torch.randn(N, dtype=dtype, device=device, requires_grad=True)
    eps = 1e-6

    # ---------- Forward ----------
    fwd_ms = {}

    def _ours_infer():
        with torch.no_grad():
            return fast_rms_norm(x.detach(), w.detach(), eps, offset=offset, casting_mode=casting_mode, mode="infer")

    def _ours_train_fwd_only():
        # 'train' triggers autotune on first call per shape; do_bench warmup absorbs it.
        with torch.no_grad():
            return fast_rms_norm(x.detach(), w.detach(), eps, offset=offset, casting_mode=casting_mode, mode="train")

    fwd_ms["ours_infer"] = _bench(_ours_infer)
    fwd_ms["ours_train"] = _bench(_ours_train_fwd_only)

    if HAS_LIGER:
        def _liger_fwd():
            with torch.no_grad():
                return LigerRMSNormFunction.apply(x.detach(), w.detach(), eps, offset, casting_mode, True, None)
        fwd_ms["liger"] = _bench(_liger_fwd)

    # ---------- Backward ----------
    bwd_ms = {}
    g = torch.randn_like(x)

    def _ours_bwd():
        x_ = x.detach().clone().requires_grad_(True)
        w_ = w.detach().clone().requires_grad_(True)
        out = fast_rms_norm(x_, w_, eps, offset=offset, casting_mode=casting_mode, mode="train", in_place=False)
        out.backward(g)

    bwd_ms["ours_train"] = _bench(_ours_bwd)

    if HAS_LIGER:
        def _liger_bwd():
            x_ = x.detach().clone().requires_grad_(True)
            w_ = w.detach().clone().requires_grad_(True)
            out = LigerRMSNormFunction.apply(x_, w_, eps, offset, casting_mode, False, None)
            out.backward(g)
        bwd_ms["liger"] = _bench(_liger_bwd)

    return Result(
        label=label,
        shape=(M, N),
        dtype=dtype,
        fwd_ms=fwd_ms,
        bwd_ms=bwd_ms,
        bytes_moved_fwd=_bytes_for_fwd(M, N, dtype, has_w=True),
    )


def _format_table(results: list[Result], device: torch.device) -> str:
    peak = _peak_hbm_bw_gbs(device)
    out = []
    out.append(f"GPU: {torch.cuda.get_device_name(device)}  peak HBM: {peak:.0f} GB/s\n")
    header = (
        f"{'shape':<22}{'dtype':<10}{'mode':<14}{'fwd ms':>10}{'fwd GB/s':>12}{'bwd ms':>10}"
    )
    out.append(header)
    out.append("-" * len(header))
    for r in results:
        out.append(f"\n[{r.label}]  shape={r.shape}  dtype={r.dtype}")
        for impl, fwd_ms in r.fwd_ms.items():
            gbs = (r.bytes_moved_fwd / (fwd_ms * 1e-3)) / 1e9 if fwd_ms > 0 else 0
            util = f" ({100*gbs/peak:.0f}%)" if peak else ""
            bwd = r.bwd_ms.get(impl, None)
            bwd_str = f"{bwd:>10.4f}" if bwd is not None else f"{'—':>10}"
            out.append(
                f"  {' ':<20}{' ':<10}{impl:<14}{fwd_ms:>10.4f}{gbs:>9.0f}{util:>3}{bwd_str}"
            )
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Run a small subset for sanity")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    if not HAS_LIGER:
        print("warning: liger-kernel not installed; skipping Liger comparison", file=sys.stderr)

    # Plan shapes
    shapes_full = [
        ("llama3-preblock-small", 1024, 4096, torch.bfloat16, "llama", 0.0),
        ("llama3-preblock-med",   8192, 4096, torch.bfloat16, "llama", 0.0),
        ("llama3-preblock-large", 65536, 4096, torch.bfloat16, "llama", 0.0),
        ("qwen3-qknorm-med",      65536, 128,  torch.bfloat16, "llama", 0.0),
        ("qwen3-qknorm-large",    262144, 128, torch.bfloat16, "llama", 0.0),
        ("gemma-offset",          8192, 4096, torch.bfloat16, "gemma", 1.0),
    ]
    shapes_quick = [
        ("llama3-preblock-small", 1024, 4096, torch.bfloat16, "llama", 0.0),
        ("qwen3-qknorm-med",      65536, 128, torch.bfloat16, "llama", 0.0),
    ]
    shapes = shapes_quick if args.quick else shapes_full

    results = []
    for label, M, N, dtype, casting_mode, offset in shapes:
        print(f"running {label}  (M={M}, N={N})...", flush=True)
        results.append(run_one(label, M, N, dtype, casting_mode, offset))

    print()
    print(_format_table(results, device))


if __name__ == "__main__":
    sys.exit(main() or 0)
