"""Forge-compliant RMSNorm correctness and isolation benchmark suite.

This is the canonical benchmark entrypoint for this repo:

    python -m benchmark.rmsnorm_forge_suite correctness --quick
    python -m benchmark.rmsnorm_forge_suite isolation --quick
    python -m benchmark.rmsnorm_forge_suite isolation --full --strict-competitors
    python -m benchmark.rmsnorm_forge_suite report --input benchmark/results/<run>.json

Phase 1 covers Forge Part A: correctness, isolation sweeps, timing statistics,
VRAM, arithmetic intensity, peak utilization, and a markdown report. Phase 2
will add model-level B1-B3 measurements.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import platform
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


RESULTS_DIR = Path("benchmark/results")
DEFAULT_EPS = 1e-6
THRESHOLDS = {
    "float32": {"atol": 1e-5, "rtol": 1e-5},
    "float16": {"atol": 1e-3, "rtol": 1e-3},
    "bfloat16": {"atol": 1e-3, "rtol": 1e-3},
}

SEQ_SWEEP = [512, 1024, 2048, 4096, 8192]
BATCH_SWEEP = [1, 2, 4, 8, 16]
HIDDEN_SWEEP = [1024, 2048, 4096, 8192, 11008]
REFERENCE_CONFIG = {"batch": 4, "seq": 2048, "hidden": 4096, "dtype": "bfloat16"}

GPU_PEAKS = [
    {"match": "A100-SXM4-80GB", "bf16_tflops": 312.0, "mem_gbs": 2039.0},
    {"match": "A100-SXM4-40GB", "bf16_tflops": 312.0, "mem_gbs": 1555.0},
    {"match": "A100", "bf16_tflops": 312.0, "mem_gbs": 1555.0},
    {"match": "H100", "bf16_tflops": 1979.0, "mem_gbs": 3350.0},
    {"match": "H200", "bf16_tflops": 1979.0, "mem_gbs": 4800.0},
    {"match": "L4", "bf16_tflops": 242.0, "mem_gbs": 300.0},
    {"match": "T4", "bf16_tflops": 65.0, "mem_gbs": 320.0},
    {"match": "V100", "bf16_tflops": 125.0, "mem_gbs": 900.0},
    {"match": "RTX 4090", "bf16_tflops": 330.0, "mem_gbs": 1008.0},
]


def _load_torch():
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install benchmark deps with `pip install -e .[bench]`."
        ) from exc
    return torch


def _load_triton_version() -> str:
    try:
        triton = importlib.import_module("triton")
    except ImportError:
        return "not installed"
    return getattr(triton, "__version__", "unknown")


def _require_cuda(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Use Colab GPU/RunPod GPU, not TPU or CPU runtime.")


def _dtype_from_name(torch, name: str):
    if name == "auto":
        return torch.bfloat16 if _supports_dtype(torch, torch.bfloat16) else torch.float16
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}. Expected auto/bf16/fp16/fp32.")
    return mapping[name]


def _dtype_name(dtype) -> str:
    text = str(dtype).replace("torch.", "")
    return {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(text, text)


def _supports_dtype(torch, dtype) -> bool:
    if dtype == torch.bfloat16:
        return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return dtype in (torch.float16, torch.float32)


def _stats_from_times(torch, times_us: list[float]) -> dict[str, float]:
    times = torch.tensor(times_us, dtype=torch.float64)
    std = times.std(unbiased=False).item() if len(times_us) > 1 else 0.0
    return {
        "mean_us": times.mean().item(),
        "median_us": times.median().item(),
        "p50_us": times.quantile(0.50).item(),
        "p95_us": times.quantile(0.95).item(),
        "p99_us": times.quantile(0.99).item(),
        "std_us": std,
        "min_us": times.min().item(),
        "max_us": times.max().item(),
        "runs": len(times_us),
        "times_us": [float(x) for x in times_us],
    }


def _l2_flush_elements(torch, device) -> int:
    props = torch.cuda.get_device_properties(device)
    l2_bytes = getattr(props, "L2_cache_size", None) or getattr(props, "l2_cache_size", None)
    # Forge's example uses 32 MB. That is not enough for A100/H100-class L2.
    # Use at least 32 MB, otherwise 2x device L2 to force eviction.
    flush_bytes = max(32 * 1024 * 1024, int(2 * l2_bytes) if l2_bytes else 0)
    return math.ceil(flush_bytes / 4)


def benchmark_kernel(
    fn: Callable[[], Any],
    *,
    warmup: int = 3,
    runs: int = 10,
    flush_l2: bool = True,
    torch=None,
) -> dict[str, Any]:
    """Standard Forge timing harness using CUDA events and L2 flush."""
    torch = torch or _load_torch()
    _require_cuda(torch)

    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    flush_buf = None
    if flush_l2:
        flush_buf = torch.empty(_l2_flush_elements(torch, torch.device("cuda")), dtype=torch.float32, device="cuda")

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]

    for i in range(runs):
        if flush_buf is not None:
            flush_buf.zero_()
            torch.cuda.synchronize()
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(start_events, end_events)]
    return _stats_from_times(torch, times_us)


def rms_norm_llama(torch, x, w, eps: float, offset: float = 0.0):
    in_dtype = x.dtype
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(-1, keepdim=True)
    out = x32 * torch.rsqrt(var + eps)
    out = out.to(in_dtype)
    if w is not None:
        out = out * (offset + w)
    return out


def rms_norm_gemma(torch, x, w, eps: float, offset: float = 1.0):
    in_dtype = x.dtype
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(-1, keepdim=True)
    out = x32 * torch.rsqrt(var + eps)
    if w is not None:
        out = out * (offset + w.to(torch.float32))
    return out.to(in_dtype)


def rms_norm_none(torch, x, w, eps: float, offset: float = 0.0):
    var = x.pow(2).mean(-1, keepdim=True)
    out = x * torch.rsqrt(var + eps)
    if w is not None:
        out = out * (offset + w)
    return out


REFS = {
    "llama": rms_norm_llama,
    "gemma": rms_norm_gemma,
    "none": rms_norm_none,
}


def _max_atol_rtol(torch, got, ref) -> tuple[float, float]:
    diff = (got - ref).abs().to(torch.float32)
    denom = ref.abs().to(torch.float32).clamp_min(1e-12)
    return diff.max().item(), (diff / denom).max().item()


def _threshold_for(dtype) -> dict[str, float]:
    return THRESHOLDS[_dtype_name(dtype)]


def _verdict(max_atol: float, max_rtol: float, dtype) -> str:
    threshold = _threshold_for(dtype)
    return "PASS" if max_atol <= threshold["atol"] and max_rtol <= threshold["rtol"] else "FAIL"


def _make_tensor(torch, shape: tuple[int, int], dtype, *, requires_grad: bool = False):
    return torch.randn(shape, device="cuda", dtype=dtype, requires_grad=requires_grad)


def _correctness_cases(torch, quick: bool) -> list[dict[str, Any]]:
    dtypes = [torch.float32, torch.float16]
    if _supports_dtype(torch, torch.bfloat16):
        dtypes.append(torch.bfloat16)

    if quick:
        shapes = [(16, 256), (64, 1024)]
        modes = [("llama", 0.0), ("gemma", 1.0)]
        weights = [True, False]
    else:
        shapes = [(128, 4096), (1024, 4096), (4, 8192), (40000, 64), (40000, 128), (40000, 256)]
        modes = [("llama", 0.0), ("gemma", 1.0), ("gemma", 0.0), ("none", 0.0)]
        weights = [True, False]

    cases = []
    for shape in shapes:
        for dtype in dtypes:
            for casting_mode, offset in modes:
                for with_weight in weights:
                    cases.append(
                        {
                            "shape": shape,
                            "dtype": dtype,
                            "casting_mode": casting_mode,
                            "offset": offset,
                            "with_weight": with_weight,
                        }
                    )
    return cases


def _run_forward_correctness(torch, rms_norm, case: dict[str, Any]) -> dict[str, Any]:
    m, n = case["shape"]
    dtype = case["dtype"]
    x = _make_tensor(torch, (m, n), dtype)
    w = _make_tensor(torch, (n,), dtype) if case["with_weight"] else None
    out = rms_norm(
        x.clone(),
        w,
        DEFAULT_EPS,
        offset=case["offset"],
        casting_mode=case["casting_mode"],
        in_place=False,
        mode="infer",
    )
    ref = REFS[case["casting_mode"]](torch, x, w, DEFAULT_EPS, case["offset"])
    max_atol, max_rtol = _max_atol_rtol(torch, out, ref)
    return {
        "kind": "forward",
        "shape": [m, n],
        "dtype": _dtype_name(dtype),
        "casting_mode": case["casting_mode"],
        "offset": case["offset"],
        "with_weight": case["with_weight"],
        "max_atol": max_atol,
        "max_rtol": max_rtol,
        "threshold": _threshold_for(dtype),
        "verdict": _verdict(max_atol, max_rtol, dtype),
    }


def _run_backward_correctness(torch, rms_norm, case: dict[str, Any]) -> dict[str, Any]:
    m, n = case["shape"]
    dtype = case["dtype"]
    x = _make_tensor(torch, (m, n), dtype, requires_grad=True)
    w = _make_tensor(torch, (n,), dtype, requires_grad=True) if case["with_weight"] else None
    grad = torch.randn((m, n), device="cuda", dtype=dtype)

    out = rms_norm(
        x,
        w,
        DEFAULT_EPS,
        offset=case["offset"],
        casting_mode=case["casting_mode"],
        in_place=False,
        mode="train",
    )
    grad_targets = [x] + ([w] if w is not None else [])
    grads = torch.autograd.grad(out, grad_targets, grad, retain_graph=False)

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True) if w is not None else None
    ref = REFS[case["casting_mode"]](torch, x_ref, w_ref, DEFAULT_EPS, case["offset"])
    ref_targets = [x_ref] + ([w_ref] if w_ref is not None else [])
    grads_ref = torch.autograd.grad(ref, ref_targets, grad.detach(), retain_graph=False)

    dx_atol, dx_rtol = _max_atol_rtol(torch, grads[0], grads_ref[0])
    if w is not None:
        dw_atol, dw_rtol = _max_atol_rtol(torch, grads[1], grads_ref[1])
    else:
        dw_atol, dw_rtol = 0.0, 0.0
    max_atol = max(dx_atol, dw_atol)
    max_rtol = max(dx_rtol, dw_rtol)
    return {
        "kind": "backward",
        "shape": [m, n],
        "dtype": _dtype_name(dtype),
        "casting_mode": case["casting_mode"],
        "offset": case["offset"],
        "with_weight": case["with_weight"],
        "max_atol": max_atol,
        "max_rtol": max_rtol,
        "dx_max_atol": dx_atol,
        "dx_max_rtol": dx_rtol,
        "dw_max_atol": dw_atol,
        "dw_max_rtol": dw_rtol,
        "threshold": _threshold_for(dtype),
        "verdict": _verdict(max_atol, max_rtol, dtype),
    }


def _run_cache_mode_correctness(torch, rms_norm, quick: bool) -> list[dict[str, Any]]:
    shapes = [(64, 1024)] if quick else [(256, 4096), (40000, 128)]
    out = []
    for shape in shapes:
        for dtype in [torch.float32, torch.bfloat16 if _supports_dtype(torch, torch.bfloat16) else torch.float16]:
            m, n = shape
            x = _make_tensor(torch, (m, n), dtype, requires_grad=True)
            w = _make_tensor(torch, (n,), dtype, requires_grad=True)
            y_a = rms_norm(x, w, DEFAULT_EPS, in_place=False, cache_rstd=True, mode="train")
            grad = torch.randn_like(y_a)
            dx_a, dw_a = torch.autograd.grad(y_a, [x, w], grad)

            x_b = x.detach().clone().requires_grad_(True)
            w_b = w.detach().clone().requires_grad_(True)
            y_b = rms_norm(x_b, w_b, DEFAULT_EPS, in_place=False, cache_rstd=False, mode="train")
            dx_b, dw_b = torch.autograd.grad(y_b, [x_b, w_b], grad.detach())

            y_atol, y_rtol = _max_atol_rtol(torch, y_b, y_a)
            dx_atol, dx_rtol = _max_atol_rtol(torch, dx_b, dx_a)
            dw_atol, dw_rtol = _max_atol_rtol(torch, dw_b, dw_a)
            max_atol = max(y_atol, dx_atol, dw_atol)
            max_rtol = max(y_rtol, dx_rtol, dw_rtol)
            out.append(
                {
                    "kind": "cache_rstd",
                    "shape": [m, n],
                    "dtype": _dtype_name(dtype),
                    "max_atol": max_atol,
                    "max_rtol": max_rtol,
                    "threshold": _threshold_for(dtype),
                    "verdict": _verdict(max_atol, max_rtol, dtype),
                }
            )
    return out


def _run_fold_correctness(torch, quick: bool) -> list[dict[str, Any]]:
    from fast_rmsnorm.transformers import FastRMSNorm, FoldPair, fold_rmsnorm_gamma_into_next_linear
    import torch.nn as nn

    class _MiniAttn(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.q_proj = nn.Linear(h, h, bias=False)
            self.k_proj = nn.Linear(h, h, bias=False)
            self.v_proj = nn.Linear(h, h, bias=False)
            self.o_proj = nn.Linear(h, h, bias=False)

        def forward(self, x):
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            return self.o_proj(q + k + v)

    class _MiniMLP(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.gate_proj = nn.Linear(h, h * 2, bias=False)
            self.up_proj = nn.Linear(h, h * 2, bias=False)
            self.down_proj = nn.Linear(h * 2, h, bias=False)

        def forward(self, x):
            return self.down_proj(self.gate_proj(x) * self.up_proj(x))

    class _MiniLayer(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.input_layernorm = FastRMSNorm(h)
            self.self_attn = _MiniAttn(h)
            self.post_attention_layernorm = FastRMSNorm(h)
            self.mlp = _MiniMLP(h)

        def forward(self, x):
            x = x + self.self_attn(self.input_layernorm(x))
            x = x + self.mlp(self.post_attention_layernorm(x))
            return x

    class _MiniInner(nn.Module):
        def __init__(self, h, num_layers):
            super().__init__()
            self.layers = nn.ModuleList([_MiniLayer(h) for _ in range(num_layers)])
            self.norm = FastRMSNorm(h)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.norm(x)

    class _MiniLlama(nn.Module):
        def __init__(self, h=128, vocab=64, num_layers=1):
            super().__init__()
            self.model = _MiniInner(h, num_layers)
            self.lm_head = nn.Linear(h, vocab, bias=False)

        def forward(self, x):
            return self.lm_head(self.model(x))

    out = []
    dtypes = [torch.float32]
    if _supports_dtype(torch, torch.bfloat16):
        dtypes.append(torch.bfloat16)

    for dtype in dtypes:
        h = 128 if quick else 256
        layers = 1 if quick else 2
        model = _MiniLlama(h=h, vocab=64, num_layers=layers).cuda().to(dtype).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, FastRMSNorm) and module.weight is not None:
                    module.weight.data.uniform_(0.5, 1.5)
        x = torch.randn(2, 8 if quick else 16, h, device="cuda", dtype=dtype)
        with torch.no_grad():
            before = model(x)
        folded = fold_rmsnorm_gamma_into_next_linear(model, arch="llama")
        with torch.no_grad():
            after = model(x)
        max_atol, max_rtol = _max_atol_rtol(torch, after, before)
        threshold = {"atol": 1e-5, "rtol": 1e-5} if dtype == torch.float32 else {"atol": 2e-2, "rtol": 2e-2}
        verdict = "PASS" if max_atol <= threshold["atol"] and max_rtol <= threshold["rtol"] else "FAIL"
        out.append(
            {
                "kind": "fold_equivalence",
                "dtype": _dtype_name(dtype),
                "folded_pairs": folded,
                "max_atol": max_atol,
                "max_rtol": max_rtol,
                "threshold": threshold,
                "verdict": verdict,
            }
        )

    class _GemmaLike(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.norm = FastRMSNorm(h, offset=1.0, casting_mode="gemma")
            self.linear = nn.Linear(h, h, bias=False)

        def forward(self, x):
            return self.linear(self.norm(x))

    gemma_dtype = torch.bfloat16 if _supports_dtype(torch, torch.bfloat16) else torch.float16
    gemma = _GemmaLike(64).cuda().to(gemma_dtype).eval()
    for p in gemma.parameters():
        p.requires_grad_(False)
    x = torch.randn(2, 64, device="cuda", dtype=gemma_dtype)
    with torch.no_grad():
        before = gemma(x)
        gemma.norm.weight.data.uniform_(-0.5, 0.5)
        before = gemma(x)
    folded = fold_rmsnorm_gamma_into_next_linear(
        gemma, pairs=[FoldPair(norm=gemma.norm, linears=[gemma.linear], name="gemma_custom")]
    )
    with torch.no_grad():
        after = gemma(x)
    max_atol, max_rtol = _max_atol_rtol(torch, after, before)
    out.append(
        {
            "kind": "fold_equivalence_gemma_offset",
            "dtype": _dtype_name(gemma_dtype),
            "folded_pairs": folded,
            "max_atol": max_atol,
            "max_rtol": max_rtol,
            "threshold": {"atol": 2e-2, "rtol": 2e-2},
            "verdict": "PASS" if max_atol <= 2e-2 and max_rtol <= 2e-2 else "FAIL",
        }
    )
    return out


def _gradcheck_status() -> dict[str, Any]:
    return {
        "kind": "gradcheck_fp64",
        "verdict": "BLOCKED",
        "reason": (
            "Forge asks for torch.autograd.gradcheck in fp64, but the current Triton "
            "dispatcher only maps float32/float16/bfloat16 in dtype_id(). Add float64 "
            "kernel support or a dedicated fp64 test wrapper before marking this PASS."
        ),
    }


def run_correctness(*, quick: bool) -> dict[str, Any]:
    torch = _load_torch()
    _require_cuda(torch)
    torch.manual_seed(0)

    from fast_rmsnorm.transformers import rms_norm

    results: list[dict[str, Any]] = []
    cases = _correctness_cases(torch, quick)
    for case in cases:
        if not _supports_dtype(torch, case["dtype"]):
            continue
        results.append(_run_forward_correctness(torch, rms_norm, case))

    backward_cases = [
        case
        for case in cases
        if case["casting_mode"] in ("llama", "gemma")
        and case["offset"] in (0.0, 1.0)
        and case["dtype"] in (torch.float32, torch.bfloat16 if _supports_dtype(torch, torch.bfloat16) else torch.float16)
        and case["shape"] in ([(16, 256), (64, 1024)] if quick else [(128, 4096), (40000, 128)])
    ]
    for case in backward_cases:
        if not _supports_dtype(torch, case["dtype"]):
            continue
        results.append(_run_backward_correctness(torch, rms_norm, case))

    results.extend(_run_cache_mode_correctness(torch, rms_norm, quick))
    results.extend(_run_fold_correctness(torch, quick))
    results.append(_gradcheck_status())

    failed = [r for r in results if r.get("verdict") == "FAIL"]
    return {
        "ok": not failed,
        "quick": quick,
        "results": results,
        "failures": failed,
    }


@dataclass
class Adapter:
    name: str
    available: bool
    reason: str = ""
    forward_fn: Callable[..., Any] | None = None


def _load_adapters(torch, *, strict: bool) -> list[Adapter]:
    from fast_rmsnorm.transformers import rms_norm as forge_rms_norm

    adapters = [
        Adapter(
            name="PyTorch",
            available=True,
            forward_fn=lambda x, w, eps, offset, casting_mode, mode="infer": REFS[casting_mode](torch, x, w, eps, offset),
        ),
        Adapter(
            name="Forge",
            available=True,
            forward_fn=lambda x, w, eps, offset, casting_mode, mode="infer": forge_rms_norm(
                x, w, eps, offset=offset, casting_mode=casting_mode, in_place=False, mode=mode
            ),
        ),
    ]

    try:
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction

        adapters.append(
            Adapter(
                name="Liger",
                available=True,
                forward_fn=lambda x, w, eps, offset, casting_mode, mode="infer": LigerRMSNormFunction.apply(
                    x, w, eps, offset, casting_mode, False, None
                ),
            )
        )
    except Exception as exc:
        adapters.append(
            Adapter(
                name="Liger",
                available=False,
                reason=f"Install with `pip install liger-kernel` or `pip install -e .[bench]`. Import error: {exc}",
            )
        )

    try:
        import unsloth.kernels.rms_layernorm as unsloth_rms

        fast_rms_layernorm = getattr(unsloth_rms, "fast_rms_layernorm", None)
        if fast_rms_layernorm is None:
            raise AttributeError("unsloth.kernels.rms_layernorm.fast_rms_layernorm not found")

        class _UnslothShim:
            def __init__(self, weight, eps):
                self.weight = weight
                self.variance_epsilon = eps

        def _unsloth_forward(x, w, eps, offset, casting_mode, mode="infer"):
            if offset not in (0.0, 1.0):
                raise RuntimeError("Unsloth adapter only supports offset 0.0/1.0")
            return fast_rms_layernorm(_UnslothShim(w, eps), x, gemma=(offset == 1.0))

        adapters.append(Adapter(name="Unsloth", available=True, forward_fn=_unsloth_forward))
    except Exception as exc:
        adapters.append(
            Adapter(
                name="Unsloth",
                available=False,
                reason=(
                    "Install with `pip install unsloth` for direct comparison. If this still fails, "
                    f"the installed Unsloth version may not expose a direct RMSNorm adapter. Import error: {exc}"
                ),
            )
        )

    if strict:
        missing = [a for a in adapters if not a.available]
        if missing:
            details = "\n".join(f"- {a.name}: {a.reason}" for a in missing)
            raise RuntimeError(f"Missing required competitors under --strict-competitors:\n{details}")
    return adapters


def _environment(torch) -> dict[str, Any]:
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    peak = _gpu_peak(torch.cuda.get_device_name(device))
    cuda_version = getattr(torch.version, "cuda", None) or getattr(torch.version, "hip", None) or "unknown"
    return {
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
        "l2_cache_bytes": getattr(props, "L2_cache_size", None) or getattr(props, "l2_cache_size", None),
        "cuda_or_hip": cuda_version,
        "torch": torch.__version__,
        "triton": _load_triton_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "peak_bf16_tflops": peak["bf16_tflops"],
        "peak_mem_gbs": peak["mem_gbs"],
        "ridge_point_flops_per_byte": peak["bf16_tflops"] * 1000.0 / peak["mem_gbs"] if peak["mem_gbs"] else None,
    }


def _gpu_peak(device_name: str) -> dict[str, float | None]:
    for row in GPU_PEAKS:
        if row["match"] in device_name:
            return {"bf16_tflops": row["bf16_tflops"], "mem_gbs": row["mem_gbs"]}
    return {"bf16_tflops": None, "mem_gbs": None}


def arithmetic_intensity(m: int, n: int, dtype_bytes: int, pass_name: str, has_weight: bool = True) -> dict[str, float]:
    if pass_name == "fwd":
        flops = 3 * m * n
        bytes_moved = (2 * m * n + (n if has_weight else 0)) * dtype_bytes + m * 4
    elif pass_name == "bwd":
        flops = 8 * m * n
        bytes_moved = (3 * m * n + (n if has_weight else 0)) * dtype_bytes + (m + n) * 4
    elif pass_name == "fwd_bwd":
        fwd = arithmetic_intensity(m, n, dtype_bytes, "fwd", has_weight)
        bwd = arithmetic_intensity(m, n, dtype_bytes, "bwd", has_weight)
        flops = fwd["flops"] + bwd["flops"]
        bytes_moved = fwd["bytes"] + bwd["bytes"]
    else:
        raise ValueError(f"Unknown pass {pass_name!r}")
    return {"flops": float(flops), "bytes": float(bytes_moved), "ai": float(flops) / float(bytes_moved)}


def _make_shape(batch: int, seq: int, hidden: int) -> tuple[int, int]:
    return batch * seq, hidden


def _bench_adapter_shape(
    torch,
    adapter: Adapter,
    *,
    batch: int,
    seq: int,
    hidden: int,
    dtype,
    warmup: int,
    runs: int,
    flush_l2: bool,
    casting_mode: str = "llama",
    offset: float = 0.0,
) -> dict[str, Any]:
    m, n = _make_shape(batch, seq, hidden)
    x = torch.randn((m, n), device="cuda", dtype=dtype)
    w = torch.randn((n,), device="cuda", dtype=dtype)
    grad = torch.randn((m, n), device="cuda", dtype=dtype)

    if not adapter.available or adapter.forward_fn is None:
        return {"status": "NOT_RUN", "reason": adapter.reason}

    def fwd():
        with torch.no_grad():
            adapter.forward_fn(x, w, DEFAULT_EPS, offset, casting_mode, "infer")

    def fwd_bwd():
        x_req = x.detach().requires_grad_(True)
        w_req = w.detach().requires_grad_(True)
        y = adapter.forward_fn(x_req, w_req, DEFAULT_EPS, offset, casting_mode, "train")
        torch.autograd.grad(y, [x_req, w_req], grad, retain_graph=False)

    torch.cuda.reset_peak_memory_stats()
    fwd()
    torch.cuda.synchronize()
    fwd_peak = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    fwd_bwd()
    torch.cuda.synchronize()
    fwd_bwd_peak = torch.cuda.max_memory_allocated()

    fwd_stats = benchmark_kernel(fwd, warmup=warmup, runs=runs, flush_l2=flush_l2, torch=torch)
    fwd_bwd_stats = benchmark_kernel(fwd_bwd, warmup=warmup, runs=runs, flush_l2=flush_l2, torch=torch)

    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    fwd_profile = arithmetic_intensity(m, n, dtype_bytes, "fwd")
    full_profile = arithmetic_intensity(m, n, dtype_bytes, "fwd_bwd")

    fwd_gbs = fwd_profile["bytes"] / (fwd_stats["mean_us"] * 1e-6) / 1e9
    full_gbs = full_profile["bytes"] / (fwd_bwd_stats["mean_us"] * 1e-6) / 1e9
    fwd_tflops = fwd_profile["flops"] / (fwd_stats["mean_us"] * 1e-6) / 1e12
    full_tflops = full_profile["flops"] / (fwd_bwd_stats["mean_us"] * 1e-6) / 1e12

    return {
        "status": "OK",
        "shape": {"batch": batch, "seq": seq, "hidden": hidden, "M": m, "N": n},
        "dtype": _dtype_name(dtype),
        "fwd": {**fwd_stats, "measured_gbs": fwd_gbs, "measured_tflops": fwd_tflops},
        "fwd_bwd": {**fwd_bwd_stats, "measured_gbs": full_gbs, "measured_tflops": full_tflops},
        "estimated_bwd_mean_us": max(0.0, fwd_bwd_stats["mean_us"] - fwd_stats["mean_us"]),
        "peak_vram_bytes": max(fwd_peak, fwd_bwd_peak),
        "profile": {"fwd": fwd_profile, "fwd_bwd": full_profile},
    }


def _sweep_configs(quick: bool) -> list[dict[str, Any]]:
    if quick:
        return [
            {"sweep": "seq", "batch": 2, "seq": 512, "hidden": 1024},
            {"sweep": "batch", "batch": 1, "seq": 512, "hidden": 1024},
            {"sweep": "hidden", "batch": 2, "seq": 512, "hidden": 1024},
            {"sweep": "reference", "batch": 2, "seq": 512, "hidden": 1024},
        ]

    configs = []
    for seq in SEQ_SWEEP:
        configs.append({"sweep": "seq", "batch": 4, "seq": seq, "hidden": 4096})
    for batch in BATCH_SWEEP:
        configs.append({"sweep": "batch", "batch": batch, "seq": 2048, "hidden": 4096})
    for hidden in HIDDEN_SWEEP:
        configs.append({"sweep": "hidden", "batch": 4, "seq": 2048, "hidden": hidden})
    configs.append({"sweep": "reference", **REFERENCE_CONFIG})
    return configs


def run_isolation(args) -> dict[str, Any]:
    torch = _load_torch()
    _require_cuda(torch)
    torch.manual_seed(args.seed)

    correctness = run_correctness(quick=args.quick)
    if not correctness["ok"]:
        raise RuntimeError("Correctness failed; refusing to benchmark. Run `correctness --quick` for details.")

    dtype = _dtype_from_name(torch, args.dtype)
    if not _supports_dtype(torch, dtype):
        raise RuntimeError(f"GPU does not support dtype {_dtype_name(dtype)}")

    adapters = _load_adapters(torch, strict=args.strict_competitors)
    env = _environment(torch)
    configs = _sweep_configs(args.quick)
    warmup = args.warmup
    runs = args.runs

    results: dict[str, Any] = {
        "suite": "rmsnorm_forge",
        "phase": "isolation",
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "quick": args.quick,
        "environment": env,
        "settings": {
            "warmup": warmup,
            "runs": runs,
            "flush_l2": not args.no_flush_l2,
            "strict_competitors": args.strict_competitors,
            "dtype": _dtype_name(dtype),
        },
        "competitors": [{"name": a.name, "available": a.available, "reason": a.reason} for a in adapters],
        "correctness": correctness,
        "isolation": [],
        "warnings": [],
    }

    for config in configs:
        print(
            f"running {config['sweep']} batch={config['batch']} seq={config['seq']} hidden={config['hidden']}",
            flush=True,
        )
        row = {"config": config, "frameworks": {}}
        for adapter in adapters:
            print(f"  {adapter.name}", flush=True)
            bench_data = _bench_adapter_shape(
                torch,
                adapter,
                batch=config["batch"],
                seq=config["seq"],
                hidden=config["hidden"],
                dtype=dtype,
                warmup=warmup,
                runs=runs,
                flush_l2=not args.no_flush_l2,
            )
            if (
                bench_data.get("status") == "OK"
                and env.get("peak_mem_gbs")
                and bench_data.get("fwd_bwd", {}).get("measured_gbs", 0) > 1.10 * env["peak_mem_gbs"]
            ):
                results["warnings"].append(
                    f"{adapter.name} measured {bench_data['fwd_bwd']['measured_gbs']:.1f} GB/s "
                    f"above advertised peak {env['peak_mem_gbs']:.1f} GB/s for {config}; "
                    "check timing noise or byte model."
                )
            row["frameworks"][adapter.name] = bench_data
        results["isolation"].append(row)

    results["report_markdown"] = render_report(results)
    return results


def _fmt(x: Any, digits: int = 1, suffix: str = "") -> str:
    if x is None:
        return "N/A"
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float)):
        if math.isnan(float(x)):
            return "N/A"
        return f"{x:.{digits}f}{suffix}"
    return str(x)


def _framework_names(results: dict[str, Any]) -> list[str]:
    return [c["name"] for c in results.get("competitors", [])]


def _find_rows(results: dict[str, Any], sweep: str) -> list[dict[str, Any]]:
    return [row for row in results.get("isolation", []) if row["config"]["sweep"] == sweep]


def _speedup(framework_row: dict[str, Any], baseline_row: dict[str, Any], key: str = "fwd_bwd") -> str:
    if framework_row.get("status") != "OK" or baseline_row.get("status") != "OK":
        return "N/A"
    val = framework_row[key]["mean_us"]
    base = baseline_row[key]["mean_us"]
    return _fmt(base / val, 2, "x") if val > 0 else "N/A"


def _latency_table(results: dict[str, Any], sweep: str, metric: str, title: str) -> str:
    rows = _find_rows(results, sweep)
    frameworks = _framework_names(results)
    varied = {"seq": "seq", "batch": "batch", "hidden": "hidden"}.get(sweep, "config")
    lines = [f"### {title}", ""]
    lines.append("| " + " | ".join([varied] + [f"{fw} (us)" for fw in frameworks] + ["Forge speedup vs PyTorch"]) + " |")
    lines.append("|" + "|".join(["---"] * (len(frameworks) + 2)) + "|")
    for row in rows:
        config = row["config"]
        values = [str(config[varied])]
        for fw in frameworks:
            data = row["frameworks"].get(fw, {})
            if data.get("status") == "OK":
                if metric == "bwd":
                    values.append(_fmt(data["estimated_bwd_mean_us"]))
                else:
                    values.append(_fmt(data[metric]["mean_us"]))
            else:
                values.append("NOT_RUN")
        values.append(_speedup(row["frameworks"].get("Forge", {}), row["frameworks"].get("PyTorch", {}), "fwd_bwd"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _detail_stats_table(results: dict[str, Any]) -> str:
    rows = _find_rows(results, "reference")
    if not rows:
        rows = _find_rows(results, "seq")
    row = rows[0]
    frameworks = _framework_names(results)
    stats = [
        ("Mean", "mean_us"),
        ("Median", "median_us"),
        ("P50", "p50_us"),
        ("P95", "p95_us"),
        ("P99", "p99_us"),
        ("Std Dev", "std_us"),
        ("Min", "min_us"),
        ("Max", "max_us"),
    ]
    lines = ["### A4. Detailed Run Statistics", ""]
    lines.append("| Statistic | " + " | ".join(f"{fw} (us)" for fw in frameworks) + " |")
    lines.append("|" + "|".join(["---"] * (len(frameworks) + 1)) + "|")
    for label, key in stats:
        values = [label]
        for fw in frameworks:
            data = row["frameworks"].get(fw, {})
            values.append(_fmt(data["fwd_bwd"][key]) if data.get("status") == "OK" else "NOT_RUN")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _vram_table(results: dict[str, Any]) -> str:
    rows = _find_rows(results, "seq")
    frameworks = _framework_names(results)
    lines = ["### A7. Peak VRAM Usage", ""]
    lines.append("| seq_len | " + " | ".join(f"{fw} (MB)" for fw in frameworks) + " | Forge saved vs PyTorch |")
    lines.append("|" + "|".join(["---"] * (len(frameworks) + 2)) + "|")
    for row in rows:
        values = [str(row["config"]["seq"])]
        for fw in frameworks:
            data = row["frameworks"].get(fw, {})
            values.append(_fmt(data["peak_vram_bytes"] / (1024 * 1024)) if data.get("status") == "OK" else "NOT_RUN")
        forge = row["frameworks"].get("Forge", {})
        pytorch = row["frameworks"].get("PyTorch", {})
        if forge.get("status") == "OK" and pytorch.get("status") == "OK" and pytorch["peak_vram_bytes"] > 0:
            saved = 100.0 * (1.0 - forge["peak_vram_bytes"] / pytorch["peak_vram_bytes"])
            values.append(_fmt(saved, 1, "%"))
        else:
            values.append("N/A")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _profiling_section(results: dict[str, Any]) -> str:
    rows = _find_rows(results, "reference") or _find_rows(results, "seq")
    row = rows[0]
    env = results["environment"]
    ridge = env.get("ridge_point_flops_per_byte")
    peak_bw = env.get("peak_mem_gbs")
    peak_tflops = env.get("peak_bf16_tflops")
    forge = row["frameworks"].get("Forge", {})
    profile = forge.get("profile", {}) if forge.get("status") == "OK" else {}

    lines = ["### A8. Performance Profiling", ""]
    lines.append("#### Arithmetic Intensity")
    lines.append("")
    lines.append("| Pass | FLOPs | Bytes | AI (FLOPs/Byte) | Ridge Point | Classification |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for pass_name, label in [("fwd", "Forward"), ("fwd_bwd", "Forward+Backward")]:
        p = profile.get(pass_name, {})
        ai = p.get("ai")
        classification = "memory-bound" if ridge and ai and ai < ridge else "compute-bound"
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _fmt(p.get("flops"), 0),
                    _fmt(p.get("bytes"), 0),
                    _fmt(ai, 3),
                    _fmt(ridge, 1),
                    classification,
                ]
            )
            + " |"
        )

    lines.extend(["", "#### Peak Utilization", ""])
    lines.append("| Framework | TFLOPS | Compute Util | GB/s | Memory Util | Bottleneck |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for fw in _framework_names(results):
        data = row["frameworks"].get(fw, {})
        if data.get("status") != "OK":
            lines.append(f"| {fw} | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | {data.get('reason', '')} |")
            continue
        tflops = data["fwd_bwd"].get("measured_tflops")
        gbs = data["fwd_bwd"].get("measured_gbs")
        comp_util = 100.0 * tflops / peak_tflops if peak_tflops and tflops is not None else None
        mem_util = 100.0 * gbs / peak_bw if peak_bw and gbs is not None else None
        bottleneck = "memory" if ridge and profile.get("fwd_bwd", {}).get("ai", 0) < ridge else "compute"
        lines.append(
            "| "
            + " | ".join([fw, _fmt(tflops, 3), _fmt(comp_util, 1, "%"), _fmt(gbs, 1), _fmt(mem_util, 1, "%"), bottleneck])
            + " |"
        )

    lines.extend(["", "#### Numerical Correctness", ""])
    lines.append("| Check | dtype | max atol | max rtol | Verdict |")
    lines.append("|---|---|---:|---:|---|")
    for check in results["correctness"]["results"]:
        name = check["kind"]
        dtype = check.get("dtype", "-")
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    dtype,
                    _fmt(check.get("max_atol"), 3),
                    _fmt(check.get("max_rtol"), 3),
                    check.get("verdict", "UNKNOWN"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _report_card(results: dict[str, Any]) -> str:
    correctness = results["correctness"]["results"]
    failures = [r for r in correctness if r.get("verdict") == "FAIL"]
    blocked = [r for r in correctness if r.get("verdict") == "BLOCKED"]
    ref_rows = _find_rows(results, "reference") or _find_rows(results, "seq")
    row = ref_rows[0]
    forge = row["frameworks"].get("Forge", {})
    pytorch = row["frameworks"].get("PyTorch", {})
    liger = row["frameworks"].get("Liger", {})
    unsloth = row["frameworks"].get("Unsloth", {})
    env = results["environment"]
    mem_util = None
    if forge.get("status") == "OK" and env.get("peak_mem_gbs") and forge.get("fwd_bwd", {}).get("measured_gbs"):
        mem_util = 100.0 * forge["fwd_bwd"]["measured_gbs"] / env["peak_mem_gbs"]

    lines = ["## Kernel Report Card", ""]
    lines.append("| Check | Status | Details |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Numerical correctness | {'PASS' if not failures else 'FAIL'} | {len(failures)} failed checks; {len(blocked)} blocked checks |"
    )
    for item in blocked:
        lines.append(f"| {item['kind']} | BLOCKED | {item.get('reason', '')} |")
    lines.append(f"| Isolation speedup vs PyTorch | {_speedup(forge, pytorch)} | Reference Fwd+Bwd config |")
    lines.append(
        f"| Peak HW utilization | {_fmt(mem_util, 1, '%')} | Memory bandwidth utilization for Forge Fwd+Bwd |"
    )
    lines.append(f"| Forge vs Liger | {_speedup(forge, liger)} | Values >1 mean Forge faster |")
    lines.append(f"| Forge vs Unsloth | {_speedup(forge, unsloth)} | Values >1 mean Forge faster; NOT_RUN means no direct adapter |")
    lines.append("| Model-level B1-B3 | PENDING | Phase 2; not included in this isolation report |")
    return "\n".join(lines)


def render_report(results: dict[str, Any]) -> str:
    env = results["environment"]
    settings = results["settings"]
    lines = [
        "# Forge RMSNorm Benchmark Report",
        "",
        "## 0. Test Environment",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| GPU | {env['gpu']} |",
        f"| Compute Capability | {env['compute_capability']} |",
        f"| Peak BF16 TFLOPS | {_fmt(env.get('peak_bf16_tflops'), 1)} |",
        f"| Peak Memory BW | {_fmt(env.get('peak_mem_gbs'), 1, ' GB/s')} |",
        f"| Ridge Point | {_fmt(env.get('ridge_point_flops_per_byte'), 1, ' FLOPs/Byte')} |",
        f"| CUDA/HIP Version | {env['cuda_or_hip']} |",
        f"| PyTorch Version | {env['torch']} |",
        f"| Triton Version | {env['triton']} |",
        "| Kernel Under Test | RMSNorm Forward + Backward |",
        f"| Dtype | {settings['dtype']} |",
        f"| Warmup Runs | {settings['warmup']} |",
        f"| Timed Runs | {settings['runs']} |",
        f"| L2 Cache | {'Flushed between timed runs' if settings['flush_l2'] else 'Not flushed'} |",
        "| Timing Method | torch.cuda.Event |",
        "",
        "## 1. Measurement Code",
        "",
        "The suite uses `benchmark_kernel()` from `benchmark/rmsnorm_forge_suite.py`: CUDA events, discarded warmups, "
        "per-run L2 flush, and mean/median/p50/p95/p99/std/min/max reporting.",
        "",
        "## Part A - Isolation Benchmarks",
        "",
        "## Warnings",
        "",
        "\n".join(f"- {warning}" for warning in results.get("warnings", [])) if results.get("warnings") else "No benchmark sanity warnings.",
        "",
        _latency_table(results, "seq", "fwd", "A1. Forward Pass Latency - Sequence Length Sweep"),
        "",
        _latency_table(results, "seq", "bwd", "A2. Estimated Backward Pass Latency - Sequence Length Sweep"),
        "",
        _latency_table(results, "seq", "fwd_bwd", "A3. Forward + Backward Combined - Sequence Length Sweep"),
        "",
        _detail_stats_table(results),
        "",
        _latency_table(results, "batch", "fwd_bwd", "A5. Scaling Analysis - Batch Size Sweep"),
        "",
        _latency_table(results, "hidden", "fwd_bwd", "A6. Scaling Analysis - Hidden Dimension Sweep"),
        "",
        _vram_table(results),
        "",
        _profiling_section(results),
        "",
        "## Part B - Model-Level Benchmarks",
        "",
        "PENDING for Phase 2. Use `python -m benchmark.rmsnorm_forge_suite model --help` for the reserved interface.",
        "",
        _report_card(results),
        "",
        "## Submission Checklist",
        "",
        "- Environment table: present",
        "- CUDA event timing: present",
        "- L2 flush: present unless explicitly disabled",
        "- 3+ warmups and 10 timed runs: configurable; check settings above",
        "- Isolation forward/backward/combined tables: present",
        "- Sequence, batch, hidden sweeps: present",
        "- Full timing statistics: present in JSON and A4 table",
        "- Peak VRAM: present",
        "- Arithmetic intensity and roofline: present",
        "- Numerical correctness: present",
        "- Model-level benchmark and convergence: Phase 2 pending",
        "- All numbers are measured by this run, not copied from the Forge template",
    ]
    return "\n".join(lines)


def _write_results(results: dict[str, Any], *, prefix: str) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gpu = results.get("environment", {}).get("gpu", "unknown_gpu").lower().replace(" ", "_").replace("/", "_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"{prefix}_{gpu}_{stamp}.json"
    md_path = RESULTS_DIR / f"{prefix}_{gpu}_{stamp}.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if "report_markdown" in results:
        md_path.write_text(results["report_markdown"], encoding="utf-8")
    return json_path, md_path


def cmd_correctness(args) -> int:
    results = run_correctness(quick=args.quick)
    print(_correctness_summary(results))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {path}")
    return 0 if results["ok"] else 1


def _correctness_summary(results: dict[str, Any]) -> str:
    total = len(results["results"])
    failed = len(results["failures"])
    blocked = sum(1 for r in results["results"] if r.get("verdict") == "BLOCKED")
    lines = [f"Correctness {'PASS' if results['ok'] else 'FAIL'}: {total} checks, {failed} failures, {blocked} blocked."]
    if failed:
        lines.append("Failures:")
        for item in results["failures"][:20]:
            lines.append(
                f"- {item['kind']} shape={item.get('shape')} dtype={item.get('dtype')} "
                f"atol={item.get('max_atol')} rtol={item.get('max_rtol')}"
            )
    for item in results["results"]:
        if item.get("verdict") == "BLOCKED":
            lines.append(f"Blocked: {item['kind']} - {item.get('reason')}")
    return "\n".join(lines)


def cmd_isolation(args) -> int:
    results = run_isolation(args)
    json_path, md_path = _write_results(results, prefix="isolation")
    print(results["report_markdown"])
    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")
    return 0


def cmd_report(args) -> int:
    path = Path(args.input)
    data = json.loads(path.read_text(encoding="utf-8"))
    markdown = render_report(data)
    output = Path(args.output) if args.output else path.with_suffix(".md")
    output.write_text(markdown, encoding="utf-8")
    print(f"wrote {output}")
    return 0


def cmd_model(args) -> int:
    print(
        textwrap.dedent(
            """
            Model-level benchmarking is Phase 2.

            Planned interface:
              python -m benchmark.rmsnorm_forge_suite model --model llama-3.2-1b --steps 1100
              python -m benchmark.rmsnorm_forge_suite model --model qwen3-8b --steps 1100

            Phase 1 intentionally stops at correctness + isolation so Colab can
            catch kernel bugs before spending RunPod/A100 time.
            """
        ).strip()
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("correctness", help="Run Forge correctness gate only")
    p.add_argument("--quick", action="store_true", help="Small Colab smoke matrix")
    p.add_argument("--output", help="Optional JSON output path")
    p.set_defaults(func=cmd_correctness)

    p = sub.add_parser("isolation", help="Run correctness then Forge Part A isolation benchmark")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Small Colab smoke sweep")
    mode.add_argument("--full", action="store_true", help="Full Forge isolation sweep")
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
        help="Benchmark dtype. auto uses bf16 when supported, otherwise fp16 for Colab T4.",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-flush-l2", action="store_true", help="Disable L2 flushing; not Forge-compliant")
    p.add_argument("--strict-competitors", action="store_true", help="Require Liger and Unsloth imports")
    p.set_defaults(func=cmd_isolation)

    p = sub.add_parser("report", help="Regenerate markdown report from suite JSON")
    p.add_argument("--input", required=True)
    p.add_argument("--output")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("model", help="Reserved Phase 2 model-level benchmark interface")
    p.add_argument("--model", default="llama-3.2-1b")
    p.add_argument("--steps", type=int, default=1100)
    p.set_defaults(func=cmd_model)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) == "isolation" and not args.quick and not args.full:
        args.quick = True
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
