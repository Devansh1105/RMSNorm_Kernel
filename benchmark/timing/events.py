"""CUDA-event timing utilities for isolation benchmarks."""
from __future__ import annotations

import math
import statistics
from collections.abc import Callable


_FLUSH_TENSORS = {}


def _percentile(samples: list[float], percentile: float) -> float:
    if not samples:
        return math.nan
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def summarize_samples(samples: list[float]) -> dict:
    """Return latency summary statistics in milliseconds."""
    if not samples:
        return {
            "samples": [],
            "mean": None,
            "median": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "samples": samples,
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "p50": _percentile(samples, 0.50),
        "p95": _percentile(samples, 0.95),
        "p99": _percentile(samples, 0.99),
        "std": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "min": min(samples),
        "max": max(samples),
    }


def _flush_l2(torch, flush_tensor) -> None:
    # A large device write is enough to evict most useful benchmark data from L2.
    flush_tensor.add_(1.0)


def _make_flush_tensor(torch):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = int(getattr(props, "l2_cache_size", 0) or 0)
    flush_bytes = max(l2_bytes * 2, 64 * 1024 * 1024)
    elements = max(flush_bytes // 4, 1)
    return torch.empty(elements, device="cuda", dtype=torch.float32)


def prepare_l2_flush(torch):
    """Allocate the reusable L2 flush tensor outside timed/memory regions."""
    device = torch.cuda.current_device()
    tensor = _FLUSH_TENSORS.get(device)
    if tensor is None:
        tensor = _make_flush_tensor(torch)
        _FLUSH_TENSORS[device] = tensor
    return tensor


def benchmark_kernel(fn: Callable[[], object], warmup: int, runs: int) -> dict:
    """Benchmark ``fn`` using CUDA events.

    Warmups are run first and discarded. Each timed sample flushes L2 outside
    the measured event interval and synchronizes before and after the sample.
    """
    import torch

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if runs <= 0:
        raise ValueError("runs must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CUDA-event timing.")

    flush_tensor = prepare_l2_flush(torch)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(runs):
        torch.cuda.synchronize()
        _flush_l2(torch, flush_tensor)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))

    return summarize_samples(samples)
