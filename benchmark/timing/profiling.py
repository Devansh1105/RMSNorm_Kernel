"""Approximate roofline-style metrics for RMSNorm timing rows."""
from __future__ import annotations


def dtype_size_bytes(dtype_name: str) -> int:
    if dtype_name in {"float16", "bfloat16", "half"}:
        return 2
    if dtype_name in {"float32", "float"}:
        return 4
    return 2


def estimate_work(pass_name: str, m: int, n: int, dtype_name: str) -> tuple[int, int]:
    """Return estimated FLOPs and bytes moved.

    These are interpretation estimates, not hardware profiler counters.
    """
    element_size = dtype_size_bytes(dtype_name)
    elements = m * n

    fwd_flops = 2 * elements
    fwd_bytes = elements * element_size + n * element_size + elements * element_size

    bwd_flops = 6 * elements
    bwd_bytes = (
        elements * element_size  # x read
        + elements * element_size  # dy read
        + n * element_size  # weight read
        + elements * element_size  # dx write
        + n * 4  # dweight write / partial reduction estimate
    )

    if pass_name == "fwd":
        return fwd_flops, fwd_bytes
    if pass_name == "fwd_bwd":
        return fwd_flops + bwd_flops, fwd_bytes + bwd_bytes
    if pass_name == "bwd_derived":
        return bwd_flops, bwd_bytes
    raise ValueError(f"unknown pass: {pass_name}")


def _peak_tflops(dtype_name: str, peak: dict) -> float | None:
    if dtype_name == "bfloat16":
        return peak.get("bf16_tflops")
    if dtype_name == "float16":
        return peak.get("fp16_tflops")
    return peak.get("fp16_tflops")


def annotate_row(row: dict, peak: dict) -> None:
    if row.get("status") != "OK":
        row.update(
            {
                "flops": 0,
                "bytes": 0,
                "arithmetic_intensity": None,
                "gbps": None,
                "peak_utilization_pct": None,
                "roofline": "unknown",
            }
        )
        return

    shape = row["shape"]
    flops, bytes_moved = estimate_work(row["pass"], int(shape["m"]), int(shape["n"]), row["dtype"])
    stats = row.get("stats_ms") or {}
    latency_ms = stats.get("median") or stats.get("mean")
    arithmetic_intensity = flops / bytes_moved if bytes_moved else None
    gbps = None
    if latency_ms and latency_ms > 0:
        gbps = bytes_moved / (latency_ms / 1000.0) / 1e9

    bandwidth = peak.get("bandwidth_gbps")
    peak_tflops = _peak_tflops(row["dtype"], peak)
    roofline = "unknown"
    utilization = None
    if bandwidth and peak_tflops and arithmetic_intensity is not None:
        machine_balance = (peak_tflops * 1e12) / (bandwidth * 1e9)
        roofline = "memory-bound" if arithmetic_intensity < machine_balance else "compute-bound"
        if roofline == "memory-bound" and gbps is not None:
            utilization = 100.0 * gbps / bandwidth
        elif latency_ms and latency_ms > 0:
            achieved_tflops = flops / (latency_ms / 1000.0) / 1e12
            utilization = 100.0 * achieved_tflops / peak_tflops

    row.update(
        {
            "flops": flops,
            "bytes": bytes_moved,
            "arithmetic_intensity": arithmetic_intensity,
            "gbps": gbps,
            "peak_utilization_pct": utilization,
            "roofline": roofline,
        }
    )

