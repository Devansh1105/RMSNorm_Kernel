"""Phase 2 timing-only isolation benchmark for RMSNorm.

Run:
    python -m benchmark.scripts.bench_isolation --quick
    python -m benchmark.scripts.bench_isolation --full
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from benchmark.common.environment import collect_environment, load_torch, require_cuda
from benchmark.timing.competitors import (
    ForgeAdapter,
    LigerAdapter,
    NotRunnableError,
    PyTorchAdapter,
    RMSNormAdapter,
    UnslothAdapter,
)
from benchmark.timing.configs import (
    DEFAULT_CASTING_MODE,
    DEFAULT_EPS,
    DEFAULT_OFFSET,
    DEFAULT_SEED,
    configs_for_mode,
    dtype_name,
    dtype_for_device,
    mode_settings,
    peak_for_gpu,
)
from benchmark.timing.events import benchmark_kernel, prepare_l2_flush, summarize_samples
from benchmark.timing.profiling import annotate_row
from benchmark.timing.reporting import markdown_report, print_report


RESULTS_DIR = Path("benchmark/results")
REQUIRED_ADAPTERS = {"PyTorch", "Forge"}
PASSES = ("fwd", "fwd_bwd", "bwd_derived")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase 2 RMSNorm isolation timing benchmark. Correctness is "
            "not run or enforced here."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true", help="Run Colab/T4-sized smoke timing.")
    group.add_argument("--full", action="store_true", help="Run the full A100/H100 timing matrix.")
    return parser.parse_args(argv)


def _adapter_instances() -> list[RMSNormAdapter]:
    return [PyTorchAdapter(), ForgeAdapter(), LigerAdapter(), UnslothAdapter()]


def _check_availability(adapters: list[RMSNormAdapter]) -> tuple[dict, list[str]]:
    availability = {}
    warnings = []
    for adapter in adapters:
        available, reason = adapter.available()
        availability[adapter.name] = {"available": available, "reason": reason}
        if not available and adapter.name in REQUIRED_ADAPTERS:
            raise RuntimeError(f"required adapter {adapter.name} is unavailable: {reason}")
        if not available:
            warnings.append(f"{adapter.name} not run: {reason}")
    return availability, warnings


def _base_row(config, framework: str, pass_name: str, dtype: str, status: str = "OK", notes: str = "") -> dict:
    return {
        "sweep": config.sweep,
        "framework": framework,
        "pass": pass_name,
        "shape": config.shape,
        "dtype": dtype,
        "casting_mode": DEFAULT_CASTING_MODE,
        "offset": DEFAULT_OFFSET,
        "stats_ms": {},
        "vram_bytes": 0,
        "flops": 0,
        "bytes": 0,
        "arithmetic_intensity": None,
        "gbps": None,
        "peak_utilization_pct": None,
        "roofline": "unknown",
        "status": status,
        "notes": notes,
    }


def _not_run_rows(config, adapter: RMSNormAdapter, dtype: str, reason: str) -> list[dict]:
    return [_base_row(config, adapter.name, pass_name, dtype, status="NOT_RUN", notes=reason) for pass_name in PASSES]


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _measure_cuda_peak(torch, fn, warmup: int, runs: int) -> tuple[dict, int]:
    prepare_l2_flush(torch)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    stats = benchmark_kernel(fn, warmup, runs)
    torch.cuda.synchronize()
    peak = max(int(torch.cuda.max_memory_allocated() - baseline), 0)
    return stats, peak


def _timed_row(torch, config, adapter: RMSNormAdapter, dtype: str, pass_name: str, fn, warmup: int, runs: int) -> dict:
    row = _base_row(config, adapter.name, pass_name, dtype)
    try:
        stats, peak = _measure_cuda_peak(torch, fn, warmup, runs)
        row.update({"stats_ms": stats, "vram_bytes": peak})
    except NotRunnableError as exc:
        row.update({"status": "NOT_RUN", "notes": str(exc)})
    except Exception as exc:  # noqa: BLE001 - benchmark should keep going by row.
        row.update({"status": "ERROR", "notes": _error_text(exc)})
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    return row


def _derived_row(config, adapter: RMSNormAdapter, dtype: str, fwd_row: dict, fwd_bwd_row: dict) -> tuple[dict, str | None]:
    row = _base_row(config, adapter.name, "bwd_derived", dtype)
    if fwd_row["status"] != "OK" or fwd_bwd_row["status"] != "OK":
        status = "NOT_RUN" if "NOT_RUN" in {fwd_row["status"], fwd_bwd_row["status"]} else "ERROR"
        notes = f"requires OK fwd and fwd_bwd rows; got fwd={fwd_row['status']} fwd_bwd={fwd_bwd_row['status']}"
        row.update({"status": status, "notes": notes})
        return row, None

    fwd_samples = (fwd_row.get("stats_ms") or {}).get("samples") or []
    fwd_bwd_samples = (fwd_bwd_row.get("stats_ms") or {}).get("samples") or []
    if len(fwd_samples) == len(fwd_bwd_samples) and fwd_samples:
        samples = [bwd - fwd for fwd, bwd in zip(fwd_samples, fwd_bwd_samples)]
    else:
        samples = [
            (fwd_bwd_row.get("stats_ms") or {}).get("mean", 0.0)
            - (fwd_row.get("stats_ms") or {}).get("mean", 0.0)
        ]

    stats = summarize_samples(samples)
    row.update({"stats_ms": stats, "vram_bytes": fwd_bwd_row.get("vram_bytes", 0)})
    if (stats.get("mean") is not None and stats["mean"] < 0) or (stats.get("median") is not None and stats["median"] < 0):
        warning = f"derived backward time is negative due to noise: {adapter.name} {config.label}"
        row["notes"] = "negative derived latency; timing noise"
        return row, warning
    return row, None


def _make_inputs(torch, config, dtype):
    x = torch.randn((config.m, config.n), device="cuda", dtype=dtype)
    weight = torch.randn((config.n,), device="cuda", dtype=dtype)
    x_bwd = x.detach().clone().requires_grad_(True)
    weight_bwd = weight.detach().clone().requires_grad_(True)
    grad_out = torch.randn((config.m, config.n), device="cuda", dtype=dtype)
    return x, weight, x_bwd, weight_bwd, grad_out


def _run_config(torch, config, adapters, availability: dict, dtype, dtype_text: str, warmup: int, runs: int) -> tuple[list[dict], list[str]]:
    rows = []
    warnings = []
    x, weight, x_bwd, weight_bwd, grad_out = _make_inputs(torch, config, dtype)

    for adapter in adapters:
        adapter_availability = availability[adapter.name]
        if not adapter_availability["available"]:
            rows.extend(_not_run_rows(config, adapter, dtype_text, adapter_availability["reason"]))
            continue

        def fwd_fn(adapter=adapter):
            with torch.no_grad():
                return adapter.forward(x, weight, DEFAULT_EPS, DEFAULT_OFFSET, DEFAULT_CASTING_MODE)

        def fwd_bwd_fn(adapter=adapter):
            return adapter.forward_backward(
                x_bwd,
                weight_bwd,
                grad_out,
                DEFAULT_EPS,
                DEFAULT_OFFSET,
                DEFAULT_CASTING_MODE,
            )

        fwd_row = _timed_row(torch, config, adapter, dtype_text, "fwd", fwd_fn, warmup, runs)
        rows.append(fwd_row)
        fwd_bwd_row = _timed_row(torch, config, adapter, dtype_text, "fwd_bwd", fwd_bwd_fn, warmup, runs)
        rows.append(fwd_bwd_row)
        derived_row, warning = _derived_row(config, adapter, dtype_text, fwd_row, fwd_bwd_row)
        rows.append(derived_row)
        if warning:
            warnings.append(warning)
        x_bwd.grad = None
        weight_bwd.grad = None

    del x, weight, x_bwd, weight_bwd, grad_out
    gc.collect()
    torch.cuda.empty_cache()
    return rows, warnings


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "gpu"


def _write_results(report: dict) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu = _slug(report["environment"]["gpu"])
    base = RESULTS_DIR / f"isolation_{gpu}_{timestamp}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    return json_path, md_path


def run(args: argparse.Namespace) -> int:
    mode = "quick" if args.quick else "full"
    print("Reminder: run Phase 1 correctness separately before trusting timing numbers.")

    torch = load_torch()
    require_cuda(torch)
    torch.manual_seed(DEFAULT_SEED)
    torch.cuda.manual_seed_all(DEFAULT_SEED)

    env = collect_environment(torch)
    dtype = dtype_for_device(torch)
    dtype_text = dtype_name(dtype)
    warmup, runs = mode_settings(mode)
    peak = peak_for_gpu(env["gpu"])

    adapters = _adapter_instances()
    availability, warnings = _check_availability(adapters)
    if peak.get("bandwidth_gbps") is None:
        warnings.append(f"unknown GPU peak bandwidth for {env['gpu']}; utilization fields will be '-'")

    results = []
    for config in configs_for_mode(mode):
        config_rows, config_warnings = _run_config(
            torch,
            config,
            adapters,
            availability,
            dtype,
            dtype_text,
            warmup,
            runs,
        )
        results.extend(config_rows)
        warnings.extend(config_warnings)

    for row in results:
        annotate_row(row, peak)
        if row.get("status") == "OK" and peak.get("bandwidth_gbps") and row.get("gbps"):
            if row["gbps"] > peak["bandwidth_gbps"]:
                warnings.append(
                    f"measured GB/s above configured peak: {row['framework']} {row['pass']} "
                    f"{row['sweep']} {row['gbps']:.1f} > {peak['bandwidth_gbps']:.1f}"
                )

    report = {
        "environment": env,
        "mode": mode,
        "dtype": dtype_text,
        "warmup": warmup,
        "runs": runs,
        "competitor_availability": availability,
        "settings": {
            "seed": DEFAULT_SEED,
            "eps": DEFAULT_EPS,
            "casting_mode": DEFAULT_CASTING_MODE,
            "offset": DEFAULT_OFFSET,
            "backward_in_place": False,
            "peak_profile": peak["name"],
            "peak_bandwidth_gbps": peak["bandwidth_gbps"],
            "peak_fp16_tflops": peak["fp16_tflops"],
            "peak_bf16_tflops": peak["bf16_tflops"],
        },
        "results": results,
        "warnings": list(dict.fromkeys(warnings)),
    }
    json_path, md_path = _write_results(report)
    print_report(report)
    print()
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote Markdown: {md_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except RuntimeError as exc:
        print(f"Isolation benchmark error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
