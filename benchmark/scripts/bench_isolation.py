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
from benchmark.timing.cold_start import measure_cold_start
from benchmark.timing.competitors import (
    ForgeAdapter,
    LigerAdapter,
    NotRunnableError,
    OperationRequest,
    PyTorchAdapter,
    RMSNormAdapter,
    UnslothAdapter,
)
from benchmark.timing.configs import (
    DEFAULT_SEED,
    dtype_for_device,
    dtype_name,
    mode_settings,
    peak_for_gpu,
    quick_configs,
)
from benchmark.timing.events import benchmark_kernel, prepare_l2_flush, summarize_samples
from benchmark.timing.profiling import annotate_row
from benchmark.timing.reporting import markdown_report, print_report
from benchmark.timing.scenarios import HORIZONS, Horizon, canary_config, configs_for_horizon, horizon_eps


RESULTS_DIR = Path("benchmark/results")
REQUIRED_ADAPTERS = {"PyTorch", "Forge"}
STATUSES_WITH_TIMING = {"OK", "EXTRA_WORK_REFERENCE"}


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
    # Unsloth stays last because importing it can globally patch other libraries.
    return [PyTorchAdapter(), ForgeAdapter(), LigerAdapter(), UnslothAdapter()]


def _adapter_availability(adapter: RMSNormAdapter) -> tuple[dict, str | None]:
    available, reason = adapter.available()
    if not available and adapter.name in REQUIRED_ADAPTERS:
        raise RuntimeError(f"required adapter {adapter.name} is unavailable: {reason}")
    warning = None if available else f"{adapter.name} not run: {reason}"
    return {"available": available, "reason": reason}, warning


def _request_for_horizon(horizon: Horizon) -> OperationRequest:
    return OperationRequest(
        eps=horizon_eps(horizon),
        offset=horizon.offset,
        casting_mode=horizon.casting_mode,
        in_place=horizon.in_place,
        mode="train" if horizon.is_backward else "infer",
    )


def _shape_text(shape: dict) -> str:
    if shape.get("batch") is not None and shape.get("seq") is not None:
        return f"b{shape['batch']} s{shape['seq']} h{shape['n']}"
    return f"m{shape['m']} n{shape['n']}"


def _seed_for(horizon: Horizon, config_index: int) -> int:
    return DEFAULT_SEED + config_index + (sum(ord(ch) for ch in horizon.id) % 10000)


def _make_inputs(torch, config, dtype, horizon: Horizon):
    x = torch.randn((config.m, config.n), device="cuda", dtype=dtype)
    weight = None if horizon.weight_mode == "no_gamma" else torch.randn((config.n,), device="cuda", dtype=dtype)
    grad_out = torch.randn((config.m, config.n), device="cuda", dtype=dtype)
    if horizon.is_backward:
        x = x.detach().clone().requires_grad_(True)
        if weight is not None and horizon.weight_mode == "trainable_gamma":
            weight = weight.detach().clone().requires_grad_(True)
        elif weight is not None:
            weight = weight.detach().clone()
    return x, weight, grad_out


def _observed_grads(x, weight) -> tuple[str, ...]:
    observed = []
    if getattr(x, "grad", None) is not None:
        observed.append("x")
    if weight is not None and getattr(weight, "grad", None) is not None:
        observed.append("gamma")
    return tuple(observed)


def _canary_status(torch, adapter: RMSNormAdapter, horizon: Horizon, dtype) -> dict:
    from benchmark.correctness.references import REFERENCES

    config = canary_config()
    request = _request_for_horizon(horizon)
    x, weight, grad_out = _make_inputs(torch, config, dtype, horizon)
    try:
        if horizon.is_backward:
            out = adapter.backward(x, weight, grad_out, request)
        else:
            with torch.no_grad():
                out = adapter.forward(x, weight, request)
        torch.cuda.synchronize()
    except NotRunnableError as exc:
        return {"status": "NOT_RUN", "observed_grads": (), "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001 - canary failures are reported, not raised.
        return {"status": "ERROR", "observed_grads": (), "reason": f"{type(exc).__name__}: {exc}"}

    _, ref_fn = REFERENCES[horizon.casting_mode]
    ref = ref_fn(torch, x.detach(), weight.detach() if weight is not None else None, horizon_eps(horizon), horizon.offset)
    max_abs = float((out.detach() - ref).abs().max().detach().cpu())
    if max_abs > 5e-2:
        return {
            "status": "NOT_COMPARABLE",
            "observed_grads": _observed_grads(x, weight),
            "reason": f"forward semantic canary mismatch max_abs={max_abs:.3e}",
        }

    observed = _observed_grads(x, weight)
    expected = set(horizon.expected_grads)
    missing = expected.difference(observed)
    unexpected_gamma = horizon.operation == "backward_dx_only" and "gamma" in observed
    if missing:
        return {
            "status": "NOT_COMPARABLE",
            "observed_grads": observed,
            "reason": "missing " + ",".join(sorted(missing)),
        }
    if unexpected_gamma:
        return {
            "status": "NOT_COMPARABLE",
            "observed_grads": observed,
            "reason": "unexpected gamma grad in dx-only horizon",
        }
    return {"status": "OK", "observed_grads": observed, "reason": ""}


def _should_attempt(adapter: RMSNormAdapter, horizon: Horizon) -> bool:
    if adapter.name in horizon.eligible or adapter.name in horizon.include_extra_work:
        return True
    if adapter.name == "Unsloth" and horizon.id in {
        "full_training_backward_safe",
        "folded_gamma_forward_no_gamma",
        "no_gamma_backward_dx_only",
    }:
        return True
    return False


def _role_for(adapter: RMSNormAdapter, horizon: Horizon, canary: dict) -> tuple[str, bool, str]:
    if adapter.name in horizon.include_extra_work and canary["status"] == "OK":
        return "EXTRA_WORK_REFERENCE", False, "extra_dGamma_work"
    if adapter.name in horizon.eligible and canary["status"] == "OK":
        return "OK", True, ""
    if adapter.name == "Unsloth" and horizon.id == "full_training_backward_safe" and canary["status"] == "OK":
        return "OK", True, ""
    if adapter.name == "Unsloth" and horizon.id in {
        "folded_gamma_forward_no_gamma",
        "no_gamma_backward_dx_only",
    } and canary["status"] == "OK":
        return "OK", True, ""
    return canary["status"], False, canary["reason"]


def _base_row(
    config,
    horizon: Horizon,
    adapter: RMSNormAdapter,
    dtype_text: str,
    *,
    status: str,
    included: bool,
    exclusion_reason: str = "",
    observed_grads: tuple[str, ...] = (),
    notes: str = "",
) -> dict:
    request = _request_for_horizon(horizon)
    return {
        "horizon_id": horizon.id,
        "horizon_name": horizon.name,
        "horizon_description": horizon.description,
        "shape_group": config.sweep,
        "sweep": config.sweep,
        "framework": adapter.name,
        "pass": "bwd_derived" if horizon.is_backward else "fwd",
        "operation": horizon.operation,
        "shape": config.shape,
        "shape_text": _shape_text(config.shape),
        "dtype": dtype_text,
        "casting_mode": horizon.casting_mode,
        "offset": horizon.offset,
        "eps": horizon_eps(horizon),
        "weight_mode": horizon.weight_mode,
        "in_place": horizon.in_place,
        "expected_grads": list(horizon.expected_grads),
        "observed_grads": list(observed_grads),
        "included_in_speedup": included,
        "exclusion_reason": exclusion_reason,
        "stats_ms": {},
        "fwd_bwd_stats_ms": {},
        "forward_baseline_stats_ms": {},
        "vram_bytes": 0,
        "flops": 0,
        "bytes": 0,
        "arithmetic_intensity": None,
        "gbps": None,
        "peak_utilization_pct": None,
        "roofline": "unknown",
        "speedup_vs_pytorch": None,
        "speedup_vs_liger": None,
        "path": adapter.path(int(config.m), int(config.n), request),
        "status": status,
        "notes": notes,
    }


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _measure_steady(torch, fn, warmup: int, runs: int) -> tuple[dict, int]:
    prepare_l2_flush(torch)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    stats = benchmark_kernel(fn, 0, runs)
    torch.cuda.synchronize()
    peak = max(int(torch.cuda.max_memory_allocated() - baseline), 0)
    return stats, peak


def _grad_pool(torch, grad_out, count: int) -> list:
    return [grad_out.detach().clone() for _ in range(count)]


def _run_forward_row(torch, config, horizon, adapter, dtype_text, x, weight, warmup, runs, status, included, reason, observed):
    row = _base_row(
        config,
        horizon,
        adapter,
        dtype_text,
        status=status,
        included=included,
        exclusion_reason=reason,
        observed_grads=observed,
    )
    if status not in STATUSES_WITH_TIMING:
        return row, None
    request = _request_for_horizon(horizon)

    def fn():
        with torch.no_grad():
            return adapter.forward(x, weight, request)

    try:
        stats, peak = _measure_steady(torch, fn, warmup, runs)
        row.update({"stats_ms": stats, "vram_bytes": peak})
    except NotRunnableError as exc:
        row.update({"status": "NOT_RUN", "included_in_speedup": False, "exclusion_reason": str(exc), "notes": str(exc)})
    except Exception as exc:  # noqa: BLE001 - benchmark should continue by row.
        row.update({"status": "ERROR", "included_in_speedup": False, "exclusion_reason": _error_text(exc), "notes": _error_text(exc)})
    return row, None


def _run_backward_row(torch, config, horizon, adapter, dtype_text, x, weight, grad_out, warmup, runs, status, included, reason, observed):
    row = _base_row(
        config,
        horizon,
        adapter,
        dtype_text,
        status=status,
        included=included,
        exclusion_reason=reason,
        observed_grads=observed,
    )
    if status not in STATUSES_WITH_TIMING:
        return row, None

    request = _request_for_horizon(horizon)
    grad_pool = _grad_pool(torch, grad_out, warmup + runs + 2)
    grad_index = {"value": 0}

    def fwd_fn():
        with torch.no_grad():
            return adapter.forward(x.detach(), weight, request)

    def fwd_bwd_fn():
        idx = grad_index["value"]
        grad_index["value"] += 1
        return adapter.backward(x, weight, grad_pool[idx], request)

    try:
        fwd_stats, _ = _measure_steady(torch, fwd_fn, warmup, runs)
        fwd_bwd_stats, peak = _measure_steady(torch, fwd_bwd_fn, warmup, runs)
        fwd_samples = fwd_stats.get("samples") or []
        fwd_bwd_samples = fwd_bwd_stats.get("samples") or []
        if len(fwd_samples) == len(fwd_bwd_samples) and fwd_samples:
            samples = [bwd - fwd for fwd, bwd in zip(fwd_samples, fwd_bwd_samples)]
        else:
            samples = [(fwd_bwd_stats.get("mean") or 0.0) - (fwd_stats.get("mean") or 0.0)]
        stats = summarize_samples(samples)
        row.update(
            {
                "stats_ms": stats,
                "fwd_bwd_stats_ms": fwd_bwd_stats,
                "forward_baseline_stats_ms": fwd_stats,
                "vram_bytes": peak,
            }
        )
        if (stats.get("mean") is not None and stats["mean"] < 0) or (
            stats.get("median") is not None and stats["median"] < 0
        ):
            warning = f"derived backward time is negative due to noise: {adapter.name} {horizon.name} {config.label}"
            row["notes"] = "negative derived latency; timing noise"
            return row, warning
    except NotRunnableError as exc:
        row.update({"status": "NOT_RUN", "included_in_speedup": False, "exclusion_reason": str(exc), "notes": str(exc)})
    except Exception as exc:  # noqa: BLE001 - benchmark should continue by row.
        row.update({"status": "ERROR", "included_in_speedup": False, "exclusion_reason": _error_text(exc), "notes": _error_text(exc)})
    return row, None


def _run_adapter_horizon(torch, adapter, horizon, availability, dtype, dtype_text, warmup, runs):
    rows = []
    exclusions = []
    warnings = []
    if not availability["available"]:
        exclusions.append(
            {
                "horizon_id": horizon.id,
                "horizon_name": horizon.name,
                "framework": adapter.name,
                "reason": availability["reason"],
                "status": "NOT_RUN",
            }
        )
        return rows, exclusions, warnings, {"status": "NOT_RUN", "observed_grads": (), "reason": availability["reason"]}

    if not _should_attempt(adapter, horizon):
        reason = "not eligible for this horizon"
        exclusions.append(
            {
                "horizon_id": horizon.id,
                "horizon_name": horizon.name,
                "framework": adapter.name,
                "reason": reason,
                "status": "NOT_COMPARABLE",
            }
        )
        return rows, exclusions, warnings, {"status": "NOT_COMPARABLE", "observed_grads": (), "reason": reason}

    canary = _canary_status(torch, adapter, horizon, dtype)
    status, included, reason = _role_for(adapter, horizon, canary)
    if status not in STATUSES_WITH_TIMING:
        exclusions.append(
            {
                "horizon_id": horizon.id,
                "horizon_name": horizon.name,
                "framework": adapter.name,
                "reason": reason,
                "status": status,
                "observed_grads": list(canary.get("observed_grads") or ()),
                "expected_grads": list(horizon.expected_grads),
            }
        )

    for config_index, config in enumerate(configs_for_horizon("full" if runs > 3 else "quick", horizon)):
        torch.manual_seed(_seed_for(horizon, config_index))
        torch.cuda.manual_seed_all(_seed_for(horizon, config_index))
        x, weight, grad_out = _make_inputs(torch, config, dtype, horizon)
        if horizon.is_backward:
            row, warning = _run_backward_row(
                torch,
                config,
                horizon,
                adapter,
                dtype_text,
                x,
                weight,
                grad_out,
                warmup,
                runs,
                status,
                included,
                reason,
                tuple(canary.get("observed_grads") or ()),
            )
        else:
            row, warning = _run_forward_row(
                torch,
                config,
                horizon,
                adapter,
                dtype_text,
                x,
                weight,
                warmup,
                runs,
                status,
                included,
                reason,
                tuple(canary.get("observed_grads") or ()),
            )
        rows.append(row)
        if warning:
            warnings.append(warning)
        del x, weight, grad_out
        gc.collect()
        torch.cuda.empty_cache()
    return rows, exclusions, warnings, canary


def _add_speedups(rows: list[dict]) -> None:
    groups = {}
    for row in rows:
        key = (row["horizon_id"], row["shape_text"], row["casting_mode"])
        groups.setdefault(key, []).append(row)

    for group_rows in groups.values():
        baselines = {}
        for row in group_rows:
            if not row.get("included_in_speedup") or row.get("status") != "OK":
                continue
            median = (row.get("stats_ms") or {}).get("median")
            if median and median > 0 and row["framework"] in {"PyTorch", "Liger"}:
                baselines[row["framework"]] = median
        for row in group_rows:
            median = (row.get("stats_ms") or {}).get("median")
            if not median or median <= 0 or not row.get("included_in_speedup"):
                continue
            if baselines.get("PyTorch"):
                row["speedup_vs_pytorch"] = baselines["PyTorch"] / median
            if baselines.get("Liger"):
                row["speedup_vs_liger"] = baselines["Liger"] / median


def _folding_canary(torch, dtype) -> dict:
    from benchmark.correctness.references import rms_norm_llama

    torch.manual_seed(DEFAULT_SEED + 999)
    x = torch.randn((4, 16), device="cuda", dtype=dtype)
    gamma = torch.randn((16,), device="cuda", dtype=dtype)
    linear = torch.randn((8, 16), device="cuda", dtype=dtype)
    eps = 1e-6
    pre = rms_norm_llama(torch, x, gamma, eps, 0.0) @ linear.t()
    folded_linear = linear * gamma[None, :]
    post = rms_norm_llama(torch, x, None, eps, 0.0) @ folded_linear.t()
    max_abs = float((pre - post).abs().max().detach().cpu())
    return {"status": "PASS" if max_abs < 5e-2 else "WARN", "max_abs": max_abs}


def _cold_start_rows(mode: str) -> list[dict]:
    configs = quick_configs()
    reference = configs[0]
    qk = configs[1]
    targets = [
        ("Forge", "standard_forward_affine", reference),
        ("Liger", "standard_forward_affine", reference),
        ("Forge", "full_training_backward_safe", qk),
        ("Liger", "full_training_backward_safe", qk),
    ]
    rows = []
    for adapter, horizon_id, config in targets:
        payload = measure_cold_start(adapter, horizon_id, config.m, config.n)
        payload.update(
            {
                "mode": mode,
                "horizon_id": horizon_id,
                "shape": config.shape,
                "shape_text": _shape_text(config.shape),
                "framework": adapter,
            }
        )
        rows.append(payload)
    return rows


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
    availability = {}
    warnings = []
    results = []
    exclusions = []
    capability_matrix = []

    if peak.get("bandwidth_gbps") is None:
        warnings.append(f"unknown GPU peak bandwidth for {env['gpu']}; utilization fields will be '-'")

    for adapter in adapters:
        adapter_availability, availability_warning = _adapter_availability(adapter)
        availability[adapter.name] = adapter_availability
        if availability_warning:
            warnings.append(availability_warning)

        for horizon in HORIZONS:
            rows, horizon_exclusions, horizon_warnings, canary = _run_adapter_horizon(
                torch,
                adapter,
                horizon,
                adapter_availability,
                dtype,
                dtype_text,
                warmup,
                runs,
            )
            results.extend(rows)
            exclusions.extend(horizon_exclusions)
            warnings.extend(horizon_warnings)
            capability_matrix.append(
                {
                    "framework": adapter.name,
                    "horizon_id": horizon.id,
                    "horizon_name": horizon.name,
                    "status": canary["status"],
                    "expected_grads": list(horizon.expected_grads),
                    "observed_grads": list(canary.get("observed_grads") or ()),
                    "reason": canary.get("reason", ""),
                }
            )

    for row in results:
        annotate_row(row, peak)
        if row.get("status") in STATUSES_WITH_TIMING and peak.get("bandwidth_gbps") and row.get("gbps"):
            if row["gbps"] > peak["bandwidth_gbps"]:
                warnings.append(
                    f"measured GB/s above configured peak: {row['framework']} {row['horizon_name']} "
                    f"{row['shape_text']} {row['gbps']:.1f} > {peak['bandwidth_gbps']:.1f}"
                )
    _add_speedups(results)

    report = {
        "environment": env,
        "mode": mode,
        "dtype": dtype_text,
        "warmup": warmup,
        "runs": runs,
        "competitor_availability": availability,
        "settings": {
            "seed": DEFAULT_SEED,
            "peak_profile": peak["name"],
            "peak_bandwidth_gbps": peak["bandwidth_gbps"],
            "peak_fp16_tflops": peak["fp16_tflops"],
            "peak_bf16_tflops": peak["bf16_tflops"],
        },
        "horizons": [
            {
                "id": horizon.id,
                "name": horizon.name,
                "description": horizon.description,
                "eligible": list(horizon.eligible),
                "include_extra_work": list(horizon.include_extra_work),
            }
            for horizon in HORIZONS
        ],
        "capability_matrix": capability_matrix,
        "fairness_exclusions": exclusions,
        "folding_canary": _folding_canary(torch, dtype),
        "cold_start": _cold_start_rows(mode),
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
