"""Phase 1 correctness gate for fast_rmsnorm.

Run:
    python -m benchmark.scripts.check_correctness
    python -m benchmark.scripts.check_correctness --full
"""
from __future__ import annotations

import argparse
import sys

from benchmark.common.environment import collect_environment, load_torch, require_cuda
from benchmark.correctness.cases import (
    backward_cases,
    cache_cases,
    forward_cases,
    inplace_cases,
    low_precision_dtype,
    mode_cases,
)
from benchmark.correctness.checks import (
    run_backward_check,
    run_cache_rstd_check,
    run_forward_check,
    run_gradcheck_status,
    run_inplace_check,
    run_mode_consistency_check,
    run_reduce_strategy_status,
)
from benchmark.correctness.folding_models import (
    run_fold_refusal_checks,
    run_gemma_fold_check,
    run_llama_fold_check,
)
from benchmark.correctness.summary import build_summary, print_correctness_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase 1 RMSNorm correctness gate. This prints tables only; "
            "it does not write JSON or benchmark timing files."
        )
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run a broader matrix before paid benchmarking. Default is the Colab-sized gate.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Torch RNG seed for reproducible inputs.")
    return parser.parse_args(argv)


def _fold_dtypes(torch) -> list:
    low_precision = low_precision_dtype(torch)
    return [torch.float32] if low_precision is torch.float32 else [torch.float32, low_precision]


def run(args: argparse.Namespace) -> int:
    torch = load_torch()
    require_cuda(torch)

    from fast_rmsnorm.transformers import rms_norm

    torch.manual_seed(args.seed)
    env = collect_environment(torch)
    rows = []

    for case in forward_cases(torch, full=args.full):
        rows.append(run_forward_check(torch, rms_norm, case))

    for case in backward_cases(torch, full=args.full):
        rows.append(run_backward_check(torch, rms_norm, case))

    for case in cache_cases(torch, full=args.full):
        rows.append(run_cache_rstd_check(torch, rms_norm, case))

    for case in inplace_cases(torch, full=args.full):
        rows.append(run_inplace_check(torch, rms_norm, case))

    for case in mode_cases(torch, full=args.full):
        rows.extend(run_mode_consistency_check(torch, rms_norm, case))

    for dtype in _fold_dtypes(torch):
        rows.append(run_llama_fold_check(torch, dtype, full=args.full))
        rows.append(run_gemma_fold_check(torch, dtype, full=args.full))

    rows.extend(run_fold_refusal_checks(torch))
    rows.append(run_reduce_strategy_status(torch))
    rows.append(run_gradcheck_status())

    summary = build_summary(rows)
    print_correctness_report(env, rows, summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except RuntimeError as exc:
        print(f"Correctness runner error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
