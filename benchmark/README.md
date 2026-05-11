# RMSNorm Benchmark Plan

This directory contains the phased benchmark and verification workflow for
`fast_rmsnorm`. Phase 1 is deliberately small: it only checks correctness and
prints readable tables. Phase 2 is timing-only isolation benchmarking and writes
JSON/markdown result files. Correctness and timing stay separate.

## Phase 1: Correctness Gate

Run this first on a CUDA GPU runtime, ideally a free Colab T4/L4 before spending
time on A100/H100 benchmarking.

```bash
pip install -e .[bench]
python -m benchmark.scripts.check_correctness
```

Before a full isolation benchmark, run the broader matrix:

```bash
python -m benchmark.scripts.check_correctness --full
```

The output is split into tables:

- `Environment`: GPU, CUDA/HIP, Torch, Triton, and bf16 support.
- `Summary`: total PASS/FAIL/BLOCKED counts.
- `Forward Correctness`: Forge output vs explicit PyTorch formulas.
- `Backward Correctness`: `dx` and `dweight` vs PyTorch autograd.
- `Path And Option Checks`: `cache_rstd`, `in_place`, `mode`, row path, and block path coverage.
- `Gamma Fold Checks`: pre-fold vs post-fold model outputs and refusal cases.
  Gemma fp16/bf16 folding is expected to refuse by default because it is an
  approximate rounding-order change.
- `Blocked Checks`: Forge-required checks that cannot honestly pass yet.

The script prints PASS/FAIL/BLOCKED rows and exits normally after printing the
tables. Treat any `FAIL` row as a stop sign before benchmarking.

Expected v1 result:

- `FAIL` should be zero before any benchmarking.
- `gradcheck_fp64` is expected to be `BLOCKED` because the Triton dispatcher
  currently supports fp32/fp16/bf16, not fp64.
- `reduce_strategy_direct` is expected to be `BLOCKED` until the public API has
  a debug flag to force atomic vs scratch dweight reduction. The automatic path
  is still exercised indirectly through the row/block correctness cases.

## Phase 2: Isolation Timing

Run Phase 1 separately before trusting timing numbers. Phase 2 does not call or
enforce correctness.

```bash
python -m benchmark.scripts.bench_isolation --quick
python -m benchmark.scripts.bench_isolation --full
```

`--quick` is the Colab/T4 smoke timing path. `--full` runs the reference,
sequence, batch, hidden, and QK-norm timing sweeps intended for A100/H100.
The benchmark is split into fair comparison horizons so implementations only
appear in speedup columns when they do the same work.

Phase 2 output includes:

- Environment and run settings.
- Competitor availability for PyTorch, Forge, Liger, and Unsloth.
- Capability and fairness-exclusion tables.
- Separate horizon tables for standard forward, full trainable backward,
  in-place backward speed mode, frozen-gamma `dX`-only backward, folded/no-gamma
  forward, no-gamma backward, and casting semantics.
- Cold-start/autotune timing separated from steady-state CUDA-event timing.
- Peak steady-state VRAM per row.
- Estimated FLOPs, bytes moved, GB/s, utilization, and roofline labels.
- Non-fatal warnings.

Results are written under `benchmark/results/`:

```text
isolation_<gpu>_<timestamp>.json
isolation_<gpu>_<timestamp>.md
```

## What Phase 1 Covers

- Llama, Gemma, and no-cast reference formulas.
- Affine and non-affine RMSNorm.
- fp32 plus available low-precision dtype. On bf16-capable GPUs this includes
  bf16; on T4 it uses fp16.
- Row-per-program kernel path.
- Block kernel path for many rows and small hidden dimension.
- Forward-only inference mode.
- Backward train mode.
- `mode="train"`, `mode="infer"`, and `mode="auto"`.
- `cache_rstd=True` vs `cache_rstd=False`.
- `in_place=True` vs `in_place=False`.
- Gamma folding equivalence for Llama-like and Gemma-offset fp32 models.
- Gamma folding refusal for trainable params, non-Linear targets, double fold,
  and Gemma fp16/bf16 default policy.

## Troubleshooting

`Correctness runner error: PyTorch is not installed`

Run:

```bash
pip install -e .[bench]
```

`Correctness runner error: CUDA GPU is required`

Switch the runtime to a CUDA GPU. CPU and Colab TPU cannot run Triton kernels.

`FAIL` rows in backward

Do not benchmark yet. The failure table shows the path, shape, dtype, reference,
and compared path. Start with the smallest failing shape and rerun after fixing
the kernel or tolerance bug.

`BLOCKED` rows

Blocked rows are not silently ignored. They document Forge requirements that are
not implemented yet. For v1, fp64 gradcheck and forced reduce-strategy comparison
are known blocked items.

## Later Phases

Phase 3 will add model-level benchmarking: HuggingFace patching, LoRA SFT step
time, convergence/loss checks, and model-level comparisons against PyTorch,
Liger, and Unsloth.

Phase 4 will add report generation and runbook polish once the timing and
model-level scripts exist.
