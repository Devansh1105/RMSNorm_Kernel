# RMSNorm Forge Benchmark Suite

This directory contains the unified correctness and isolation benchmark flow for
`fast_rmsnorm`. The first gate is designed for Colab GPU so kernel bugs are
caught before spending money on RunPod/A100. The full RunPod pass produces the
Forge Part A report.

Phase 1 is implemented:

- correctness vs PyTorch reference formulas
- isolation sweeps for sequence length, batch size, and hidden dimension
- CUDA event timing with warmups and L2 flushing
- mean, median, p50, p95, p99, std, min, max
- VRAM, arithmetic intensity, peak utilization, and roofline classification
- markdown report generation

Phase 2 is not implemented yet:

- HuggingFace model-level B1-B3 benchmarking
- LoRA SFT step timing
- convergence/loss-curve comparison

## Prerequisites

Use a CUDA GPU runtime. Triton kernels do not run on Colab TPU.

```bash
git clone <this-repo-url>
cd rmsnorm
pip install -e .[bench]
```

If `unsloth` does not install cleanly in the environment, run the quick checks
without strict competitors first. The suite will print a `NOT_RUN`/install
message for missing competitors instead of silently omitting them. The final
RunPod report should use `--strict-competitors`.

## 1. Colab Correctness Gate

Run this first on a free Colab GPU, typically T4:

```bash
python -m benchmark.rmsnorm_forge_suite correctness --quick
```

Expected result:

```text
Correctness PASS: ... checks, 0 failures, 1 blocked.
Blocked: gradcheck_fp64 - Forge asks for torch.autograd.gradcheck in fp64...
```

The blocked gradcheck row is intentional for v1. The current Triton dispatcher
only maps fp32/fp16/bf16, so Forge's fp64 gradcheck requirement cannot honestly
be marked PASS until fp64 support or a dedicated fp64 test wrapper is added.

Do not continue to paid benchmarking if this command reports any `FAIL`.

## 2. Colab Quick Isolation Smoke

After correctness passes:

```bash
python -m benchmark.rmsnorm_forge_suite isolation --quick
```

This runs a tiny sequence, batch, hidden, and reference sweep. It writes:

```text
benchmark/results/isolation_<gpu>_<timestamp>.json
benchmark/results/isolation_<gpu>_<timestamp>.md
```

Missing Liger or Unsloth imports are shown in the report as `NOT_RUN` with an
install or adapter note.

The benchmark dtype defaults to `auto`: bf16 when the GPU supports it, otherwise
fp16. This keeps T4 smoke runs usable while full A100/H100 runs still use bf16.

## 3. RunPod Full Isolation Report

Use an A100/H100 CUDA image with this repo installed:

```bash
pip install -e .[bench]
python -m benchmark.rmsnorm_forge_suite isolation --full --strict-competitors
```

This is the Forge Part A report run. It requires all direct competitors to be
importable. If Unsloth is installed but no direct RMSNorm adapter is exposed by
that version, the suite fails with a clear message rather than producing an
incomplete competitor table.

Default timing settings match the Forge rules:

- 3 warmup runs
- 10 timed runs
- CUDA event timing
- L2 flush before every timed run

Override only for debugging:

```bash
python -m benchmark.rmsnorm_forge_suite isolation --full --warmup 1 --runs 3
python -m benchmark.rmsnorm_forge_suite isolation --quick --no-flush-l2
```

Runs with `--no-flush-l2` are not Forge-compliant.

## 4. Regenerate A Report

Reports are deterministic from the JSON payload:

```bash
python -m benchmark.rmsnorm_forge_suite report \
  --input benchmark/results/isolation_<gpu>_<timestamp>.json
```

This rewrites the sibling `.md` report.

## 5. Reading The Report

- A1: forward latency across sequence length.
- A2: estimated backward latency. The suite measures forward+backward and
  reports backward as `Fwd+Bwd - Fwd`, so compare this consistently across
  frameworks.
- A3: combined forward+backward latency.
- A4: full timing distribution for the reference configuration.
- A5/A6: scaling with batch size and hidden dimension.
- A7: peak CUDA memory allocated.
- A8: arithmetic intensity, roofline classification, peak utilization, and
  correctness rows.
- Kernel Report Card: quick decision table for correctness, speedup, hardware
  utilization, and missing Phase 2 items.

## 6. Troubleshooting

`CUDA GPU is required`

You are on CPU or TPU. Switch Colab runtime to GPU.

`PyTorch is not installed`

Run:

```bash
pip install -e .[bench]
```

`Missing required competitors under --strict-competitors`

Install the named dependency. For Liger:

```bash
pip install liger-kernel
```

For Unsloth:

```bash
pip install unsloth
```

If Unsloth imports but does not expose a direct RMSNorm adapter, record that in
the report notes and compare Unsloth at model-level in Phase 2.

`Correctness failed; refusing to benchmark`

Run:

```bash
python -m benchmark.rmsnorm_forge_suite correctness --quick --output /tmp/rmsnorm_correctness.json
```

Inspect the failing rows before starting RunPod.

## 7. Not Covered Yet

- Forge Part B model-level training benchmark.
- 1000-step convergence/loss-curve check.
- C++/CUDA Apex-style competitor column.
- AMD/ROCm benchmarking.
- Persistent autotune-choice cache validation.
