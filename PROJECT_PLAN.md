# fast_rmsnorm Project Plan

This is the living checklist for the RMSNorm kernel, benchmark suite, and later
kernel versions. Update it after every GPU run and before changing v1.5/v2 scope.

## Context And Goal

- Research docs:
  - `RMSNorm_ Forward, Backward and Intermidiates.pdf`
  - `Forge_Kernel_Testing__Benchmarking_Rules.pdf`
- Package: `fast_rmsnorm`
- Main kernel file: `src/fast_rmsnorm/ops/rms_norm.py`
- Main module wrapper: `src/fast_rmsnorm/transformers/rms_norm.py`
- Active Phase 1 correctness entry point:
  `python -m benchmark.scripts.check_correctness`
- Active Phase 2 isolation timing entry points:
  `python -m benchmark.scripts.bench_isolation --quick`
  `python -m benchmark.scripts.bench_isolation --full`
- Goal: standalone Triton RMSNorm forward + backward that matches or beats Liger
  on realistic LLM shapes, plus inference-time gamma folding.
- Later goal: add fused variants only where measurement shows value, including
  RMSNorm + residual add, RMSNorm + RoPE, RMSNorm + `lm_head`, and matmul
  epilogue RMSNorm variants.

## Current Snapshot

- [x] v1 RMSNorm kernel exists.
- [x] v1 module, functional API, and gamma-fold utility exist.
- [x] Old pytest tests were removed from the active workflow.
- [x] Old prototype benchmark script was removed from the active workflow.
- [x] Benchmark plan changed from one large script to small phased scripts.
- [x] Phase 1 correctness gate implemented as modular files.
- [ ] Phase 1 correctness gate run on Colab GPU after this rewrite.
- [ ] Phase 1 full correctness matrix run on A100/H100.
- [x] Phase 2 isolation benchmark implemented.
- [ ] Phase 3 model-level benchmark implemented.
- [ ] v1 benchmark results copied into this plan.
- [ ] v1.5 scope finalized from actual benchmark results.

## Result Placeholders

Fill this table after runs finish.

| Run | GPU | Command | Status | Notes |
|---|---|---|---|---|
| Colab correctness | TBD | `python -m benchmark.scripts.check_correctness` | Pending | Expected: no FAIL; fp64 gradcheck may be BLOCKED |
| Full correctness | TBD | `python -m benchmark.scripts.check_correctness --full` | Pending | Run before paid isolation benchmark |
| Isolation quick | TBD | `python -m benchmark.scripts.bench_isolation --quick` | Pending | Phase 2 timing smoke; run after separate correctness check |
| Isolation full | TBD | `python -m benchmark.scripts.bench_isolation --full` | Pending | Phase 2 timing, A100/H100; run after separate full correctness check |
| Model-level | TBD | TBD | Pending | Phase 3 |

## v1 Kernel Checklist

### Implemented

- [x] Forked a Liger-style RMSNorm structure into `fast_rmsnorm`.
- [x] Forward row-per-program path.
- [x] Forward block path for many rows and small hidden dimension.
- [x] Backward computes `dx` and `dweight` partials in one pass.
- [x] Llama casting mode.
- [x] Gemma offset/casting mode.
- [x] No-cast mode.
- [x] Non-affine no-weight path.
- [x] `mode="train"`, `mode="infer"`, and `mode="auto"`.
- [x] Triton autotune variants for training mode.
- [x] Row-kernel autotune over warp/stage choices.
- [x] Block-kernel autotune over `BLOCK_ROW`.
- [x] dweight reduce strategy branch: scratch buffer or atomic add.
- [x] Runtime L2-fit heuristic for dweight reduce strategy.
- [x] Public/debug `reduce_strategy="auto"|"atomic"|"scratch"` knob for
  forced reducer ablation. Default remains `auto`.
- [x] fp32 epsilon pinning inside Triton kernels.
- [x] `cache_rstd` flag.
- [x] `cache_rstd=False` recompute path has an isolation timing horizon.
- [x] Inference-only gamma folding.
- [x] Gemma fp16/bf16 gamma folding refuses by default; approximate fold requires
  explicit opt-in.
- [x] Backward autotune split for in-place vs non-in-place paths so autotune
  reset/restore overhead is not paid by the non-in-place path.

### Needs GPU Testing

- [ ] Colab T4/L4 correctness for forward and backward.
- [ ] Full A100/H100 correctness matrix.
- [ ] Gamma folding equivalence on Llama-like and Gemma-offset fp32 paths.
- [ ] Gemma fp16/bf16 fold refusal path on Colab/A100.
- [ ] `cache_rstd=True` vs `cache_rstd=False` correctness.
- [ ] `in_place=True` vs `in_place=False` correctness after backward autotune split.
- [ ] `mode="auto"` behavior when models are in `eval()` but params still have
  `requires_grad=True`.
- [ ] Atomic vs scratch strategy correctness on shapes that choose each branch.

### Known Issues Or Watch Items

- [ ] Forge fp64 gradcheck is blocked because the current Triton dtype map only
  supports fp32/fp16/bf16.
- [ ] Forced atomic-vs-scratch correctness and timing need GPU validation on
  T4/L4 and A100/H100.
- [ ] Feature dimensions above Triton's fused block limit still need future
  streaming support.
- [ ] Persistent autotune-choice cache is not implemented yet.
- [ ] Direct Unsloth RMSNorm adapter is source-audited against
  `fast_rms_layernorm(layernorm, X, gemma=...)`; installed versions still need
  runtime canary validation.

## Benchmark Rewrite

The benchmark suite is split into small phases so each part can be reviewed and
validated independently.

### Phase 1: Correctness Gate

Purpose: catch correctness bugs on a cheap Colab GPU before running any timing
or paid A100/H100 benchmark.

Implemented files:

- [x] `benchmark/common/environment.py`
  - loads PyTorch lazily
  - requires CUDA
  - prints GPU, CUDA/HIP, Torch, Triton, Python, platform, and bf16 support
- [x] `benchmark/common/tolerances.py`
  - central dtype tolerances
  - max absolute and relative error helpers
  - fold-specific low-precision tolerance
- [x] `benchmark/common/table.py`
  - dependency-free terminal tables
  - compact scientific formatting for small/large values
- [x] `benchmark/correctness/references.py`
  - explicit PyTorch Llama formula
  - explicit PyTorch Gemma formula
  - explicit PyTorch no-cast formula
- [x] `benchmark/correctness/cases.py`
  - quick and full case matrices
  - row path and block path coverage
  - dtype selection for bf16-capable GPUs and T4-style fp16 GPUs
- [x] `benchmark/correctness/checks.py`
  - forward output vs PyTorch reference
  - backward `dx` and `dweight` vs PyTorch autograd
  - `cache_rstd` comparison
  - `in_place` comparison
  - `mode` comparison
  - blocked rows for fp64 gradcheck and forced reduce-strategy comparison
- [x] `benchmark/correctness/folding_models.py`
  - tiny Llama-like model for fold equivalence
  - Gemma-offset fp32 fold equivalence
  - fold refusal checks for trainable params, non-Linear target, double fold,
    and Gemma fp16/bf16 default policy
- [x] `benchmark/correctness/summary.py`
  - formatted environment, summary, forward, backward, option, fold, blocked,
    and failure tables
- [x] `benchmark/scripts/check_correctness.py`
  - simple CLI
  - default Colab-sized matrix
  - `--full` broader pre-benchmark matrix
  - no JSON output
  - no timing output
  - prints PASS/FAIL/BLOCKED tables instead of asserting numeric checks
- [x] `benchmark/README.md`
  - Phase 1 runbook and expected output guidance

Phase 1 checks:

- [x] Forward correctness: row path and block path.
- [x] Forward correctness: fp32 and available low-precision dtype.
- [x] Forward correctness: Llama, Gemma, and no-cast modes.
- [x] Forward correctness: affine and non-affine paths.
- [x] Backward correctness: `dx` vs PyTorch autograd.
- [x] Backward correctness: `dweight` vs PyTorch autograd.
- [x] Backward correctness: row path and block path.
- [x] `cache_rstd=True` vs `cache_rstd=False`.
- [x] `in_place=True` vs `in_place=False`.
- [x] `mode="train"`, `"infer"`, and `"auto"`.
- [x] Gamma fold equivalence for exact/default-supported paths.
- [x] Gamma fold refusal cases.
- [x] Blocked Forge fp64 gradcheck shown explicitly.
- [x] Blocked forced reduce-strategy comparison shown explicitly.

Phase 1 validation still required:

- [ ] Run `python -m benchmark.scripts.check_correctness` on Colab GPU.
- [ ] Fix any FAIL rows before benchmarking.
- [ ] Run `python -m benchmark.scripts.check_correctness --full` on A100/H100.
- [ ] Copy the summary counts and failure details, if any, into this file.

### Phase 2: Isolation Timing Benchmark

Purpose: timing-only Forge Part A isolation benchmark. Correctness remains a
separate Phase 1 workflow and is not run or enforced by Phase 2.

Implemented files:

- [x] `benchmark/timing/events.py`
  - CUDA event timing only
  - warmup runs
  - timed runs
  - L2 flush before timed runs
  - CUDA synchronization before and after each timed sample
  - mean, median, p50, p95, p99, std, min, max
- [x] `benchmark/timing/configs.py`
  - quick reference and QK-norm configs
  - full reference, sequence, batch, hidden, and QK-norm sweeps
  - dtype policy: fp16 below compute capability 8.0, bf16 at 8.0+
  - GPU peak bandwidth/FLOPS table for T4, L4, A100, and H100
- [x] `benchmark/timing/competitors.py`
  - PyTorch explicit-formula adapter
  - Forge adapter using `fast_rmsnorm.transformers.rms_norm`
  - Liger adapter using `liger_kernel.ops.LigerRMSNormFunction`
  - Unsloth direct adapter using the audited
    `unsloth.kernels.rms_layernorm.fast_rms_layernorm(layernorm, X, gemma=...)`
    API, or explicit `NOT_RUN` reason
- [x] `benchmark/timing/contracts.py`
  - source-audited API contract table for PyTorch, Forge, Liger, and Unsloth
  - records supported semantics and caveats used to decide fair horizons
- [x] `benchmark/timing/scenarios.py`
  - horizon definitions for apples-to-apples comparison
  - full trainable backward, in-place speed mode, frozen-gamma dx-only,
    folded/no-gamma, no-gamma backward, cache-rstd recompute ablation, and
    casting semantics
  - forced Forge atomic/scratch reducer ablation horizons
- [x] `benchmark/timing/cold_start.py`
  - representative cold-start/autotune timing in a fresh subprocess
  - temporary Triton cache so cold-start is not mixed with steady-state timing
- [x] `benchmark/timing/profiling.py`
  - estimated FLOPs
  - estimated bytes moved
  - arithmetic intensity
  - achieved GB/s
  - peak utilization percentage where peak data is known
  - roofline label
- [x] `benchmark/timing/reporting.py`
  - Environment
  - Run Settings
  - Competitor Availability
  - Capability Matrix
  - Fairness Exclusions
  - Source-Audited API Contracts
  - Cold Start / Autotune
  - separate steady-state tables per benchmark horizon
  - Reducer Policy Comparison summary for Forge auto heuristic vs Forge forced
    atomic vs Forge forced scratch/Liger-style vs Liger scratch baseline
  - VRAM
  - Profiling
  - Path Coverage
  - Warnings
- [x] `benchmark/scripts/bench_isolation.py`
  - simple CLI with only `--quick` and `--full`
  - no correctness invocation or correctness gating
  - prints a reminder to run Phase 1 separately
  - deterministic seeded inputs built outside timed regions
  - records horizon-aware forward and derived backward rows
  - reports `cache_rstd=False` rows relative to cached Forge baseline
  - reports reducer policy rows relative to Forge auto, Forge scratch, and
    Liger scratch baselines
  - validates expected gradients before timing each adapter/horizon
  - reports autograd-visible grads separately from source-inferred kernel work
  - excludes non-comparable rows from speedup calculations
  - records `NOT_RUN`, `NOT_COMPARABLE`, and `EXTRA_WORK_REFERENCE` states
  - continues through per-row errors
  - writes JSON and markdown to `benchmark/results/`

Phase 2 validation still required:

- [x] Run `python -m compileall benchmark src` locally.
- [x] Run `python -m benchmark.scripts.bench_isolation --help` locally.
- [ ] Run `python -m benchmark.scripts.check_correctness` on Colab GPU.
- [ ] Run `python -m benchmark.scripts.bench_isolation --quick` on Colab GPU.
- [ ] Run `python -m benchmark.scripts.check_correctness --full` on A100/H100.
- [ ] Run `python -m benchmark.scripts.bench_isolation --full` on A100/H100.
- [ ] Copy generated JSON/markdown result filenames and summary numbers into
  this plan.

### Phase 3: Model-Level Benchmark

Purpose: implement Forge Part B after isolation timing is stable.

- [ ] HuggingFace model patch helpers.
- [ ] PyTorch baseline model path.
- [ ] Forge-patched model path.
- [ ] Liger-patched model path.
- [ ] Unsloth model-level comparison path.
- [ ] Framework-specific fair-shot run structure:
  - import and patch each framework in the order its docs require
  - isolate globally patching frameworks, especially Unsloth, in separate
    processes so their monkey patches do not contaminate PyTorch, Forge, or
    Liger runs
  - audit each framework's patch helper before using it; record whether it
    patches only RMSNorm or also attention, MLP, cross entropy, optimizers,
    tokenization, RoPE, fused losses, or other modules
  - prefer RMSNorm-only patch paths for RMSNorm attribution; if a framework
    exposes only bundled/global patching, report that separately and do not use
    it as an RMSNorm-only comparison without an explicit decision
  - report the exact import order, patch helper, trainable parameter set, and
    patched module counts per framework
  - keep LoRA/frozen-base comparisons separate from full-finetune comparisons
- [ ] LoRA SFT step-time loop.
- [ ] Fixed-seed convergence/loss comparison.
- [ ] Peak VRAM and tokens/sec reporting.
- [ ] B1/B2/B3 formatted tables.

Target smoke model:

- [ ] Llama-3.2-1B or another small Llama-like model that fits the available
  Colab GPU.

Target final model:

- [ ] Qwen3-8B or the chosen Forge reference model on A100/H100.

### Phase 4: Report Generation And Runbook

- [ ] Combine Phase 1, Phase 2, and Phase 3 outputs into one Forge report.
- [ ] Kernel report card.
- [ ] Submission checklist.
- [ ] README update with final commands and expected outputs.
- [ ] Troubleshooting section from actual observed failures.

## v1 Results Placeholder

Fill after Phase 2 full isolation benchmark.

| Question | Result | Decision |
|---|---|---|
| Does Phase 1 correctness pass? | TBD | TBD |
| Forge vs PyTorch speedup at reference config | TBD | TBD |
| Forge vs Liger at reference config | TBD | TBD |
| Forge vs Unsloth direct RMSNorm | TBD | TBD |
| Peak memory bandwidth utilization | TBD | TBD |
| Any shape regresses by more than 5 percent? | TBD | TBD |
| Atomic dweight better for QK-norm shapes? | TBD | TBD |
| Scratch dweight better for long-context shapes? | TBD | TBD |
| Forge auto heuristic better than forced scratch? | TBD | Use Reducer Policy Comparison by shape/GPU |
| `cache_rstd=False` worth changing default? | TBD | Use cache ablation horizon; compare `vs Cached` by shape/GPU |
| Any autotune cold-start issue? | TBD | TBD |

## v1.5 Candidate Work

Choose from this list only after real benchmark results land.

- [ ] Fix any Phase 1 correctness failures.
- [x] Add a public debug flag to force atomic vs scratch dweight reduction.
- [ ] Add persistent autotune-choice cache keyed by GPU and kernel key.
- [ ] Add public API control for autotune policy, e.g. heuristic/default vs
  force-autotune, so users can explicitly pay cold-start cost when they expect
  enough repeated calls to amortize it.
- [ ] Add performance regression guard using achieved bandwidth thresholds.
- [ ] Revisit `cache_rstd` default from measured memory/time tradeoff.
- [ ] Add grid-multiplier autotune for backward if bandwidth utilization shows
  latency-hiding headroom.
- [ ] Add per-call-site specialization if QK-norm or final norm still has a gap.
- [ ] Improve reduce-strategy heuristic if measurements disagree with L2-fit.
- [ ] Broaden gamma-fold recipes beyond current Llama-like patterns.
- [ ] Add explicit quantized-linear refusal coverage for gamma folding.
- [ ] Add optional approximate Gemma fp16/bf16 fold validation against real
  model outputs if that path is worth keeping.

## v2 And Later Candidate Work

- [ ] Streaming reduction for hidden sizes above Triton's fused block limit.
- [ ] RMSNorm + residual add fusion.
- [ ] RMSNorm + RoPE fusion.
- [ ] RMSNorm + `lm_head` specialization or fusion.
- [ ] Matmul epilogue writes output plus row sum-of-squares.
- [ ] Full RMSNorm in matmul epilogue.
- [ ] Add adaptive autotune policy once final benchmarks quantify cold-start
  cost vs steady-state speedup. Candidate behavior: start with heuristic kernels,
  track repeated calls per shape/GPU key, and trigger Triton autotune once call
  count exceeds a threshold `x`; choose `x` from measured amortization data.
- [ ] AMD/ROCm tuning pass after the NVIDIA path is stable.
