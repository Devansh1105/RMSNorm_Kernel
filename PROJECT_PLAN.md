# fast_rmsnorm Project Plan

This is the living checklist for the RMSNorm kernel, benchmark suite, and next
versions. Update this file whenever benchmark results land or a version scope
changes.

## Context

- Research docs:
  - `RMSNorm_ Forward, Backward and Intermidiates.pdf`
  - `Forge_Kernel_Testing__Benchmarking_Rules.pdf`
- Goal: standalone Triton RMSNorm forward + backward that matches or beats
  Liger on realistic LLM shapes, plus inference-time gamma folding. Later
  versions should extend this into fused variants where measurements show real
  value, such as RMSNorm plus residual add, RoPE, `lm_head`, or matmul epilogues.
- Why it matters: decoder-only LLMs call RMSNorm many times per forward pass,
  so small per-call wins can matter at model scale.
- Current package: `fast_rmsnorm`
- Reference implementation source: `refs/liger-kernel/`
- Main kernel file: `src/fast_rmsnorm/ops/rms_norm.py`
- Main benchmark runner: `benchmark/rmsnorm_forge_suite.py`

## Current Status Snapshot

- [x] v1 kernel implementation exists.
- [x] Old pytest tests were removed from the active workflow.
- [x] Old prototype benchmark script was removed from the active workflow.
- [x] Forge-style Phase 1 benchmark suite exists.
- [ ] Colab GPU correctness pass confirmed after the latest fixes.
- [ ] Colab quick isolation report generated.
- [ ] RunPod A100/H100 full isolation report generated.
- [ ] v1 benchmark results copied into this file.
- [ ] v1.5 scope finalized from actual measurements.
- [ ] Phase 2 model-level benchmark implemented.

## Result Placeholders

Fill these in after each benchmark run.

| Run | GPU | Command | Result JSON | Report MD | Status | Notes |
|---|---|---|---|---|---|---|
| Colab correctness | TBD | `python -m benchmark.rmsnorm_forge_suite correctness --quick` | TBD | N/A | Pending | Expected: no FAIL, fp64 gradcheck may be BLOCKED |
| Colab quick isolation | TBD | `python -m benchmark.rmsnorm_forge_suite isolation --quick` | TBD | TBD | Pending | Smoke only, not final performance |
| RunPod full isolation | TBD | `python -m benchmark.rmsnorm_forge_suite isolation --full --strict-competitors` | TBD | TBD | Pending | Forge Part A final report |
| Model-level benchmark | TBD | TBD | TBD | TBD | Pending | Phase 2 |

## v1 Kernel Checklist

### Implemented

- [x] Forked Liger-style RMSNorm structure into `fast_rmsnorm`.
- [x] Forward kernel supports row-per-program path.
- [x] Forward kernel supports block path for small hidden dimension and many rows.
- [x] Backward kernel computes `dx` and `dweight` partials in one pass.
- [x] Supports Llama-style casting mode.
- [x] Supports Gemma-style offset and casting mode.
- [x] Supports no-cast mode.
- [x] Supports non-affine no-weight path.
- [x] Supports `mode="train"`, `mode="infer"`, and `mode="auto"`.
- [x] Adds Triton autotune variants for training mode.
- [x] Autotunes `num_warps` and `num_stages` for row kernels.
- [x] Autotunes `BLOCK_ROW` for block kernels.
- [x] Adds dweight reduction strategy branch: scratch buffer or atomic add.
- [x] Picks dweight reduction strategy from a runtime L2-fit heuristic.
- [x] Pins epsilon to fp32 inside Triton kernels.
- [x] Adds `cache_rstd` flag.
- [x] Adds inference-only gamma folding utility.
- [x] Adds module and functional API wrappers.

### Needs Testing

- [ ] Colab T4 correctness for forward and backward after latest changes.
- [ ] Colab T4 correctness for gamma folding after latest harness tolerance fix.
- [ ] A100/H100 correctness for full shape matrix.
- [ ] A100/H100 isolation performance vs PyTorch, Liger, and Unsloth.
- [ ] Atomic vs scratch dweight strategy behavior on A100/H100.
- [ ] `cache_rstd=True` vs `cache_rstd=False` performance.
- [ ] `mode="auto"` behavior in inference models where params still have
  `requires_grad=True`.
- [ ] Gamma folding output equivalence on at least Llama-like and Qwen-like
  structures.

### Known Issues Or Watch Items

- [ ] fp64 Forge gradcheck is blocked because the current Triton dtype map only
  supports fp32/fp16/bf16.
- [ ] Backward autotune side effects were patched with `reset_to_zero` and
  `restore_value`; this needs GPU retesting.
- [ ] Direct Unsloth RMSNorm adapter may depend on the installed Unsloth API.
- [ ] `mode="auto"` may choose train/autotune when weights still require grad,
  even in `model.eval()`.
- [ ] Feature dims above the fused Triton limit still require future streaming
  support.

## Forge Benchmark Suite Checklist

### Implemented

- [x] Single canonical runner: `benchmark/rmsnorm_forge_suite.py`.
- [x] `correctness` subcommand.
- [x] `isolation` subcommand.
- [x] `report` subcommand.
- [x] `model` placeholder subcommand for Phase 2.
- [x] CUDA event timing.
- [x] Warmup runs and timed runs.
- [x] L2 flush before timed runs.
- [x] Device-aware L2 flush size instead of fixed 32 MB only.
- [x] Mean, median, p50, p95, p99, std, min, max.
- [x] Sequence length sweep.
- [x] Batch size sweep.
- [x] Hidden dimension sweep.
- [x] Peak VRAM measurement.
- [x] Arithmetic intensity calculation.
- [x] Peak utilization calculation.
- [x] Roofline classification.
- [x] Markdown report generation.
- [x] JSON result dump.
- [x] Missing competitor warning path.
- [x] Strict competitor mode for final runs.
- [x] Colab and RunPod runbook in `benchmark/README.md`.

### Needs Testing

- [ ] `correctness --quick` on Colab GPU.
- [ ] `isolation --quick` on Colab GPU.
- [ ] `isolation --full --strict-competitors` on A100/H100.
- [ ] Generated report visually checked against Forge template.
- [ ] Missing Unsloth behavior checked in a real benchmark environment.
- [ ] JSON-to-markdown regeneration checked from real output.
- [ ] Report sanity warning for measured GB/s above peak checked.

### Not Implemented Yet

- [ ] Forge Part B model-level benchmark.
- [ ] LoRA SFT step-time measurement.
- [ ] Loss-curve convergence check at fixed steps.
- [ ] HuggingFace model patching utilities.
- [ ] Model-level PyTorch vs Liger vs Unsloth vs Forge comparison.
- [ ] Optional C++/CUDA competitor column.

## Benchmark Runbook

### 1. Colab Correctness Gate

Run on Colab GPU before paid GPU work.

```bash
pip install -e .[bench]
python -m benchmark.rmsnorm_forge_suite correctness --quick
```

Expected:

- No `FAIL` rows.
- `gradcheck_fp64` may remain `BLOCKED` for v1.

If it fails:

```bash
python -m benchmark.rmsnorm_forge_suite correctness --quick --output /tmp/rmsnorm_correctness.json
```

Copy the failure summary and keep the JSON.

### 2. Colab Quick Isolation

Run only after correctness has no FAIL rows.

```bash
python -m benchmark.rmsnorm_forge_suite isolation --quick
```

Expected outputs:

- `benchmark/results/isolation_<gpu>_<timestamp>.json`
- `benchmark/results/isolation_<gpu>_<timestamp>.md`

### 3. RunPod Full Isolation

Run on A100/H100 after Colab correctness and quick isolation pass.

```bash
pip install -e .[bench]
python -m benchmark.rmsnorm_forge_suite isolation --full --strict-competitors
```

If strict competitors fails because Unsloth has no direct RMSNorm adapter, run
once without strict mode and record the exact `NOT_RUN` reason.

```bash
python -m benchmark.rmsnorm_forge_suite isolation --full
```

### 4. Report Regeneration

```bash
python -m benchmark.rmsnorm_forge_suite report \
  --input benchmark/results/isolation_<gpu>_<timestamp>.json
```

## v1 Result Summary Placeholder

Fill after the RunPod full isolation report.

| Question | Result | Decision |
|---|---|---|
| Does correctness pass for Forge thresholds? | TBD | TBD |
| Forge vs PyTorch speedup at reference config | TBD | TBD |
| Forge vs Liger at reference config | TBD | TBD |
| Forge vs Unsloth direct RMSNorm | TBD | TBD |
| Peak memory bandwidth utilization | TBD | TBD |
| Any shape regresses by more than 5 percent? | TBD | TBD |
| Atomic dweight better for QK-norm shapes? | TBD | TBD |
| Scratch dweight better for long-context shapes? | TBD | TBD |
| `cache_rstd=False` worth changing default? | TBD | TBD |
| Any autotune cold-start issue? | TBD | TBD |

## v1.5 Candidate Work

These items should be selected based on v1 benchmark results.

- [ ] Fix any correctness or harness bugs found in the Colab gate.
- [ ] Add persistent autotune-choice cache keyed by GPU and kernel key.
- [ ] Add performance regression guard using achieved bandwidth thresholds.
- [ ] Decide whether to keep `cache_rstd=True` default or switch to recompute.
- [ ] Add grid-multiplier autotune for backward if bandwidth utilization shows
  latency-hiding headroom.
- [ ] Add per-call-site specialization if QK-norm or final norm still has a
  measurable gap.
- [ ] Improve reduce-strategy heuristic if atomic vs scratch results disagree
  with the L2-fit proxy.
- [ ] Broaden gamma-fold recipes beyond the current Llama-like pattern.
- [ ] Add explicit tests for quantized-linear refusal in gamma folding.

## Phase 2 Benchmark Work

This is Forge Part B.

- [ ] Add `benchmark/harness/patching.py` or equivalent model patch helpers.
- [ ] Patch HuggingFace RMSNorm modules with `FastRMSNorm`.
- [ ] Add Liger model patch path.
- [ ] Add Unsloth model-level path or document why only model-level comparison
  is valid for that package.
- [ ] Add LoRA SFT step-time benchmark.
- [ ] Add fixed-seed convergence check.
- [ ] Record loss at required checkpoints.
- [ ] Report step time mean, median, p50, p95, p99, std.
- [ ] Report tokens per second.
- [ ] Report peak VRAM.
- [ ] Add B1-B3 sections to generated report.
- [ ] Add model-level rows to kernel report card.

Target smoke model:

- [ ] Llama-3.2-1B or another small Llama-like model that fits the available
  Colab GPU.

Target final model:

- [ ] Qwen3-8B or the chosen Forge reference model on A100/H100.

## v2 Candidate Work

- [ ] Streaming sum-of-squares path for feature dims above current fused limit.
- [ ] RMSNorm plus `lm_head` specialization if measurements justify it.
- [ ] Matmul epilogue fusion, phase 2a: matmul emits row sum of squares.
- [ ] Matmul epilogue fusion, phase 2b: full RMSNorm in matmul epilogue.
- [ ] RMSNorm plus residual-add fusion.
- [ ] RMSNorm plus RoPE fusion.
- [ ] AMD/ROCm support and tuning.
- [ ] Optional C++/CUDA competitor comparison if a relevant competitor appears.

## Decision Log

Add dated entries here as results come in.

| Date | Decision | Evidence |
|---|---|---|
| TBD | TBD | TBD |

## Immediate Next Actions

- [ ] Rerun `python -m benchmark.rmsnorm_forge_suite correctness --quick` on
  Colab GPU after the latest backward autotune and harness fixes.
- [ ] If correctness has no FAIL rows, run quick isolation on Colab GPU.
- [ ] If quick isolation works, run full isolation on A100/H100.
- [ ] Paste result paths and key numbers into the placeholders above.
- [ ] Re-scope v1.5 from the measured bottlenecks.
