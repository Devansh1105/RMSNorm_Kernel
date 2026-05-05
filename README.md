# fast_rmsnorm

A Triton RMSNorm kernel with forward, backward, and an inference-only
FlashNorm-style gamma-fold path.

Status: v1 kernel code is present. The old pytest tests and prototype benchmark
have been replaced by a Forge-style benchmark suite under `benchmark/`.

## Install

```bash
pip install -e .
```

Benchmark dependencies:

```bash
pip install -e .[bench]
```

## Use

```python
import torch
from fast_rmsnorm.transformers import FastRMSNorm

module = FastRMSNorm(hidden_size=4096, eps=1e-6).cuda().bfloat16()
x = torch.randn(8, 1024, 4096, device="cuda", dtype=torch.bfloat16)
y = module(x)
```

Functional API:

```python
from fast_rmsnorm.transformers import rms_norm

y = rms_norm(x, weight, eps=1e-6, mode="auto")
```

Important knobs:

- `mode`: `"train"` uses Triton autotune, `"infer"` uses the heuristic ladder, `"auto"` picks from `requires_grad`.
- `cache_rstd`: default `True`; set `False` to recompute rstd in backward.
- `casting_mode`: `"llama"`, `"gemma"`, or `"none"`.
- `offset`: `0.0` for Llama-style RMSNorm, `1.0` for Gemma-style `(1 + weight)`.
- `in_place`: default `True`; reuses backward gradient storage for `dX`.

## Layout

```text
src/fast_rmsnorm/
  ops/
    rms_norm.py          # Triton kernels, dispatch, autograd Function
    utils.py             # dtype/casting helpers and heuristics
  transformers/
    rms_norm.py          # FastRMSNorm nn.Module
    functional.py        # rms_norm() helper
    folding.py           # inference-only gamma folding utility

benchmark/
  rmsnorm_forge_suite.py # correctness + Forge Part A isolation suite
  README.md              # Colab and RunPod runbook

refs/liger-kernel/       # reference Liger source
```

## Verification

Use a CUDA GPU runtime. Triton does not run on Colab TPU.

Initial Colab correctness gate:

```bash
python -m benchmark.rmsnorm_forge_suite correctness --quick
```

Quick Colab isolation smoke:

```bash
python -m benchmark.rmsnorm_forge_suite isolation --quick
```

The benchmark dtype defaults to `auto`: bf16 when supported, otherwise fp16 for
T4-style smoke runs.

Full RunPod/A100 isolation report:

```bash
python -m benchmark.rmsnorm_forge_suite isolation --full --strict-competitors
```

The suite writes JSON and markdown reports under `benchmark/results/`. See
[benchmark/README.md](benchmark/README.md) for the full runbook.

## Implemented v1 Deltas

- Autotuned forward/backward variants for training mode.
- Block-row autotune for the small-hidden, many-row path.
- Runtime dgamma reduction strategy selection: scratch buffer or atomic add.
- Single backward kernel path for `dX` plus `dweight` partials.
- fp32 epsilon pinning in the Triton kernels.
- `cache_rstd` flag for activation-checkpointing workflows.
- Inference-only gamma folding into following `nn.Linear` weights.

## Known Limitations

- Feature dimensions above Triton's fused block limit raise; streaming fallback is future work.
- Forge fp64 `gradcheck` is reported as blocked because the current kernel dtype map supports fp32/fp16/bf16 only.
- Model-level Forge Part B benchmarking is planned as Phase 2.
- Gamma folding supports plain `nn.Linear` targets only; quantized linears are out of scope for v1.
- Persistent autotune-choice caching is not implemented yet.

## References

- Zhang & Sennrich 2019, Root Mean Square Layer Normalization.
- Graef 2024, FlashNorm.
- Hsu et al. 2024, Liger Kernel.
- Liger-Kernel source under `refs/liger-kernel/`.
