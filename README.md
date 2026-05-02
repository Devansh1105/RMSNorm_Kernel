# fast_rmsnorm

A Triton RMSNorm kernel (forward + backward) that aims to match or beat
[Liger-Kernel](https://github.com/linkedin/Liger-Kernel) across realistic LLM
shapes on A100/H100, plus an inference-time γ-fold path
([FlashNorm](https://arxiv.org/abs/2407.09577)) for additional speedup.

**Status:** v1 code complete. Pending Colab T4 correctness verification and
A100 performance numbers. v1.5 / v2 backlog at the bottom.

## What RMSNorm computes

Per token row $x \in \mathbb{R}^N$ with weight $\gamma \in \mathbb{R}^N$:

$$
m = \tfrac{1}{N}\sum_i x_i^2,\quad
s = \tfrac{1}{\sqrt{m + \varepsilon}},\quad
y_i = (\text{offset} + \gamma_i)\, x_i\, s
$$

`offset = 0` for Llama, `offset = 1` for Gemma. Backward is in
`docs`-style notation in `ops/rms_norm.py`.

## Install / use

```bash
pip install -e .
```

```python
import torch
from fast_rmsnorm.transformers import FastRMSNorm

m = FastRMSNorm(hidden_size=4096).cuda().bfloat16()
x = torch.randn(8, 1024, 4096, device='cuda', dtype=torch.bfloat16)
y = m(x)
```

Or as a function:

```python
from fast_rmsnorm.transformers import rms_norm
y = rms_norm(x, weight, eps=1e-6, mode='auto')
```

Module-level knobs that map to kernel constexprs:
- `mode`         — `'train'` (autotune) | `'infer'` (heuristic ladder) | `'auto'` (picks by `requires_grad`)
- `cache_rstd`   — `True` (default) caches rstd; `False` recomputes in bwd (for activation checkpointing)
- `casting_mode` — `'llama'` | `'gemma'` | `'none'` (Liger's three-mode design)
- `offset`       — `0.0` (Llama) | `1.0` (Gemma's `(1+γ)`)
- `in_place`     — `True` (default) reuses dY storage for dX in backward

## Layout (mirrors Liger)

```
src/fast_rmsnorm/
  ops/
    rms_norm.py           # Triton kernels + dispatch + autograd Function
    utils.py              # heuristics, dtype map, casting modes, REDUCE_STRATEGY
  transformers/
    rms_norm.py           # FastRMSNorm nn.Module
    functional.py         # rms_norm() functional API
    folding.py            # γ-fold utility for inference (FlashNorm)

test/transformers/
  test_rms_norm.py        # correctness vs PyTorch reference (Colab T4 OK)
  test_folding.py         # numerical-equivalence tests for the fold path
  reference.py            # PyTorch reference impls

benchmark/scripts/
  benchmark_rms_norm.py   # perf comparison vs Liger (A100/H100)

refs/liger-kernel/        # pristine Liger source, kept for line-by-line reference
```

## What's in v1 (implemented)

Every delta below is in code; correctness pending GPU verification.

| ID | Delta | Why it might help | Where it lives |
|---|---|---|---|
| **E1** | `@triton.autotune` wraps each kernel as a `*_at` variant. Tunes `num_warps × num_stages` for row-impl; tunes `BLOCK_ROW × num_warps × num_stages` for block-impl backward (Liger hard-codes `BLOCK_ROW=16`). `BLOCK_SIZE` stays forced to `next_pow2(N)`. Wrapper exposes `mode='train'/'infer'/'auto'` so inference paths can skip cold-start. | Liger and Unsloth use a static `if BLOCK_SIZE >= 32768: nw=32 elif ...` ladder regardless of GPU/dtype/M. Autotune picks per-shape. The block-impl `BLOCK_ROW` autotune is where QK-norm wins are likely. | `ops/rms_norm.py` (`_FORWARD_AUTOTUNE_CONFIGS`, `_BACKWARD_AUTOTUNE_CONFIGS`, `_BLOCK_BACKWARD_AUTOTUNE_CONFIGS`) |
| **E2** | `REDUCE_STRATEGY: tl.constexpr` branches dgamma write between `tl.atomic_add` (single fp32 buffer) and Liger's scratch `[num_blocks, N]` + final sum. Wrapper picks at runtime via L2-fit heuristic: `num_blocks * N * 4 < 0.5 * L2_size → ATOMIC`. | Atomic wins on H100 + small working set; scratch wins for huge M (long context). Liger always uses scratch. | `ops/rms_norm.py` (kernel branches), `ops/utils.py::pick_reduce_strategy` |
| **E3** | Fused dx + per-block dγ partial in one backward kernel. | Match Liger; do **not** regress to Unsloth (which omits dγ from Triton, costing one extra M×N HBM read of dY through PyTorch autograd). | `ops/rms_norm.py` |
| **E6** | `eps = eps.to(tl.float32)` at every kernel entry. | No-op on NVIDIA. Defensive against type-promotion bugs in `CASTING_NONE` on AMD/HIP. | `ops/rms_norm.py` |
| **I1** | `fold_rmsnorm_gamma_into_next_linear(model, arch=...)` — pre-multiplies `(offset + γ)` into the next Linear's weight at model-load. Kernel then runs as a pure normalize via `_gamma_folded` flag on the Module. Per-architecture recipes for Llama / Mistral / Qwen2 / Qwen3 / Gemma. Refuses on `requires_grad=True`, quantized weights, non-Linear next op, or already-folded module. | Inference only; saves ~33% of the kernel's HBM traffic (skips the γ load). FlashNorm (Graef 2024). | `transformers/folding.py` |
| **I3** | `cache_rstd: bool = True` flag. When `False`, fwd kernel skips the rstd store (`STORE_RSTD: tl.constexpr`) and bwd recomputes from X (`RECOMPUTE_RSTD: tl.constexpr`). | Useful with activation checkpointing (X is recomputed anyway, so rstd is "free"). Documented, not advertised. | `ops/rms_norm.py` |

### Tests already written

`test/transformers/test_rms_norm.py`
- All 18 (shape × dtype × casting_mode × with_weight × offset) combinations of forward
  vs PyTorch reference, both row-impl and block-impl dispatch paths.
- Backward (dx + dγ) for a focused subset.
- `in_place=True` matches `in_place=False`.
- `mode='train'/'infer'/'auto'` consistency.
- `cache_rstd=False` matches `cache_rstd=True` for fwd + bwd.
- `REDUCE_STRATEGY` correctness on shapes that hit ATOMIC vs SCRATCH.

`test/transformers/test_folding.py`
- Build a mini-Llama, fold γ, assert output unchanged within bf16 tolerance.
- Refusal under `requires_grad=True`.
- Idempotency (no double-fold).
- Gemma-style `(1+γ)` offset path.

## What's NOT in v1 (deferred)

These are deliberately out — either out of scope, awaiting v1 measurements, or
needing more design work.

### Dropped from v1 (with reasoning)

- **E4 — performance regression guard (achieved-HBM-bandwidth threshold + alignment audit).**
  Useful but reviewer-flagged: the assembly inspection part was overclaiming
  ("4× latency", brittle bf16 PTX assertions). Deferred until v1 numbers exist;
  then we add a real bandwidth-threshold test as the regression guard.
- **E5 — persistent kernel for launch overhead.**
  Misframed in the brainstorm: the per-launch overhead in a forward pass is
  *between* RMSNorm and other ops (attention/MLP), not within one RMSNorm call.
  CUDA graphs (caller-side) and Phase-2 op-fusion (v2 below) are the real fixes.
- **E7 — streaming BLOCK_N tile loop for N > 32 K.**
  Not needed today; modern LLM hidden ≤ 16 K. Becomes relevant when we fuse
  `RMSNorm + lm_head` (vocab dim ~128 K). Wrapper currently asserts `N ≤ 32 K`.

### v1.5 backlog (after Colab + A100 verification)

- **I4 — per-call-site specialization.** Decide based on measured Liger-vs-ours
  numbers on Qwen3 QK-norm. If E1's `BLOCK_ROW` autotune already closes the gap,
  drop. If a measurable gap remains for the lm_head call (M tiny, latency-bound),
  add a `lmhead` specialization sharing source via constexpr branching.
- **Grid-multiplier autotune for backward.** Currently `grid=(sm_count,)`
  hard-coded for both row- and block-impl backward. Multiplying by 2 (so 2
  programs per SM concurrent) buys latency hiding *but* costs ~1.5% extra
  HBM traffic (more dW scratch / more atomics / extra final reduction).
  **Two-condition gate before implementing:**
    1. A100 bench shows `ours_train` bwd < 85% of peak HBM bandwidth (room
       for latency hiding to clear the overhead).
    2. Autotune is consistently picking max `num_warps=32` for that shape —
       indicating warp-level parallelism inside one program is already
       maxed and adding cross-program concurrency could help.
  If either fails, skip — the win won't clear the overhead, and in the
  saturated regime grid_mult=2 *regresses* perf. Prune to `grid_mult ∈ {1, 2}`
  if we add it; 4× rarely wins and doubles cold-start.
- **Default `cache_rstd=False`?** Recompute path is wired and tested, but
  defaults to caching. Bench cache vs recompute on A100; if recompute is
  within 2% of cache, flip the default — saves M fp32 scalars of HBM traffic
  per layer per direction at essentially-free compute cost.
- **E4 reinstated** — a real performance regression test once we have bandwidth
  numbers to threshold against.
- **Persistent autotune cache to disk.** Triton's compile cache already
  persists; the autotune *choice* cache is in-process. Pickle keyed on
  `(GPU UUID, kernel_key)` so a given GPU pays the cold-start cost once
  across runs — not just once per process.
- **Broader architecture coverage for the fold utility** — Mistral, Phi-3,
  DeepSeek, Gemma-3 explicit recipes (current `_RECIPES` table maps them to
  `_llama_pairs` which works in most but not all cases).

### v2 backlog

- **I2 — streaming Σx² for N > BLOCK_SIZE_max.** Pairs naturally with the
  fused `RMSNorm + lm_head` (vocab dim ~128 K) below.
- **I5 — matmul-epilogue fusion.**
  - Phase 2a: matmul writes `(output, row_sum_of_squares)`; RMSNorm skips its
    first pass. Saves one M×N HBM read per layer.
  - Phase 2b: full RMSNorm in matmul epilogue. Saves both M×N reads. Couples
    RMSNorm tightly to the matmul kernel.
- **Other fusion variants:** `RMSNorm + residual add`, `RMSNorm + RoPE`.
- **AMD/ROCm port.** Triton portability is partial; some constexpr branches
  need HIP-specific tuning. Discuss after the NVIDIA path is stable.

## Verification

```bash
# Correctness — Colab T4 is enough
pytest test/ -x -q

# Performance vs Liger — needs A100/H100 for meaningful numbers
pip install liger-kernel
python -m benchmark.scripts.benchmark_rms_norm           # full sweep
python -m benchmark.scripts.benchmark_rms_norm --quick   # smoke
```

The benchmark sweep covers:
- Llama-3 pre-block (M ∈ {1024, 8192, 65536}, N=4096)
- Qwen3 QK-norm (M ∈ {65536, 262144}, N=128)
- Gemma offset variant (`casting_mode='gemma'`, offset=1.0)

For each, prints fwd ms (ours_infer / ours_train / Liger), bwd ms (ours_train / Liger),
and achieved HBM GB/s with utilization vs the device's advertised peak.

## Known limitations

- `N > 32768` (= MAX_FUSED_SIZE / 2 for fp16) raises — streaming fallback is v1.5 work.
- γ-fold is inference-only; the utility refuses if any param requires grad.
- γ-fold currently only handles `nn.Linear` next-ops (no quantized linears).
- Persistent autotune cache is in-process only (Triton default); v1.5 will pickle to disk.
- Liger's `DTensor` (tensor-parallel) handling is preserved in the autograd Function but untested in this fork.

## References

- Zhang & Sennrich 2019, *Root Mean Square Layer Normalization* — [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- Graef 2024, *FlashNorm* — [arXiv:2407.09577](https://arxiv.org/abs/2407.09577)
- Hsu et al. 2024, *Liger Kernel* — [arXiv:2410.10989](https://arxiv.org/abs/2410.10989)
- Milakov & Gimelshein 2018, *Online softmax* — [arXiv:1805.02867](https://arxiv.org/abs/1805.02867) (relevant for v2 streaming)
- Jiang et al. 2023, *Pre-RMSNorm equivalences* — [arXiv:2305.14858](https://arxiv.org/abs/2305.14858)

## Credits

Built on Liger-Kernel and Unsloth (both Apache 2.0). FlashNorm idea from Graef 2024.
