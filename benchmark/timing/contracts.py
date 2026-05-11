"""Source-audited competitor API contracts for timing benchmarks."""
from __future__ import annotations


def adapter_contracts() -> dict[str, dict]:
    """Return static source/API contracts used by the timing adapters.

    These records intentionally describe the audited API surface, not what a
    particular installed environment successfully imports. Runtime availability
    and canaries still verify the installed packages.
    """
    return {
        "PyTorch": {
            "api": "benchmark.correctness.references explicit formulas + autograd",
            "source": "local benchmark reference formulas",
            "capabilities": "forward; dX; dGamma; affine; no-gamma; llama/gemma/none semantics",
            "caveats": "baseline formulas are not torch.nn.RMSNorm and are not optimized kernels",
        },
        "Forge": {
            "api": "fast_rmsnorm.transformers.rms_norm(x, weight, eps, ...)",
            "source": "local src/fast_rmsnorm/transformers/functional.py and src/fast_rmsnorm/ops/rms_norm.py",
            "capabilities": (
                "forward; dX; dGamma; affine; no-gamma; llama/gemma/none semantics; "
                "in_place; cache_rstd=True/False; train autotune; infer heuristic; reduce=auto/atomic/scratch"
            ),
            "caveats": "atomic/scratch reducer force is a benchmark/debug knob; default remains auto",
        },
        "Liger": {
            "api": "liger_kernel.ops.LigerRMSNormFunction.apply(X, W, eps, offset, casting_mode, in_place, row_mode)",
            "source": "github.com/linkedin/Liger-Kernel@ed6b2ffc src/liger_kernel/ops/rms_norm.py",
            "capabilities": "forward; dX; dGamma; affine; no-gamma; llama/gemma/none semantics; in_place",
            "caveats": "dGamma uses scratch-style per-SM partials plus torch sum reduction",
        },
        "Unsloth": {
            "api": "unsloth.kernels.rms_layernorm.fast_rms_layernorm(layernorm, X, gemma=False)",
            "source": "github.com/unslothai/unsloth@23cebfaf unsloth/kernels/rms_layernorm.py",
            "capabilities": "forward; dX-only backward; affine; llama semantics; gemma semantics via gemma=True",
            "caveats": "direct RMSNorm autograd returns no gamma grad and requires a real weight tensor",
        },
    }
