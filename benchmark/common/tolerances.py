"""Tolerance and comparison helpers for correctness checks."""
from __future__ import annotations


THRESHOLDS = {
    "float32": {"atol": 1e-5, "rtol": 1e-5},
    "float16": {"atol": 1e-3, "rtol": 1e-3},
    "bfloat16": {"atol": 1e-3, "rtol": 1e-3},
}

# Folding moves gamma into a later matmul. Algebra is equivalent, but rounding
# boundaries change, especially in bf16/fp16.
FOLD_LOW_PRECISION_THRESHOLD = {"atol": 2e-2, "rtol": 2e-2}

# No-cast mode intentionally keeps reductions in input dtype. Low precision
# drift is much larger than Llama/Gemma fp32-accumulation modes.
NO_CAST_LOW_PRECISION_THRESHOLD = {"atol": 1e-1, "rtol": 1e-1}


def dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


def threshold_for(dtype, *, casting_mode: str | None = None, fold: bool = False) -> dict[str, float]:
    name = dtype_name(dtype)
    if fold and name in ("float16", "bfloat16"):
        return FOLD_LOW_PRECISION_THRESHOLD
    if casting_mode == "none" and name in ("float16", "bfloat16"):
        return NO_CAST_LOW_PRECISION_THRESHOLD
    return THRESHOLDS[name]


def max_atol_rtol(torch, got, ref) -> tuple[float, float]:
    diff = (got - ref).abs().to(torch.float32)
    denom = ref.abs().to(torch.float32).clamp_min(1e-12)
    return diff.max().item(), (diff / denom).max().item()


def is_close(torch, got, ref, threshold: dict[str, float]) -> bool:
    diff = (got - ref).abs().to(torch.float32)
    limit = threshold["atol"] + threshold["rtol"] * ref.abs().to(torch.float32)
    return bool(torch.all(diff <= limit).item())


def verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def format_tolerance(threshold: dict[str, float]) -> str:
    return f"{threshold['atol']:.0e}/{threshold['rtol']:.0e}"
