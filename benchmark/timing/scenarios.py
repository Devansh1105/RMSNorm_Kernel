"""Benchmark horizons for apples-to-apples RMSNorm timing."""
from __future__ import annotations

from dataclasses import dataclass

from benchmark.timing.configs import DEFAULT_EPS, TimingConfig, full_configs, quick_configs


@dataclass(frozen=True)
class Horizon:
    id: str
    name: str
    description: str
    eligible: tuple[str, ...]
    operation: str
    weight_mode: str
    expected_grads: tuple[str, ...]
    in_place: bool
    casting_mode: str
    offset: float
    full_sweep: bool
    cache_rstd: bool = True
    reduce_strategy: str = "auto"
    include_extra_work: tuple[str, ...] = ()

    @property
    def is_backward(self) -> bool:
        return self.operation in {"backward_full", "backward_dx_only"}


HORIZONS = [
    Horizon(
        id="standard_forward_affine",
        name="Standard Forward (Affine)",
        description="Normal RMSNorm forward with gamma.",
        eligible=("PyTorch", "Forge", "Liger", "Unsloth"),
        operation="forward",
        weight_mode="affine",
        expected_grads=(),
        in_place=False,
        casting_mode="llama",
        offset=0.0,
        full_sweep=True,
    ),
    Horizon(
        id="full_training_backward_safe",
        name="Full Training Backward (Safe)",
        description="Full trainable RMSNorm backward: dX + dGamma, in_place=False.",
        eligible=("PyTorch", "Forge", "Liger"),
        operation="backward_full",
        weight_mode="trainable_gamma",
        expected_grads=("x", "gamma"),
        in_place=False,
        casting_mode="llama",
        offset=0.0,
        full_sweep=True,
    ),
    Horizon(
        id="full_training_backward_inplace",
        name="Full Training Backward (In-Place Speed Mode)",
        description="Full trainable RMSNorm backward: dX + dGamma, in_place=True. Alias-sensitive speed mode.",
        eligible=("Forge", "Liger"),
        operation="backward_full",
        weight_mode="trainable_gamma",
        expected_grads=("x", "gamma"),
        in_place=True,
        casting_mode="llama",
        offset=0.0,
        full_sweep=True,
    ),
    Horizon(
        id="cache_ablation_recompute_rstd_safe_backward",
        name="Cache Ablation: Recompute RSTD Backward",
        description="Full trainable Forge backward with cache_rstd=False, in_place=False. Compares against cached RSTD.",
        eligible=("Forge",),
        operation="backward_full",
        weight_mode="trainable_gamma",
        expected_grads=("x", "gamma"),
        in_place=False,
        casting_mode="llama",
        offset=0.0,
        full_sweep=True,
        cache_rstd=False,
    ),
    Horizon(
        id="frozen_gamma_backward_dx_only",
        name="Frozen-Gamma Backward (dX Only)",
        description="LoRA-style backward where gamma is frozen and only dX is required.",
        eligible=("PyTorch", "Unsloth"),
        operation="backward_dx_only",
        weight_mode="frozen_gamma",
        expected_grads=("x",),
        in_place=True,
        casting_mode="llama",
        offset=0.0,
        full_sweep=True,
        include_extra_work=("Forge", "Liger"),
    ),
    Horizon(
        id="reducer_ablation_atomic_safe_backward",
        name="Reducer Ablation: Atomic Safe Backward",
        description="Full trainable RMSNorm backward with Forge dGamma reducer forced to atomic, in_place=False.",
        eligible=("Forge", "Liger"),
        operation="backward_full",
        weight_mode="trainable_gamma",
        expected_grads=("x", "gamma"),
        in_place=False,
        casting_mode="llama",
        offset=0.0,
        full_sweep=True,
        reduce_strategy="atomic",
    ),
    Horizon(
        id="reducer_ablation_scratch_safe_backward",
        name="Reducer Ablation: Scratch Safe Backward",
        description="Full trainable RMSNorm backward with Forge dGamma reducer forced to scratch, in_place=False.",
        eligible=("Forge", "Liger"),
        operation="backward_full",
        weight_mode="trainable_gamma",
        expected_grads=("x", "gamma"),
        in_place=False,
        casting_mode="llama",
        offset=0.0,
        full_sweep=True,
        reduce_strategy="scratch",
    ),
    Horizon(
        id="reducer_ablation_atomic_inplace_backward",
        name="Reducer Ablation: Atomic In-Place Backward",
        description="Full trainable RMSNorm backward with Forge dGamma reducer forced to atomic, in_place=True.",
        eligible=("Forge", "Liger"),
        operation="backward_full",
        weight_mode="trainable_gamma",
        expected_grads=("x", "gamma"),
        in_place=True,
        casting_mode="llama",
        offset=0.0,
        full_sweep=True,
        reduce_strategy="atomic",
    ),
    Horizon(
        id="reducer_ablation_scratch_inplace_backward",
        name="Reducer Ablation: Scratch In-Place Backward",
        description="Full trainable RMSNorm backward with Forge dGamma reducer forced to scratch, in_place=True.",
        eligible=("Forge", "Liger"),
        operation="backward_full",
        weight_mode="trainable_gamma",
        expected_grads=("x", "gamma"),
        in_place=True,
        casting_mode="llama",
        offset=0.0,
        full_sweep=True,
        reduce_strategy="scratch",
    ),
    Horizon(
        id="folded_gamma_forward_no_gamma",
        name="Folded-Gamma Forward (No Gamma Parameter)",
        description="Post-fold/no-affine RMSNorm forward with weight=None.",
        eligible=("PyTorch", "Forge", "Liger"),
        operation="forward",
        weight_mode="no_gamma",
        expected_grads=(),
        in_place=False,
        casting_mode="llama",
        offset=0.0,
        full_sweep=False,
    ),
    Horizon(
        id="no_gamma_backward_dx_only",
        name="No-Gamma Backward (dX Only)",
        description="RMSNorm backward with no gamma parameter, so only dX is required.",
        eligible=("PyTorch", "Forge", "Liger"),
        operation="backward_dx_only",
        weight_mode="no_gamma",
        expected_grads=("x",),
        in_place=False,
        casting_mode="llama",
        offset=0.0,
        full_sweep=False,
    ),
    Horizon(
        id="casting_mode_llama_forward",
        name="Casting Semantics: Llama Forward",
        description="Affine forward using Llama casting semantics.",
        eligible=("PyTorch", "Forge", "Liger", "Unsloth"),
        operation="forward",
        weight_mode="affine",
        expected_grads=(),
        in_place=False,
        casting_mode="llama",
        offset=0.0,
        full_sweep=False,
    ),
    Horizon(
        id="casting_mode_gemma_forward",
        name="Casting Semantics: Gemma Forward",
        description="Affine forward using Gemma casting semantics and offset=1.0.",
        eligible=("PyTorch", "Forge", "Liger", "Unsloth"),
        operation="forward",
        weight_mode="affine",
        expected_grads=(),
        in_place=False,
        casting_mode="gemma",
        offset=1.0,
        full_sweep=False,
    ),
    Horizon(
        id="casting_mode_none_forward",
        name="Casting Semantics: No-Cast Forward",
        description="Affine forward using no-cast RMSNorm semantics.",
        eligible=("PyTorch", "Forge", "Liger"),
        operation="forward",
        weight_mode="affine",
        expected_grads=(),
        in_place=False,
        casting_mode="none",
        offset=0.0,
        full_sweep=False,
    ),
]


HORIZON_BY_ID = {horizon.id: horizon for horizon in HORIZONS}


def configs_for_horizon(mode: str, horizon: Horizon) -> list[TimingConfig]:
    if mode == "quick":
        return quick_configs()
    if mode != "full":
        raise ValueError(f"unknown benchmark mode: {mode}")
    return full_configs() if horizon.full_sweep else quick_configs()


def canary_config() -> TimingConfig:
    return TimingConfig("canary", m=8, n=16)


def horizon_eps(_: Horizon) -> float:
    return DEFAULT_EPS
