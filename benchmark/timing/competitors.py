"""Competitor adapters for RMSNorm isolation timing."""
from __future__ import annotations

import importlib
from dataclasses import dataclass

from benchmark.correctness.references import REFERENCES


class NotRunnableError(RuntimeError):
    """Raised when an optional competitor cannot run a requested operation."""


@dataclass(frozen=True)
class OperationRequest:
    eps: float
    offset: float
    casting_mode: str
    in_place: bool
    mode: str


class RMSNormAdapter:
    name = "base"

    def available(self) -> tuple[bool, str]:
        raise NotImplementedError

    def forward(self, x, weight, request: OperationRequest):
        raise NotImplementedError

    def backward(self, x, weight, grad_out, request: OperationRequest):
        x.grad = None
        if weight is not None:
            weight.grad = None
        out = self.forward(x, weight, request)
        import torch

        torch.autograd.backward(out, grad_out)
        return out

    def path(self, m: int, n: int, request: OperationRequest) -> str:
        return "framework"


class PyTorchAdapter(RMSNormAdapter):
    name = "PyTorch"

    def available(self) -> tuple[bool, str]:
        return True, "explicit PyTorch formulas"

    def forward(self, x, weight, request: OperationRequest):
        _, ref_fn = REFERENCES[request.casting_mode]
        import torch

        return ref_fn(torch, x, weight, request.eps, request.offset)

    def path(self, m: int, n: int, request: OperationRequest) -> str:
        return "torch-autograd" if request.mode == "train" else "torch-eager"


class ForgeAdapter(RMSNormAdapter):
    name = "Forge"

    def __init__(self):
        self._rms_norm = None

    def available(self) -> tuple[bool, str]:
        try:
            module = importlib.import_module("fast_rmsnorm.transformers")
            self._rms_norm = module.rms_norm
        except Exception as exc:  # noqa: BLE001 - availability should report import cause.
            return False, f"{type(exc).__name__}: {exc}"
        return True, "fast_rmsnorm.transformers.rms_norm"

    def _fn(self):
        if self._rms_norm is None:
            ok, reason = self.available()
            if not ok:
                raise RuntimeError(reason)
        return self._rms_norm

    def forward(self, x, weight, request: OperationRequest):
        return self._fn()(
            x,
            weight,
            request.eps,
            offset=request.offset,
            casting_mode=request.casting_mode,
            in_place=request.in_place,
            mode=request.mode,
            cache_rstd=True,
        )

    def path(self, m: int, n: int, request: OperationRequest) -> str:
        kernel_path = "block" if m >= 32768 and n <= 256 else "row"
        reduce_strategy = "-"
        if request.mode == "train":
            try:
                import torch

                from fast_rmsnorm.ops.utils import (
                    REDUCE_STRATEGY_ATOMIC,
                    pick_reduce_strategy,
                )

                sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
                reduce_strategy = (
                    "atomic"
                    if pick_reduce_strategy(sm_count, n, torch.device("cuda")) == REDUCE_STRATEGY_ATOMIC.value
                    else "scratch"
                )
            except Exception:
                reduce_strategy = "unknown"
        return f"forge-{kernel_path} mode={request.mode} in_place={request.in_place} reduce={reduce_strategy}"


class LigerAdapter(RMSNormAdapter):
    name = "Liger"

    def __init__(self):
        self._function = None

    def available(self) -> tuple[bool, str]:
        try:
            ops = importlib.import_module("liger_kernel.ops")
            self._function = ops.LigerRMSNormFunction
        except Exception as exc:  # noqa: BLE001 - availability should report import cause.
            return False, f"{type(exc).__name__}: {exc}"
        return True, "liger_kernel.ops.LigerRMSNormFunction"

    def _fn(self):
        if self._function is None:
            ok, reason = self.available()
            if not ok:
                raise NotRunnableError(reason)
        return self._function

    def forward(self, x, weight, request: OperationRequest):
        return self._fn().apply(
            x,
            weight,
            request.eps,
            request.offset,
            request.casting_mode,
            request.in_place,
        )

    def path(self, m: int, n: int, request: OperationRequest) -> str:
        kernel_path = "block" if m >= 32768 and n <= 256 else "row"
        return f"liger-{kernel_path} in_place={request.in_place}"


@dataclass
class _LayerNormShim:
    weight: object
    variance_epsilon: float

    @property
    def eps(self) -> float:
        return self.variance_epsilon


class UnslothAdapter(RMSNormAdapter):
    name = "Unsloth"

    def __init__(self):
        self._module = None

    def available(self) -> tuple[bool, str]:
        try:
            self._module = importlib.import_module("unsloth.kernels.rms_layernorm")
        except Exception as exc:  # noqa: BLE001 - availability should report import cause.
            return False, f"{type(exc).__name__}: {exc}"

        candidates = ("fast_rms_layernorm", "rms_layernorm", "Fast_RMS_Layernorm")
        if not any(hasattr(self._module, candidate) for candidate in candidates):
            return False, "unsloth.kernels.rms_layernorm has no known direct RMSNorm API"
        return True, "unsloth.kernels.rms_layernorm direct API"

    def _api_module(self):
        if self._module is None:
            ok, reason = self.available()
            if not ok:
                raise NotRunnableError(reason)
        return self._module

    def forward(self, x, weight, request: OperationRequest):
        if request.offset != 0.0:
            raise NotRunnableError("Unsloth direct adapter only supports offset=0.0 in this benchmark")
        module = self._api_module()
        layernorm = _LayerNormShim(weight=weight, variance_epsilon=request.eps)
        gemma = request.casting_mode == "gemma"
        errors = []

        for name in ("fast_rms_layernorm", "rms_layernorm"):
            fn = getattr(module, name, None)
            if fn is None:
                continue
            for args, kwargs in (
                ((layernorm, x), {"gemma": gemma}),
                ((layernorm, x), {}),
                ((x, weight, request.eps), {}),
            ):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - try next known signature.
                    errors.append(f"{name}: {type(exc).__name__}: {exc}")

        function = getattr(module, "Fast_RMS_Layernorm", None)
        if function is not None and hasattr(function, "apply"):
            for args in ((x, weight, request.eps), (x, weight, request.eps, gemma)):
                try:
                    return function.apply(*args)
                except Exception as exc:  # noqa: BLE001 - try next known signature.
                    errors.append(f"Fast_RMS_Layernorm.apply: {type(exc).__name__}: {exc}")

        detail = "; ".join(errors[-3:]) if errors else "no known callable found"
        raise NotRunnableError(f"Unsloth direct RMSNorm API incompatible: {detail}")

    def path(self, m: int, n: int, request: OperationRequest) -> str:
        return "unsloth-direct"

