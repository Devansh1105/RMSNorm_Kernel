"""Competitor adapters for RMSNorm isolation timing."""
from __future__ import annotations

import importlib
from dataclasses import dataclass

from benchmark.correctness.references import REFERENCES


class NotRunnableError(RuntimeError):
    """Raised when an optional competitor is present but cannot run directly."""


class RMSNormAdapter:
    name = "base"

    def available(self) -> tuple[bool, str]:
        raise NotImplementedError

    def forward(self, x, weight, eps, offset, casting_mode):
        raise NotImplementedError

    def forward_backward(self, x, weight, grad_out, eps, offset, casting_mode):
        x.grad = None
        if weight is not None:
            weight.grad = None
        out = self.forward(x, weight, eps, offset, casting_mode)
        import torch

        torch.autograd.backward(out, grad_out)
        return out


class PyTorchAdapter(RMSNormAdapter):
    name = "PyTorch"

    def available(self) -> tuple[bool, str]:
        return True, "explicit PyTorch formulas"

    def forward(self, x, weight, eps, offset, casting_mode):
        _, ref_fn = REFERENCES[casting_mode]
        import torch

        return ref_fn(torch, x, weight, eps, offset)


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

    def forward(self, x, weight, eps, offset, casting_mode):
        return self._fn()(
            x,
            weight,
            eps,
            offset=offset,
            casting_mode=casting_mode,
            in_place=False,
            mode="infer",
            cache_rstd=True,
        )

    def forward_backward(self, x, weight, grad_out, eps, offset, casting_mode):
        x.grad = None
        if weight is not None:
            weight.grad = None
        out = self._fn()(
            x,
            weight,
            eps,
            offset=offset,
            casting_mode=casting_mode,
            in_place=False,
            mode="train",
            cache_rstd=True,
        )
        import torch

        torch.autograd.backward(out, grad_out)
        return out


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

    def forward(self, x, weight, eps, offset, casting_mode):
        return self._fn().apply(x, weight, eps, offset, casting_mode, False)

    def forward_backward(self, x, weight, grad_out, eps, offset, casting_mode):
        x.grad = None
        if weight is not None:
            weight.grad = None
        out = self._fn().apply(x, weight, eps, offset, casting_mode, False)
        import torch

        torch.autograd.backward(out, grad_out)
        return out


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

    def _call_direct(self, x, weight, eps, offset, casting_mode):
        if offset != 0.0:
            raise NotRunnableError("Unsloth direct adapter only supports offset=0.0 in this benchmark")
        module = self._api_module()
        layernorm = _LayerNormShim(weight=weight, variance_epsilon=eps)
        gemma = casting_mode == "gemma"
        errors = []

        for name in ("fast_rms_layernorm", "rms_layernorm"):
            fn = getattr(module, name, None)
            if fn is None:
                continue
            for args, kwargs in (
                ((layernorm, x), {"gemma": gemma}),
                ((layernorm, x), {}),
                ((x, weight, eps), {}),
            ):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - try next known signature.
                    errors.append(f"{name}: {type(exc).__name__}: {exc}")

        function = getattr(module, "Fast_RMS_Layernorm", None)
        if function is not None and hasattr(function, "apply"):
            for args in ((x, weight, eps), (x, weight, eps, gemma)):
                try:
                    return function.apply(*args)
                except Exception as exc:  # noqa: BLE001 - try next known signature.
                    errors.append(f"Fast_RMS_Layernorm.apply: {type(exc).__name__}: {exc}")

        detail = "; ".join(errors[-3:]) if errors else "no known callable found"
        raise NotRunnableError(f"Unsloth direct RMSNorm API incompatible: {detail}")

    def forward(self, x, weight, eps, offset, casting_mode):
        return self._call_direct(x, weight, eps, offset, casting_mode)

