"""Environment loading and reporting for benchmark scripts."""
from __future__ import annotations

import importlib
import platform


def load_torch():
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed. Run `pip install -e .[bench]`.") from exc


def _triton_version() -> str:
    try:
        triton = importlib.import_module("triton")
    except ImportError:
        return "not installed"
    return getattr(triton, "__version__", "unknown")


def require_cuda(torch) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Use a Colab GPU or CUDA GPU runtime, not TPU/CPU.")


def collect_environment(torch) -> dict:
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    cuda_or_hip = getattr(torch.version, "cuda", None) or getattr(torch.version, "hip", None) or "unknown"
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return {
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": f"{props.major}.{props.minor}",
        "cuda_or_hip": cuda_or_hip,
        "torch": torch.__version__,
        "triton": _triton_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "bf16_supported": bf16_supported,
    }
