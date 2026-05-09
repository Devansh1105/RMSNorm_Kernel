"""Fixed shape, dtype, and hardware-peak configuration for timing runs."""
from __future__ import annotations

from dataclasses import dataclass


DEFAULT_EPS = 1e-6
DEFAULT_CASTING_MODE = "llama"
DEFAULT_OFFSET = 0.0
DEFAULT_SEED = 0

QUICK_WARMUP = 1
QUICK_RUNS = 3
FULL_WARMUP = 3
FULL_RUNS = 10


@dataclass(frozen=True)
class TimingConfig:
    sweep: str
    m: int
    n: int
    batch: int | None = None
    seq: int | None = None

    @property
    def shape(self) -> dict:
        return {
            "batch": self.batch,
            "seq": self.seq,
            "m": self.m,
            "n": self.n,
        }

    @property
    def label(self) -> str:
        if self.batch is not None and self.seq is not None:
            return f"b{self.batch} s{self.seq} h{self.n}"
        return f"m{self.m} n{self.n}"


GPU_PEAKS = {
    "T4": {"bandwidth_gbps": 320.0, "fp16_tflops": 65.0, "bf16_tflops": None},
    "L4": {"bandwidth_gbps": 300.0, "fp16_tflops": 121.0, "bf16_tflops": 121.0},
    "A100": {"bandwidth_gbps": 1555.0, "fp16_tflops": 312.0, "bf16_tflops": 312.0},
    "H100": {"bandwidth_gbps": 3350.0, "fp16_tflops": 989.0, "bf16_tflops": 989.0},
}


def dtype_for_device(torch):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if (props.major, props.minor) >= (8, 0):
        return torch.bfloat16
    return torch.float16


def dtype_name(dtype) -> str:
    return str(dtype).removeprefix("torch.")


def mode_settings(mode: str) -> tuple[int, int]:
    if mode == "quick":
        return QUICK_WARMUP, QUICK_RUNS
    if mode == "full":
        return FULL_WARMUP, FULL_RUNS
    raise ValueError(f"unknown mode: {mode}")


def quick_configs() -> list[TimingConfig]:
    return [
        TimingConfig("reference", batch=4, seq=2048, m=4 * 2048, n=4096),
        TimingConfig("qk_norm", m=65536, n=128),
    ]


def full_configs() -> list[TimingConfig]:
    configs = [TimingConfig("reference", batch=4, seq=2048, m=4 * 2048, n=4096)]
    configs.extend(
        TimingConfig("seq", batch=4, seq=seq, m=4 * seq, n=4096)
        for seq in (512, 1024, 2048, 4096, 8192)
    )
    configs.extend(
        TimingConfig("batch", batch=batch, seq=2048, m=batch * 2048, n=4096)
        for batch in (1, 2, 4, 8, 16)
    )
    configs.extend(
        TimingConfig("hidden", batch=4, seq=2048, m=4 * 2048, n=hidden)
        for hidden in (1024, 2048, 4096, 8192, 11008)
    )
    configs.extend(
        TimingConfig("qk_norm", m=m, n=n)
        for m, n in ((65536, 128), (262144, 128))
    )
    return configs


def configs_for_mode(mode: str) -> list[TimingConfig]:
    if mode == "quick":
        return quick_configs()
    if mode == "full":
        return full_configs()
    raise ValueError(f"unknown mode: {mode}")


def peak_for_gpu(gpu_name: str) -> dict:
    upper_name = gpu_name.upper()
    for key, peak in GPU_PEAKS.items():
        if key in upper_name:
            return {"name": key, **peak}
    return {
        "name": "unknown",
        "bandwidth_gbps": None,
        "fp16_tflops": None,
        "bf16_tflops": None,
    }

