"""Cold-start timing worker for representative RMSNorm benchmark operations."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from benchmark.timing.configs import dtype_for_device, dtype_name
from benchmark.timing.competitors import ForgeAdapter, LigerAdapter, OperationRequest, PyTorchAdapter, UnslothAdapter
from benchmark.timing.scenarios import HORIZON_BY_ID, horizon_eps


ADAPTERS = {
    "PyTorch": PyTorchAdapter,
    "Forge": ForgeAdapter,
    "Liger": LigerAdapter,
    "Unsloth": UnslothAdapter,
}


def _make_inputs(torch, m: int, n: int, dtype, weight_mode: str):
    x = torch.randn((m, n), device="cuda", dtype=dtype)
    weight = None if weight_mode == "no_gamma" else torch.randn((n,), device="cuda", dtype=dtype)
    grad_out = torch.randn((m, n), device="cuda", dtype=dtype)
    return x, weight, grad_out


def worker_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--horizon", required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    args = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print(json.dumps({"status": "NOT_RUN", "notes": "CUDA unavailable"}))
        return 0

    horizon = HORIZON_BY_ID[args.horizon]
    adapter = ADAPTERS[args.adapter]()
    available, reason = adapter.available()
    if not available:
        print(json.dumps({"status": "NOT_RUN", "notes": reason}))
        return 0

    dtype = dtype_for_device(torch)
    request = OperationRequest(
        eps=horizon_eps(horizon),
        offset=horizon.offset,
        casting_mode=horizon.casting_mode,
        in_place=horizon.in_place,
        mode="train" if horizon.is_backward else "infer",
    )
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    x, weight, grad_out = _make_inputs(torch, args.m, args.n, dtype, horizon.weight_mode)
    if horizon.is_backward:
        x = x.detach().clone().requires_grad_(True)
        if weight is not None and horizon.weight_mode == "trainable_gamma":
            weight = weight.detach().clone().requires_grad_(True)
        elif weight is not None:
            weight = weight.detach().clone()

    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        if horizon.is_backward:
            adapter.backward(x, weight, grad_out, request)
        else:
            with torch.no_grad():
                adapter.forward(x, weight, request)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        peak = max(int(torch.cuda.max_memory_allocated() - baseline), 0)
        payload = {
            "status": "OK",
            "elapsed_ms": elapsed_ms,
            "vram_bytes": peak,
            "dtype": dtype_name(dtype),
            "notes": "",
        }
    except Exception as exc:  # noqa: BLE001 - report worker failures as rows.
        payload = {"status": "ERROR", "notes": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(payload))
    return 0


def measure_cold_start(adapter: str, horizon_id: str, m: int, n: int, timeout: int = 300) -> dict:
    with tempfile.TemporaryDirectory(prefix="rmsnorm_triton_cache_") as cache_dir:
        env = os.environ.copy()
        env["TRITON_CACHE_DIR"] = cache_dir
        repo = Path.cwd()
        src = repo / "src"
        env["PYTHONPATH"] = (
            str(src)
            + os.pathsep
            + str(repo)
            + os.pathsep
            + env.get("PYTHONPATH", "")
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmark.timing.cold_start",
                "--adapter",
                adapter,
                "--horizon",
                horizon_id,
                "--m",
                str(m),
                "--n",
                str(n),
            ],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    payload = {}
    if lines:
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError:
            payload = {}
    if not payload:
        payload = {"status": "ERROR", "notes": (proc.stderr or proc.stdout or "no worker output").strip()}
    payload.update(
        {
            "adapter": adapter,
            "horizon_id": horizon_id,
            "m": m,
            "n": n,
            "returncode": proc.returncode,
        }
    )
    if proc.stderr.strip():
        payload["stderr"] = proc.stderr.strip()[-2000:]
    return payload


if __name__ == "__main__":
    raise SystemExit(worker_main())

