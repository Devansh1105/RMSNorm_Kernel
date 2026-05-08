"""Correctness case generation.

The default matrix is intentionally targeted: it exercises every implemented
path without exploding into a long benchmark run. The full matrix broadens
shape coverage before paid GPU benchmarking.
"""
from __future__ import annotations


DEFAULT_EPS = 1e-6


def supported_dtypes(torch) -> list:
    dtypes = [torch.float32, torch.float16]
    if bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
        dtypes.append(torch.bfloat16)
    return dtypes


def low_precision_dtype(torch):
    return torch.bfloat16 if bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()) else torch.float16


def quick_dtypes(torch) -> list:
    dtypes = [torch.float32, low_precision_dtype(torch)]
    return list(dict.fromkeys(dtypes))


def _path_for_shape(shape: tuple[int, int]) -> str:
    m, n = shape
    return "block" if m >= 32768 and n <= 256 else "row"


def _case(
    torch,
    *,
    shape,
    dtype,
    casting_mode,
    offset,
    with_weight=True,
    in_place=False,
    cache_rstd=True,
    mode="infer",
):
    return {
        "path": _path_for_shape(shape),
        "shape": shape,
        "dtype": dtype,
        "casting_mode": casting_mode,
        "offset": offset,
        "with_weight": with_weight,
        "in_place": in_place,
        "cache_rstd": cache_rstd,
        "mode": mode,
        "eps": DEFAULT_EPS,
    }


def casting_cases(full: bool = False) -> list[tuple[str, float]]:
    cases = [("llama", 0.0), ("gemma", 1.0), ("none", 0.0)]
    if full:
        cases.append(("gemma", 0.0))
    return cases


def forward_cases(torch, full: bool = False) -> list[dict]:
    shapes = [(16, 256), (40000, 128)]
    if full:
        shapes.extend([(64, 1024), (128, 4096), (1024, 4096), (4, 8192), (40000, 64), (40000, 256)])
    dtypes = supported_dtypes(torch) if full else quick_dtypes(torch)

    cases = []
    for shape in shapes:
        for dtype in dtypes:
            for casting_mode, offset in casting_cases(full):
                for with_weight in (True, False):
                    cases.append(
                        _case(
                            torch,
                            shape=shape,
                            dtype=dtype,
                            casting_mode=casting_mode,
                            offset=offset,
                            with_weight=with_weight,
                            mode="infer",
                        )
                    )
    return cases


def backward_cases(torch, full: bool = False) -> list[dict]:
    dtypes = supported_dtypes(torch) if full else quick_dtypes(torch)
    base_shapes = [(16, 256), (40000, 128)]
    if full:
        base_shapes.extend([(128, 4096), (40000, 64), (40000, 256)])

    cases = []

    # Reference correctness across casting and affine paths.
    for shape in base_shapes:
        for dtype in dtypes:
            for casting_mode, offset in casting_cases(full):
                cases.append(
                    _case(
                        torch,
                        shape=shape,
                        dtype=dtype,
                        casting_mode=casting_mode,
                        offset=offset,
                        with_weight=True,
                        in_place=False,
                        cache_rstd=True,
                        mode="train",
                    )
                )
            cases.append(
                _case(
                    torch,
                    shape=shape,
                    dtype=dtype,
                    casting_mode="llama",
                    offset=0.0,
                    with_weight=False,
                    in_place=False,
                    cache_rstd=True,
                    mode="train",
                )
            )

    # Backward should also work when mode="infer" selects the heuristic kernels.
    for shape in [(16, 256), (40000, 128)]:
        for dtype in dtypes:
            cases.append(
                _case(
                    torch,
                    shape=shape,
                    dtype=dtype,
                    casting_mode="llama",
                    offset=0.0,
                    with_weight=True,
                    in_place=False,
                    cache_rstd=True,
                    mode="infer",
                )
            )
    return cases


def cache_cases(torch, full: bool = False) -> list[dict]:
    shapes = [(64, 1024), (40000, 128)]
    if full:
        shapes.append((128, 4096))
    dtypes = supported_dtypes(torch) if full else quick_dtypes(torch)
    return [
        _case(
            torch,
            shape=shape,
            dtype=dtype,
            casting_mode="llama",
            offset=0.0,
            with_weight=True,
            in_place=False,
            cache_rstd=True,
            mode="train",
        )
        for shape in shapes
        for dtype in dtypes
    ]


def inplace_cases(torch, full: bool = False) -> list[dict]:
    shapes = [(64, 1024), (40000, 128)]
    if full:
        shapes.append((128, 4096))
    dtypes = supported_dtypes(torch) if full else quick_dtypes(torch)
    return [
        _case(
            torch,
            shape=shape,
            dtype=dtype,
            casting_mode="llama",
            offset=0.0,
            with_weight=True,
            in_place=False,
            cache_rstd=True,
            mode="train",
        )
        for shape in shapes
        for dtype in dtypes
    ]


def mode_cases(torch, full: bool = False) -> list[dict]:
    shapes = [(16, 256), (40000, 128)]
    if full:
        shapes.extend([(8, 4096), (40000, 256)])
    dtypes = supported_dtypes(torch) if full else quick_dtypes(torch)
    return [
        _case(
            torch,
            shape=shape,
            dtype=dtype,
            casting_mode="llama",
            offset=0.0,
            with_weight=True,
            in_place=False,
            cache_rstd=True,
            mode="infer",
        )
        for shape in shapes
        for dtype in dtypes
    ]
