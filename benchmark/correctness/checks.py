"""Correctness checks for RMSNorm paths."""
from __future__ import annotations

from benchmark.common.tolerances import (
    dtype_name,
    format_tolerance,
    is_close,
    max_atol_rtol,
    threshold_for,
    verdict,
)
from benchmark.correctness.references import REFERENCES


def _shape_text(shape: tuple[int, int]) -> str:
    return f"{shape[0]}x{shape[1]}"


def _weight_text(with_weight: bool) -> str:
    return "yes" if with_weight else "no"


def _make_tensor(torch, shape, dtype, *, requires_grad: bool = False):
    return torch.randn(shape, device="cuda", dtype=dtype, requires_grad=requires_grad)


def _base_row(case: dict, *, group: str, check: str, reference: str, compared: str) -> dict:
    threshold = threshold_for(case["dtype"], casting_mode=case.get("casting_mode"))
    return {
        "group": group,
        "check": check,
        "path": case.get("path", "-"),
        "shape": _shape_text(case["shape"]),
        "dtype": dtype_name(case["dtype"]),
        "casting": case.get("casting_mode", "-"),
        "offset": case.get("offset", "-"),
        "weight": _weight_text(case.get("with_weight", False)),
        "mode": case.get("mode", "-"),
        "in_place": str(case.get("in_place", "-")),
        "cache_rstd": str(case.get("cache_rstd", "-")),
        "reference": reference,
        "compared": compared,
        "tolerance": format_tolerance(threshold),
        "verdict": "FAIL",
        "notes": "",
    }


def _failure_row(case: dict, *, group: str, check: str, reference: str, compared: str, error: Exception) -> dict:
    row = _base_row(case, group=group, check=check, reference=reference, compared=compared)
    row["notes"] = f"{type(error).__name__}: {error}"
    return row


def run_forward_check(torch, rms_norm, case: dict) -> dict:
    ref_name, ref_fn = REFERENCES[case["casting_mode"]]
    row = _base_row(case, group="forward", check="forward", reference=ref_name, compared="Forge forward")
    try:
        m, n = case["shape"]
        x = _make_tensor(torch, (m, n), case["dtype"])
        weight = _make_tensor(torch, (n,), case["dtype"]) if case["with_weight"] else None
        out = rms_norm(
            x.clone(),
            weight,
            case["eps"],
            offset=case["offset"],
            casting_mode=case["casting_mode"],
            in_place=case["in_place"],
            mode=case["mode"],
            cache_rstd=case["cache_rstd"],
        )
        ref = ref_fn(torch, x, weight, case["eps"], case["offset"])
        max_abs, max_rel = max_atol_rtol(torch, out, ref)
        threshold = threshold_for(case["dtype"], casting_mode=case["casting_mode"])
        row.update(
            {
                "max_abs": max_abs,
                "max_rel": max_rel,
                "verdict": verdict(is_close(torch, out, ref, threshold)),
            }
        )
    except Exception as exc:
        return _failure_row(case, group="forward", check="forward", reference=ref_name, compared="Forge forward", error=exc)
    return row


def run_backward_check(torch, rms_norm, case: dict) -> dict:
    ref_name, ref_fn = REFERENCES[case["casting_mode"]]
    row = _base_row(case, group="backward", check="backward", reference=f"{ref_name} grads", compared="Forge grads")
    try:
        m, n = case["shape"]
        x = _make_tensor(torch, (m, n), case["dtype"], requires_grad=True)
        weight = _make_tensor(torch, (n,), case["dtype"], requires_grad=True) if case["with_weight"] else None
        grad_out = torch.randn((m, n), device="cuda", dtype=case["dtype"])

        out = rms_norm(
            x,
            weight,
            case["eps"],
            offset=case["offset"],
            casting_mode=case["casting_mode"],
            in_place=case["in_place"],
            mode=case["mode"],
            cache_rstd=case["cache_rstd"],
        )
        targets = [x] + ([weight] if weight is not None else [])
        grads = torch.autograd.grad(out, targets, grad_out, retain_graph=False)

        x_ref = x.detach().clone().requires_grad_(True)
        weight_ref = weight.detach().clone().requires_grad_(True) if weight is not None else None
        ref = ref_fn(torch, x_ref, weight_ref, case["eps"], case["offset"])
        ref_targets = [x_ref] + ([weight_ref] if weight_ref is not None else [])
        ref_grads = torch.autograd.grad(ref, ref_targets, grad_out.detach(), retain_graph=False)

        threshold = threshold_for(case["dtype"], casting_mode=case["casting_mode"])
        dx_abs, dx_rel = max_atol_rtol(torch, grads[0], ref_grads[0])
        passed = is_close(torch, grads[0], ref_grads[0], threshold)
        if weight is not None:
            dw_abs, dw_rel = max_atol_rtol(torch, grads[1], ref_grads[1])
            passed = passed and is_close(torch, grads[1], ref_grads[1], threshold)
        else:
            dw_abs, dw_rel = None, None
        row.update(
            {
                "dx_abs": dx_abs,
                "dx_rel": dx_rel,
                "dw_abs": dw_abs,
                "dw_rel": dw_rel,
                "max_abs": max(v for v in (dx_abs, dw_abs) if v is not None),
                "max_rel": max(v for v in (dx_rel, dw_rel) if v is not None),
                "verdict": verdict(passed),
            }
        )
    except Exception as exc:
        return _failure_row(case, group="backward", check="backward", reference=f"{ref_name} grads", compared="Forge grads", error=exc)
    return row


def _forge_grads(torch, rms_norm, case: dict, x, weight, grad_out, *, cache_rstd: bool, in_place: bool):
    out = rms_norm(
        x,
        weight,
        case["eps"],
        offset=case["offset"],
        casting_mode=case["casting_mode"],
        in_place=in_place,
        mode=case["mode"],
        cache_rstd=cache_rstd,
    )
    targets = [x] + ([weight] if weight is not None else [])
    return out, torch.autograd.grad(out, targets, grad_out, retain_graph=False)


def run_cache_rstd_check(torch, rms_norm, case: dict) -> dict:
    row = _base_row(
        case,
        group="internal",
        check="cache_rstd",
        reference="Forge cache_rstd=True",
        compared="Forge cache_rstd=False",
    )
    try:
        m, n = case["shape"]
        x = _make_tensor(torch, (m, n), case["dtype"], requires_grad=True)
        weight = _make_tensor(torch, (n,), case["dtype"], requires_grad=True)
        grad_out = torch.randn((m, n), device="cuda", dtype=case["dtype"])

        out_a, grads_a = _forge_grads(torch, rms_norm, case, x, weight, grad_out, cache_rstd=True, in_place=False)
        x_b = x.detach().clone().requires_grad_(True)
        w_b = weight.detach().clone().requires_grad_(True)
        out_b, grads_b = _forge_grads(torch, rms_norm, case, x_b, w_b, grad_out.detach(), cache_rstd=False, in_place=False)

        threshold = threshold_for(case["dtype"], casting_mode=case["casting_mode"])
        y_abs, y_rel = max_atol_rtol(torch, out_b, out_a)
        dx_abs, dx_rel = max_atol_rtol(torch, grads_b[0], grads_a[0])
        dw_abs, dw_rel = max_atol_rtol(torch, grads_b[1], grads_a[1])
        passed = (
            is_close(torch, out_b, out_a, threshold)
            and is_close(torch, grads_b[0], grads_a[0], threshold)
            and is_close(torch, grads_b[1], grads_a[1], threshold)
        )
        row.update(
            {
                "max_abs": max(y_abs, dx_abs, dw_abs),
                "max_rel": max(y_rel, dx_rel, dw_rel),
                "dx_abs": dx_abs,
                "dx_rel": dx_rel,
                "dw_abs": dw_abs,
                "dw_rel": dw_rel,
                "verdict": verdict(passed),
            }
        )
    except Exception as exc:
        return _failure_row(case, group="internal", check="cache_rstd", reference="Forge cache_rstd=True", compared="Forge cache_rstd=False", error=exc)
    return row


def run_inplace_check(torch, rms_norm, case: dict) -> dict:
    row = _base_row(
        case,
        group="internal",
        check="in_place",
        reference="Forge in_place=False",
        compared="Forge in_place=True",
    )
    try:
        m, n = case["shape"]
        x = _make_tensor(torch, (m, n), case["dtype"], requires_grad=True)
        weight = _make_tensor(torch, (n,), case["dtype"], requires_grad=True)
        grad_out = torch.randn((m, n), device="cuda", dtype=case["dtype"])

        _, grads_a = _forge_grads(torch, rms_norm, case, x, weight, grad_out, cache_rstd=True, in_place=False)
        x_b = x.detach().clone().requires_grad_(True)
        w_b = weight.detach().clone().requires_grad_(True)
        _, grads_b = _forge_grads(torch, rms_norm, case, x_b, w_b, grad_out.detach(), cache_rstd=True, in_place=True)

        threshold = threshold_for(case["dtype"], casting_mode=case["casting_mode"])
        dx_abs, dx_rel = max_atol_rtol(torch, grads_b[0], grads_a[0])
        dw_abs, dw_rel = max_atol_rtol(torch, grads_b[1], grads_a[1])
        passed = is_close(torch, grads_b[0], grads_a[0], threshold) and is_close(torch, grads_b[1], grads_a[1], threshold)
        row.update(
            {
                "max_abs": max(dx_abs, dw_abs),
                "max_rel": max(dx_rel, dw_rel),
                "dx_abs": dx_abs,
                "dx_rel": dx_rel,
                "dw_abs": dw_abs,
                "dw_rel": dw_rel,
                "verdict": verdict(passed),
            }
        )
    except Exception as exc:
        return _failure_row(case, group="internal", check="in_place", reference="Forge in_place=False", compared="Forge in_place=True", error=exc)
    return row


def run_mode_consistency_check(torch, rms_norm, case: dict) -> list[dict]:
    rows = []
    ref_name, ref_fn = REFERENCES[case["casting_mode"]]
    try:
        m, n = case["shape"]
        x = _make_tensor(torch, (m, n), case["dtype"])
        weight = _make_tensor(torch, (n,), case["dtype"])
        ref = ref_fn(torch, x, weight, case["eps"], case["offset"])
        threshold = threshold_for(case["dtype"], casting_mode=case["casting_mode"])
        for mode in ("train", "infer", "auto"):
            mode_case = {**case, "mode": mode}
            row = _base_row(mode_case, group="internal", check="mode", reference=ref_name, compared=f"Forge mode={mode}")
            out = rms_norm(
                x.clone(),
                weight,
                case["eps"],
                offset=case["offset"],
                casting_mode=case["casting_mode"],
                in_place=False,
                mode=mode,
                cache_rstd=True,
            )
            max_abs, max_rel = max_atol_rtol(torch, out, ref)
            row.update(
                {
                    "max_abs": max_abs,
                    "max_rel": max_rel,
                    "verdict": verdict(is_close(torch, out, ref, threshold)),
                }
            )
            rows.append(row)
    except Exception as exc:
        rows.append(_failure_row(case, group="internal", check="mode", reference=ref_name, compared="Forge mode consistency", error=exc))
    return rows


def run_reduce_strategy_status(torch) -> dict:
    from fast_rmsnorm.transformers import rms_norm

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    case = {
        "path": "block",
        "shape": (40000, 128),
        "dtype": dtype,
        "casting_mode": "llama",
        "offset": 0.0,
        "with_weight": True,
        "mode": "infer",
        "in_place": False,
        "cache_rstd": True,
        "eps": 1e-6,
    }
    row = _base_row(
        case,
        group="internal",
        check="reduce_strategy_direct",
        reference="Forge reduce_strategy=auto",
        compared="Forge reduce_strategy=atomic/scratch",
    )
    try:
        m, n = case["shape"]
        x = _make_tensor(torch, (m, n), case["dtype"], requires_grad=True)
        weight = _make_tensor(torch, (n,), case["dtype"], requires_grad=True)
        grad_out = torch.randn((m, n), device="cuda", dtype=case["dtype"])

        def run_strategy(strategy: str):
            x_i = x.detach().clone().requires_grad_(True)
            w_i = weight.detach().clone().requires_grad_(True)
            out = rms_norm(
                x_i,
                w_i,
                case["eps"],
                offset=case["offset"],
                casting_mode=case["casting_mode"],
                in_place=case["in_place"],
                mode=case["mode"],
                cache_rstd=case["cache_rstd"],
                reduce_strategy=strategy,
            )
            grads = torch.autograd.grad(out, [x_i, w_i], grad_out.detach(), retain_graph=False)
            return out, grads

        out_auto, grads_auto = run_strategy("auto")
        out_atomic, grads_atomic = run_strategy("atomic")
        out_scratch, grads_scratch = run_strategy("scratch")
        threshold = threshold_for(case["dtype"], casting_mode=case["casting_mode"])
        checks = [
            max_atol_rtol(torch, out_atomic, out_auto),
            max_atol_rtol(torch, grads_atomic[0], grads_auto[0]),
            max_atol_rtol(torch, grads_atomic[1], grads_auto[1]),
            max_atol_rtol(torch, out_scratch, out_auto),
            max_atol_rtol(torch, grads_scratch[0], grads_auto[0]),
            max_atol_rtol(torch, grads_scratch[1], grads_auto[1]),
        ]
        passed = (
            is_close(torch, out_atomic, out_auto, threshold)
            and is_close(torch, grads_atomic[0], grads_auto[0], threshold)
            and is_close(torch, grads_atomic[1], grads_auto[1], threshold)
            and is_close(torch, out_scratch, out_auto, threshold)
            and is_close(torch, grads_scratch[0], grads_auto[0], threshold)
            and is_close(torch, grads_scratch[1], grads_auto[1], threshold)
        )
        row.update(
            {
                "max_abs": max(item[0] for item in checks),
                "max_rel": max(item[1] for item in checks),
                "dx_abs": max(checks[1][0], checks[4][0]),
                "dx_rel": max(checks[1][1], checks[4][1]),
                "dw_abs": max(checks[2][0], checks[5][0]),
                "dw_rel": max(checks[2][1], checks[5][1]),
                "verdict": verdict(passed),
            }
        )
    except Exception as exc:
        return _failure_row(
            case,
            group="internal",
            check="reduce_strategy_direct",
            reference="Forge reduce_strategy=auto",
            compared="Forge reduce_strategy=atomic/scratch",
            error=exc,
        )
    return row


def run_gradcheck_status() -> dict:
    return {
        "group": "blocked",
        "check": "gradcheck_fp64",
        "path": "-",
        "shape": "-",
        "dtype": "float64",
        "casting": "-",
        "weight": "-",
        "mode": "-",
        "in_place": "-",
        "cache_rstd": "-",
        "reference": "torch.autograd.gradcheck",
        "compared": "Forge fp64",
        "tolerance": "-",
        "max_abs": None,
        "max_rel": None,
        "dx_abs": None,
        "dx_rel": None,
        "dw_abs": None,
        "dw_rel": None,
        "verdict": "BLOCKED",
        "notes": "Current kernel dtype map supports fp32/fp16/bf16 only.",
    }
