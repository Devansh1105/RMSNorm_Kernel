"""Small models for gamma-fold correctness checks.

Gemma-style RMSNorm folding into fp16/bf16 nn.Linear weights is refused by
default because it changes rounding order.
"""

from __future__ import annotations

import copy

from benchmark.common.tolerances import (
    dtype_name,
    format_tolerance,
    is_close,
    max_atol_rtol,
    threshold_for,
    verdict,
)


def _fold_row(torch, *, check: str, dtype, reference: str, compared: str, fold: bool = True) -> dict:
    threshold = threshold_for(dtype, fold=fold)
    return {
        "group": "fold",
        "check": check,
        "path": "-",
        "shape": "-",
        "dtype": dtype_name(dtype),
        "casting": "-",
        "weight": "yes",
        "mode": "infer",
        "in_place": "-",
        "cache_rstd": "-",
        "reference": reference,
        "compared": compared,
        "tolerance": format_tolerance(threshold),
        "max_abs": None,
        "max_rel": None,
        "dx_abs": None,
        "dx_rel": None,
        "dw_abs": None,
        "dw_rel": None,
        "verdict": "FAIL",
        "notes": "",
    }


def _build_llama_like(torch, dtype, *, hidden_size: int, layers: int, offset: float, casting_mode: str):
    import torch.nn as nn

    from fast_rmsnorm.transformers import FastRMSNorm

    class _MiniSelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, x):
            return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))

    class _MiniMlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, x):
            return self.down_proj(self.gate_proj(x) * self.up_proj(x))

    class _MiniLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = FastRMSNorm(
                hidden_size,
                eps=1e-6,
                offset=offset,
                casting_mode=casting_mode,
                mode="infer",
                in_place=False,
            )
            self.self_attn = _MiniSelfAttn()
            self.post_attention_layernorm = FastRMSNorm(
                hidden_size,
                eps=1e-6,
                offset=offset,
                casting_mode=casting_mode,
                mode="infer",
                in_place=False,
            )
            self.mlp = _MiniMlp()

        def forward(self, x):
            x = x + self.self_attn(self.input_layernorm(x))
            return x + self.mlp(self.post_attention_layernorm(x))

    class _MiniInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_MiniLayer() for _ in range(layers)])
            self.norm = FastRMSNorm(
                hidden_size,
                eps=1e-6,
                offset=offset,
                casting_mode=casting_mode,
                mode="infer",
                in_place=False,
            )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.norm(x)

    class _MiniLlama(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _MiniInner()
            self.lm_head = nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, x):
            return self.lm_head(self.model(x))

    model = _MiniLlama().cuda().to(dtype=dtype)
    for param in model.parameters():
        param.requires_grad_(False)
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "weight") and module.weight is not None and module.weight.numel() > 0:
                module.weight.uniform_(-0.25, 0.25)
    model.eval()
    return model


def run_llama_fold_check(torch, dtype, *, full: bool = False) -> dict:
    from fast_rmsnorm.transformers import fold_rmsnorm_gamma_into_next_linear

    hidden_size = 256 if full else 128
    layers = 2 if full else 1
    row = _fold_row(
        torch,
        check="fold_equivalence_llama",
        dtype=dtype,
        reference="Mini Llama before fold",
        compared="Mini Llama after fold",
    )
    try:
        torch.manual_seed(1234)
        model = _build_llama_like(
            torch,
            dtype,
            hidden_size=hidden_size,
            layers=layers,
            offset=0.0,
            casting_mode="llama",
        )
        folded = copy.deepcopy(model)
        x = torch.randn((2, 8, hidden_size), device="cuda", dtype=dtype)
        with torch.no_grad():
            expected = model(x)
            folded_count = fold_rmsnorm_gamma_into_next_linear(folded, arch="llama")
            got = folded(x)
        threshold = threshold_for(dtype, fold=True)
        max_abs, max_rel = max_atol_rtol(torch, got, expected)
        row.update(
            {
                "shape": f"2x8x{hidden_size}",
                "max_abs": max_abs,
                "max_rel": max_rel,
                "verdict": verdict(is_close(torch, got, expected, threshold)),
                "notes": f"folded_pairs={folded_count}",
            }
        )
    except Exception as exc:
        row["notes"] = f"{type(exc).__name__}: {exc}"
    return row


def run_gemma_fold_check(torch, dtype, *, full: bool = False) -> dict:
    from fast_rmsnorm.transformers import fold_rmsnorm_gamma_into_next_linear

    hidden_size = 256 if full else 128
    if dtype in (torch.float16, torch.bfloat16):
        row = _fold_row(
            torch,
            check="fold_refusal_gemma_low_precision",
            dtype=dtype,
            reference="Gemma fp16/bf16 fold should require opt-in",
            compared="default fold policy",
            fold=False,
        )
        try:
            torch.manual_seed(5678)
            model = _build_llama_like(
                torch,
                dtype,
                hidden_size=hidden_size,
                layers=1,
                offset=1.0,
                casting_mode="gemma",
            )
            fold_rmsnorm_gamma_into_next_linear(model, arch="llama")
            row["notes"] = "fold unexpectedly succeeded"
        except RuntimeError as exc:
            row.update(
                {
                    "shape": f"2x8x{hidden_size}",
                    "verdict": "PASS",
                    "notes": str(exc),
                }
            )
        except Exception as exc:
            row["notes"] = f"{type(exc).__name__}: {exc}"
        return row

    row = _fold_row(
        torch,
        check="fold_equivalence_gemma_offset",
        dtype=dtype,
        reference="Mini Gemma-style model before fold",
        compared="Mini Gemma-style model after fold",
    )
    try:
        torch.manual_seed(5678)
        model = _build_llama_like(
            torch,
            dtype,
            hidden_size=hidden_size,
            layers=1,
            offset=1.0,
            casting_mode="gemma",
        )
        folded = copy.deepcopy(model)
        x = torch.randn((2, 8, hidden_size), device="cuda", dtype=dtype)
        with torch.no_grad():
            expected = model(x)
            folded_count = fold_rmsnorm_gamma_into_next_linear(folded, arch="llama")
            got = folded(x)
        threshold = threshold_for(dtype, fold=True)
        max_abs, max_rel = max_atol_rtol(torch, got, expected)
        row.update(
            {
                "shape": f"2x8x{hidden_size}",
                "max_abs": max_abs,
                "max_rel": max_rel,
                "verdict": verdict(is_close(torch, got, expected, threshold)),
                "notes": f"folded_pairs={folded_count}",
            }
        )
    except Exception as exc:
        row["notes"] = f"{type(exc).__name__}: {exc}"
    return row


def run_fold_refusal_checks(torch) -> list[dict]:
    import torch.nn as nn

    from fast_rmsnorm.transformers import FastRMSNorm, FoldPair, fold_rmsnorm_gamma_into_next_linear

    rows = []

    row = _fold_row(
        torch,
        check="fold_refusal_training",
        dtype=torch.float32,
        reference="fold should reject trainable params",
        compared="requires_grad=True model",
        fold=False,
    )
    try:
        norm = FastRMSNorm(32, mode="infer").cuda()
        linear = nn.Linear(32, 32, bias=False).cuda()
        fold_rmsnorm_gamma_into_next_linear(None, pairs=[FoldPair(norm=norm, linears=[linear], name="training")])
        row["notes"] = "fold unexpectedly succeeded"
    except RuntimeError as exc:
        row.update({"verdict": "PASS", "notes": str(exc)})
    except Exception as exc:
        row["notes"] = f"{type(exc).__name__}: {exc}"
    rows.append(row)

    row = _fold_row(
        torch,
        check="fold_refusal_non_linear",
        dtype=torch.float32,
        reference="fold should reject non-Linear target",
        compared="ReLU target",
        fold=False,
    )
    try:
        norm = FastRMSNorm(32, mode="infer").cuda()
        linear = nn.ReLU().cuda()
        for param in norm.parameters():
            param.requires_grad_(False)
        fold_rmsnorm_gamma_into_next_linear(None, pairs=[FoldPair(norm=norm, linears=[linear], name="nonlinear")])
        row["notes"] = "fold unexpectedly succeeded"
    except TypeError as exc:
        row.update({"verdict": "PASS", "notes": str(exc)})
    except Exception as exc:
        row["notes"] = f"{type(exc).__name__}: {exc}"
    rows.append(row)

    row = _fold_row(
        torch,
        check="fold_refusal_double_fold",
        dtype=torch.float32,
        reference="fold should reject second fold",
        compared="already folded norm",
        fold=False,
    )
    try:
        norm = FastRMSNorm(32, mode="infer").cuda()
        linear = nn.Linear(32, 32, bias=False).cuda()
        for param in list(norm.parameters()) + list(linear.parameters()):
            param.requires_grad_(False)
        pair = FoldPair(norm=norm, linears=[linear], name="double")
        fold_rmsnorm_gamma_into_next_linear(None, pairs=[pair])
        fold_rmsnorm_gamma_into_next_linear(None, pairs=[pair])
        row["notes"] = "second fold unexpectedly succeeded"
    except RuntimeError as exc:
        row.update({"verdict": "PASS", "notes": str(exc)})
    except Exception as exc:
        row["notes"] = f"{type(exc).__name__}: {exc}"
    rows.append(row)

    return rows
