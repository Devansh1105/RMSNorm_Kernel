"""FlashNorm-style γ-fold utility.

Algebraic identity (for any RMSNorm immediately followed by a Linear):

    y = ((offset + γ) ⊙ x_norm) @ W^T
      = x_norm @ ((offset + γ) ⊙ W)^T          # column-wise scaling of W rows

So we can pre-multiply the next Linear's weight column-wise by (offset + γ),
zero out γ, and the RMSNorm becomes a pure normalize. Saves ~33% of the
RMSNorm kernel's HBM traffic at inference.

The fold is done **in-place** on the model. Refuses to fold when:
  - any parameter still requires grad (training)
  - the next op is not a plain ``nn.Linear``
  - the Linear is quantized (int8/int4) — out of scope for v1
  - the RMSNorm has already been folded (idempotent guard)

References:
  - Graef 2024, FlashNorm (arxiv.org/abs/2407.09577)
  - Liger discussion in your study doc
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from fast_rmsnorm.transformers.rms_norm import FastRMSNorm


@dataclass
class FoldPair:
    """One RMSNorm → list-of-Linears fold target.

    Multiple Linears are fine — Llama's pre-attn norm feeds q/k/v in parallel,
    pre-MLP norm feeds gate/up. Each Linear's weight gets scaled the same way.
    """
    norm: FastRMSNorm
    linears: list[nn.Linear]
    name: str = ""  # for diagnostics


# ---------------------------------------------------------------------------
# Pattern recipes — each yields FoldPairs for a known model family
# ---------------------------------------------------------------------------


def _llama_pairs(model: nn.Module) -> Iterable[FoldPair]:
    """Llama / Mistral / Qwen2 / similar.

    Walks ``model.model.layers`` and pairs:
      - ``input_layernorm`` → q_proj, k_proj, v_proj
      - ``post_attention_layernorm`` → gate_proj, up_proj
    Plus the final norm → lm_head.
    """
    inner = model.model if hasattr(model, "model") else model
    for i, layer in enumerate(inner.layers):
        if isinstance(layer.input_layernorm, FastRMSNorm):
            yield FoldPair(
                norm=layer.input_layernorm,
                linears=[layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
                name=f"layers.{i}.input_layernorm",
            )
        if isinstance(layer.post_attention_layernorm, FastRMSNorm):
            yield FoldPair(
                norm=layer.post_attention_layernorm,
                linears=[layer.mlp.gate_proj, layer.mlp.up_proj],
                name=f"layers.{i}.post_attention_layernorm",
            )
    if isinstance(getattr(inner, "norm", None), FastRMSNorm) and hasattr(model, "lm_head"):
        yield FoldPair(norm=inner.norm, linears=[model.lm_head], name="model.norm")


_RECIPES = {
    "llama": _llama_pairs,
    "mistral": _llama_pairs,
    "qwen2": _llama_pairs,
    "qwen3": _llama_pairs,  # QK-norm not foldable; treated like Llama for the rest
    "yi": _llama_pairs,
    "phi3": _llama_pairs,
    "deepseek": _llama_pairs,
}


# ---------------------------------------------------------------------------
# The fold itself
# ---------------------------------------------------------------------------


def _fold_pair(pair: FoldPair) -> None:
    """In-place fold of one (norm, [linears]) target."""
    norm = pair.norm
    if norm._gamma_folded:
        raise RuntimeError(f"{pair.name}: γ already folded — refusing to double-fold.")
    if not norm.elementwise_affine:
        # No γ to fold; just mark folded so the kernel stays in the no-W path.
        norm._gamma_folded = True
        return

    if norm.weight.requires_grad:
        raise RuntimeError(
            f"{pair.name}: weight.requires_grad=True. Folding is inference-only; "
            "call model.eval() and ensure no grads are required."
        )

    # offset + γ  (Gemma uses offset=1, Llama uses 0)
    scale = (norm.offset + norm.weight.detach()).to(torch.float32)

    for lin in pair.linears:
        if not isinstance(lin, nn.Linear):
            raise TypeError(
                f"{pair.name}: target {type(lin).__name__} is not nn.Linear. "
                "Quantized / non-Linear targets are not supported in v1."
            )
        if lin.weight.requires_grad:
            raise RuntimeError(f"{pair.name}: target Linear requires_grad=True; fold is inference-only.")
        # nn.Linear weight is (out_features, in_features); scale along in_features (dim=1).
        target_dtype = lin.weight.dtype
        lin.weight.data.mul_(scale.to(target_dtype).unsqueeze(0))

    # Zero out γ (and the offset semantics) so the kernel's no-W path is correct.
    norm._gamma_folded = True
    with torch.no_grad():
        norm.weight.data = torch.empty(0, dtype=norm.weight.dtype, device=norm.weight.device)


def fold_rmsnorm_gamma_into_next_linear(
    model: nn.Module,
    arch: str | None = None,
    *,
    pairs: Iterable[FoldPair] | None = None,
    strict: bool = True,
) -> int:
    """Fold every RMSNorm γ in ``model`` into the immediately-following Linear's weight.

    Args:
        model: a transformer with FastRMSNorm modules in known positions.
        arch: one of ``_RECIPES`` keys, e.g. 'llama', 'qwen3'. Ignored if ``pairs`` given.
        pairs: explicit list of FoldPair — overrides the recipe; useful for custom archs.
        strict: when True (default), every step must succeed or we raise. When False,
                a per-pair failure is logged and the rest continue.

    Returns:
        Number of (norm, linears) pairs that were successfully folded.
    """
    if pairs is None:
        if arch is None:
            raise ValueError("Pass either arch=... (one of: " + ", ".join(_RECIPES) + ") or pairs=[...]")
        if arch not in _RECIPES:
            raise ValueError(f"Unknown arch {arch!r}. Known: {list(_RECIPES)}")
        pairs = list(_RECIPES[arch](model))

    folded = 0
    for pair in pairs:
        try:
            _fold_pair(pair)
            folded += 1
        except Exception:
            if strict:
                raise
    return folded
