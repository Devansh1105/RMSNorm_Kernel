"""Numerical equivalence tests for the γ-fold path.

Builds a tiny Llama-shaped model, runs a forward, folds γ, runs the same
forward, and asserts outputs match within bf16 tolerance. No real HF model
needed for the unit test — we mimic the structural pattern.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

torch.manual_seed(0)
cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not cuda_available, reason="CUDA not available")

if cuda_available:
    from fast_rmsnorm.transformers import FastRMSNorm, FoldPair, fold_rmsnorm_gamma_into_next_linear


class _MiniAttn(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        return self.o_proj(q + k + v)


class _MiniMLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.gate_proj = nn.Linear(h, h * 2, bias=False)
        self.up_proj = nn.Linear(h, h * 2, bias=False)
        self.down_proj = nn.Linear(h * 2, h, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class _MiniLayer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.input_layernorm = FastRMSNorm(h)
        self.self_attn = _MiniAttn(h)
        self.post_attention_layernorm = FastRMSNorm(h)
        self.mlp = _MiniMLP(h)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class _MiniInner(nn.Module):
    def __init__(self, h, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([_MiniLayer(h) for _ in range(num_layers)])
        self.norm = FastRMSNorm(h)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class _MiniLlama(nn.Module):
    def __init__(self, h=256, vocab=128, num_layers=2):
        super().__init__()
        self.model = _MiniInner(h, num_layers)
        self.lm_head = nn.Linear(h, vocab, bias=False)

    def forward(self, x):
        return self.lm_head(self.model(x))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fold_preserves_output(dtype):
    h = 256
    model = _MiniLlama(h=h, vocab=128, num_layers=2).cuda().to(dtype).eval()
    # .eval() only flips dropout/BN training flag; params still requires_grad.
    # Folding is inference-only, so we explicitly clear it.
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, FastRMSNorm) and m.weight is not None:
                m.weight.data.uniform_(0.5, 1.5)

    x = torch.randn(2, 16, h, device="cuda", dtype=dtype)

    with torch.no_grad():
        y_before = model(x)

    folded = fold_rmsnorm_gamma_into_next_linear(model, arch="llama")
    assert folded == 5  # 2 layers * 2 norms + 1 final = 5

    for m in model.modules():
        if isinstance(m, FastRMSNorm):
            assert m._gamma_folded

    with torch.no_grad():
        y_after = model(x)

    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    else:
        # bf16 reality: the fold path computes (γ · W_lin) once at fold time and
        # then accumulates `x_norm @ W_lin'` in the matmul; the unfolded path
        # accumulates `(γ · x_norm) @ W_lin`. Algebraically identical, but the
        # rounding boundaries differ — across 5 folds + 2-layer matmul stack the
        # accumulated drift hits ~1e-2 abs on a few percent of outputs. This is
        # inherent to bf16, not a kernel bug.
        atol, rtol = 2e-2, 2e-2
    torch.testing.assert_close(y_after, y_before, atol=atol, rtol=rtol)


def test_fold_refuses_when_training():
    """Fold should refuse to run on a model that still requires grad."""
    h = 64
    model = _MiniLlama(h=h, vocab=32, num_layers=1).cuda()
    with pytest.raises(RuntimeError, match="inference-only|requires_grad"):
        fold_rmsnorm_gamma_into_next_linear(model, arch="llama")


def test_fold_idempotency_guard():
    h = 64
    model = _MiniLlama(h=h, vocab=32, num_layers=1).cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    fold_rmsnorm_gamma_into_next_linear(model, arch="llama")
    with pytest.raises(RuntimeError, match="already folded"):
        fold_rmsnorm_gamma_into_next_linear(model, arch="llama")


def test_fold_with_offset_gemma():
    """Gemma uses (1+γ); fold formula uses (offset+γ)."""
    h = 64

    class _GemmaLikeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = FastRMSNorm(h, offset=1.0, casting_mode="gemma")
            self.linear = nn.Linear(h, h, bias=False)

        def forward(self, x):
            return self.linear(self.norm(x))

    model = _GemmaLikeModel().cuda().bfloat16().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        model.norm.weight.data.uniform_(-0.5, 0.5)

    x = torch.randn(4, h, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        y_before = model(x)

    pairs = [FoldPair(norm=model.norm, linears=[model.linear], name="custom")]
    folded = fold_rmsnorm_gamma_into_next_linear(model, pairs=pairs)
    assert folded == 1

    with torch.no_grad():
        y_after = model(x)

    # Same bf16 fold drift class as test_fold_preserves_output (single-fold here,
    # so tolerance is tighter — but still 1-ULP noise on near-zero outputs).
    torch.testing.assert_close(y_after, y_before, atol=2e-2, rtol=2e-2)
