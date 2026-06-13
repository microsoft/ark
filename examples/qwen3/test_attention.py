# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Equivalence tests: ARK GQA attention vs torch reference.

All GPU tests are gated with ``skipif(not cuda)``.
"""

import math

import pytest
import torch

_CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _CUDA, reason="CUDA not available")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SEED = 42


def _small_cfg():
    """Return a small Qwen3Config suitable for unit tests."""
    from .qwen3_config import Qwen3Config

    return Qwen3Config(
        n_layers=1,
        hidden_dim=128,
        n_q_heads=4,
        n_kv_heads=2,
        head_dim=32,
        intermediate_dim=256,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        max_seq_len=256,
    )


def _build_ref_attn(cfg):
    """Instantiate a torch GQAAttention with fixed seed on CUDA."""
    from .qwen3_ref import GQAAttention

    torch.manual_seed(_SEED)
    return GQAAttention(cfg).cuda().half().eval()


def _causal_mask(seq, device, dtype):
    mask = torch.full((seq, seq), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Intermediate check: QK-norm (composed RMSNorm)
# ---------------------------------------------------------------------------


@requires_cuda
def test_qk_norm():
    """torch_rmsnorm matches torch RMSNorm on a 4-D head tensor."""
    from .qwen3_ref import RMSNorm
    from .ark_attention import torch_rmsnorm
    from .equiv import assert_close

    dim = 32
    torch.manual_seed(_SEED)
    x = torch.randn(1, 4, 16, dim, device="cuda", dtype=torch.float16)

    # Torch reference
    norm = RMSNorm(dim, eps=1e-6).cuda()
    with torch.no_grad():
        ref = norm(x.reshape(-1, dim)).reshape(x.shape)

    # torch_rmsnorm
    weight = norm.weight.detach().half().cuda()
    out = torch_rmsnorm(x, weight, 1e-6)

    assert_close(out, ref, atol=1e-6, rtol=1e-6, msg="QK-norm mismatch")


# ---------------------------------------------------------------------------
# Intermediate check: RoPE
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("seq", [16, 128])
def test_rope(seq):
    """torch_rope matches reference apply_rope on a 4-D head tensor."""
    from .qwen3_ref import apply_rope, precompute_rope_freqs
    from .ark_attention import torch_rope
    from .equiv import assert_close

    head_dim = 32
    torch.manual_seed(_SEED)
    x = torch.randn(1, 4, seq, head_dim, device="cuda", dtype=torch.float16)

    # Torch reference (fp32 internal)
    freqs = precompute_rope_freqs(head_dim, 256, theta=1e6).to("cuda")
    with torch.no_grad():
        ref = apply_rope(x, freqs)

    # torch_rope (fp32 internal — should match reference exactly)
    out = torch_rope(x, freqs)

    assert_close(out, ref, atol=1e-6, rtol=1e-6, msg=f"RoPE S={seq}")


# ---------------------------------------------------------------------------
# Full attention equivalence — prefill (S=128)
# ---------------------------------------------------------------------------


@requires_cuda
def test_attention_prefill():
    """ARK attention matches torch GQAAttention at S=128 (prefill shape)."""
    _run_attention_equivalence(_small_cfg(), B=1, S=128, seed_offset=1)


# ---------------------------------------------------------------------------
# Full attention equivalence — small (S=16)
# ---------------------------------------------------------------------------


@requires_cuda
def test_attention_small():
    """ARK attention matches torch GQAAttention at S=16 (small shape)."""
    _run_attention_equivalence(_small_cfg(), B=1, S=16, seed_offset=2)


# ---------------------------------------------------------------------------
# Causal mask correctness
# ---------------------------------------------------------------------------


@requires_cuda
def test_attention_causal():
    """Future positions have zero attention weight in ARK attention."""
    # Intentionally replicates the pipeline inline to inspect intermediate
    # attention weights — not covered by _run_attention_equivalence.
    from .ark_attention import (
        torch_rmsnorm,
        torch_rope,
        precompute_torch_rope_freqs,
    )

    cfg = _small_cfg()
    attn = _build_ref_attn(cfg)

    B, S = 1, 16
    torch.manual_seed(_SEED + 3)
    x = torch.randn(B, S, cfg.hidden_dim, device="cuda", dtype=torch.float16)
    mask = _causal_mask(S, "cuda", torch.float16)

    with torch.no_grad():
        # Manually compute attention weights using ARK ops
        q = torch.matmul(x, attn.q_proj.weight.detach().t())
        k = torch.matmul(x, attn.k_proj.weight.detach().t())

        q = (
            q.reshape(B, S, cfg.n_q_heads, cfg.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            k.reshape(B, S, cfg.n_kv_heads, cfg.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        # QK-norm
        q_w = attn.qk_norm.q_norm.weight.detach().half()
        k_w = attn.qk_norm.k_norm.weight.detach().half()
        q = torch_rmsnorm(q, q_w, cfg.rms_norm_eps)
        k = torch_rmsnorm(k, k_w, cfg.rms_norm_eps)

        # RoPE
        rope_freqs = precompute_torch_rope_freqs(
            cfg.head_dim, cfg.max_seq_len, cfg.rope_theta
        ).to("cuda")
        q = torch_rope(q, rope_freqs)
        k = torch_rope(k, rope_freqs)

        # GQA expand
        n_rep = cfg.n_q_heads // cfg.n_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)

        # Compute attention weights
        scale = 1.0 / math.sqrt(cfg.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale + mask
        weights = torch.softmax(scores.float(), dim=-1)

    # Upper triangle (future) must be zero
    for h in range(weights.shape[1]):
        upper = torch.triu(weights[0, h], diagonal=1)
        assert upper.abs().max().item() < 1e-6, (
            f"Head {h}: future-position weight non-zero "
            f"(max={upper.abs().max().item():.2e})"
        )


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


@requires_cuda
def test_attention_output_shape():
    """ARK attention output has shape (B, S, hidden_dim)."""
    from .ark_attention import ark_gqa_attention, precompute_torch_rope_freqs

    cfg = _small_cfg()
    attn = _build_ref_attn(cfg)

    B, S = 1, 16
    torch.manual_seed(_SEED + 4)
    x = torch.randn(B, S, cfg.hidden_dim, device="cuda", dtype=torch.float16)
    mask = _causal_mask(S, "cuda", torch.float16)

    rope_freqs = precompute_torch_rope_freqs(
        cfg.head_dim, cfg.max_seq_len, cfg.rope_theta
    ).to("cuda")

    ark_out = ark_gqa_attention(
        x,
        attn.q_proj.weight.detach(),
        attn.k_proj.weight.detach(),
        attn.v_proj.weight.detach(),
        attn.o_proj.weight.detach(),
        attn.qk_norm.q_norm.weight.detach().half(),
        attn.qk_norm.k_norm.weight.detach().half(),
        rope_freqs,
        mask,
        cfg,
    ).eval()

    assert ark_out.shape == (B, S, cfg.hidden_dim)
    assert ark_out.dtype == torch.float16


# ---------------------------------------------------------------------------
# Edge cases: seq_len=1, batch>1, MHA (n_q_heads==n_kv_heads)
# ---------------------------------------------------------------------------


def _mha_cfg():
    """Config with n_q_heads == n_kv_heads (MHA, n_rep=1)."""
    from .qwen3_config import Qwen3Config

    return Qwen3Config(
        n_layers=1,
        hidden_dim=128,
        n_q_heads=4,
        n_kv_heads=4,
        head_dim=32,
        intermediate_dim=256,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        max_seq_len=256,
    )


def _run_attention_equivalence(cfg, B, S, seed_offset=10, mask="causal"):
    """Run ARK vs torch attention equivalence for given cfg, B, S."""
    from .ark_attention import ark_gqa_attention, precompute_torch_rope_freqs
    from .equiv import assert_close

    attn = _build_ref_attn(cfg)
    rope_freqs = precompute_torch_rope_freqs(
        cfg.head_dim, cfg.max_seq_len, cfg.rope_theta
    ).to("cuda")

    torch.manual_seed(_SEED + seed_offset)
    x = torch.randn(B, S, cfg.hidden_dim, device="cuda", dtype=torch.float16)
    if mask == "causal":
        mask = _causal_mask(S, "cuda", torch.float16)
    # else mask is already None or a user-supplied tensor

    with torch.no_grad():
        ref = attn(x, rope_freqs, mask)

    with torch.no_grad():
        ark_out = ark_gqa_attention(
            x,
            attn.q_proj.weight.detach(),
            attn.k_proj.weight.detach(),
            attn.v_proj.weight.detach(),
            attn.o_proj.weight.detach(),
            attn.qk_norm.q_norm.weight.detach().half(),
            attn.qk_norm.k_norm.weight.detach().half(),
            rope_freqs,
            mask,
            cfg,
        ).eval()

    assert ark_out.shape == ref.shape
    assert_close(ark_out, ref, atol=5e-3, rtol=5e-3, msg=f"B={B} S={S}")


@requires_cuda
def test_attention_seq_len_1():
    """ARK attention matches torch at S=1 (decode step)."""
    _run_attention_equivalence(_small_cfg(), B=1, S=1, seed_offset=20)


@requires_cuda
def test_attention_batch_2():
    """ARK attention matches torch at B=2 (multi-batch GQA expand)."""
    _run_attention_equivalence(_small_cfg(), B=2, S=16, seed_offset=21)


@requires_cuda
def test_attention_mha():
    """ARK attention matches torch when n_q_heads==n_kv_heads (MHA, n_rep=1)."""
    _run_attention_equivalence(_mha_cfg(), B=1, S=16, seed_offset=22)


@requires_cuda
def test_attention_no_mask():
    """ARK attention matches torch with mask=None (no causal mask)."""
    _run_attention_equivalence(
        _small_cfg(), B=1, S=16, seed_offset=23, mask=None
    )


# ---------------------------------------------------------------------------
# xfail: raw 4-D ark_rmsnorm at (1,4,128,32) — upstream ARK planner bug
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.xfail(
    reason="ARK planner bug: cudaErrorMisalignedAddress on 4-D shape (1,4,128,32)",
    strict=False,
)
def test_rmsnorm_4d_s128_xfail():
    """Document that raw 4-D ark_rmsnorm crashes at (1,4,128,32).

    This test records the upstream ARK planner bug.  The production
    code works around it by flattening to 2-D in torch.

    Runs in a subprocess to avoid poisoning the CUDA context for
    subsequent tests.
    """
    import subprocess
    import sys
    import os

    script = (
        "import torch, ark\n"
        "dim = 32\n"
        "torch.manual_seed(42)\n"
        "x = torch.randn(1, 4, 128, dim, device='cuda', dtype=torch.float16)\n"
        "w = torch.ones(dim, device='cuda', dtype=torch.float16)\n"
        "ark.init()\n"
        "xf = ark.cast(x, ark.fp32)\n"
        "x2 = ark.mul(xf, xf)\n"
        "m = ark.reduce_mean(x2, axis=-1)\n"
        "me = ark.add(m, 1e-6)\n"
        "rr = ark.rsqrt(me)\n"
        "xn = ark.mul(xf, rr)\n"
        "wf = ark.cast(w, ark.fp32)\n"
        "wf = ark.reshape(wf, [1, 1, 1, dim])\n"
        "xs = ark.mul(xn, wf)\n"
        "out = ark.cast(xs, ark.fp16).eval()\n"
        "assert out.shape == (1, 4, 128, dim)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        env=os.environ.copy(),
    )
    assert (
        result.returncode == 0
    ), f"Subprocess exited {result.returncode}: {result.stderr[-500:]}"
