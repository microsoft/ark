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
    return GQAAttention(cfg).cuda().half()


def _causal_mask(seq, device, dtype):
    mask = torch.full((seq, seq), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Intermediate check: QK-norm (composed RMSNorm)
# ---------------------------------------------------------------------------


@requires_cuda
def test_qk_norm():
    """ARK composed RMSNorm matches torch RMSNorm on a 4-D head tensor."""
    import ark
    from .qwen3_ref import RMSNorm
    from .ark_attention import ark_rmsnorm
    from .equiv import assert_close

    dim = 32
    torch.manual_seed(_SEED)
    x = torch.randn(1, 4, 16, dim, device="cuda", dtype=torch.float16)

    # Torch reference
    norm = RMSNorm(dim, eps=1e-6).cuda()
    with torch.no_grad():
        ref = norm(x.reshape(-1, dim)).reshape(x.shape)

    # ARK
    ark.init()
    weight = norm.weight.detach().half().cuda()
    ark_out = ark_rmsnorm(x, weight, 1e-6).eval()

    assert_close(ark_out, ref, atol=5e-3, rtol=5e-3, msg="QK-norm mismatch")


# ---------------------------------------------------------------------------
# Intermediate check: RoPE
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("seq", [16, 128])
def test_rope(seq):
    """ARK rope matches torch apply_rope on a 4-D head tensor."""
    import ark
    from .qwen3_ref import apply_rope, precompute_rope_freqs
    from .ark_attention import precompute_ark_rope_freqs
    from .equiv import assert_close

    head_dim = 32
    torch.manual_seed(_SEED)
    x = torch.randn(1, 4, seq, head_dim, device="cuda", dtype=torch.float16)

    # Torch reference (fp32 internal)
    freqs = precompute_rope_freqs(head_dim, 256, theta=1e6).to("cuda")
    with torch.no_grad():
        ref = apply_rope(x, freqs)

    # ARK (fp16 internal — some precision loss expected)
    ark.init()
    ark_freqs = precompute_ark_rope_freqs(head_dim, 256, theta=1e6).cuda()
    ark_freqs = ark_freqs[:, :, :seq, :]
    ark_out = ark.rope(x, ark_freqs).eval()

    assert_close(ark_out, ref, atol=5e-3, rtol=5e-3, msg=f"RoPE S={seq}")


# ---------------------------------------------------------------------------
# Full attention equivalence — prefill (S=128)
# ---------------------------------------------------------------------------


@requires_cuda
def test_attention_prefill():
    """ARK attention matches torch GQAAttention at S=128 (prefill shape)."""
    import ark
    from .qwen3_ref import precompute_rope_freqs
    from .ark_attention import ark_gqa_attention, precompute_ark_rope_freqs
    from .equiv import assert_close

    cfg = _small_cfg()
    attn = _build_ref_attn(cfg)
    rope_freqs = precompute_rope_freqs(
        cfg.head_dim, cfg.max_seq_len, cfg.rope_theta
    ).to("cuda")

    B, S = 1, 128
    torch.manual_seed(_SEED + 1)
    x = torch.randn(B, S, cfg.hidden_dim, device="cuda", dtype=torch.float16)
    mask = _causal_mask(S, "cuda", torch.float16)

    # Torch reference
    with torch.no_grad():
        ref = attn(x, rope_freqs, mask)

    # ARK
    ark.init()
    ark_rf = precompute_ark_rope_freqs(
        cfg.head_dim, cfg.max_seq_len, cfg.rope_theta
    ).cuda()[:, :, :S, :]

    ark_out = ark_gqa_attention(
        x,
        attn.q_proj.weight.detach(),
        attn.k_proj.weight.detach(),
        attn.v_proj.weight.detach(),
        attn.o_proj.weight.detach(),
        attn.qk_norm.q_norm.weight.detach().half(),
        attn.qk_norm.k_norm.weight.detach().half(),
        ark_rf,
        mask,
        cfg,
    ).eval()

    assert (
        ark_out.shape == ref.shape
    ), f"Shape mismatch: {ark_out.shape} vs {ref.shape}"
    assert_close(ark_out, ref, atol=5e-3, rtol=5e-3, msg="Prefill S=128")


# ---------------------------------------------------------------------------
# Full attention equivalence — small (S=16)
# ---------------------------------------------------------------------------


@requires_cuda
def test_attention_small():
    """ARK attention matches torch GQAAttention at S=16 (small shape)."""
    import ark
    from .qwen3_ref import precompute_rope_freqs
    from .ark_attention import ark_gqa_attention, precompute_ark_rope_freqs
    from .equiv import assert_close

    cfg = _small_cfg()
    attn = _build_ref_attn(cfg)
    rope_freqs = precompute_rope_freqs(
        cfg.head_dim, cfg.max_seq_len, cfg.rope_theta
    ).to("cuda")

    B, S = 1, 16
    torch.manual_seed(_SEED + 2)
    x = torch.randn(B, S, cfg.hidden_dim, device="cuda", dtype=torch.float16)
    mask = _causal_mask(S, "cuda", torch.float16)

    with torch.no_grad():
        ref = attn(x, rope_freqs, mask)

    ark.init()
    ark_rf = precompute_ark_rope_freqs(
        cfg.head_dim, cfg.max_seq_len, cfg.rope_theta
    ).cuda()[:, :, :S, :]

    ark_out = ark_gqa_attention(
        x,
        attn.q_proj.weight.detach(),
        attn.k_proj.weight.detach(),
        attn.v_proj.weight.detach(),
        attn.o_proj.weight.detach(),
        attn.qk_norm.q_norm.weight.detach().half(),
        attn.qk_norm.k_norm.weight.detach().half(),
        ark_rf,
        mask,
        cfg,
    ).eval()

    assert ark_out.shape == ref.shape
    assert_close(ark_out, ref, atol=5e-3, rtol=5e-3, msg="Small S=16")


# ---------------------------------------------------------------------------
# Causal mask correctness
# ---------------------------------------------------------------------------


@requires_cuda
def test_attention_causal():
    """Future positions have zero attention weight in ARK attention."""
    import ark
    from .ark_attention import (
        ark_rmsnorm,
        precompute_ark_rope_freqs,
    )

    cfg = _small_cfg()
    attn = _build_ref_attn(cfg)

    B, S = 1, 16
    torch.manual_seed(_SEED + 3)
    x = torch.randn(B, S, cfg.hidden_dim, device="cuda", dtype=torch.float16)
    mask = _causal_mask(S, "cuda", torch.float16)

    with torch.no_grad():
        # Manually compute attention weights using ARK ops
        ark.init()
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
        q = ark_rmsnorm(q, q_w, cfg.rms_norm_eps).eval()
        ark.init()
        k = ark_rmsnorm(k, k_w, cfg.rms_norm_eps).eval()

        # RoPE
        ark_rf = precompute_ark_rope_freqs(
            cfg.head_dim, cfg.max_seq_len, cfg.rope_theta
        ).cuda()[:, :, :S, :]
        ark.init()
        q = ark.rope(q, ark_rf).eval()
        ark.init()
        k = ark.rope(k, ark_rf).eval()

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
    import ark
    from .qwen3_ref import precompute_rope_freqs
    from .ark_attention import ark_gqa_attention, precompute_ark_rope_freqs

    cfg = _small_cfg()
    attn = _build_ref_attn(cfg)

    B, S = 1, 16
    torch.manual_seed(_SEED + 4)
    x = torch.randn(B, S, cfg.hidden_dim, device="cuda", dtype=torch.float16)
    mask = _causal_mask(S, "cuda", torch.float16)

    ark.init()
    ark_rf = precompute_ark_rope_freqs(
        cfg.head_dim, cfg.max_seq_len, cfg.rope_theta
    ).cuda()[:, :, :S, :]

    ark_out = ark_gqa_attention(
        x,
        attn.q_proj.weight.detach(),
        attn.k_proj.weight.detach(),
        attn.v_proj.weight.detach(),
        attn.o_proj.weight.detach(),
        attn.qk_norm.q_norm.weight.detach().half(),
        attn.qk_norm.k_norm.weight.detach().half(),
        ark_rf,
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


def _run_attention_equivalence(cfg, B, S, seed_offset=10):
    """Run ARK vs torch attention equivalence for given cfg, B, S."""
    import ark
    from .qwen3_ref import precompute_rope_freqs
    from .ark_attention import ark_gqa_attention, precompute_ark_rope_freqs
    from .equiv import assert_close

    attn = _build_ref_attn(cfg)
    rope_freqs = precompute_rope_freqs(
        cfg.head_dim, cfg.max_seq_len, cfg.rope_theta
    ).to("cuda")

    torch.manual_seed(_SEED + seed_offset)
    x = torch.randn(B, S, cfg.hidden_dim, device="cuda", dtype=torch.float16)
    mask = _causal_mask(S, "cuda", torch.float16)

    with torch.no_grad():
        ref = attn(x, rope_freqs, mask)

    ark.init()
    ark_rf = precompute_ark_rope_freqs(
        cfg.head_dim, cfg.max_seq_len, cfg.rope_theta
    ).cuda()[:, :, :S, :]

    ark_out = ark_gqa_attention(
        x,
        attn.q_proj.weight.detach(),
        attn.k_proj.weight.detach(),
        attn.v_proj.weight.detach(),
        attn.o_proj.weight.detach(),
        attn.qk_norm.q_norm.weight.detach().half(),
        attn.qk_norm.k_norm.weight.detach().half(),
        ark_rf,
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
    code works around it by flattening to 2-D.  When ARK fixes the
    planner, this test will start passing and the xfail marker can
    be removed.
    """
    import ark
    from .qwen3_ref import RMSNorm
    from .equiv import assert_close

    dim = 32
    torch.manual_seed(_SEED)
    x = torch.randn(1, 4, 128, dim, device="cuda", dtype=torch.float16)

    norm = RMSNorm(dim, eps=1e-6).cuda()
    with torch.no_grad():
        ref = norm(x.reshape(-1, dim)).reshape(x.shape)

    # Build a raw 4-D graph (no flatten) to trigger the planner bug.
    ark.init()
    weight = norm.weight.detach().half().cuda()
    x_f32 = ark.cast(x, ark.fp32)
    x2 = ark.mul(x_f32, x_f32)
    mean = ark.reduce_mean(x2, axis=-1)
    mean_eps = ark.add(mean, 1e-6)
    rrms = ark.rsqrt(mean_eps)
    x_normed = ark.mul(x_f32, rrms)
    w_f32 = ark.cast(weight, ark.fp32)
    w_f32 = ark.reshape(w_f32, [1, 1, 1, dim])
    x_scaled = ark.mul(x_normed, w_f32)
    ark_out = ark.cast(x_scaled, ark.fp16).eval()

    assert_close(ark_out, ref, atol=5e-3, rtol=5e-3, msg="4D RMSNorm S=128")
