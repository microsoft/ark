# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the Qwen3 component harness.

All GPU tests are skipped when CUDA is unavailable.
"""

import pytest
import torch

_CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _CUDA, reason="CUDA not available")


# ---------------------------------------------------------------------------
# Reference model tests
# ---------------------------------------------------------------------------


@requires_cuda
def test_ref_forward_shape():
    """Reference model produces logits of correct shape (1 layer, seq=128)."""
    from .qwen3_config import Qwen3Config
    from .qwen3_ref import Qwen3Model

    cfg = Qwen3Config(n_layers=1)
    model = Qwen3Model(cfg, seed=42).cuda().half()

    batch, seq = 1, 128
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device="cuda")

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (
        batch,
        seq,
        cfg.vocab_size,
    ), f"Expected shape ({batch}, {seq}, {cfg.vocab_size}), got {logits.shape}"
    assert logits.dtype == torch.float16


@requires_cuda
def test_rmsnorm_unit_rms():
    """RMSNorm output has approximately unit RMS (within eps tolerance)."""
    from .qwen3_ref import RMSNorm

    dim = 128
    eps = 1e-6
    norm = RMSNorm(dim, eps=eps).cuda()

    x = torch.randn(2, 64, dim, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        y = norm(x)

    # RMS of each vector should be close to 1 (since weight is ones)
    rms = y.float().pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(
        rms,
        torch.ones_like(rms),
        atol=0.05,
        rtol=0.05,
        msg="RMSNorm output should have approximately unit RMS",
    )


@requires_cuda
def test_attention_is_causal():
    """Causal attention zeroes future positions in attention weights.

    Verifies that the upper-triangular portion of the attention-weight
    matrix is zero (future tokens cannot attend to past).
    """
    import math
    from .qwen3_config import Qwen3Config
    from .qwen3_ref import GQAAttention, precompute_rope_freqs

    cfg = Qwen3Config(
        n_layers=1, hidden_dim=128, n_q_heads=4, n_kv_heads=2, head_dim=32
    )
    attn = GQAAttention(cfg).cuda().half()
    rope_freqs = precompute_rope_freqs(
        cfg.head_dim, cfg.max_seq_len, cfg.rope_theta
    ).to("cuda")

    batch, seq = 1, 16
    x = torch.randn(
        batch, seq, cfg.hidden_dim, device="cuda", dtype=torch.float16
    )

    # Build causal mask
    mask = torch.full(
        (seq, seq), float("-inf"), device="cuda", dtype=torch.float16
    )
    mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        # Manually compute attention weights to inspect them
        q = (
            attn.q_proj(x)
            .reshape(batch, seq, cfg.n_q_heads, cfg.head_dim)
            .transpose(1, 2)
        )
        k = (
            attn.k_proj(x)
            .reshape(batch, seq, cfg.n_kv_heads, cfg.head_dim)
            .transpose(1, 2)
        )

        from .qwen3_ref import apply_rope

        q, k = attn.qk_norm(q, k)
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        if cfg.n_q_heads // cfg.n_kv_heads > 1:
            k = k.repeat_interleave(cfg.n_q_heads // cfg.n_kv_heads, dim=1)

        scale = 1.0 / math.sqrt(cfg.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale + mask
        weights = torch.softmax(scores.float(), dim=-1)

    # Upper triangle (future positions) must be zero
    for h in range(weights.shape[1]):
        upper = torch.triu(weights[0, h], diagonal=1)
        assert upper.abs().max().item() < 1e-6, (
            f"Head {h}: future-position attention weight is non-zero "
            f"(max={upper.abs().max().item():.2e})"
        )


@requires_cuda
def test_rope_applied_per_head():
    """RoPE modifies Q/K values (not a no-op)."""
    from .qwen3_ref import apply_rope, precompute_rope_freqs

    head_dim = 64
    freqs = precompute_rope_freqs(head_dim, 256, theta=1e6).to("cuda")
    x = torch.randn(1, 4, 32, head_dim, device="cuda", dtype=torch.float16)

    y = apply_rope(x, freqs)

    # RoPE should change the tensor
    assert not torch.allclose(
        x, y, atol=1e-4
    ), "RoPE had no effect on the tensor"
    # Shape must be preserved
    assert x.shape == y.shape


# ---------------------------------------------------------------------------
# Equivalence helper tests
# ---------------------------------------------------------------------------


def test_equiv_pass_identical():
    """assert_close passes for identical tensors."""
    from .equiv import assert_close

    t = torch.randn(4, 8, device="cpu", dtype=torch.float16)
    assert_close(t, t.clone())  # should not raise


def test_equiv_fail_perturbed():
    """assert_close raises AssertionError on intentional mismatch."""
    from .equiv import assert_close

    t = torch.randn(4, 8, device="cpu", dtype=torch.float16)
    perturbed = t + 10.0  # large perturbation

    with pytest.raises(AssertionError, match="not close"):
        assert_close(perturbed, t, atol=1e-3, rtol=1e-3)


def test_get_dtype_invalid():
    """_get_dtype raises ValueError for invalid dtype strings."""
    from .qwen3_config import Qwen3Config
    from .qwen3_ref import _get_dtype

    cfg = Qwen3Config(dtype="not_a_dtype")
    with pytest.raises(ValueError, match="Invalid dtype"):
        _get_dtype(cfg)


def test_config_invalid_head_ratio():
    """Qwen3Config rejects n_q_heads not divisible by n_kv_heads."""
    from .qwen3_config import Qwen3Config

    with pytest.raises(ValueError):
        Qwen3Config(n_layers=1, n_q_heads=5, n_kv_heads=2, head_dim=32)


def test_equiv_shape_mismatch():
    """assert_close raises on shape mismatch."""
    from .equiv import assert_close

    a = torch.randn(4, 8, device="cpu")
    b = torch.randn(4, 9, device="cpu")

    with pytest.raises(AssertionError, match="Shape mismatch"):
        assert_close(a, b)


# ---------------------------------------------------------------------------
# Microbench helper tests
# ---------------------------------------------------------------------------


@requires_cuda
def test_microbench_returns_positive():
    """microbench returns dict with mean_us > 0 for a trivial matmul."""
    from .microbench import microbench

    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.float16)

    def fn():
        torch.matmul(a, b)

    result = microbench(fn, n_iters=10, use_cuda_graph=False, flush_l2=False)

    assert isinstance(result, dict)
    assert "mean_us" in result
    assert "std_us" in result
    assert "n_iters" in result
    assert (
        result["mean_us"] > 0
    ), f"Expected mean_us > 0, got {result['mean_us']}"
    assert result["n_iters"] > 0


@requires_cuda
def test_microbench_with_cuda_graph():
    """microbench works with CUDA graph capture."""
    from .microbench import microbench

    a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    b = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    c = torch.empty(128, 128, device="cuda", dtype=torch.float16)

    def fn():
        torch.mm(a, b, out=c)

    result = microbench(fn, n_iters=10, use_cuda_graph=True, flush_l2=False)

    assert result["mean_us"] > 0
    assert result["n_iters"] > 0


@requires_cuda
def test_microbench_with_flush_l2():
    """microbench works with L2 flush enabled."""
    from .microbench import microbench

    a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    b = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    c = torch.empty(128, 128, device="cuda", dtype=torch.float16)

    def fn():
        torch.mm(a, b, out=c)

    result = microbench(fn, n_iters=5, use_cuda_graph=False, flush_l2=True)
    assert result["mean_us"] > 0
    assert result["n_iters"] > 0
