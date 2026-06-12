#!/usr/bin/env python3
"""Unit tests for profile_sglang.py and classify_kernels.py.

CPU-only — no GPU, no SGLang server, no network required.
Tests: kernel classifier correctness, profiler schedule construction,
argument parsing, and module imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add bench directory to path so we can import the modules under test.
_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

import classify_kernels  # noqa: E402
import profile_sglang  # noqa: E402

# =========================================================================
# classify_kernels tests
# =========================================================================


class TestClassifyKernel:
    """Test classify_kernel() with known kernel name patterns."""

    # --- Attention kernels ---

    @pytest.mark.parametrize(
        "name",
        [
            "flash_fwd_splitkv_kernel",
            "fmha_v2_flash_attention_fp16_128_64_S_128_kernel",
            "flashinfer_batch_prefill_ragged",
            "cudnn_sdpa_fwd_fp16",
            "sdpa_backward_kernel",
        ],
    )
    def test_attention_kernels(self, name: str) -> None:
        assert (
            classify_kernels.classify_kernel(name) == classify_kernels.ATTENTION
        )

    # --- NCCL kernels ---

    @pytest.mark.parametrize(
        "name",
        [
            "ncclDevKernel_AllReduce_Sum_f16",
            "nccl_allreduce_ring_ll_bf16",
            "ncclKernel_ReduceScatter_fp32",
        ],
    )
    def test_nccl_kernels(self, name: str) -> None:
        assert classify_kernels.classify_kernel(name) == classify_kernels.NCCL

    # --- Norms / RoPE / activations ---

    @pytest.mark.parametrize(
        "name",
        [
            "rms_norm_kernel_fp16",
            "rmsnorm_fwd",
            "layernorm_forward_cuda",
            "layer_norm_kernel",
            "fused_rms_norm_kernel",
            "rotary_embedding_kernel",
            "rope_forward_kernel",
            "silu_and_mul_kernel",
            "gelu_forward",
            "elementwise_kernel_add",
        ],
    )
    def test_norms_rope_kernels(self, name: str) -> None:
        assert (
            classify_kernels.classify_kernel(name)
            == classify_kernels.NORMS_ROPE
        )

    # --- GEMM kernels with shape disambiguation ---

    def test_gemm_mlp_gate_up(self) -> None:
        """GEMM with MLP gate/up dimension (1792 for TP=8)."""
        result = classify_kernels.classify_kernel(
            "cutlass_80_tensorop_f16_s16816gemm_f16_256x128_32x3_nn",
            shapes=[[2048, 4096], [4096, 1792]],
            tp=8,
        )
        assert result == classify_kernels.GEMM_MLP

    def test_gemm_mlp_fused_gate_up(self) -> None:
        """GEMM with fused gate+up dimension (3584 for TP=8)."""
        result = classify_kernels.classify_kernel(
            "cublas_GemmEx_f16",
            shapes=[[2048, 4096], [4096, 3584]],
            tp=8,
        )
        assert result == classify_kernels.GEMM_MLP

    def test_gemm_mlp_down(self) -> None:
        """GEMM with MLP down dimension (1792 input for TP=8)."""
        result = classify_kernels.classify_kernel(
            "sm80_xmma_gemm_f16f16_f32_f32_tn_n_tilesize128x128x32",
            shapes=[[2048, 1792], [1792, 4096]],
            tp=8,
        )
        assert result == classify_kernels.GEMM_MLP

    def test_gemm_attention_qkv(self) -> None:
        """GEMM with fused QKV dimension (768 for TP=8)."""
        result = classify_kernels.classify_kernel(
            "cutlass_80_tensorop_f16_gemm",
            shapes=[[2048, 4096], [4096, 768]],
            tp=8,
        )
        assert result == classify_kernels.GEMM_ATTENTION

    def test_gemm_attention_q_proj(self) -> None:
        """GEMM with Q projection dimension (512 for TP=8)."""
        result = classify_kernels.classify_kernel(
            "cutlass_80_tensorop_f16_gemm",
            shapes=[[2048, 4096], [4096, 512]],
            tp=8,
        )
        assert result == classify_kernels.GEMM_ATTENTION

    def test_gemm_attention_kv_proj(self) -> None:
        """GEMM with K/V projection dimension (128 for TP=8)."""
        result = classify_kernels.classify_kernel(
            "cublas_GemmEx_bf16",
            shapes=[[2048, 4096], [4096, 128]],
            tp=8,
        )
        assert result == classify_kernels.GEMM_ATTENTION

    def test_gemm_embed_lm_head(self) -> None:
        """GEMM with vocab dimension (151936)."""
        result = classify_kernels.classify_kernel(
            "cutlass_80_tensorop_f16_gemm",
            shapes=[[1, 4096], [4096, 151936]],
            tp=8,
        )
        assert result == classify_kernels.EMBED_LM_HEAD

    def test_gemm_embed_lm_head_sharded(self) -> None:
        """GEMM with sharded vocab dimension (151936 / 8 = 18992)."""
        result = classify_kernels.classify_kernel(
            "cutlass_80_tensorop_f16_gemm",
            shapes=[[1, 4096], [4096, 18992]],
            tp=8,
        )
        assert result == classify_kernels.EMBED_LM_HEAD

    def test_gemm_no_shapes_falls_to_other(self) -> None:
        """GEMM without shapes cannot be disambiguated → OTHER."""
        result = classify_kernels.classify_kernel(
            "cutlass_80_tensorop_f16_gemm",
            shapes=None,
            tp=8,
        )
        assert result == classify_kernels.OTHER

    def test_gemm_unknown_shapes_falls_to_other(self) -> None:
        """GEMM with unrecognised shapes → OTHER."""
        result = classify_kernels.classify_kernel(
            "cutlass_80_tensorop_f16_gemm",
            shapes=[[32, 32], [32, 32]],
            tp=8,
        )
        assert result == classify_kernels.OTHER

    # --- Unknown kernels ---

    def test_unknown_kernel(self) -> None:
        assert (
            classify_kernels.classify_kernel("some_random_kernel")
            == classify_kernels.OTHER
        )

    def test_empty_name(self) -> None:
        assert classify_kernels.classify_kernel("") == classify_kernels.OTHER

    # --- TP=1 shapes ---

    def test_gemm_mlp_tp1(self) -> None:
        """MLP gate dimension at TP=1 is 14336."""
        result = classify_kernels.classify_kernel(
            "cutlass_80_tensorop_f16_gemm",
            shapes=[[2048, 4096], [4096, 14336]],
            tp=1,
        )
        assert result == classify_kernels.GEMM_MLP

    def test_gemm_tp1_hidden_is_not_attention(self) -> None:
        """At TP=1 q_dim == HIDDEN; HIDDEN is discarded so this is 'other'."""
        result = classify_kernels.classify_kernel(
            "cutlass_80_tensorop_f16_gemm",
            shapes=[[1, 4096], [4096, 4096]],
            tp=1,
        )
        assert result == classify_kernels.OTHER

    def test_gemm_tp1_fused_qkv_is_attention(self) -> None:
        """At TP=1 fused_qkv = 4096 + 2*1024 = 6144, a distinctive dim."""
        result = classify_kernels.classify_kernel(
            "cutlass_80_tensorop_f16_gemm",
            shapes=[[1, 4096], [4096, 6144]],
            tp=1,
        )
        assert result == classify_kernels.GEMM_ATTENTION


class TestClassifyTraceEvents:
    """Test classify_trace_events() aggregation."""

    def test_aggregation(self) -> None:
        events = [
            {"name": "flash_fwd_kernel", "dur": 100.0},
            {"name": "flash_fwd_kernel", "dur": 200.0},
            {"name": "ncclDevKernel_AllReduce", "dur": 50.0},
            {"name": "rms_norm_kernel", "dur": 30.0},
            {"name": "unknown_kernel_xyz", "dur": 10.0},
        ]
        budgets = classify_kernels.classify_trace_events(events, tp=8)

        assert budgets[classify_kernels.ATTENTION].total_us == pytest.approx(
            300.0
        )
        assert budgets[classify_kernels.ATTENTION].kernel_count == 2
        assert budgets[classify_kernels.NCCL].total_us == pytest.approx(50.0)
        assert budgets[classify_kernels.NORMS_ROPE].total_us == pytest.approx(
            30.0
        )
        assert budgets[classify_kernels.OTHER].total_us == pytest.approx(10.0)

    def test_empty_events(self) -> None:
        budgets = classify_kernels.classify_trace_events([])
        for comp in classify_kernels.ALL_COMPONENTS:
            assert budgets[comp].total_us == 0.0
            assert budgets[comp].kernel_count == 0

    def test_shapes_in_events(self) -> None:
        events = [
            {
                "name": "cutlass_80_gemm",
                "dur": 500.0,
                "args": {"shapes": [[2048, 4096], [4096, 1792]]},
            },
        ]
        budgets = classify_kernels.classify_trace_events(events, tp=8)
        assert budgets[classify_kernels.GEMM_MLP].total_us == pytest.approx(
            500.0
        )


class TestComponentBudget:
    """Test ComponentBudget dataclass helpers."""

    def test_total_ms(self) -> None:
        b = classify_kernels.ComponentBudget(component="test", total_us=1500.0)
        assert b.total_ms == pytest.approx(1.5)

    def test_pct_of(self) -> None:
        b = classify_kernels.ComponentBudget(component="test", total_us=250.0)
        assert b.pct_of(1000.0) == pytest.approx(25.0)

    def test_pct_of_zero(self) -> None:
        b = classify_kernels.ComponentBudget(component="test", total_us=100.0)
        assert b.pct_of(0.0) == pytest.approx(0.0)


class TestFormatBudgetTable:
    """Test format_budget_table() output structure."""

    def test_markdown_table(self) -> None:
        budgets = {
            comp: classify_kernels.ComponentBudget(component=comp)
            for comp in classify_kernels.ALL_COMPONENTS
        }
        budgets[classify_kernels.ATTENTION].total_us = 1000.0
        budgets[classify_kernels.GEMM_MLP].total_us = 2000.0

        table = classify_kernels.format_budget_table(budgets, "Test Phase")
        assert "## Test Phase" in table
        assert "| Component |" in table
        assert "gemm_mlp" in table
        assert "attention" in table
        assert "**Total**" in table


# =========================================================================
# profile_sglang tests
# =========================================================================


class TestParseArgs:
    """Test profile_sglang.parse_args()."""

    def test_defaults(self) -> None:
        args = profile_sglang.parse_args([])
        assert args.port == 30000
        assert args.host == "localhost"
        assert args.phase == "both"
        assert args.prompt_len == 2048
        assert args.decode_tokens == 128
        assert args.trials == 5
        assert args.output_dir == "/tmp/sglang_profile"

    def test_custom_args(self) -> None:
        args = profile_sglang.parse_args(
            [
                "--port",
                "8080",
                "--host",
                "gpu-node",
                "--phase",
                "prefill",
                "--prompt-len",
                "1024",
                "--decode-tokens",
                "64",
                "--trials",
                "3",
                "--output-dir",
                "/data/traces",
            ]
        )
        assert args.port == 8080
        assert args.host == "gpu-node"
        assert args.phase == "prefill"
        assert args.prompt_len == 1024
        assert args.decode_tokens == 64
        assert args.trials == 3
        assert args.output_dir == "/data/traces"


class TestBuildProfilerConfig:
    """Test profile_sglang.build_profiler_config()."""

    def test_config_structure(self) -> None:
        config = profile_sglang.build_profiler_config()
        assert config["record_shapes"] is True
        assert config["with_stack"] is True
        assert config["schedule"]["wait"] == 2
        assert config["schedule"]["warmup"] == 1
        assert config["schedule"]["active"] == 3
        assert "activities" in config

    def test_custom_output_dir(self) -> None:
        config = profile_sglang.build_profiler_config("/custom/dir")
        assert config["output_dir"] == "/custom/dir"


class TestBuildProfilerSchedule:
    """Test profile_sglang.build_profiler_schedule()."""

    def test_returns_callable(self) -> None:
        pytest.importorskip("torch")
        schedule = profile_sglang.build_profiler_schedule()
        assert callable(schedule)

    def test_schedule_phases(self) -> None:
        """Verify the schedule produces the expected action sequence."""
        torch_profiler = pytest.importorskip("torch.profiler")

        schedule = profile_sglang.build_profiler_schedule()
        # wait=2 → NONE, NONE; warmup=1 → WARMUP; active=3 → RECORD x3
        expected = [
            torch_profiler.ProfilerAction.NONE,  # step 0 (wait)
            torch_profiler.ProfilerAction.NONE,  # step 1 (wait)
            torch_profiler.ProfilerAction.WARMUP,  # step 2 (warmup)
            torch_profiler.ProfilerAction.RECORD,  # step 3 (active)
            torch_profiler.ProfilerAction.RECORD,  # step 4 (active)
            torch_profiler.ProfilerAction.RECORD_AND_SAVE,  # step 5 (last active)
        ]
        for step, exp_action in enumerate(expected):
            assert schedule(step) == exp_action


class TestBuildPrompt:
    """Test profile_sglang.build_prompt()."""

    def test_prompt_length(self) -> None:
        prompt = profile_sglang.build_prompt(100)
        # Overgenerate by 1.2×, each "hello " is one word
        words = prompt.strip().split()
        assert len(words) == 120  # 100 * 1.2

    def test_prompt_not_empty(self) -> None:
        prompt = profile_sglang.build_prompt(1)
        assert len(prompt) > 0


class TestRequireRequests:
    """Test _require_requests() raises ImportError when requests is missing."""

    def test_raises_when_requests_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(profile_sglang, "_requests", None)
        with pytest.raises(ImportError, match="requests"):
            profile_sglang._require_requests()


class TestClassifyKernelsMain:
    """Test classify_kernels.main() CLI entry point."""

    def test_main_prints_budget_table(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() loads a trace JSON, classifies events, and prints a budget table."""
        trace = {
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "flash_fwd_kernel",
                    "dur": 100.0,
                },
            ]
        }
        trace_file = tmp_path / "trace.json"
        import json

        trace_file.write_text(json.dumps(trace))

        classify_kernels.main([str(trace_file), "--tp", "8"])

        captured = capsys.readouterr().out
        assert "| Component |" in captured
        assert "attention" in captured


class TestBuildProfilerScheduleNoTorch:
    """Test build_profiler_schedule() when torch is absent."""

    def test_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(profile_sglang, "_torch_profiler", None)
        with pytest.raises(RuntimeError, match="torch is required"):
            profile_sglang.build_profiler_schedule()


class TestConstants:
    """Verify module-level constants are consistent."""

    def test_sglang_image_tag(self) -> None:
        assert (
            profile_sglang.SGLANG_IMAGE_TAG
            == "lmsysorg/sglang:v0.4.6.post1-cu124"
        )

    def test_all_components_complete(self) -> None:
        assert len(classify_kernels.ALL_COMPONENTS) == 7
        assert classify_kernels.ATTENTION in classify_kernels.ALL_COMPONENTS
        assert (
            classify_kernels.GEMM_ATTENTION in classify_kernels.ALL_COMPONENTS
        )
        assert classify_kernels.GEMM_MLP in classify_kernels.ALL_COMPONENTS
        assert classify_kernels.NCCL in classify_kernels.ALL_COMPONENTS
        assert classify_kernels.NORMS_ROPE in classify_kernels.ALL_COMPONENTS
        assert classify_kernels.EMBED_LM_HEAD in classify_kernels.ALL_COMPONENTS
        assert classify_kernels.OTHER in classify_kernels.ALL_COMPONENTS
