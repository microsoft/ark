#!/usr/bin/env python3
"""Unit tests for measure_baseline pure functions."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from measure_baseline import (
    build_prompt,
    capture_gpu_clocks,
    measure_decode,
    measure_ttft,
    send_request,
)


class TestBuildPrompt:
    def test_zero_tokens_returns_empty(self):
        assert build_prompt(0) == ""

    def test_one_token_returns_at_least_one_word(self):
        result = build_prompt(1)
        assert "hello " in result

    def test_100_tokens_has_at_least_120_repetitions(self):
        result = build_prompt(100)
        # 1.2x multiplier means at least 120 repetitions
        assert result.count("hello ") >= 120


class TestCaptureGpuClocks:
    def test_returns_string(self):
        # Even without nvidia-smi, should return a string
        result = capture_gpu_clocks()
        assert isinstance(result, str)

    @patch("measure_baseline.subprocess.run", side_effect=FileNotFoundError)
    def test_missing_nvidia_smi_returns_fallback(self, mock_run):
        result = capture_gpu_clocks()
        assert result == "nvidia-smi not available"

    @patch(
        "measure_baseline.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10),
    )
    def test_timeout_returns_fallback(self, mock_run):
        result = capture_gpu_clocks()
        assert result == "nvidia-smi not available"


class TestSendRequest:
    @patch("measure_baseline.requests.post")
    def test_warns_when_completion_tokens_missing(self, mock_post, capsys):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "hello world", "meta_info": {}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = send_request(
            "http://localhost:30000", "prompt", max_new_tokens=128
        )

        captured = capsys.readouterr()
        assert "completion_tokens missing" in captured.err
        # Falls back to word-count estimation
        assert result["output_tokens"] == 2  # "hello world" -> 2 words

    @patch("measure_baseline.requests.post")
    def test_uses_completion_tokens_from_meta(self, mock_post, capsys):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "text": "hello world",
            "meta_info": {"completion_tokens": 10},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = send_request(
            "http://localhost:30000", "prompt", max_new_tokens=128
        )

        captured = capsys.readouterr()
        assert captured.err == ""
        assert result["output_tokens"] == 10


class TestMeasureTtft:
    @patch("measure_baseline.send_request")
    def test_returns_total_ms(self, mock_send):
        mock_send.return_value = {"total_ms": 42.5}
        result = measure_ttft("http://localhost:30000", "prompt")
        assert result == 42.5
        mock_send.assert_called_once_with(
            "http://localhost:30000", "prompt", max_new_tokens=1
        )


class TestMeasureDecode:
    @patch("measure_baseline.send_request")
    def test_normal_path(self, mock_send):
        mock_send.return_value = {
            "total_ms": 1000.0,
            "output_tokens": 10,
            "output_text": "text",
        }
        result = measure_decode("http://localhost:30000", "prompt", 128)
        assert result["total_ms"] == 1000.0
        assert result["output_tokens"] == 10
        assert result["per_token_ms"] == 100.0

    @patch("measure_baseline.send_request")
    def test_zero_output_tokens_uses_guard(self, mock_send, capsys):
        mock_send.return_value = {
            "total_ms": 500.0,
            "output_tokens": 0,
            "output_text": "",
        }
        result = measure_decode("http://localhost:30000", "prompt", 128)
        assert result["per_token_ms"] == 500.0
        captured = capsys.readouterr()
        assert "0 output tokens" in captured.err
