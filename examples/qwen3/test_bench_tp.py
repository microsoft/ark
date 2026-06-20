# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""No-GPU tests for the Qwen3 TP perf-gate CLI contract."""

import os
import sys

import pytest

try:
    from . import bench_tp
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bench_tp


_VALID_SHA = "0123456789abcdef0123456789abcdef01234567"


def _perf_gate_lines(stdout):
    return [
        line for line in stdout.splitlines() if line.startswith("PERF_GATE ")
    ]


def test_main_emits_single_success_perf_gate(monkeypatch, capsys):
    """A passing benchmark prints one packet-route PERF_GATE line."""
    monkeypatch.setattr(sys, "argv", ["bench_tp.py"])
    monkeypatch.setattr(
        bench_tp,
        "run_bench",
        lambda world_size, timeout, hidden_size: (
            0.01,
            "all_reduce_packet",
            False,
        ),
    )
    monkeypatch.setattr(bench_tp, "_resolve_head_sha", lambda: _VALID_SHA)

    bench_tp.main()

    lines = _perf_gate_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert "name=tp" in lines[0]
    assert "route=all_reduce_packet" in lines[0]
    assert f"head_sha={_VALID_SHA}" in lines[0]


def test_main_fails_closed_with_unknown_route_and_sha(monkeypatch, capsys):
    """Failure still prints exactly one unknown-route PERF_GATE line."""
    monkeypatch.setattr(sys, "argv", ["bench_tp.py"])
    monkeypatch.setattr(
        bench_tp,
        "run_bench",
        lambda world_size, timeout, hidden_size: (
            bench_tp._SENTINEL_MS,
            "unknown",
            True,
        ),
    )
    monkeypatch.setattr(bench_tp, "_resolve_head_sha", lambda: "unknown")

    with pytest.raises(SystemExit) as exc_info:
        bench_tp.main()

    assert exc_info.value.code == 1
    lines = _perf_gate_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert "name=tp" in lines[0]
    assert "route=unknown" in lines[0]
    assert "head_sha=unknown" in lines[0]


@pytest.mark.parametrize(
    ("ark_ms", "route", "head_sha"),
    [
        (0.01, "unknown", _VALID_SHA),
        (0.01, "all_reduce_packet", "unknown"),
        (bench_tp._TP_TARGET_MS, "all_reduce_packet", _VALID_SHA),
    ],
)
def test_main_fails_closed_for_independent_gate_failures(
    monkeypatch, capsys, ark_ms, route, head_sha
):
    """Route, SHA, and threshold gates each fail closed independently."""
    monkeypatch.setattr(sys, "argv", ["bench_tp.py"])
    monkeypatch.setattr(
        bench_tp,
        "run_bench",
        lambda world_size, timeout, hidden_size: (ark_ms, route, False),
    )
    monkeypatch.setattr(bench_tp, "_resolve_head_sha", lambda: head_sha)

    with pytest.raises(SystemExit) as exc_info:
        bench_tp.main()

    assert exc_info.value.code == 1
    lines = _perf_gate_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert "name=tp" in lines[0]
    assert f"route={route}" in lines[0]
    assert f"head_sha={head_sha}" in lines[0]


def test_run_bench_fails_closed_when_env_resolution_fails(monkeypatch, capsys):
    """Worker environment errors return sentinel latency and unknown route."""

    def fail_env(world_size):
        raise RuntimeError("no ark")

    monkeypatch.setattr(bench_tp, "_subprocess_env", fail_env)

    ark_ms, route, failed = bench_tp.run_bench(
        world_size=1,
        timeout=1,
        hidden_size=bench_tp.HIDDEN_SIZE,
    )

    assert (ark_ms, route, failed) == (bench_tp._SENTINEL_MS, "unknown", True)
    assert "ERROR: cannot build worker env: no ark" in capsys.readouterr().err


def test_run_bench_marks_incomplete_worker_results_unknown(monkeypatch, capsys):
    """Missing worker JSON forces sentinel latency and unknown route."""

    class FakeProcess:
        def __init__(self, rank):
            self.rank = rank
            self.returncode = 0

        def communicate(self, timeout):
            if self.rank == 0:
                return (
                    b'{"route":"all_reduce_packet","latency_ms":0.01}\n',
                    b"",
                )
            return b"", b""

        def kill(self):
            pass

        def wait(self):
            pass

    def fake_popen(cmd, stdout, stderr, cwd, env):
        return FakeProcess(rank=int(cmd[-3]))

    monkeypatch.setattr(bench_tp, "_subprocess_env", lambda world_size: {})
    monkeypatch.setattr(bench_tp.subprocess, "Popen", fake_popen)

    result = bench_tp.run_bench(
        world_size=2,
        timeout=1,
        hidden_size=bench_tp.HIDDEN_SIZE,
    )

    assert result == (bench_tp._SENTINEL_MS, "unknown", True)
    assert "incomplete worker results" in capsys.readouterr().err
