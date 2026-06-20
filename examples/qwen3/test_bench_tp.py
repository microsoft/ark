# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""No-GPU tests for the Qwen3 TP perf-gate CLI contract."""

import os
import subprocess
import sys

import pytest

try:
    from . import bench_tp
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bench_tp


_VALID_SHA = "0123456789abcdef0123456789abcdef01234567"
_VALID_BASE_SHA = "fedcba9876543210fedcba9876543210fedcba98"


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
    monkeypatch.setattr(
        bench_tp, "_resolve_base_sha", lambda: _VALID_BASE_SHA
    )

    bench_tp.main()

    lines = _perf_gate_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert "name=tp" in lines[0]
    assert "route=all_reduce_packet" in lines[0]
    assert f"head_sha={_VALID_SHA}" in lines[0]
    assert f"base_sha={_VALID_BASE_SHA}" in lines[0]


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
    monkeypatch.setattr(bench_tp, "_resolve_base_sha", lambda: "unknown")

    with pytest.raises(SystemExit) as exc_info:
        bench_tp.main()

    assert exc_info.value.code == 1
    lines = _perf_gate_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert "name=tp" in lines[0]
    assert "route=unknown" in lines[0]
    assert "head_sha=unknown" in lines[0]
    assert "base_sha=unknown" in lines[0]


@pytest.mark.parametrize(
    ("ark_ms", "route", "head_sha", "base_sha"),
    [
        (0.01, "unknown", _VALID_SHA, _VALID_BASE_SHA),
        (0.01, "all_reduce_packet", "unknown", _VALID_BASE_SHA),
        (0.01, "all_reduce_packet", _VALID_SHA, "unknown"),
        (
            bench_tp._TP_TARGET_MS,
            "all_reduce_packet",
            _VALID_SHA,
            _VALID_BASE_SHA,
        ),
    ],
)
def test_main_fails_closed_for_independent_gate_failures(
    monkeypatch, capsys, ark_ms, route, head_sha, base_sha
):
    """Route, SHA, and threshold gates each fail closed independently."""
    monkeypatch.setattr(sys, "argv", ["bench_tp.py"])
    monkeypatch.setattr(
        bench_tp,
        "run_bench",
        lambda world_size, timeout, hidden_size: (ark_ms, route, False),
    )
    monkeypatch.setattr(bench_tp, "_resolve_head_sha", lambda: head_sha)
    monkeypatch.setattr(bench_tp, "_resolve_base_sha", lambda: base_sha)

    with pytest.raises(SystemExit) as exc_info:
        bench_tp.main()

    assert exc_info.value.code == 1
    lines = _perf_gate_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert "name=tp" in lines[0]
    assert f"route={route}" in lines[0]
    assert f"head_sha={head_sha}" in lines[0]
    assert f"base_sha={base_sha}" in lines[0]


def test_resolve_base_sha_prefers_env_override(monkeypatch):
    """A valid explicit base SHA wins over git fallback."""
    monkeypatch.setenv("ARK_BASE_SHA", _VALID_BASE_SHA)

    def fail_check_output(*args, **kwargs):
        raise AssertionError("git fallback should not be called")

    monkeypatch.setattr(bench_tp.subprocess, "check_output", fail_check_output)

    assert bench_tp._resolve_base_sha() == _VALID_BASE_SHA


def test_resolve_base_sha_uses_qwen_target_refs(monkeypatch):
    """Base SHA falls back only to qwen3-allreduce-bench refs."""
    calls = []

    def fake_check_output(cmd, text, stderr):
        calls.append(cmd)
        if cmd[-1] == "qwen3-allreduce-bench":
            return _VALID_BASE_SHA + "\n"
        raise bench_tp.subprocess.CalledProcessError(1, cmd)

    monkeypatch.delenv("ARK_BASE_SHA", raising=False)
    monkeypatch.delenv("GITHUB_BASE_SHA", raising=False)
    monkeypatch.delenv("BASE_SHA", raising=False)
    monkeypatch.setattr(bench_tp, "_repo_root", lambda: "/repo")
    monkeypatch.setattr(bench_tp.subprocess, "check_output", fake_check_output)

    assert bench_tp._resolve_base_sha() == _VALID_BASE_SHA
    assert calls == [
        ["git", "-C", "/repo", "rev-parse", "origin/qwen3-allreduce-bench"],
        ["git", "-C", "/repo", "rev-parse", "qwen3-allreduce-bench"],
    ]


def test_resolve_base_sha_returns_unknown_without_qwen_refs(monkeypatch):
    """Missing qwen3-allreduce-bench refs do not fall back to main."""
    calls = []

    def fake_check_output(cmd, text, stderr):
        calls.append(cmd)
        raise bench_tp.subprocess.CalledProcessError(1, cmd)

    monkeypatch.delenv("ARK_BASE_SHA", raising=False)
    monkeypatch.delenv("GITHUB_BASE_SHA", raising=False)
    monkeypatch.delenv("BASE_SHA", raising=False)
    monkeypatch.setattr(bench_tp, "_repo_root", lambda: "/repo")
    monkeypatch.setattr(bench_tp.subprocess, "check_output", fake_check_output)

    assert bench_tp._resolve_base_sha() == "unknown"
    assert calls == [
        ["git", "-C", "/repo", "rev-parse", "origin/qwen3-allreduce-bench"],
        ["git", "-C", "/repo", "rev-parse", "qwen3-allreduce-bench"],
    ]


def test_run_bench_rejects_invalid_world_size(monkeypatch, capsys):
    """Invalid world sizes fail closed before worker env resolution."""

    def fail_env(world_size):
        raise AssertionError("worker env should not be resolved")

    monkeypatch.setattr(bench_tp, "_subprocess_env", fail_env)

    result = bench_tp.run_bench(
        world_size=0,
        timeout=1,
        hidden_size=bench_tp.HIDDEN_SIZE,
    )

    assert result == (bench_tp._SENTINEL_MS, "unknown", True)
    assert "ERROR: invalid world_size=0" in capsys.readouterr().err


def test_main_fails_closed_for_invalid_world_size(monkeypatch, capsys):
    """CLI world-size validation still prints one PERF_GATE line."""
    monkeypatch.setattr(sys, "argv", ["bench_tp.py", "--world-size", "0"])
    monkeypatch.setattr(bench_tp, "_resolve_head_sha", lambda: _VALID_SHA)
    monkeypatch.setattr(
        bench_tp, "_resolve_base_sha", lambda: _VALID_BASE_SHA
    )

    with pytest.raises(SystemExit) as exc_info:
        bench_tp.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    lines = _perf_gate_lines(captured.out)
    assert len(lines) == 1
    assert "ark_ms=999999.0000" in lines[0]
    assert "route=unknown" in lines[0]
    assert "ERROR: invalid world_size=0" in captured.err


def test_perf_gate_wrapper_emits_sentinel_when_child_prints_no_gate(
    tmp_path,
):
    """Shell wrapper preserves one PERF_GATE line on child startup failure."""
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(bench_tp.__file__), "..", "..")
    )
    fake_python = tmp_path / "python3"
    fake_python.write_text("#!/usr/bin/env sh\nexit 42\n", encoding="utf-8")
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}",
    }

    result = subprocess.run(
        ["bash", os.path.join(repo_root, "__perf_gate__.sh")],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    lines = _perf_gate_lines(result.stdout)
    assert lines == [
        "PERF_GATE name=tp ark_ms=999999.0000 sglang_ms=0.3268 "
        "ratio=3060223.3127 route=unknown head_sha=unknown "
        "base_sha=unknown"
    ]


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
