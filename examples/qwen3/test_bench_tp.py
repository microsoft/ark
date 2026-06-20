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
    monkeypatch.setattr(bench_tp, "_resolve_base_sha", lambda: _VALID_BASE_SHA)

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


def test_resolve_head_sha_prefers_pull_request_event(monkeypatch, tmp_path):
    """Head SHA uses the PR head event before the merge SHA."""
    event_path = tmp_path / "event.json"
    event_path.write_text(
        '{"pull_request":{"head":{"sha":"'
        + _VALID_SHA
        + '"}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "ARK_HEAD_SHA", "1111111111111111111111111111111111111111"
    )
    monkeypatch.setenv(
        "GITHUB_HEAD_SHA", "2222222222222222222222222222222222222222"
    )
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_SHA", _VALID_BASE_SHA)

    assert bench_tp._resolve_head_sha() == _VALID_SHA


def test_resolve_base_sha_uses_pull_request_event(monkeypatch, tmp_path):
    """Base SHA can be recovered from the GitHub PR event payload."""
    event_path = tmp_path / "event.json"
    event_path.write_text(
        '{"pull_request":{"base":{"sha":"'
        + _VALID_BASE_SHA
        + '"}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "ARK_BASE_SHA", "1111111111111111111111111111111111111111"
    )
    monkeypatch.setenv(
        "GITHUB_BASE_SHA", "2222222222222222222222222222222222222222"
    )
    monkeypatch.setenv("BASE_SHA", "3333333333333333333333333333333333333333")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))

    def fail_check_output(*args, **kwargs):
        raise AssertionError("git fallback should not be called")

    monkeypatch.setattr(bench_tp.subprocess, "check_output", fail_check_output)

    assert bench_tp._resolve_base_sha() == _VALID_BASE_SHA


def test_resolve_base_sha_ignores_malformed_event(monkeypatch, tmp_path):
    """Malformed GitHub event payloads fall back to git refs."""
    event_path = tmp_path / "event.json"
    event_path.write_text("{", encoding="utf-8")

    def fake_check_output(cmd, text, stderr):
        if cmd[-1] == "origin/qwen3-allreduce-bench":
            return _VALID_BASE_SHA + "\n"
        raise bench_tp.subprocess.CalledProcessError(1, cmd)

    monkeypatch.delenv("ARK_BASE_SHA", raising=False)
    monkeypatch.delenv("GITHUB_BASE_SHA", raising=False)
    monkeypatch.delenv("BASE_SHA", raising=False)
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setattr(bench_tp, "_repo_root", lambda: "/repo")
    monkeypatch.setattr(bench_tp.subprocess, "check_output", fake_check_output)

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
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
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
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    monkeypatch.setattr(bench_tp, "_repo_root", lambda: "/repo")
    monkeypatch.setattr(bench_tp.subprocess, "check_output", fake_check_output)

    assert bench_tp._resolve_base_sha() == "unknown"
    assert calls == [
        ["git", "-C", "/repo", "rev-parse", "origin/qwen3-allreduce-bench"],
        ["git", "-C", "/repo", "rev-parse", "qwen3-allreduce-bench"],
    ]


@pytest.mark.parametrize("world_size", [0, 1])
def test_run_bench_rejects_invalid_world_size(monkeypatch, capsys, world_size):
    """Invalid world sizes fail closed before worker env resolution."""

    def fail_env(world_size):
        raise AssertionError("worker env should not be resolved")

    monkeypatch.setattr(bench_tp, "_subprocess_env", fail_env)

    result = bench_tp.run_bench(
        world_size=world_size,
        timeout=1,
        hidden_size=bench_tp.HIDDEN_SIZE,
    )

    assert result == (bench_tp._SENTINEL_MS, "unknown", True)
    assert (
        "ERROR: all_reduce_packet requires "
        f"world_size >= 2 (got {world_size})"
    ) in capsys.readouterr().err


def test_main_fails_closed_for_invalid_world_size(monkeypatch, capsys):
    """CLI world-size validation still prints one PERF_GATE line."""
    monkeypatch.setattr(sys, "argv", ["bench_tp.py", "--world-size", "0"])
    monkeypatch.setattr(bench_tp, "_resolve_head_sha", lambda: _VALID_SHA)
    monkeypatch.setattr(bench_tp, "_resolve_base_sha", lambda: _VALID_BASE_SHA)

    with pytest.raises(SystemExit) as exc_info:
        bench_tp.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    lines = _perf_gate_lines(captured.out)
    assert len(lines) == 1
    assert "ark_ms=999999.0000" in lines[0]
    assert "route=unknown" in lines[0]
    assert (
        "ERROR: all_reduce_packet requires world_size >= 2 (got 0)"
        in captured.err
    )


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


def test_perf_gate_wrapper_does_not_inject_git_shas(tmp_path):
    """Shell wrapper leaves SHA precedence to bench_tp.py."""
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(bench_tp.__file__), "..", "..")
    )
    git_head_sha = "1111111111111111111111111111111111111111"
    git_base_sha = "2222222222222222222222222222222222222222"
    fake_git = tmp_path / "git"
    fake_git.write_text(
        "#!/usr/bin/env sh\n"
        "case \"$*\" in\n"
        f"  *'rev-parse HEAD') echo '{git_head_sha}' ; exit 0 ;;\n"
        "  *'rev-parse origin/qwen3-allreduce-bench') "
        f"echo '{git_base_sha}' ; exit 0 ;;\n"
        "esac\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    fake_python = tmp_path / "python3"
    fake_python.write_text(
        "#!/usr/bin/env sh\n"
        "if [ -n \"${ARK_HEAD_SHA:-}\" ] || "
        "[ -n \"${ARK_BASE_SHA:-}\" ]; then\n"
        "  exit 43\n"
        "fi\n"
        "echo \"PERF_GATE name=tp ark_ms=0.1000 sglang_ms=0.3268 "
        "ratio=0.3060 route=all_reduce_packet "
        "head_sha=$GITHUB_HEAD_SHA base_sha=$GITHUB_BASE_SHA\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "GITHUB_HEAD_SHA": _VALID_SHA,
        "GITHUB_BASE_SHA": _VALID_BASE_SHA,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    env.pop("ARK_HEAD_SHA", None)
    env.pop("ARK_BASE_SHA", None)

    result = subprocess.run(
        ["bash", os.path.join(repo_root, "__perf_gate__.sh")],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert _perf_gate_lines(result.stdout) == [
        "PERF_GATE name=tp ark_ms=0.1000 sglang_ms=0.3268 "
        f"ratio=0.3060 route=all_reduce_packet head_sha={_VALID_SHA} "
        f"base_sha={_VALID_BASE_SHA}"
    ]


@pytest.mark.parametrize(
    "child_line",
    [
        (
            "PERF_GATE name=tp ark_ms=0.1000 sglang_ms=0.3268 "
            f"ratio=0.3060 route=unknown head_sha={_VALID_SHA} "
            f"base_sha={_VALID_BASE_SHA}"
        ),
        (
            "PERF_GATE name=tp ark_ms=0.1000 sglang_ms=0.3268 "
            f"ratio=0.3060 route=all_reduce_packet head_sha=unknown "
            f"base_sha={_VALID_BASE_SHA}"
        ),
        (
            "PERF_GATE name=tp ark_ms=0.1000 sglang_ms=0.3268 "
            f"ratio=0.3060 route=all_reduce_packet head_sha={_VALID_SHA}"
        ),
        (
            "PERF_GATE name=tp sglang_ms=0.3268 ratio=0.3060 "
            f"route=all_reduce_packet head_sha={_VALID_SHA} "
            f"base_sha={_VALID_BASE_SHA}"
        ),
        (
            "PERF_GATE name=tp ark_ms=0.1000 sglang_ms=0.3268 "
            f"ratio=fast route=all_reduce_packet head_sha={_VALID_SHA} "
            f"base_sha={_VALID_BASE_SHA}"
        ),
        (
            "PERF_GATE name=tp ark_ms=-1.0000 sglang_ms=0.3268 "
            f"ratio=-3.0600 route=all_reduce_packet head_sha={_VALID_SHA} "
            f"base_sha={_VALID_BASE_SHA}"
        ),
        (
            "PERF_GATE name=tp ark_ms=0.1000 sglang_ms=0.3268 "
            f"ratio=-0.3060 route=all_reduce_packet head_sha={_VALID_SHA} "
            f"base_sha={_VALID_BASE_SHA}"
        ),
        (
            "PERF_GATE name=tp ark_ms=0.4000 sglang_ms=0.3268 "
            f"ratio=0.3060 route=all_reduce_packet head_sha={_VALID_SHA} "
            f"base_sha={_VALID_BASE_SHA}"
        ),
        (
            "PERF_GATE name=tp ark_ms=0.1000 sglang_ms=1.0000 "
            f"ratio=0.1000 route=all_reduce_packet head_sha={_VALID_SHA} "
            f"base_sha={_VALID_BASE_SHA}"
        ),
        (
            "PERF_GATE name=tp ark_ms=2.0000 sglang_ms=1.0000 "
            f"ratio=2.0000 route=all_reduce_packet head_sha={_VALID_SHA} "
            f"base_sha={_VALID_BASE_SHA}"
        ),
    ],
)
def test_perf_gate_wrapper_rejects_malformed_success_line(tmp_path, child_line):
    """Shell wrapper rejects each invalid child PERF_GATE field."""
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(bench_tp.__file__), "..", "..")
    )
    fake_python = tmp_path / "python3"
    fake_python.write_text(
        "#!/usr/bin/env sh\n" f"echo '{child_line}'\n",
        encoding="utf-8",
    )
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
    assert _perf_gate_lines(result.stdout) == [
        "PERF_GATE name=tp ark_ms=999999.0000 sglang_ms=0.3268 "
        "ratio=3060223.3127 route=unknown head_sha=unknown "
        "base_sha=unknown"
    ]


def test_perf_gate_wrapper_preserves_rounded_passing_line(tmp_path):
    """Shell wrapper gates on ratio, not rounded ark_ms text."""
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(bench_tp.__file__), "..", "..")
    )
    child_line = (
        "PERF_GATE name=tp ark_ms=0.3268 sglang_ms=0.3268 "
        f"ratio=0.9999 route=all_reduce_packet head_sha={_VALID_SHA} "
        f"base_sha={_VALID_BASE_SHA}"
    )
    fake_python = tmp_path / "python3"
    fake_python.write_text(
        "#!/usr/bin/env sh\n" f"echo '{child_line}'\n",
        encoding="utf-8",
    )
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

    assert result.returncode == 0
    assert _perf_gate_lines(result.stdout) == [child_line]


def test_perf_gate_wrapper_preserves_valid_slow_perf_line(tmp_path):
    """Shell wrapper reports real packet evidence when only ratio fails."""
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(bench_tp.__file__), "..", "..")
    )
    child_line = (
        "PERF_GATE name=tp ark_ms=0.4000 sglang_ms=0.3268 "
        f"ratio=1.2239 route=all_reduce_packet head_sha={_VALID_SHA} "
        f"base_sha={_VALID_BASE_SHA}"
    )
    fake_python = tmp_path / "python3"
    fake_python.write_text(
        "#!/usr/bin/env sh\n" f"echo '{child_line}'\n",
        encoding="utf-8",
    )
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
    assert _perf_gate_lines(result.stdout) == [child_line]


def test_perf_gate_wrapper_resolves_bench_relative_to_script(tmp_path):
    """Shell wrapper finds bench_tp.py when cwd is not the repo root."""
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(bench_tp.__file__), "..", "..")
    )
    argv_file = tmp_path / "argv.txt"
    pythonpath_file = tmp_path / "pythonpath.txt"
    fake_python = tmp_path / "python3"
    fake_python.write_text(
        "#!/usr/bin/env sh\n"
        'printf \'%s\\n\' "$1" > "$ARGV_FILE"\n'
        'printf \'%s\\n\' "$PYTHONPATH" > "$PYTHONPATH_FILE"\n'
        "echo 'PERF_GATE name=tp ark_ms=0.1000 sglang_ms=0.3268 "
        "ratio=0.3060 route=all_reduce_packet head_sha="
        f"{_VALID_SHA} base_sha={_VALID_BASE_SHA}'\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "ARGV_FILE": str(argv_file),
        "PYTHONPATH_FILE": str(pythonpath_file),
        "PATH": f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    env.pop("ARK_ROOT", None)
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        ["bash", os.path.join(repo_root, "__perf_gate__.sh")],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert _perf_gate_lines(result.stdout) == [
        "PERF_GATE name=tp ark_ms=0.1000 sglang_ms=0.3268 "
        f"ratio=0.3060 route=all_reduce_packet head_sha={_VALID_SHA} "
        f"base_sha={_VALID_BASE_SHA}"
    ]
    assert argv_file.read_text(encoding="utf-8").strip() == os.path.join(
        repo_root, "examples", "qwen3", "bench_tp.py"
    )
    pythonpath = pythonpath_file.read_text(encoding="utf-8").strip()
    assert pythonpath == os.path.join(repo_root, "python")
    assert "" not in pythonpath.split(os.pathsep)


def test_run_bench_returns_max_latency_for_successful_workers(monkeypatch):
    """Successful worker JSON is aggregated with max-rank latency."""
    launches = []

    class FakeProcess:
        def __init__(self, rank):
            self.rank = rank
            self.returncode = 0

        def communicate(self, timeout):
            latency = 0.01 if self.rank == 0 else 0.03
            return (
                (
                    '{"route":"all_reduce_packet",'
                    '"route_proof":"AllReducePacketFused",'
                    '"latency_ms":'
                    f"{latency}}}\n"
                ).encode(),
                b"",
            )

        def kill(self):
            pass

        def wait(self):
            pass

    def fake_popen(cmd, stdout, stderr, cwd, env):
        launches.append((cmd, cwd, env))
        return FakeProcess(rank=int(cmd[-3]))

    monkeypatch.setattr(bench_tp, "_subprocess_env", lambda world_size: {})
    monkeypatch.setattr(bench_tp.subprocess, "Popen", fake_popen)

    result = bench_tp.run_bench(
        world_size=2,
        timeout=1,
        hidden_size=bench_tp.HIDDEN_SIZE,
    )

    assert result == (0.03, "all_reduce_packet", False)
    assert [launch[1] for launch in launches] == ["/", "/"]
    assert [launch[0][-1] for launch in launches] == [
        str(bench_tp.HIDDEN_SIZE)
    ] * 2


def test_run_bench_fails_closed_when_env_resolution_fails(monkeypatch, capsys):
    """Worker environment errors return sentinel latency and unknown route."""

    def fail_env(world_size):
        raise RuntimeError("no ark")

    monkeypatch.setattr(bench_tp, "_subprocess_env", fail_env)

    ark_ms, route, failed = bench_tp.run_bench(
        world_size=2,
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
                    b'{"route":"all_reduce_packet",'
                    b'"route_proof":"AllReducePacketFused",'
                    b'"latency_ms":0.01}\n',
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


def test_run_bench_rejects_worker_result_missing_route_proof(
    monkeypatch, capsys
):
    """Worker JSON must prove the planned packet route."""

    class FakeProcess:
        returncode = 0

        def communicate(self, timeout):
            return (
                b'{"route":"all_reduce_packet","latency_ms":0.01}\n',
                b"",
            )

        def kill(self):
            pass

        def wait(self):
            pass

    def fake_popen(cmd, stdout, stderr, cwd, env):
        return FakeProcess()

    monkeypatch.setattr(bench_tp, "_subprocess_env", lambda world_size: {})
    monkeypatch.setattr(bench_tp.subprocess, "Popen", fake_popen)

    result = bench_tp.run_bench(
        world_size=2,
        timeout=1,
        hidden_size=bench_tp.HIDDEN_SIZE,
    )

    assert result == (bench_tp._SENTINEL_MS, "unknown", True)
    assert "invalid worker result schema" in capsys.readouterr().err


def test_run_bench_rejects_worker_result_missing_latency(monkeypatch, capsys):
    """Malformed worker JSON forces sentinel latency and unknown route."""

    class FakeProcess:
        returncode = 0

        def communicate(self, timeout):
            return b'{"route":"all_reduce_packet"}\n', b""

        def kill(self):
            pass

        def wait(self):
            pass

    def fake_popen(cmd, stdout, stderr, cwd, env):
        return FakeProcess()

    monkeypatch.setattr(bench_tp, "_subprocess_env", lambda world_size: {})
    monkeypatch.setattr(bench_tp.subprocess, "Popen", fake_popen)

    result = bench_tp.run_bench(
        world_size=2,
        timeout=1,
        hidden_size=bench_tp.HIDDEN_SIZE,
    )

    assert result == (bench_tp._SENTINEL_MS, "unknown", True)
    assert "invalid worker result schema" in capsys.readouterr().err


def test_run_bench_rejects_worker_result_negative_latency(monkeypatch, capsys):
    """Impossible negative latency forces sentinel latency."""

    class FakeProcess:
        returncode = 0

        def communicate(self, timeout):
            return (
                b'{"route":"all_reduce_packet",'
                b'"route_proof":"AllReducePacketFused",'
                b'"latency_ms":-0.01}\n',
                b"",
            )

        def kill(self):
            pass

        def wait(self):
            pass

    def fake_popen(cmd, stdout, stderr, cwd, env):
        return FakeProcess()

    monkeypatch.setattr(bench_tp, "_subprocess_env", lambda world_size: {})
    monkeypatch.setattr(bench_tp.subprocess, "Popen", fake_popen)

    result = bench_tp.run_bench(
        world_size=2,
        timeout=1,
        hidden_size=bench_tp.HIDDEN_SIZE,
    )

    assert result == (bench_tp._SENTINEL_MS, "unknown", True)
    assert "invalid worker result schema" in capsys.readouterr().err
