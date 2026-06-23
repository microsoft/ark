# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""CPU-only tests for Qwen3 all-reduce benchmark orchestration."""

import json
import os
import sys

import pytest

try:
    from . import bench_allreduce
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bench_allreduce


class _FakeProcess:
    def __init__(self, argv):
        self.argv = argv
        self.returncode = 0

    def communicate(self, timeout=None):
        del timeout
        rank = int(self.argv[3])
        world_size = int(self.argv[4])
        n_elements = int(self.argv[5])
        shape = self.argv[6]
        label = self.argv[7]
        mode = self.argv[8]
        latency_us = (200.0 if mode == "external" else 500.0) + (
            100.0 * rank
        )
        result = {
            "rank": rank,
            "shape": shape,
            "label": label,
            "input_mode": mode,
            "world_size": world_size,
            "n_elements": n_elements,
            "latency_us": latency_us,
        }
        return json.dumps(result).encode(), b""

    def kill(self):
        pass

    def wait(self):
        pass


def _patch_popen(monkeypatch):
    launches = []

    def fake_popen(argv, **kwargs):
        launches.append((argv, kwargs))
        return _FakeProcess(argv)

    monkeypatch.setattr(bench_allreduce.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        bench_allreduce,
        "_subprocess_env",
        lambda world_size: {"WS": str(world_size)},
    )
    return launches


def test_run_bench_all_input_modes_launches_and_aggregates(monkeypatch):
    launches = _patch_popen(monkeypatch)

    results, any_failed = bench_allreduce.run_bench(
        world_size=2, timeout=7, shape="decode", input_mode="all"
    )

    assert not any_failed
    assert [entry[0][3:] for entry in launches] == [
        ["0", "2", "4096", "decode", "decode  (1, 4096)", "external"],
        ["1", "2", "4096", "decode", "decode  (1, 4096)", "external"],
        ["0", "2", "4096", "decode", "decode  (1, 4096)", "internal"],
        ["1", "2", "4096", "decode", "decode  (1, 4096)", "internal"],
    ]
    assert [entry[1]["env"] for entry in launches] == [{"WS": "2"}] * 4
    assert results == [
        {
            "shape": "decode",
            "label": "decode  (1, 4096)",
            "input_mode": "external",
            "world_size": 2,
            "n_elements": 4096,
            "max_rank": 1,
            "latency_us": 300.0,
            "rank_latencies_us": [200.0, 300.0],
        },
        {
            "shape": "decode",
            "label": "decode  (1, 4096)",
            "input_mode": "internal",
            "world_size": 2,
            "n_elements": 4096,
            "max_rank": 1,
            "latency_us": 600.0,
            "rank_latencies_us": [500.0, 600.0],
        },
    ]


@pytest.mark.parametrize("world_size", (2, 8))
def test_run_bench_decode_reports_both_required_modes(
    monkeypatch, world_size
):
    _patch_popen(monkeypatch)

    results, any_failed = bench_allreduce.run_bench(
        world_size=world_size, timeout=7, shape="decode", input_mode="all"
    )

    assert not any_failed
    assert {
        (result["world_size"], result["shape"], result["input_mode"])
        for result in results
    } == {
        (world_size, "decode", "external"),
        (world_size, "decode", "internal"),
    }
    assert {result["n_elements"] for result in results} == {4096}


def test_run_bench_internal_mode_only_preserves_metadata(monkeypatch):
    launches = _patch_popen(monkeypatch)

    results, any_failed = bench_allreduce.run_bench(
        world_size=2, timeout=7, shape="decode", input_mode="internal"
    )

    assert not any_failed
    assert [entry[0][8] for entry in launches] == ["internal", "internal"]
    assert len(results) == 1
    assert results[0]["input_mode"] == "internal"
    assert results[0]["shape"] == "decode"
    assert results[0]["label"] == "decode  (1, 4096)"


def test_main_default_reports_all_modes_and_keeps_external_gate(
    monkeypatch, capsys
):
    results = [
        {
            "n_elements": 4096,
            "input_mode": "internal",
            "latency_us": 100.0,
        },
        {
            "n_elements": 4096,
            "input_mode": "external",
            "latency_us": 2000.0,
        },
    ]
    calls = []

    def fake_run_bench(world_size, timeout, shape, input_mode):
        calls.append((world_size, timeout, shape, input_mode))
        return results, False

    monkeypatch.setattr(bench_allreduce, "run_bench", fake_run_bench)
    monkeypatch.setattr(sys, "argv", ["bench_allreduce.py"])

    bench_allreduce.main()

    out, err = capsys.readouterr()
    assert calls == [(2, 120, "all", "all")]
    assert "PERF_GATE name=allreduce ark_ms=2.0000" in out
    assert err == ""


def test_main_perf_gate_does_not_fallback_to_internal(monkeypatch, capsys):
    results = [
        {
            "n_elements": 4096,
            "input_mode": "internal",
            "latency_us": 100.0,
        }
    ]
    monkeypatch.setattr(
        bench_allreduce, "run_bench", lambda *args: (results, False)
    )
    monkeypatch.setattr(sys, "argv", ["bench_allreduce.py"])

    with pytest.raises(SystemExit) as exc_info:
        bench_allreduce.main()

    out, err = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "PERF_GATE name=allreduce ark_ms=999999.0000" in out
    assert (
        "ERROR: external decode benchmark produced no result for PERF_GATE"
        in err
    )
