# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""No-torch tests for Qwen3 worker environment resolution."""

import importlib.machinery
import os
import sys

try:
    from . import _env as qwen3_env
    from ._env import _subprocess_env
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _env as qwen3_env
    from _env import _subprocess_env


def _fake_ark_package(parent_dir, compiled, suffix=None):
    """Create a minimal ark package tree for _subprocess_env tests."""
    ark_pkg = parent_dir / "ark"
    ark_pkg.mkdir(parents=True)
    (ark_pkg / "__init__.py").write_text("# fake ark\n", encoding="utf-8")
    if compiled:
        suffix = suffix or importlib.machinery.EXTENSION_SUFFIXES[0]
        (ark_pkg / f"core{suffix}").write_text("", encoding="utf-8")


def test_subprocess_env_prefers_ark_root_python(monkeypatch, tmp_path):
    """Worker PYTHONPATH starts with the built ark package under ARK_ROOT."""
    build_root = tmp_path / "build"
    build_python = build_root / "python"
    source_python = tmp_path / "source" / "python"
    repo_root = tmp_path / "repo"
    _fake_ark_package(build_python, compiled=True)
    _fake_ark_package(source_python, compiled=False)
    repo_root.mkdir()

    monkeypatch.setattr(qwen3_env, "_REPO_ROOT", str(repo_root))
    monkeypatch.setattr(
        qwen3_env.importlib.util, "find_spec", lambda name: None
    )
    monkeypatch.setattr(sys, "path", [str(source_python)])
    monkeypatch.setenv("ARK_ROOT", str(build_root))
    monkeypatch.setenv("PYTHONPATH", str(source_python))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    env = _subprocess_env(world_size=2)
    paths = env["PYTHONPATH"].split(os.pathsep)

    assert paths[:2] == [str(build_python), str(repo_root)]
    assert paths.count(str(build_python)) == 1
    assert paths.count(str(source_python)) == 1
    assert env["ARK_ROOT"] == str(build_root)
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_subprocess_env_skips_source_only_inherited_path(monkeypatch, tmp_path):
    """A source-only inherited ark package cannot shadow compiled ark.core."""
    source_python = tmp_path / "source" / "python"
    build_python = tmp_path / "other-build" / "python"
    repo_root = tmp_path / "repo"
    _fake_ark_package(source_python, compiled=False)
    _fake_ark_package(build_python, compiled=True)
    repo_root.mkdir()

    monkeypatch.setattr(qwen3_env, "_REPO_ROOT", str(repo_root))
    monkeypatch.setattr(
        qwen3_env.importlib.util, "find_spec", lambda name: None
    )
    monkeypatch.setattr(sys, "path", [str(source_python), str(build_python)])
    monkeypatch.delenv("ARK_ROOT", raising=False)
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join([str(source_python), str(build_python)]),
    )

    env = _subprocess_env(world_size=1)
    paths = env["PYTHONPATH"].split(os.pathsep)

    assert paths[0] == str(build_python)
    assert paths.count(str(build_python)) == 1
    assert paths.count(str(source_python)) == 1
    assert paths.index(str(build_python)) < paths.index(str(source_python))


def test_subprocess_env_finds_scikit_build_wheel_dir(monkeypatch, tmp_path):
    """Worker PYTHONPATH can use pip/scikit-build's build/*/python dir."""
    repo_root = tmp_path / "repo"
    wheel_python = repo_root / "build" / "cp312-cp312-linux_x86_64" / "python"
    source_python = repo_root / "python"
    _fake_ark_package(wheel_python, compiled=True)
    _fake_ark_package(source_python, compiled=False)

    monkeypatch.setattr(qwen3_env, "_REPO_ROOT", str(repo_root))
    monkeypatch.setattr(
        qwen3_env.importlib.util, "find_spec", lambda name: None
    )
    monkeypatch.setattr(sys, "path", [str(source_python)])
    monkeypatch.delenv("ARK_ROOT", raising=False)
    monkeypatch.setenv("PYTHONPATH", str(source_python))

    env = _subprocess_env(world_size=1)
    paths = env["PYTHONPATH"].split(os.pathsep)

    assert paths[0] == str(wheel_python)
    assert paths.count(str(wheel_python)) == 1
    assert paths.count(str(source_python)) == 1
    assert env["ARK_ROOT"] == str(wheel_python.parent)


def test_subprocess_env_skips_incompatible_wheel_dir(monkeypatch, tmp_path):
    """A stale build for another interpreter cannot shadow current core."""
    repo_root = tmp_path / "repo"
    stale_python = repo_root / "build" / "aa-stale" / "python"
    wheel_python = repo_root / "build" / "zz-current" / "python"
    _fake_ark_package(stale_python, compiled=True, suffix=".cpython-stale.so")
    _fake_ark_package(wheel_python, compiled=True)

    monkeypatch.setattr(qwen3_env, "_REPO_ROOT", str(repo_root))
    monkeypatch.setattr(
        qwen3_env.importlib.util, "find_spec", lambda name: None
    )
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.delenv("ARK_ROOT", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)

    env = _subprocess_env(world_size=1)
    paths = env["PYTHONPATH"].split(os.pathsep)

    assert paths[0] == str(wheel_python)
    assert str(stale_python) not in paths
    assert env["ARK_ROOT"] == str(wheel_python.parent)
