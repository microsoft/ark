# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Shared subprocess environment helpers for Qwen3 examples.

Used by both ``bench_allreduce.py`` and ``test_allreduce.py`` to build
a consistent PYTHONPATH / CUDA_VISIBLE_DEVICES env for worker processes.
"""

import glob
import importlib.util
import os

# Repo root — used to locate the built ark Python package for subprocesses.
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)

# Directory containing this file — propagated so workers can import
# sibling modules (microbench, qwen3_config, etc.) if needed.
_EXAMPLES_QWEN3_DIR = os.path.dirname(os.path.abspath(__file__))


def _has_compiled_ark(parent_dir: str) -> bool:
    """Return True if *parent_dir*/ark/ contains the compiled C++ extension.

    The source tree's ``python/ark/`` has ``__init__.py`` but no compiled
    ``core.cpython-*.so``.  Adding it to PYTHONPATH causes workers to fail
    with ``ModuleNotFoundError: No module named 'ark.core'``.
    """
    ark_pkg = os.path.join(parent_dir, "ark")
    if not os.path.isfile(os.path.join(ark_pkg, "__init__.py")):
        return False
    # Check for compiled extension (Linux .so, Windows .pyd)
    return bool(
        glob.glob(os.path.join(ark_pkg, "core*.so"))
        or glob.glob(os.path.join(ark_pkg, "core*.pyd"))
    )


def _subprocess_env(world_size: int) -> dict:
    """Build env dict for worker subprocesses.

    Resolution order for the ``ark`` package path:
      1. ``importlib.util.find_spec("ark")`` — wherever the parent already
         resolved ark (handles build-tree, install, and namespace packages).
      2. ``$ARK_ROOT/python`` (CI sets ``ARK_ROOT=$PWD``).
      3. ``<repo>/build/python`` or ``<repo>/python``.
      4. inherited ``PYTHONPATH``.

    Also propagates the ``examples/qwen3/`` directory so workers can
    import sibling modules (microbench, qwen3_config) when needed.
    """
    extra = []  # type: list[str]

    # --- Primary: resolve from the running interpreter's import state ---
    try:
        spec = importlib.util.find_spec("ark")
        if spec is not None:
            if spec.submodule_search_locations:
                # Regular package: parent of the package directory.
                ark_pkg_dir = next(iter(spec.submodule_search_locations))
                ark_parent = os.path.dirname(ark_pkg_dir)
            elif spec.origin:
                # Single-file or namespace with origin.
                ark_parent = os.path.dirname(os.path.dirname(spec.origin))
            else:
                ark_parent = None
            if ark_parent and _has_compiled_ark(ark_parent):
                if ark_parent not in extra:
                    extra.append(ark_parent)
    except (ModuleNotFoundError, ValueError, TypeError):
        pass

    # --- Fallback: $ARK_ROOT/python ---
    ark_root = os.environ.get("ARK_ROOT", "")
    if ark_root:
        ark_root_py = os.path.join(ark_root, "python")
        if _has_compiled_ark(ark_root_py):
            if ark_root_py not in extra:
                extra.append(ark_root_py)

    # --- Fallback: repo build/python or python ---
    for subdir in ("build/python", "python"):
        candidate = os.path.join(_REPO_ROOT, subdir)
        if _has_compiled_ark(candidate):
            if candidate not in extra:
                extra.append(candidate)

    # --- Propagate examples/qwen3 for sibling module imports ---
    examples_parent = os.path.dirname(_EXAMPLES_QWEN3_DIR)
    if examples_parent not in extra:
        extra.append(examples_parent)

    # --- Inherited PYTHONPATH ---
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        extra.append(existing)

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(world_size)),
    }
    if extra:
        env["PYTHONPATH"] = os.pathsep.join(extra)
    return env
