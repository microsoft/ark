# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Shared subprocess environment helpers for Qwen3 examples.

Used by both ``bench_allreduce.py`` and ``test_allreduce.py`` to build
a consistent PYTHONPATH / CUDA_VISIBLE_DEVICES env for worker processes.
"""

import glob
import importlib.util
import os
import sys

# Repo root — used to locate the built ark Python package for subprocesses.
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)

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


def _build_root_from_python_parent(parent_dir: str):
    """Infer the CMake build root from a build-tree Python package path."""
    if os.path.basename(parent_dir) != "python":
        return None
    build_root = os.path.dirname(parent_dir)
    if os.path.isdir(build_root):
        return build_root
    return None


def _append_unique(paths, path):
    """Append *path* to *paths* once when it is non-empty."""
    if path and path not in paths:
        paths.append(path)


def _subprocess_env(world_size: int) -> dict:
    """Build env dict for worker subprocesses.

    Resolution order for the ``ark`` package path:
      1. ``importlib.util.find_spec("ark")`` — wherever the parent already
         resolved ark (handles build-tree, install, and namespace packages).
      2. ``$ARK_ROOT/python`` (CI sets ``ARK_ROOT=$PWD``).
      3. ``<repo>/build/python`` or ``<repo>/python``.
      4. inherited ``PYTHONPATH``.

    Also propagates the repo root for package imports in workers and sets
    ``ARK_ROOT`` / ``LD_LIBRARY_PATH`` when a build-tree package is found.
    """
    extra = []  # type: list[str]
    resolved_ark_root = None

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
                _append_unique(extra, ark_parent)
                if resolved_ark_root is None:
                    resolved_ark_root = _build_root_from_python_parent(
                        ark_parent
                    )
    except (ModuleNotFoundError, ValueError, TypeError):
        pass

    # --- Secondary: scan sys.path for a compiled ark package ---
    # When PYTHONPATH points at the source tree (e.g., /w/python),
    # find_spec("ark") resolves to source-only ark/ (no core*.so).
    # Keep searching for an installed/built ark with compiled extension.
    for entry in sys.path:
        if not entry:
            continue
        if _has_compiled_ark(entry):
            _append_unique(extra, entry)
            if resolved_ark_root is None:
                resolved_ark_root = _build_root_from_python_parent(entry)
            break

    # --- Fallback: $ARK_ROOT/python ---
    ark_root = os.environ.get("ARK_ROOT", "")
    if ark_root:
        ark_root = os.path.abspath(ark_root)
        ark_root_py = os.path.join(ark_root, "python")
        if _has_compiled_ark(ark_root_py):
            _append_unique(extra, ark_root_py)
            if resolved_ark_root is None:
                resolved_ark_root = ark_root

    # --- Fallback: repo build/python or python ---
    for subdir in ("build/python", "python"):
        candidate = os.path.join(_REPO_ROOT, subdir)
        if _has_compiled_ark(candidate):
            _append_unique(extra, candidate)
            if resolved_ark_root is None:
                resolved_ark_root = _build_root_from_python_parent(candidate)

    # --- Propagate repo root for examples.qwen3 package imports ---
    _append_unique(extra, _REPO_ROOT)

    # --- Inherited PYTHONPATH ---
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        extra.append(existing)

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible = os.environ["CUDA_VISIBLE_DEVICES"]
        devices = [d.strip() for d in visible.split(",") if d.strip()]
        if len(devices) < world_size:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES exposes fewer devices "
                f"({len(devices)}) than world_size ({world_size})"
            )
        cuda_visible_devices = ",".join(devices[:world_size])
    else:
        cuda_visible_devices = ",".join(str(i) for i in range(world_size))

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
    }
    if extra:
        env["PYTHONPATH"] = os.pathsep.join(extra)
    if resolved_ark_root is not None:
        env["ARK_ROOT"] = resolved_ark_root

    ld_paths = []
    for root in (resolved_ark_root, ark_root):
        if not root:
            continue
        for subdir in ("", "lib", "ark", os.path.join("python", "ark")):
            candidate = os.path.join(root, subdir)
            if os.path.isdir(candidate):
                _append_unique(ld_paths, candidate)
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if existing_ld:
        _append_unique(ld_paths, existing_ld)
    if ld_paths:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_paths)
    return env
