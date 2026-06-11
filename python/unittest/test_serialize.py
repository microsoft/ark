# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from common import ark, pytest_ark
import os
import tempfile
import numpy as np
import pytest
from ark.serialize import save, load


@pytest_ark()
def test_serialize_save_load():
    """Test round-trip save and load of a state dict."""
    state = {"w": np.ones((4, 4), dtype=np.float32)}

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name

    try:
        save(state, path)
        loaded = load(path)
        assert "w" in loaded
        assert np.array_equal(loaded["w"], state["w"])
    finally:
        os.unlink(path)


@pytest_ark()
def test_serialize_save_non_dict_warns():
    """save() with non-dict still succeeds (warns)."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name

    try:
        # Should not raise, just warn
        save([1, 2, 3], path)
        loaded = load(path)
        assert loaded == [1, 2, 3]
    finally:
        os.unlink(path)
