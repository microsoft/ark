# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for model.py edge branches not covered by test_model.py."""

from common import ark, pytest_ark
import pytest


@pytest_ark()
def test_model_set_device_id_valid():
    """set_device_id accepts 0."""
    ark.Model.set_device_id(0)
    assert ark.Model.get_device_id() == 0


@pytest_ark()
def test_model_set_device_id_negative():
    """set_device_id raises InvalidUsageError for negative value."""
    with pytest.raises(ark.InvalidUsageError):
        ark.Model.set_device_id(-1)


@pytest_ark()
def test_model_set_rank():
    """set_rank / get_rank round-trips."""
    ark.Model.set_rank(3)
    assert ark.Model.get_rank() == 3
    ark.Model.set_rank(0)


@pytest_ark()
def test_model_set_world_size():
    """set_world_size / get_world_size round-trips."""
    ark.Model.set_world_size(4)
    assert ark.Model.get_world_size() == 4
    ark.Model.set_world_size(1)


@pytest_ark()
def test_model_str():
    """Model.__str__ returns serialized JSON."""
    t = ark.tensor([64], ark.fp16)
    m = ark.Model.get_model()
    s = str(m)
    assert isinstance(s, str)
    assert "Tensors" in s or "{" in s


@pytest_ark()
def test_model_serialize_pretty_false():
    """Model.serialize(pretty=False) returns compact JSON."""
    t = ark.tensor([64], ark.fp16)
    m = ark.Model.get_model()
    compact = m.serialize(pretty=False)
    pretty = m.serialize(pretty=True)
    # Compact should be shorter (no indentation)
    assert len(compact) <= len(pretty)


@pytest_ark()
def test_set_rank_top_level():
    """ark.set_rank top-level function works."""
    ark.set_rank(1)
    assert ark.Model.get_rank() == 1
    ark.set_rank(0)


@pytest_ark()
def test_set_world_size_top_level():
    """ark.set_world_size top-level function works."""
    ark.set_world_size(8)
    assert ark.Model.get_world_size() == 8
    ark.set_world_size(1)
