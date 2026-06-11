# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from common import ark, pytest_ark
import json
import pytest


@pytest_ark()
def test_model():
    input_tensor = ark.tensor([64, 64], ark.fp16)
    other_tensor = ark.tensor([64, 64], ark.fp16)
    ark.add(input_tensor, other_tensor)

    m = ark.Model.get_model().compress()
    m_json = json.loads(m.serialize())

    assert m_json.get("Nodes", None) is not None
    assert len(m_json["Nodes"]) == 1
    assert m_json["Nodes"][0].get("Op", None) is not None
    assert m_json["Nodes"][0]["Op"].get("Type", None) == "Add"

    ark.Model.reset()

    m = ark.Model.get_model().compress()
    m_json = json.loads(m.serialize())

    assert m_json.get("Nodes", None) is not None
    assert len(m_json["Nodes"]) == 0


@pytest_ark()
def test_set_model_valid():
    """set_model accepts a Model instance."""
    m = ark.Model()
    ark.set_model(m)
    assert ark.Model.get_model() is m


@pytest_ark()
def test_set_model_invalid():
    """set_model rejects non-Model arguments."""
    with pytest.raises(ark.InvalidUsageError):
        ark.set_model("not a model")


@pytest_ark()
def test_current_model_auto_creates():
    """current_model returns a Model, auto-creating if needed."""
    m = ark.current_model()
    assert isinstance(m, ark.Model)


@pytest_ark()
def test_use_model_restores_previous():
    """use_model restores the previous model on exit."""
    outer = ark.current_model()
    inner = ark.Model()
    with ark.use_model(inner) as m:
        assert m is inner
        assert ark.Model.get_model() is inner
    assert ark.Model.get_model() is outer


@pytest_ark()
def test_use_model_none_creates_fresh():
    """use_model(None) creates a fresh model."""
    outer = ark.current_model()
    with ark.use_model(None) as m:
        assert isinstance(m, ark.Model)
        assert m is not outer
    assert ark.Model.get_model() is outer


@pytest_ark()
def test_use_model_invalid():
    """use_model rejects non-Model arguments."""
    with pytest.raises(ark.InvalidUsageError):
        with ark.use_model("not a model"):
            pass


@pytest_ark()
def test_use_model_restores_on_exception():
    """use_model restores the previous model even if body raises."""
    outer = ark.current_model()
    inner = ark.Model()
    with pytest.raises(RuntimeError):
        with ark.use_model(inner):
            raise RuntimeError("boom")
    assert ark.Model.get_model() is outer
