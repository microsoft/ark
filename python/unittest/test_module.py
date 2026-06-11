# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from common import ark, pytest_ark
import numpy as np
import pytest


@pytest_ark()
def test_module_register_parameter():
    """Test registering parameters on a module."""

    class Linear(ark.Module):
        def __init__(self):
            super().__init__()
            self.weight = ark.parameter([64, 64], ark.fp16)

    m = Linear()
    assert "weight" in m.parameters
    assert isinstance(m.parameters["weight"], ark.Parameter)


@pytest_ark()
def test_module_register_submodule():
    """Test registering submodules."""

    class Block(ark.Module):
        def __init__(self):
            super().__init__()

    class Net(ark.Module):
        def __init__(self):
            super().__init__()
            self.block = Block()

    net = Net()
    assert "block" in net.sub_modules
    assert isinstance(net.sub_modules["block"], ark.Module)


@pytest_ark()
def test_module_params_dict_nested():
    """Test params_dict with nested modules."""

    class Child(ark.Module):
        def __init__(self):
            super().__init__()
            self.w = ark.parameter([32, 32], ark.fp32)

    class Parent(ark.Module):
        def __init__(self):
            super().__init__()
            self.child = Child()
            self.bias = ark.parameter([32], ark.fp32)

    parent = Parent()
    pd = parent.params_dict()
    assert "child.w" in pd
    assert "bias" in pd


@pytest_ark()
def test_module_params_dict_prefix():
    """Test params_dict with custom prefix."""

    class M(ark.Module):
        def __init__(self):
            super().__init__()
            self.w = ark.parameter([8], ark.fp32)

    m = M()
    pd = m.params_dict(prefix="layer0.")
    assert "layer0.w" in pd


@pytest_ark()
def test_module_call_invokes_forward():
    """Test that __call__ invokes forward."""

    class Adder(ark.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return ark.add(x, 1.0)

    m = Adder()
    t = ark.tensor([64], ark.fp16)
    out = m(t)
    assert out.shape() == [64]


@pytest_ark()
def test_module_register_module_type_error():
    """register_module raises TypeError for non-Module."""

    class M(ark.Module):
        def __init__(self):
            super().__init__()

    m = M()
    with pytest.raises(TypeError):
        m.register_module("bad", "not a module")


@pytest_ark()
def test_module_register_parameter_type_error():
    """register_parameter raises TypeError for non-Parameter."""

    class M(ark.Module):
        def __init__(self):
            super().__init__()

    m = M()
    with pytest.raises(TypeError):
        m.register_parameter("bad", "not a param")
