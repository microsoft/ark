# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest_common import ark, pytest_ark


@pytest_ark(need_torch=True)
def test_torch_tracer_module():
    import torch
    from ark.torch.tracer import tracer

    @tracer
    class TestModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.randn(1024, 1024))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.param

    model = TestModule().to("cuda:0")
    x = torch.randn(1024, 1024).to("cuda:0")
    y = model(x)
    y2 = model.forward_ark(x)
    assert torch.allclose(y, y2)
