# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from common import ark, pytest_ark


@pytest_ark(need_torch=True)
def test_tensor_torch():
    import torch

    ones = torch.ones(2, 1024, device=torch.device("cuda:0"))

    t = ark.Tensor.from_torch(ones)
    t = ark.mul(t, 5)

    with ark.Runtime() as rt:
        rt.launch()
        rt.run()

        x = t.to_torch()

    assert torch.allclose(x, ones * 5)
