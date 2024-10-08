# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from common import ark, pytest_ark
import numpy as np


@pytest_ark()
def test_tensor_slice():
    t0 = ark.ones([4, 64], ark.fp16)
    t1 = t0[2:, :]
    ark.noop(t1)

    assert t1.shape() == [2, 64]
    assert t1.dtype() == ark.fp16
    assert t1.strides() == [4, 64]

    with ark.Runtime() as rt:
        rt.launch()
        rt.run()

        x = t1.to_numpy()

    assert np.allclose(x, np.ones([2, 64], np.float16))


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
