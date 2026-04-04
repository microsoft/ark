# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import torch


def quickstart_tutorial():
    # Initialize the ARK environments
    ark.init()

    M, N, K = 1024, 1024, 1024
    m0 = torch.randn(M, K, dtype=torch.float16, device="cuda:0") * 0.01
    m1 = torch.randn(N, K, dtype=torch.float16, device="cuda:0") * 0.01

    # stage 1: matmul
    with ark.PlannerContext(processor_range=[0, 108]):
        # Use SMs 0~107 (all)
        t0 = ark.matmul(m0, m1, transpose_other=True)

    # stage 2: parallel copy and matmul
    m2 = ark.tensor([M, K], ark.fp16)
    with ark.PlannerContext(processor_range=[0, 54]):
        # Use SMs 0~53
        t1 = ark.matmul(t0, m1)
    with ark.PlannerContext(processor_range=[54, 108]):
        # Use SMs 54~107
        t2 = ark.copy(input=t0, output=m2)

    # Evaluate and check results
    with ark.Runtime() as rt:
        rt.launch()
        rt.run()
        t0_result = t0.to_torch()
        t1_result = t1.to_torch()
        t2_result = t2.to_torch()

    # Check the matmul result
    expected = torch.matmul(torch.matmul(m0, m1.T), m1)
    torch.testing.assert_close(t1_result, expected, rtol=1e-3, atol=1e-3)

    # Check the copy result
    torch.testing.assert_close(t2_result, t0_result, atol=0, rtol=0)

    print("Successful!")


if __name__ == "__main__":
    quickstart_tutorial()
