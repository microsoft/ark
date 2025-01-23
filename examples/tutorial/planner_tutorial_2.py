# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import ark


def quickstart_tutorial():
    # Initialize the ARK environments
    ark.init()

    M, N, K = 1024, 1024, 1024
    m0 = ark.tensor([M, K], ark.fp16)
    m1 = ark.tensor([N, K], ark.fp16)
    m2 = ark.tensor([M, K], ark.fp16)

    # stage 1: matmul
    with ark.PlannerContext(processor_range=[0, 108]):
        # Use SMs 0~107 (all)
        t0 = ark.matmul(m0, m1, transpose_other=True)

    # stage 2: parallel copy and matmul
    with ark.PlannerContext(processor_range=[0, 54]):
        # Use SMs 0~53
        t1 = ark.matmul(t0, m1)
    with ark.PlannerContext(processor_range=[54, 108]):
        # Use SMs 54~107
        t2 = ark.copy(input=t0, output=m2)

    # Initialize the ARK runtime
    runtime = ark.Runtime()

    # Launch the ARK runtime
    runtime.launch()

    # Initialize
    m0_host = np.random.rand(M, K).astype(np.float16) * 0.01
    m0.from_numpy(m0_host)
    m1_host = np.random.rand(N, K).astype(np.float16) * 0.01
    m1.from_numpy(m1_host)

    # Run the ARK program
    runtime.run()

    # Check the matmul result
    res_host = np.matmul(np.matmul(m0_host, m1_host.T), m1_host)
    np.testing.assert_allclose(t1.to_numpy(), res_host, rtol=1e-3, atol=1e-3)

    # Check the copy result
    np.testing.assert_equal(t2.to_numpy(), t0.to_numpy())

    print("Successful!")


if __name__ == "__main__":
    quickstart_tutorial()
