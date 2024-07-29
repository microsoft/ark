# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np
import json


def test_runtime_relaunch():
    ark.init()

    empty_plan = json.dumps(
        {
            "Rank": 0,
            "WorldSize": 1,
            "NumProcessors": 1,
            "NumWarpsPerProcessor": 1,
            "TaskInfos": [],
            "ProcessorGroups": [],
        }
    )

    with ark.Runtime.get_runtime() as rt:
        assert rt.launched() == False
        rt.launch(plan=empty_plan)
        assert rt.launched() == True

    with ark.Runtime.get_runtime() as rt:
        assert rt.launched() == False
        rt.launch(plan=empty_plan)
        assert rt.launched() == True


def test_add_plans():
    ark.init()
    M, N = 64, 64
    input_tensor = ark.tensor([M, N], ark.fp16)
    other_tensor = ark.tensor([M, N], ark.fp16)
    output_tensor = ark.add(input_tensor, other_tensor)
    runtime = ark.Runtime()
    runtime.launch(loop_mode=False)
    input_tensor_host = np.random.rand(M, N).astype(np.float16)
    input_tensor.from_numpy(input_tensor_host)
    other_tensor_host = np.random.rand(M, N).astype(np.float16)
    other_tensor.from_numpy(other_tensor_host)
    runtime.run()
    output_tensor_host = output_tensor.to_numpy()
    np.testing.assert_allclose(
        output_tensor_host, input_tensor_host + other_tensor_host
    )
    runtime.reset(persist=True)
    # Add subsequent plan
    prev_output = output_tensor
    new_tensor = ark.tensor([M, N], ark.fp16)
    final_output = ark.add(prev_output, new_tensor)
    runtime.launch(loop_mode=False)
    new_tensor_host = np.random.rand(M, N).astype(np.float16)
    new_tensor.from_numpy(new_tensor_host)
    runtime.run()
    final_output_host = final_output.to_numpy()
    np.testing.assert_allclose(
        final_output_host, output_tensor_host + new_tensor_host
    )
    runtime.reset()
