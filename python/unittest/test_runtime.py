# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np


empty_plan = ark.Plan(None)


def test_runtime_relaunch():
    ark.init()
    with ark.Runtime.get_runtime() as rt:
        assert rt.launched() == False
        rt.launch(plan=empty_plan)
        assert rt.launched() == True

    with ark.Runtime.get_runtime() as rt:
        assert rt.launched() == False
        rt.launch(plan=empty_plan)
        assert rt.launched() == True


def test_multiple_runtime_launch():
    ark.init()
    num_runtimes = 5
    for i in range(num_runtimes):
        rt = ark.Runtime.get_runtime(i)
        assert rt.launched() == False
        rt.launch(plan=empty_plan, device_id=i)
        assert rt.launched() == True
    for i in range(num_runtimes):
        rt = ark.Runtime.get_runtime(i)
        assert rt.launched() == True
    ark.Runtime.delete_all_runtimes()


def test_stop_runtime():
    ark.init()
    rt1 = ark.Runtime.get_runtime(1)
    rt1.launch(plan=empty_plan, device_id=1)
    rt2 = ark.Runtime.get_runtime(2)
    rt2.launch(plan=empty_plan, device_id=2)
    rt1.stop()
    rt1.reset()
    assert rt1.state == ark.Runtime.State.Init
    assert rt2.state == ark.Runtime.State.LaunchedNotRunning
    ark.Runtime.delete_all_runtimes()


def test_reset_runtime():
    ark.init()
    rt1 = ark.Runtime.get_runtime(0)
    rt1.launch(plan=empty_plan, device_id=1)
    rt2 = ark.Runtime.get_runtime(1)
    rt2.launch(plan=empty_plan, device_id=2)
    rt1.reset()
    assert rt1.launched() == False
    assert rt2.launched() == True
    rt1.launch(plan=empty_plan)
    assert rt1.launched() == True
    ark.Runtime.delete_all_runtimes()


def test_multiple_runtimes_complex():
    ark.init()
    num_runtimes = 3
    runtime_list = [ark.Runtime.get_runtime(i) for i in range(num_runtimes)]
    default_runtime = ark.Runtime.get_runtime()
    runtime_list.append(default_runtime)
    for i, rt in enumerate(runtime_list):
        rt.launch(plan=empty_plan, device_id=i)
        assert rt.launched() == True
    runtime_list[0].stop()
    assert runtime_list[0].state == ark.Runtime.State.LaunchedNotRunning
    for rt in runtime_list[1:]:
        assert rt.launched() == True
    runtime_list[1].reset()
    assert runtime_list[1].state == ark.Runtime.State.Init
    assert runtime_list[0].state == ark.Runtime.State.LaunchedNotRunning
    assert runtime_list[2].state == ark.Runtime.State.LaunchedNotRunning
    runtime_list[1].launch(plan=empty_plan, device_id=1)
    for rt in runtime_list:
        assert rt.launched() == True
    ark.Runtime.delete_all_runtimes()


def test_runtime_state_after_reset():
    ark.init()
    rt = ark.Runtime.get_runtime()
    rt.launch(plan=empty_plan)
    rt.reset()
    assert rt.launched() == False
    assert rt.running() == False
    ark.Runtime.delete_all_runtimes()


def test_see_runtime_statuses():
    ark.init()
    num_runtimes = 3
    runtimes = [ark.Runtime.get_runtime(i) for i in range(num_runtimes)]
    runtime_statuses = ark.Runtime.see_runtime_statuses()
    assert len(runtime_statuses) == num_runtimes
    for i in range(num_runtimes):
        assert i in runtime_statuses
    for i, rt in enumerate(runtimes):
        assert runtime_statuses[i] == rt
    ark.Runtime.delete_all_runtimes()


def test_multiple_runtimes_init():
    ark.init()
    runtimes = [ark.Runtime.get_runtime(i) for i in range(3)]
    for rt in runtimes:
        assert rt.state == ark.Runtime.State.Init
    ark.init()
    runtimes = ark.Runtime.see_runtime_statuses()
    assert len(runtimes) == 0
    ark.Runtime.delete_all_runtimes()

# Executor tests


def test_adding_executors():
    ark.init()
    runtime = ark.Runtime.get_runtime()
    num_executors = 3
    for i in range(num_executors):
        runtime.launch(plan=empty_plan, executor_id=i, device_id=i)
    assert len(runtime.executor_map) == num_executors
    assert len(runtime.executor_states) == num_executors
    for i in range(num_executors):
        assert i in runtime.executor_map
        assert (
            runtime.executor_states[i] == ark.Runtime.State.LaunchedNotRunning
        )
    ark.Runtime.delete_all_runtimes()


def test_getting_executor_state():
    ark.init()
    runtime = ark.Runtime.get_runtime()
    executor_ids = [1, 2, 3]
    assert runtime.get_state(1) == ark.Runtime.State.Init
    assert runtime.get_state(2) == ark.Runtime.State.Init
    assert runtime.get_state(3) == ark.Runtime.State.Init
    for id in executor_ids:
        runtime.launch(plan=empty_plan, executor_id=id, device_id=id)
        assert runtime.get_state(id) == ark.Runtime.State.LaunchedNotRunning
    ark.Runtime.delete_all_runtimes()
    assert runtime.get_state(1) == ark.Runtime.State.Init
    assert runtime.get_state(2) == ark.Runtime.State.Init
    assert runtime.get_state(3) == ark.Runtime.State.Init


def test_distinct_exec_states():
    ark.init()
    rt1, rt2 = ark.Runtime.get_runtime(1), ark.Runtime.get_runtime(2)
    executor_id = 1
    rt1.launch(plan=empty_plan, executor_id=executor_id, device_id=1)
    assert rt1.get_state(executor_id) == ark.Runtime.State.LaunchedNotRunning
    assert rt2.get_state(executor_id) == ark.Runtime.State.Init
    rt2.launch(plan=empty_plan, executor_id=executor_id, device_id=2)
    assert rt2.get_state(executor_id) == ark.Runtime.State.LaunchedNotRunning
    rt1.reset()
    assert rt1.get_state(executor_id) == ark.Runtime.State.Init
    assert rt2.get_state(executor_id) == ark.Runtime.State.LaunchedNotRunning
    rt2.reset()
    assert rt2.get_state(executor_id) == ark.Runtime.State.Init
    ark.Runtime.delete_all_runtimes()


def test_multi_executor():
    ark.init()
    M, N = 64, 64
    input_tensor = ark.tensor([M, N], ark.fp16)
    other_tensor = ark.tensor([M, N], ark.fp16)
    output_tensor = ark.add(input_tensor, other_tensor)
    runtime = ark.Runtime()
    executor_ids = [0, 1]
    for i in executor_ids:
        runtime.launch(executor_id=i)
        input_tensor_host = np.random.rand(M, N).astype(np.float16)
        input_tensor.from_numpy(input_tensor_host, executor_id=i)
        other_tensor_host = np.random.rand(M, N).astype(np.float16)
        other_tensor.from_numpy(other_tensor_host, executor_id=i)
        runtime.run(executor_id=i)
        output_tensor_host = output_tensor.to_numpy(executor_id=i)
        np.testing.assert_allclose(
            output_tensor_host, input_tensor_host + other_tensor_host
        )
        runtime.stop(executor_id=i)
    runtime.reset()
