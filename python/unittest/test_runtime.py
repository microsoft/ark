# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark


empty_plan = ark.Plan(None)


def test_runtime_relaunch():
    ark.init()
    with ark.Runtime.get_runtime() as rt:
        assert rt.launched() == False
        rt.launch()
        assert rt.launched() == True

    with ark.Runtime.get_runtime() as rt:
        assert rt.launched() == False
        rt.launch()
        assert rt.launched() == True


# def test_multiple_runtime_launch():
#     ark.init()
#     num_runtimes = 5
#     for i in range(num_runtimes):
#         rt = ark.Runtime.get_runtime(i)
#         assert rt.launched() == False
#         rt.launch(plan=empty_plan, device_id=i)
#         assert rt.launched() == True
#     for i in range(num_runtimes):
#         rt = ark.Runtime.get_runtime(i)
#         assert rt.launched() == True
#     ark.Runtime.delete_all_runtimes()


# def test_stop_runtime():
#     ark.init()
#     rt1 = ark.Runtime.get_runtime(1)
#     rt1.launch(plan=empty_plan, device_id=1)
#     rt2 = ark.Runtime.get_runtime(2)
#     rt2.launch(plan=empty_plan, device_id=2)
#     rt1.stop()
#     rt1.reset()
#     assert rt1.state == ark.Runtime.State.Init
#     assert rt2.state == ark.Runtime.State.LaunchedNotRunning
#     ark.Runtime.delete_all_runtimes()


# def test_reset_runtime():
#     ark.init()
#     rt1 = ark.Runtime.get_runtime(0)
#     rt1.launch(plan=empty_plan, device_id=1)
#     rt2 = ark.Runtime.get_runtime(1)
#     rt2.launch(plan=empty_plan, device_id=2)
#     rt1.reset()
#     assert rt1.launched() == False
#     assert rt2.launched() == True
#     rt1.launch(plan=empty_plan)
#     assert rt1.launched() == True
#     ark.Runtime.delete_all_runtimes()


# def test_multiple_runtimes_complex():
#     ark.init()
#     num_runtimes = 3
#     runtime_list = [ark.Runtime.get_runtime(i) for i in range(num_runtimes)]
#     default_runtime = ark.Runtime.get_runtime()
#     runtime_list.append(default_runtime)
#     for i, rt in enumerate(runtime_list):
#         rt.launch(plan=empty_plan, device_id=i)
#         assert rt.launched() == True
#     runtime_list[0].stop()
#     assert runtime_list[0].state == ark.Runtime.State.LaunchedNotRunning
#     for rt in runtime_list[1:]:
#         assert rt.launched() == True
#     runtime_list[1].reset()
#     assert runtime_list[1].state == ark.Runtime.State.Init
#     assert runtime_list[0].state == ark.Runtime.State.LaunchedNotRunning
#     assert runtime_list[2].state == ark.Runtime.State.LaunchedNotRunning
#     runtime_list[1].launch(plan=empty_plan, device_id=1)
#     for rt in runtime_list:
#         assert rt.launched() == True
#     ark.Runtime.delete_all_runtimes()


# def test_runtime_state_after_reset():
#     ark.init()
#     rt = ark.Runtime.get_runtime()
#     rt.launch(plan=empty_plan)
#     rt.reset()
#     assert rt.launched() == False
#     assert rt.running() == False
#     ark.Runtime.delete_all_runtimes()


# def test_see_runtime_statuses():
#     ark.init()
#     num_runtimes = 3
#     runtimes = [ark.Runtime.get_runtime(i) for i in range(num_runtimes)]
#     runtime_statuses = ark.Runtime.see_runtime_statuses()
#     assert len(runtime_statuses) == num_runtimes
#     for i in range(num_runtimes):
#         assert i in runtime_statuses
#     for i, rt in enumerate(runtimes):
#         assert runtime_statuses[i] == rt
#     ark.Runtime.delete_all_runtimes()


# def test_multiple_runtimes_init():
#     ark.init()
#     runtimes = [ark.Runtime.get_runtime(i) for i in range(3)]
#     for rt in runtimes:
#         assert rt.state == ark.Runtime.State.Init
#     ark.init()
#     runtimes = ark.Runtime.see_runtime_statuses()
#     assert len(runtimes) == 0
#     ark.Runtime.delete_all_runtimes()
