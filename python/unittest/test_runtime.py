# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
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
