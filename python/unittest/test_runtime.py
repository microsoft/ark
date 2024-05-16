# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest
import ark
import json
import time

class TestRuntime(unittest.TestCase):
    def test_runtime(self):
        empty_plan = json.dumps({
            "Rank": 0,
            "WorldSize": 1,
            "NumProcessors": 1,
            "NumWarpsPerProcessor": 1,
            "TaskInfos": [],
            "ProcessorGroups": []
        })

        with ark.Runtime.get_runtime() as rt:
            self.assertEqual(rt.launched(), False)
            rt.launch(plan=empty_plan)
            self.assertEqual(rt.launched(), True)

        with ark.Runtime.get_runtime() as rt:
            self.assertEqual(rt.launched(), False)
            rt.launch(plan=empty_plan)
            self.assertEqual(rt.launched(), True)
