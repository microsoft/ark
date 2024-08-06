# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark


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
