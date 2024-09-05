# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from common import ark, pytest_ark


@pytest_ark()
def test_runtime_empty():
    with ark.Runtime.get_runtime() as rt:
        rt.launch()
        rt.run()
        rt.stop()
