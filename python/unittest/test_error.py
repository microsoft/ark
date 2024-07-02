# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark


def test_error():
    ark.init()
    try:
        ark.tensor([0])
    except Exception as e:
        assert isinstance(e, ark.InvalidUsageError)
