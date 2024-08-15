# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest_common import ark, pytest_ark


@pytest_ark()
def test_error():
    try:
        ark.tensor([0])
    except ark.BaseError as e:
        assert isinstance(e, ark.ModelError)
