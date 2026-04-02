# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from common import ark, pytest_ark


@pytest_ark()
def test_error():
    try:
        raise ark.InternalError("test")
    except ark.BaseError as e:
        assert isinstance(e, ark.InternalError)
        assert str(e) == "test"
