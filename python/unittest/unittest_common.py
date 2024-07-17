# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import ark


def pytest_ark(need_torch: bool = False):
    """
    Decorator for ARK unit tests.
    """

    def decorator(test_func):
        if need_torch:
            try:
                import torch
            except ImportError:
                return pytest.mark.skip(reason="torch is not installed")(
                    test_func
                )

        def wrapper(*args, **kwargs):
            ark.init()
            test_func(*args, **kwargs)

        return wrapper

    return decorator
