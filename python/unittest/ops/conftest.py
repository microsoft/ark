# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Shared fixtures and helpers for ARK op numerical tests.
"""

import sys
import os
import pytest

# Add parent directory to path so `common` is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ark


# Auto-initialize ARK before each test; yield allows future teardown.
@pytest.fixture(autouse=True)
def _ark_init():
    """Initialize ARK state before each test so tests start fresh."""
    ark.init()
    yield
