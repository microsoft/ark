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


# Note: ops/ tests use an autouse fixture for ark.init() rather than the
# @pytest_ark() decorator used in the parent directory. Both are equivalent;
# the fixture approach avoids per-test decoration.
@pytest.fixture(autouse=True)
def _ark_init():
    """Initialize ARK state before each test so tests start fresh."""
    ark.init()
