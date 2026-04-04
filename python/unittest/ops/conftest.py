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

try:
    import torch

    _no_torch = False
except ImportError:
    _no_torch = True

# Skip entire ops/ directory if torch is unavailable
pytestmark = pytest.mark.skipif(_no_torch, reason="torch not available")

DEVICE = "cuda:0"


@pytest.fixture(autouse=True)
def _ark_init():
    """Reset ARK state before each test so tests don't share models."""
    ark.init()
