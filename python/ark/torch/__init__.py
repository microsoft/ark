# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

try:
    import torch

    _no_torch = False
except ImportError:
    from . import mock as torch

    _no_torch = True
