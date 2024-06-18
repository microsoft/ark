# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, os.environ.get("ARK_ROOT", ".") + "/python")

from test_model import *
from test_runtime import *
from test_conversion import *
