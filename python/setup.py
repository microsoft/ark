# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Using python3 setup.py build_ext to build the python bindings.
# After running this script, the ark.cpython-38-x86_64-linux-gnu.so
# will be generated in the ark/python/build/lib.linux-x86_64-3.8 directory.
# Users can use export PYTHONPATH="$ARK_DIR/ark/python/build/lib.linux-x86_64-3.8:${PYTHONPATH}"
# And cd to the ark/python directory and run python3 python_test.py to test the python bindings.

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
import pybind11
import os

pybind_include_path = pybind11.get_include()

ark_dir = os.environ["ARK_ROOT"]
if ark_dir is None:
    print("ARK_ROOT is not set.")
    exit(-1)

ext_modules = [
    Pybind11Extension(
        "ark",
        ["bindings.cpp"],
        include_dirs=[pybind_include_path, ark_dir + "/include"],
        libraries=["ark"],
        library_dirs=[ark_dir + "/lib"],
        define_macros=[("DEBUG", None)],
    ),
]

setup(
    name="ark",
    version="0.1",
    ext_modules=ext_modules,
)
