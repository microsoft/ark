# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
import pybind11
import os

pybind_include_path = pybind11.get_include()

ark_root = os.environ["ARK_ROOT"]
if ark_root is None:
    print("ARK_ROOT is not set.")
    exit(-1)

this_dir = os.path.dirname(os.path.abspath(__file__))
bindings_src = this_dir + "/bindings.cpp"

ext_modules = [
    Pybind11Extension(
        "ark",
        [bindings_src],
        include_dirs=[pybind_include_path, ark_root + "/include"],
        libraries=["ark"],
        library_dirs=[ark_root + "/lib"],
        define_macros=[("DEBUG", None)],
    )
]

setup(name="ark", version="0.1", ext_modules=ext_modules)
