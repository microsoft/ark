# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
