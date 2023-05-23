from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension

ark_dir = "/home/v-lifanwu/.ark"

ext_modules = [
    Pybind11Extension(
        "ark",
        ["bindings.cpp"],
        include_dirs=[
            "/opt/conda/lib/python3.8/site-packages/pybind11/include ",ark_dir+"/include"],
        libraries=["ark"],
        library_dirs=[ark_dir+"/lib"],
    ),
]

setup(
    name="ark",
    version="0.1",
    ext_modules=ext_modules,
)
