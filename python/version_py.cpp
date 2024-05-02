// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/version.hpp>

namespace py = pybind11;

void register_version(py::module &m) { m.def("version", &ark::version); }
