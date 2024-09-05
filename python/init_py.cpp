// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/init.hpp>

namespace py = pybind11;

void register_init(py::module &m) { m.def("init", &ark::init); }
