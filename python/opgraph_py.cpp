// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ark.h"

namespace py = pybind11;

void register_opgraph(py::module &m) {
    py::class_<ark::OpGraph>(m, "_OpGraph")
        .def(py::init<ark::Model &>())
        .def("serialize", &ark::OpGraph::serialize, py::arg("indent"));
}
