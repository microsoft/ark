// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/model_graph.hpp>

namespace py = pybind11;

void register_model_graph(py::module &m) {
    py::class_<ark::ModelGraph>(m, "_ModelGraph")
        .def("serialize", &ark::ModelGraph::serialize, py::arg("indent"));
}
