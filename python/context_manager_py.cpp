// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/context_manager.hpp>

namespace py = pybind11;

void register_context_manager(py::module &m) {
    py::class_<ark::ContextManager>(m, "_ContextManager")
        .def(py::init<ark::Model&, const std::map<std::string, std::string>&>());
}
