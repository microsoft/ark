// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/plan_manager.hpp>

namespace py = pybind11;

void register_plan_manager(py::module &m) {
    py::class_<ark::PlanManager>(m, "_PlanManager")
        .def(py::init<ark::Model&, const std::string&>());
}
