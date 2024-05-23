// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/model.hpp>
#include <ark/planner.hpp>

namespace py = pybind11;

void register_planner(py::module &m) {
    py::class_<ark::DefaultPlanner>(m, "_DefaultPlanner")
        .def(py::init<const ark::Model &, int>())
        .def("install_config_rule",
             [](ark::DefaultPlanner *self, const py::function &rule) {
                 self->install_config_rule(
                     [rule](const std::string &op, const std::string &arch) {
                         return rule(op, arch).cast<std::string>();
                     });
             })
        .def("plan", &ark::DefaultPlanner::plan, py::arg("pretty") = true);
}
