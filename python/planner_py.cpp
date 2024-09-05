// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/model.hpp>
#include <ark/planner.hpp>

namespace py = pybind11;

void register_planner(py::module &m) {
    py::class_<ark::PlannerContext>(m, "CorePlannerContext")
        .def(py::init<ark::Model &>())
        .def("processor_range", &ark::PlannerContext::processor_range,
             py::arg("start"), py::arg("end"), py::arg("step") = 1)
        .def("warp_range", &ark::PlannerContext::warp_range, py::arg("start"),
             py::arg("end"), py::arg("step") = 1)
        .def("sram_range", &ark::PlannerContext::sram_range, py::arg("start"),
             py::arg("end"), py::arg("step") = 1)
        .def("sync", &ark::PlannerContext::sync, py::arg("sync"))
        .def("config", &ark::PlannerContext::config, py::arg("config"));

    py::class_<ark::Planner>(m, "CorePlanner")
        .def(py::init<const ark::Model &, int>())
        .def("install_config_rule",
             [](ark::Planner *self, const py::function &rule) {
                 self->install_config_rule(
                     [rule](const std::string &op, const std::string &arch) {
                         return rule(op, arch).cast<std::string>();
                     });
             })
        .def("plan", &ark::Planner::plan, py::arg("pretty") = true);
}
