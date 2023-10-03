// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ark.h"

namespace py = pybind11;

void register_executor(py::module &m) {
    py::class_<ark::Executor>(m, "_Executor")
        .def(py::init<int, int, ark::Model &, const std::string &, int>(),
             py::arg("rank"), py::arg("world_size"), py::arg("model"),
             py::arg("name"), py::arg("num_warps_per_sm") = 16)
        .def("compile", &ark::Executor::compile)
        .def("launch", &ark::Executor::launch)
        .def("run", &ark::Executor::run, py::arg("iter"))
        .def("wait", &ark::Executor::wait)
        .def("stop", &ark::Executor::stop);
}
