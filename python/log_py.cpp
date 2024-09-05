// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/log.hpp>

namespace py = pybind11;

void register_log(py::module &m) {
    py::enum_<ark::LogLevel>(m, "LogLevel")
        .value("DEBUG", ark::LogLevel::DEBUG)
        .value("INFO", ark::LogLevel::INFO)
        .value("WARN", ark::LogLevel::WARN)
        .value("ERROR", ark::LogLevel::ERROR)
        .export_values();
    m.def("log", &ark::log);
}
