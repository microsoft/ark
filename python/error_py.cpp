// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/error.hpp>

namespace py = pybind11;

#define REGISTER_ERROR_PY(_name)                      \
    py::register_exception<ark::_name>(m, "_" #_name, \
                                       m.attr("_BaseError").ptr())

void register_error(py::module &m) {
    py::register_exception<ark::BaseError>(m, "_BaseError");

    REGISTER_ERROR_PY(InternalError);
    REGISTER_ERROR_PY(InvalidUsageError);
    REGISTER_ERROR_PY(ModelError);
    REGISTER_ERROR_PY(PlanError);
    REGISTER_ERROR_PY(UnsupportedError);
    REGISTER_ERROR_PY(SystemError);
    REGISTER_ERROR_PY(GpuError);
}
