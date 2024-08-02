// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/error.hpp>

namespace py = pybind11;

#define REGISTER_ERROR_PY(_name) \
    py::register_exception<ark::_name>(m, "_" #_name)

void register_error(py::module &m) {
    REGISTER_ERROR_PY(InternalError);
    REGISTER_ERROR_PY(InvalidUsageError);
    REGISTER_ERROR_PY(NotFoundError);
    REGISTER_ERROR_PY(ModelError);
    REGISTER_ERROR_PY(SchedulerError);
    REGISTER_ERROR_PY(ExecutorError);
    REGISTER_ERROR_PY(SystemError);
    REGISTER_ERROR_PY(GpuError);
    REGISTER_ERROR_PY(RuntimeError);
}
