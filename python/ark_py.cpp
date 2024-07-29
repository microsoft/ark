// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

extern void register_plan_manager(py::module &m);
extern void register_data_type(py::module &m);
extern void register_dims(py::module &m);
extern void register_error(py::module &m);
extern void register_executor(py::module &m);
extern void register_init(py::module &m);
extern void register_model_graph(py::module &m);
extern void register_model(py::module &m);
extern void register_planner(py::module &m);
extern void register_random(py::module &m);
extern void register_tensor(py::module &m);
extern void register_version(py::module &m);

PYBIND11_MODULE(_ark_core, m) {
    m.doc() = "Bind ARK C++ APIs to Python";

    register_plan_manager(m);
    register_data_type(m);
    register_dims(m);
    register_error(m);
    register_executor(m);
    register_init(m);
    register_model_graph(m);
    register_model(m);
    register_planner(m);
    register_random(m);
    register_tensor(m);
    register_version(m);
}
