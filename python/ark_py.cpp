// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include "ark.h"

namespace py = pybind11;

extern void register_dims(py::module &m);
extern void register_tensor_type(py::module &m);
extern void register_tensor(py::module &m);
extern void register_model(py::module &m);
extern void register_executor(py::module &m);

PYBIND11_MODULE(_ark_core, m) {
    m.doc() = "Bind ARK C++ APIs to Python";

    m.def("version", &ark::version);
    m.def("init", &ark::init);
    m.def("srand", &ark::srand, py::arg("seed") = -1);
    m.def("rand", &ark::rand);

    register_dims(m);
    register_tensor_type(m);
    register_tensor(m);
    register_model(m);
    register_executor(m);
}
