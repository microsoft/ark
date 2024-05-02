// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/random.hpp>

namespace py = pybind11;

void register_random(py::module &m) {
    m.def("srand", &ark::srand, py::arg("seed"));
    m.def("rand", &ark::rand);
}
