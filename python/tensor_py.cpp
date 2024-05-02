// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/tensor.hpp>

namespace py = pybind11;

void register_tensor(py::module &m) {
    py::class_<ark::Tensor>(m, "_Tensor")
        .def("id", &ark::Tensor::id)
        .def("shape", &ark::Tensor::shape, py::return_value_policy::reference)
        .def("strides", &ark::Tensor::strides, py::return_value_policy::reference)
        .def("offsets", &ark::Tensor::offsets, py::return_value_policy::reference)
        .def("pads", &ark::Tensor::pads, py::return_value_policy::reference)
        .def("data_type", &ark::Tensor::data_type, py::return_value_policy::reference);

    m.attr("NullTensor") = &ark::NullTensor;
}
