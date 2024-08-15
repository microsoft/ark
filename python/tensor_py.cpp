// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/tensor.hpp>

namespace py = pybind11;

void register_tensor(py::module& m) {
    py::class_<ark::Tensor>(m, "_Tensor")
        .def("id", &ark::Tensor::id)
        .def("shape", &ark::Tensor::shape)
        .def("strides", &ark::Tensor::strides)
        .def("offsets", &ark::Tensor::offsets)
        .def("padded_shape", &ark::Tensor::padded_shape)
        .def("data_type", &ark::Tensor::data_type)
        .def("torch_strides", &ark::Tensor::torch_strides);

    m.attr("_NullTensor") = &ark::NullTensor;
}
