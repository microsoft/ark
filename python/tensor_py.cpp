// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/tensor.hpp>

namespace py = pybind11;

void register_tensor(py::module &m) {
    py::class_<ark::Tensor>(m, "CoreTensor")
        .def("id", &ark::Tensor::id)
        .def("shape", &ark::Tensor::shape)
        .def("strides", &ark::Tensor::strides)
        .def("offsets", &ark::Tensor::offsets)
        .def("padded_shape", &ark::Tensor::padded_shape)
        .def("data_type", &ark::Tensor::data_type)
        .def("torch_strides", &ark::Tensor::torch_strides)
        .def("data",
             [](const ark::Tensor& self) {
                 return reinterpret_cast<uintptr_t>(self.data());
             })
        .def(
            "data",
            [](ark::Tensor& self, uintptr_t data) {
                return self.data(reinterpret_cast<void*>(data));
            },
            py::arg("data"))
        .def("is_external", &ark::Tensor::is_external);

    m.attr("NullTensor") = &ark::NullTensor;
}
