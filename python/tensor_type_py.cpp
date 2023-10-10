// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ark.h"

namespace py = pybind11;

#define PY_REGISTER_TENSOR_TYPE(_type_name)                    \
    py::class_<ark::TensorType_##_type_name, ark::TensorType>( \
        m, "_TensorType_" #_type_name)                         \
        .def(py::init<>());                                    \
    m.attr("_" #_type_name) =                                  \
        py::cast(&ark::_type_name, py::return_value_policy::reference);

void register_tensor_type(py::module &m) {
    py::class_<ark::TensorType>(m, "_TensorType")
        .def(pybind11::self == pybind11::self)
        .def(pybind11::self != pybind11::self)
        .def("bytes", &ark::TensorType::bytes)
        .def("name", &ark::TensorType::name);

    PY_REGISTER_TENSOR_TYPE(FP32)
    PY_REGISTER_TENSOR_TYPE(FP16)
    PY_REGISTER_TENSOR_TYPE(BF16)
    PY_REGISTER_TENSOR_TYPE(INT32)
    PY_REGISTER_TENSOR_TYPE(UINT32)
    PY_REGISTER_TENSOR_TYPE(INT8)
    PY_REGISTER_TENSOR_TYPE(UINT8)
    PY_REGISTER_TENSOR_TYPE(BYTE)
}
