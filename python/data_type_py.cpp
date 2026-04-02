// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/data_type.hpp>

namespace py = pybind11;

void register_data_type(py::module &m) {
    py::class_<ark::DataType>(m, "CoreDataType")
        .def("__eq__", &ark::DataType::operator==)
        .def("__ne__", &ark::DataType::operator!=)
        .def("is_null", &ark::DataType::is_null)
        .def("bytes", &ark::DataType::bytes)
        .def("name", &ark::DataType::name, py::return_value_policy::reference)
        .def_static("from_name", &ark::DataType::from_name);

    m.attr("NONE") = &ark::NONE;
    m.attr("FP32") = &ark::FP32;
    m.attr("FP16") = &ark::FP16;
    m.attr("BF16") = &ark::BF16;
    m.attr("INT32") = &ark::INT32;
    m.attr("UINT32") = &ark::UINT32;
    m.attr("INT8") = &ark::INT8;
    m.attr("UINT8") = &ark::UINT8;
    m.attr("BYTE") = &ark::BYTE;
}
