// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/dims.hpp>
#include <sstream>

namespace py = pybind11;

void register_dims(py::module &m) {
    m.attr("DIMS_LEN") = py::int_(ark::DIMS_LEN);

    py::class_<ark::Dims>(m, "_Dims")
        .def(py::init<>())
        .def(py::init<ark::DimType>())
        .def(py::init<ark::DimType, ark::DimType>())
        .def(py::init<ark::DimType, ark::DimType, ark::DimType>())
        .def(py::init<ark::DimType, ark::DimType, ark::DimType, ark::DimType>())
        .def(py::init<const ark::Dims &>())
        .def(py::init<const std::vector<ark::DimType> &>())
        .def("nelems", &ark::Dims::nelems)
        .def("ndims", &ark::Dims::ndims)
        .def("__getitem__",
             [](const ark::Dims &d, ark::DimType idx) { return d[idx]; })
        .def("__setitem__", [](ark::Dims &d, ark::DimType idx,
                               ark::DimType value) { d[idx] = value; })
        .def("__repr__", [](const ark::Dims &d) {
            std::ostringstream os;
            os << d;
            return os.str();
        });
}
