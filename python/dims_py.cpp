// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include "ark.h"

namespace py = pybind11;

void register_dims(py::module &m) {
    m.attr("DIMS_LEN") = py::int_(static_cast<int>(ark::DIMS_LEN));
    m.attr("NO_DIM") = py::int_(static_cast<int>(ark::NO_DIM));

    py::class_<ark::Dims>(m, "_Dims")
        .def(py::init([](ark::DimType d0, ark::DimType d1, ark::DimType d2,
                         ark::DimType d3) {
                 return std::make_unique<ark::Dims>(d0, d1, d2, d3);
             }),
             py::arg_v("d0", static_cast<int>(ark::NO_DIM)),
             py::arg_v("d1", static_cast<int>(ark::NO_DIM)),
             py::arg_v("d2", static_cast<int>(ark::NO_DIM)),
             py::arg_v("d3", static_cast<int>(ark::NO_DIM)))
        .def(py::init<const ark::Dims &>())
        .def(py::init<const std::vector<ark::DimType> &>())
        .def("size", &ark::Dims::size)
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
