// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ark.h"

namespace py = pybind11;

void tensor_write(ark::Tensor *tns, py::buffer host_buffer) {
    py::buffer_info info = host_buffer.request();
    tns->write(info.ptr);
}

void tensor_read(ark::Tensor *tns, py::buffer host_buffer) {
    py::buffer_info info = host_buffer.request();
    tns->read(info.ptr);
}

void register_tensor(py::module &m) {
    py::class_<ark::TensorBuf>(m, "_TensorBuf")
        .def(py::init<const ark::DimType &, int>(), py::arg("bytes") = 0,
             py::arg("id") = -1)
        .def_readwrite("bytes", &ark::TensorBuf::bytes)
        .def_readwrite("id", &ark::TensorBuf::id)
        .def_readwrite("immutable", &ark::TensorBuf::immutable);

    py::class_<ark::Tensor>(m, "_Tensor")
        .def(py::init<const ark::Dims &, const ark::TensorType &,
                      ark::TensorBuf *, const ark::Dims &, const ark::Dims &,
                      const ark::Dims &, bool, int, int, const std::string &>(),
             py::arg("shape"), py::arg("type"), py::arg("buf"),
             py::arg("ldims"), py::arg("offs"), py::arg("pads"),
             py::arg("exported"), py::arg("imported_rank"), py::arg("id"),
             py::arg("name"))
        .def_property_readonly("shape",
                               [](const ark::Tensor &t) {
                                   py::list shape_list;
                                   for (int i = 0; i < t.ndims(); ++i) {
                                       shape_list.append((int)t.shape[i]);
                                   }
                                   return shape_list;
                               })
        .def_property_readonly("ldims",
                               [](const ark::Tensor &t) {
                                   py::list ldims_list;
                                   for (int i = 0; i < t.ndims(); ++i) {
                                       ldims_list.append((int)t.ldims[i]);
                                   }
                                   return ldims_list;
                               })
        .def_property_readonly("type",
                               [](const ark::Tensor &t) { return t.type; })
        .def("write", &tensor_write, py::arg("buf"))
        .def("read", &tensor_read, py::arg("buf"))
        .def("clear", &ark::Tensor::clear)
        .def("offset", &ark::Tensor::offset, py::arg("i0") = 0,
             py::arg("i1") = 0, py::arg("i2") = 0, py::arg("i3") = 0)
        .def("size", &ark::Tensor::size)
        .def("ndims", &ark::Tensor::ndims)
        .def("type_bytes", &ark::Tensor::type_bytes)
        .def("shape_bytes", &ark::Tensor::shape_bytes)
        .def("ldims_bytes", &ark::Tensor::ldims_bytes)
        .def("offset_bytes", &ark::Tensor::offset_bytes, py::arg("i0") = 0,
             py::arg("i1") = 0, py::arg("i2") = 0, py::arg("i3") = 0)
        .def("is_alloced", &ark::Tensor::is_alloced)
        .def("is_sequential", &ark::Tensor::is_sequential);
}
