// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/model_graph.hpp>
#include <ark/model.hpp>

namespace py = pybind11;

void register_model(py::module &m) {
    py::class_<ark::Model, ark::ModelGraph>(m, "_Model")
        .def(py::init<int, int>(), py::arg("rank"), py::arg("world_size"))
        .def("rank", &ark::Model::rank)
        .def("world_size", &ark::Model::world_size)
        .def("compress", &ark::Model::compress)
        .def("add", &ark::Model::add, py::arg("input"), py::arg("other"),
             py::arg("output"), py::arg("name"))
        .def("cast", &ark::Model::cast, py::arg("input"), py::arg("data_type"),
             py::arg("output"), py::arg("name"))
        .def("copy", &ark::Model::copy, py::arg("input"), py::arg("output"),
             py::arg("name"))
        .def("div", &ark::Model::div, py::arg("input"), py::arg("other"),
             py::arg("output"), py::arg("name"))
        .def("embedding", &ark::Model::embedding, py::arg("input"),
             py::arg("weight"), py::arg("output"), py::arg("name"))
        .def("exp", &ark::Model::exp, py::arg("input"), py::arg("output"),
             py::arg("name"))
        .def("gelu", &ark::Model::gelu, py::arg("input"), py::arg("output"),
             py::arg("name"))
        .def("matmul", &ark::Model::matmul, py::arg("input"), py::arg("other"),
             py::arg("output"), py::arg("trans_input"), py::arg("trans_other"),
             py::arg("name"))
        .def("mul", &ark::Model::mul, py::arg("input"), py::arg("other"),
             py::arg("output"), py::arg("name"))
        .def("noop", &ark::Model::noop, py::arg("input"), py::arg("name"))
        .def("reduce_max", &ark::Model::reduce_max, py::arg("input"),
             py::arg("output"), py::arg("axis"), py::arg("keepdims"),
             py::arg("name"))
        .def("reduce_mean", &ark::Model::reduce_mean, py::arg("input"),
             py::arg("output"), py::arg("axis"), py::arg("keepdims"),
             py::arg("name"))
        .def("reduce_sum", &ark::Model::reduce_sum, py::arg("input"),
             py::arg("output"), py::arg("axis"), py::arg("keepdims"),
             py::arg("name"))
        .def("relu", &ark::Model::relu, py::arg("input"), py::arg("output"),
             py::arg("name"))
        .def("reshape", &ark::Model::reshape, py::arg("input"),
             py::arg("shape"), py::arg("allowzero"), py::arg("name"))
        .def("rsqrt", &ark::Model::rsqrt, py::arg("input"), py::arg("output"),
             py::arg("name"))
        .def("scale", &ark::Model::scale, py::arg("input"), py::arg("factor"),
             py::arg("output"), py::arg("name"))
        .def("sigmoid", &ark::Model::sigmoid, py::arg("input"),
             py::arg("output"), py::arg("name"))
        .def("sqrt", &ark::Model::sqrt, py::arg("input"), py::arg("output"),
             py::arg("name"))
        .def("sub", &ark::Model::sub, py::arg("input"), py::arg("other"),
             py::arg("output"), py::arg("name"))
        .def("tensor", &ark::Model::tensor, py::arg("shape"),
             py::arg("data_type"), py::arg("strides"), py::arg("offsets"),
             py::arg("pads"), py::arg("exported"), py::arg("imported_rank"),
             py::arg("name"));
}
