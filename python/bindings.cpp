// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark.h"
#include <iostream>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

namespace py = pybind11;

void tensor_memcpy_host_to_device(ark::Executor *executor, ark::Tensor *tns,
                                  py::buffer host_buffer)
{
    py::buffer_info info = host_buffer.request();
    size_t bytes = info.size * info.itemsize;
    void *host_buffer_ptr = info.ptr;
    executor->tensor_memcpy(tns, (const void *)host_buffer_ptr, bytes);
}

void tensor_memcpy_device_to_host(ark::Executor *executor,
                                  py::buffer host_buffer, ark::Tensor *tns)
{
    py::buffer_info info = host_buffer.request();
    size_t bytes = info.size * info.itemsize;
    void *host_buffer_ptr = info.ptr;
    executor->tensor_memcpy((void *)host_buffer_ptr, tns, bytes);
}

PYBIND11_MODULE(ark, m)
{
    m.doc() = "pybind11 ark plugin"; // optional module docstring
    m.def("init", &ark::init, "A function that initializes the ark");

    m.def("srand", &ark::srand, py::arg("seed") = -1,
          "Sets the seed for the random number generator");
    m.def("rand", &ark::rand, "Generates a random integer");
    m.attr("NO_DIM") = py::int_(static_cast<int>(ark::NO_DIM));
    m.attr("DIMS_LEN") = py::int_(static_cast<int>(ark::DIMS_LEN));

    py::class_<ark::Dims>(m, "Dims")
        .def(py::init([](ark::DimType d0, ark::DimType d1, ark::DimType d2,
                         ark::DimType d3) {
                 return std::make_unique<ark::Dims>(d0, d1, d2, d3);
             }),
             py::arg_v("d0", static_cast<int>(ark::NO_DIM),
                       "default value: NO_DIM"),
             py::arg_v("d1", static_cast<int>(ark::NO_DIM),
                       "default value: NO_DIM"),
             py::arg_v("d2", static_cast<int>(ark::NO_DIM),
                       "default value: NO_DIM"),
             py::arg_v("d3", static_cast<int>(ark::NO_DIM),
                       "default value: NO_DIM"))
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

    // Register TensorType
    py::enum_<ark::TensorType>(m, "TensorType")
        .value("FP16", ark::TensorType::FP16)
        .value("FP32", ark::TensorType::FP32)
        .value("INT32", ark::TensorType::INT32)
        .export_values();

    // Register TensorBuf
    py::class_<ark::TensorBuf>(m, "TensorBuf")
        .def(py::init<const ark::DimType &, int>(), py::arg("bytes") = 0,
             py::arg("id") = -1)
        .def_readwrite("bytes", &ark::TensorBuf::bytes)
        .def_readwrite("id", &ark::TensorBuf::id)
        .def_readwrite("immutable", &ark::TensorBuf::immutable);

    // Register Tensor
    py::class_<ark::Tensor>(m, "Tensor")
        .def(py::init<const ark::Dims &, ark::TensorType, ark::TensorBuf *,
                      const ark::Dims &, const ark::Dims &, const ark::Dims &,
                      bool, bool, int, const std::string &>(),
             py::arg("shape"), py::arg("type"), py::arg("buf"),
             py::arg("ldims"), py::arg("offs"), py::arg("pads"),
             py::arg("exported"), py::arg("imported"), py::arg("id"),
             py::arg("name"))
        .def("offset", &ark::Tensor::offset, py::arg("i0") = 0,
             py::arg("i1") = 0, py::arg("i2") = 0, py::arg("i3") = 0)
        .def("size", &ark::Tensor::size)
        .def("ndims", &ark::Tensor::ndims)
        .def("padded_shape", &ark::Tensor::padded_shape)
        .def("type_bytes", &ark::Tensor::type_bytes)
        .def("shape_bytes", &ark::Tensor::shape_bytes)
        .def("ldims_bytes", &ark::Tensor::ldims_bytes)
        .def("offset_bytes", &ark::Tensor::offset_bytes, py::arg("i0") = 0,
             py::arg("i1") = 0, py::arg("i2") = 0, py::arg("i3") = 0);

    py::class_<ark::Model>(m, "Model")
        .def(py::init<>())
        .def("tensor", &ark::Model::tensor,
             py::return_value_policy::reference_internal, py::arg("shape"),
             py::arg("dtype"), py::arg("buf") = nullptr,
             py::arg("ldims") = ark::Dims(), py::arg("offs") = ark::Dims(),
             py::arg("pads") = ark::Dims(),
             py::arg("deps") = std::vector<ark::Tensor *>(),
             py::arg("exported") = false, py::arg("imported") = false,
             py::arg("name") = "tensor")
        .def("reshape",
             (ark::Tensor * (ark::Model::*)(ark::Tensor *, const ark::Dims &,
                                            bool, ark::Tensor *,
                                            const std::string &)) &
                 ark::Model::reshape,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("shape"), py::arg("allowzero") = false,
             py::arg("output") = nullptr, py::arg("name") = "reshape")
        .def("reshape",
             (ark::Tensor * (ark::Model::*)(ark::Tensor *,
                                            std::initializer_list<ark::DimType>,
                                            bool, ark::Tensor *,
                                            const std::string &)) &
                 ark::Model::reshape,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("shape"), py::arg("allowzero") = false,
             py::arg("output") = nullptr, py::arg("name") = "reshape")
        .def("identity", &ark::Model::identity,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("deps") = std::vector<ark::Tensor *>(),
             py::arg("output") = nullptr, py::arg("name") = "identity")
        .def("sharding", &ark::Model::sharding,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("axis"), py::arg("dim_per_shard"),
             py::arg("name") = "sharding")
        .def("reduce", &ark::Model::reduce,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("axis"), py::arg("output") = nullptr,
             py::arg("is_relu") = false, py::arg("name") = "reduce")
        .def("transpose", &ark::Model::transpose,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("perm"), py::arg("output") = nullptr,
             py::arg("name") = "transpose")
        .def("matmul", &ark::Model::matmul,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("other"), py::arg("output") = nullptr,
             py::arg("splitk") = 1, py::arg("trans_input") = false,
             py::arg("trans_other") = false, py::arg("is_relu") = false,
             py::arg("name") = "matmul", py::arg("gran_lev") = -1)
        .def("linear", &ark::Model::linear,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("out_features"), py::arg("bias") = true,
             py::arg("output") = nullptr, py::arg("splitk") = 1,
             py::arg("is_relu") = false, py::arg("name") = "linear",
             py::arg("gran_lev") = -1)
        .def("im2col", &ark::Model::im2col,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("kernel_height"), py::arg("kernel_width"),
             py::arg("stride_height"), py::arg("stride_width"),
             py::arg("pad_height"), py::arg("pad_width"),
             py::arg("dilation_height"), py::arg("dilation_width"),
             py::arg("output") = nullptr, py::arg("name") = "im2col")
        .def("conv2d", &ark::Model::conv2d,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("in_channels"), py::arg("out_channels"),
             py::arg("kernel_size"), py::arg("stride"), py::arg("padding"),
             py::arg("bias") = false, py::arg("output") = nullptr,
             py::arg("name") = "conv2d")
        .def("max_pool", &ark::Model::max_pool,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("kernel_size"), py::arg("stride"),
             py::arg("output") = nullptr, py::arg("name") = "max_pool")
        .def("scale", &ark::Model::scale,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("val"), py::arg("output") = nullptr,
             py::arg("name") = "scale")
        .def("glue", &ark::Model::glue,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("output") = nullptr, py::arg("name") = "glue")
        .def("add", &ark::Model::add,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("other"), py::arg("output") = nullptr,
             py::arg("name") = "add")
        .def("mul", &ark::Model::mul,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("other"), py::arg("output") = nullptr,
             py::arg("name") = "mul")
        .def("send", &ark::Model::send,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("id"), py::arg("gpu_dst"), py::arg("bytes") = 0,
             py::arg("output") = nullptr, py::arg("name") = "send")
        .def("send_done", &ark::Model::send_done,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("id"), py::arg("output") = nullptr,
             py::arg("name") = "send_done")
        .def("recv", &ark::Model::recv,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("id"), py::arg("gpu_src"), py::arg("bytes") = 0,
             py::arg("output") = nullptr, py::arg("name") = "recv")
        .def("send_mm", &ark::Model::send_mm,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("id"), py::arg("gpu_dst"), py::arg("bytes") = 0,
             py::arg("output") = nullptr, py::arg("name") = "send_mm")
        .def("recv_mm", &ark::Model::recv_mm,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("id"), py::arg("gpu_src"), py::arg("bytes") = 0,
             py::arg("output") = nullptr, py::arg("name") = "recv_mm")
        .def("all_reduce", &ark::Model::all_reduce,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("gpu_id"), py::arg("gpu_num"), py::arg("output") = nullptr,
             py::arg("name") = "all_reduce");
    // register class Executor
    py::class_<ark::Executor>(m, "Executor")
        .def(py::init<const int, int, int, const ark::Model &,
                      const std::string &>(),
             py::arg("gpu_id"), py::arg("rank"), py::arg("world_size"),
             py::arg("model"), py::arg("name"))
        .def("compile", &ark::Executor::compile)
        .def("launch", &ark::Executor::launch)
        .def("run", &ark::Executor::run, py::arg("iter"))
        .def("wait", &ark::Executor::wait)
        .def("stop", &ark::Executor::stop)
        .def("get_tensor", &ark::Executor::get_tensor, py::arg("tns"),
             py::return_value_policy::reference_internal)
        .def("tensor_memcpy_host_to_device", &tensor_memcpy_host_to_device,
             py::arg("tns"), py::arg("src"))
        .def("tensor_memcpy_device_to_host", &tensor_memcpy_device_to_host,
             py::arg("dst"), py::arg("tns"))
        .def("tensor_clear", &ark::Executor::tensor_clear, py::arg("tns"));
}
