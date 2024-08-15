// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <dlpack/dlpack.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/model.hpp>
#include <ark/model_graph.hpp>

#include "logging.hpp"

namespace py = pybind11;

struct DLTensorMetadata {
    void *data_ptr;
    int32_t device_id;
    DLDeviceType device_type;
    int32_t ndim;
    DLDataType dtype;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    uint64_t byte_offset;
};

static DLTensorMetadata extractDLTensorMetadata(DLManagedTensor *dl_tensor) {
    DLTensorMetadata metadata;
    metadata.data_ptr = dl_tensor->dl_tensor.data;
    metadata.device_id = dl_tensor->dl_tensor.device.device_id;
    metadata.device_type = dl_tensor->dl_tensor.device.device_type;
    metadata.ndim = dl_tensor->dl_tensor.ndim;
    metadata.dtype = dl_tensor->dl_tensor.dtype;
    metadata.shape.assign(
        dl_tensor->dl_tensor.shape,
        dl_tensor->dl_tensor.shape + dl_tensor->dl_tensor.ndim);
    if (dl_tensor->dl_tensor.strides != nullptr) {
        metadata.strides.assign(
            dl_tensor->dl_tensor.strides,
            dl_tensor->dl_tensor.strides + dl_tensor->dl_tensor.ndim);
    }
    metadata.byte_offset = dl_tensor->dl_tensor.byte_offset;
    return metadata;
}

static ark::DataType from_dl_dtype(const DLDataType &dl_dtype) {
    if (dl_dtype.lanes != 1) {
        ERR(ark::UnsupportedError, "unsupported data type");
    }
    ark::DataType ark_dtype;
    if (dl_dtype.code == kDLFloat && dl_dtype.bits == 32) {
        ark_dtype = ark::FP32;
    } else if (dl_dtype.code == kDLFloat && dl_dtype.bits == 16) {
        ark_dtype = ark::FP16;
    } else if (dl_dtype.code == kDLBfloat && dl_dtype.bits == 16) {
        ark_dtype = ark::BF16;
    } else if (dl_dtype.code == kDLInt && dl_dtype.bits == 32) {
        ark_dtype = ark::INT32;
    } else if (dl_dtype.code == kDLUInt && dl_dtype.bits == 32) {
        ark_dtype = ark::UINT32;
    } else if (dl_dtype.code == kDLInt && dl_dtype.bits == 8) {
        ark_dtype = ark::INT8;
    } else if (dl_dtype.code == kDLUInt && dl_dtype.bits == 8) {
        ark_dtype = ark::UINT8;
    } else {
        ERR(ark::UnsupportedError, "unsupported data type");
    }
    return ark_dtype;
}

void register_model(py::module &m) {
    py::class_<ark::Model, ark::ModelGraph>(m, "_Model")
        .def(py::init<int, int>(), py::arg("rank"), py::arg("world_size"))
        .def("rank", &ark::Model::rank)
        .def("world_size", &ark::Model::world_size)
        .def("id", &ark::Model::id)
        .def("compress", &ark::Model::compress)
        .def("add",
             py::overload_cast<ark::Tensor, ark::Tensor, ark::Tensor,
                               const std::string &>(&ark::Model::add),
             py::arg("input"), py::arg("other"), py::arg("output"),
             py::arg("name"))
        .def("add",
             py::overload_cast<ark::Tensor, float, ark::Tensor,
                               const std::string &>(&ark::Model::add),
             py::arg("input"), py::arg("other"), py::arg("output"),
             py::arg("name"))
        .def("cast", &ark::Model::cast, py::arg("input"), py::arg("data_type"),
             py::arg("output"), py::arg("name"))
        .def("constant", &ark::Model::constant, py::arg("value"),
             py::arg("shape"), py::arg("data_type"), py::arg("name"))
        .def("copy",
             py::overload_cast<ark::Tensor, ark::Tensor, const std::string &>(
                 &ark::Model::copy),
             py::arg("input"), py::arg("output"), py::arg("name"))
        .def("copy",
             py::overload_cast<float, ark::Tensor, const std::string &>(
                 &ark::Model::copy),
             py::arg("input"), py::arg("output"), py::arg("name"))
        .def("div",
             py::overload_cast<ark::Tensor, ark::Tensor, ark::Tensor,
                               const std::string &>(&ark::Model::div),
             py::arg("input"), py::arg("other"), py::arg("output"),
             py::arg("name"))
        .def("div",
             py::overload_cast<ark::Tensor, float, ark::Tensor,
                               const std::string &>(&ark::Model::div),
             py::arg("input"), py::arg("other"), py::arg("output"),
             py::arg("name"))
        .def("embedding", &ark::Model::embedding, py::arg("input"),
             py::arg("weight"), py::arg("output"), py::arg("name"))
        .def("exp", &ark::Model::exp, py::arg("input"), py::arg("output"),
             py::arg("name"))
        .def("gelu", &ark::Model::gelu, py::arg("input"), py::arg("output"),
             py::arg("name"))
        .def("identity", &ark::Model::identity, py::arg("input"),
             py::arg("deps"), py::arg("name"))
        .def("matmul", &ark::Model::matmul, py::arg("input"), py::arg("other"),
             py::arg("output"), py::arg("trans_input"), py::arg("trans_other"),
             py::arg("name"))
        .def("mul",
             py::overload_cast<ark::Tensor, ark::Tensor, ark::Tensor,
                               const std::string &>(&ark::Model::mul),
             py::arg("input"), py::arg("other"), py::arg("output"),
             py::arg("name"))
        .def("mul",
             py::overload_cast<ark::Tensor, float, ark::Tensor,
                               const std::string &>(&ark::Model::mul),
             py::arg("input"), py::arg("other"), py::arg("output"),
             py::arg("name"))
        .def("noop", &ark::Model::noop, py::arg("input"), py::arg("name"))
        .def(
            "placeholder",
            [](ark::Model &model, const ark::Dims &shape,
               const ark::DataType &data_type, const ark::Dims &strides,
               const ark::Dims &offsets, const ark::Dims &padded_shape,
               int rank, uintptr_t data, const std::string &name) {
                return model.placeholder(shape, data_type, strides, offsets,
                                         padded_shape, rank,
                                         reinterpret_cast<void *>(data), name);
            },
            py::arg("shape"), py::arg("data_type"), py::arg("strides"),
            py::arg("offsets"), py::arg("padded_shape"), py::arg("rank"),
            py::arg("data"), py::arg("name"))
        .def("reduce_max", &ark::Model::reduce_max, py::arg("input"),
             py::arg("axis"), py::arg("keepdims"), py::arg("output"),
             py::arg("name"))
        .def("reduce_mean", &ark::Model::reduce_mean, py::arg("input"),
             py::arg("axis"), py::arg("keepdims"), py::arg("output"),
             py::arg("name"))
        .def("reduce_sum", &ark::Model::reduce_sum, py::arg("input"),
             py::arg("axis"), py::arg("keepdims"), py::arg("output"),
             py::arg("name"))
        .def("relu", &ark::Model::relu, py::arg("input"), py::arg("output"),
             py::arg("name"))
        .def("reshape", &ark::Model::reshape, py::arg("input"),
             py::arg("shape"), py::arg("allowzero"), py::arg("name"))
        .def("rope", &ark::Model::rope, py::arg("input"), py::arg("other"),
             py::arg("output"), py::arg("name"))
        .def("rsqrt", &ark::Model::rsqrt, py::arg("input"), py::arg("output"),
             py::arg("name"))
        .def("sharding", &ark::Model::sharding, py::arg("input"),
             py::arg("axis"), py::arg("dim_per_shard"), py::arg("name"))
        .def("sigmoid", &ark::Model::sigmoid, py::arg("input"),
             py::arg("output"), py::arg("name"))
        .def("sqrt", &ark::Model::sqrt, py::arg("input"), py::arg("output"),
             py::arg("name"))
        .def("sub",
             py::overload_cast<ark::Tensor, ark::Tensor, ark::Tensor,
                               const std::string &>(&ark::Model::sub),
             py::arg("input"), py::arg("other"), py::arg("output"),
             py::arg("name"))
        .def("sub",
             py::overload_cast<ark::Tensor, float, ark::Tensor,
                               const std::string &>(&ark::Model::sub),
             py::arg("input"), py::arg("other"), py::arg("output"),
             py::arg("name"))
        .def("tensor", &ark::Model::tensor, py::arg("shape"),
             py::arg("data_type"), py::arg("strides"), py::arg("offsets"),
             py::arg("padded_shape"), py::arg("rank"), py::arg("name"))
        .def("placeholder",
             py::overload_cast<const ark::Dims &, const ark::DataType &,
                               const ark::Dims &, const ark::Dims &,
                               const ark::Dims &, int, const std::string &,
                               void *>(&ark::Model::placeholder),
             py::arg("shape"), py::arg("data_type"), py::arg("strides"),
             py::arg("offsets"), py::arg("padded_shape"), py::arg("rank"),
             py::arg("name"), py::arg("data"))
        .def(
            "placeholder",
            [](ark::Model &self, py::capsule input, int rank,
               const std::string &name) {
                DLManagedTensor *dl_tensor =
                    static_cast<DLManagedTensor *>(input.get_pointer());
                DLTensorMetadata metadata = extractDLTensorMetadata(dl_tensor);
                ark::DataType ark_dtype = from_dl_dtype(metadata.dtype);
                ark::Dims shape(metadata.shape);
                return self.placeholder(shape, ark_dtype, {}, {}, {}, rank,
                                        name, metadata.data_ptr);
            },
            py::arg("external_tensor"), py::arg("rank"), py::arg("name"))
        .def("transpose", &ark::Model::transpose, py::arg("input"),
             py::arg("permutation"), py::arg("output"), py::arg("name"))
        .def("all_reduce", &ark::Model::all_reduce, py::arg("input"),
             py::arg("rank"), py::arg("world_size"), py::arg("output"),
             py::arg("name"));
}
