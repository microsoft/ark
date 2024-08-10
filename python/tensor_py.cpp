// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <dlpack/dlpack.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/tensor.hpp>

#include "logging.hpp"

namespace py = pybind11;

struct DLTensorMetadata {
    void* data_ptr;
    int32_t device_id;
    DLDeviceType device_type;
    int32_t ndim;
    DLDataType dtype;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    uint64_t byte_offset;
};

static DLTensorMetadata extractDLTensorMetadata(DLManagedTensor* dl_tensor) {
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

void register_tensor(py::module& m) {
    py::class_<ark::Tensor>(m, "_Tensor")
        .def(py::init([](py::capsule capsule) {
            DLManagedTensor* dl_tensor = (DLManagedTensor*)capsule;
            if (!dl_tensor) {
                ERR(ark::InvalidUsageError,
                    "Capsule does not contain a DLManagedTensor");
            }
            DLTensorMetadata metadata = extractDLTensorMetadata(dl_tensor);
            int32_t device_id = metadata.device_id;
            void* data_ptr = metadata.data_ptr;
            auto shape = metadata.shape;

            return ark::Tensor(data_ptr, device_id, shape, from_dl_dtype(metadata.dtype));
        }))
        .def("id", &ark::Tensor::id)
        .def("shape", &ark::Tensor::shape, py::return_value_policy::reference)
        .def("strides", &ark::Tensor::strides,
             py::return_value_policy::reference)
        .def("offsets", &ark::Tensor::offsets,
             py::return_value_policy::reference)
        .def("padded_shape", &ark::Tensor::padded_shape,
             py::return_value_policy::reference)
        .def("data_type", &ark::Tensor::data_type,
             py::return_value_policy::reference);

    m.attr("_NullTensor") = &ark::NullTensor;
}
