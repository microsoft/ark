// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <dlpack/dlpack.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/tensor.hpp>

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

DLTensorMetadata extractDLTensorMetadata(DLManagedTensor* dl_tensor) {
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

void register_tensor(py::module& m) {
    py::class_<ark::Tensor>(m, "_Tensor")
        .def(py::init([](py::capsule capsule, const std::string& ark_type_str) {
            DLManagedTensor* dl_tensor = (DLManagedTensor*)capsule;
            if (!dl_tensor) {
                throw std::runtime_error(
                    "Capsule does not contain a DLManagedTensor");
            }
            DLTensorMetadata metadata = extractDLTensorMetadata(dl_tensor);
            int32_t device_id = metadata.device_id;
            void* data_ptr = metadata.data_ptr;
            int8_t dtype_bytes = metadata.dtype.bits / 8;
            auto shape = metadata.shape;

            return new ark::Tensor(data_ptr, device_id, dtype_bytes, shape, ark_type_str);
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
