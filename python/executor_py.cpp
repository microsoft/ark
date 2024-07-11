// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <dlpack/dlpack.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/executor.hpp>
#include <ark/model.hpp>
#include <iostream>
#include <stdexcept>
namespace py = pybind11;

static void tensor_write(ark::Executor *exe, const ark::Tensor &tensor,
                         py::buffer host_buffer, uintptr_t stream) {
    py::buffer_info info = host_buffer.request();
    exe->tensor_write(tensor, reinterpret_cast<void *>(info.ptr),
                      info.size * info.itemsize,
                      reinterpret_cast<ark::Stream>(stream), false);
}

static void tensor_write(ark::Executor *exe, const ark::Tensor &tensor,
                         size_t address, size_t bytes, uintptr_t stream,
                         bool is_d2d) {
    exe->tensor_write(tensor, reinterpret_cast<void *>(address), bytes,
                      reinterpret_cast<ark::Stream>(stream), is_d2d);
}

static void tensor_read(ark::Executor *exe, const ark::Tensor &tensor,
                        py::buffer host_buffer, uintptr_t stream) {
    py::buffer_info info = host_buffer.request();
    exe->tensor_read(tensor, reinterpret_cast<void *>(info.ptr),
                     info.size * info.itemsize,
                     reinterpret_cast<ark::Stream>(stream), false);
}

static void tensor_read(ark::Executor *exe, const ark::Tensor &tensor,
                        size_t address, size_t bytes, uintptr_t stream,
                        bool is_d2d) {
    exe->tensor_read(tensor, reinterpret_cast<void *>(address), bytes,
                     reinterpret_cast<ark::Stream>(stream), is_d2d);
}

static DLDataType get_dl_dtype(const ark::DataType &ark_data_type) {
    DLDataType dl_data_type;
    dl_data_type.lanes = 1;
    if (ark_data_type == ark::FP32) {
        dl_data_type.code = kDLFloat;
        dl_data_type.bits = 32;
    } else if (ark_data_type == ark::FP16) {
        dl_data_type.code = kDLFloat;
        dl_data_type.bits = 16;
    } else if (ark_data_type == ark::BF16) {
        dl_data_type.code = kDLBfloat;
        dl_data_type.bits = 16;
    } else if (ark_data_type == ark::INT32) {
        dl_data_type.code = kDLInt;
        dl_data_type.bits = 32;
    } else if (ark_data_type == ark::UINT32) {
        dl_data_type.code = kDLUInt;
        dl_data_type.bits = 32;
    } else if (ark_data_type == ark::INT8) {
        dl_data_type.code = kDLInt;
        dl_data_type.bits = 8;
    } else if (ark_data_type == ark::UINT8) {
        dl_data_type.code = kDLUInt;
        dl_data_type.bits = 8;
    } else if (ark_data_type == ark::BYTE) {
        dl_data_type.code = kDLUInt;
        dl_data_type.bits = 8;
    } else {
        throw std::runtime_error("unexpected error");
    }
    return dl_data_type;
}

static DLDeviceType get_device_type() {
#if defined(ARK_CUDA)
    return kDLCUDA;
#elif defined(ARK_ROCM)
    return kDLROCM;
#else
    return kDLCPU;
#endif
}

static DLManagedTensor *to_dlpack(ark::Executor &exe,
                                  const ark::Tensor &tensor) {
    DLTensor dl_tensor;
    dl_tensor.data = reinterpret_cast<void *>(exe.tensor_address(tensor));
    size_t offset_in_elements =
        tensor.offsets().is_no_dim() ? 0 : tensor.offsets().vector()[0];
    dl_tensor.byte_offset = offset_in_elements * tensor.data_type().bytes();
    dl_tensor.device.device_type = get_device_type();
    dl_tensor.device.device_id = static_cast<int32_t>(exe.gpu_id());
    dl_tensor.ndim = static_cast<int32_t>(tensor.shape().ndims());
    dl_tensor.dtype = get_dl_dtype(tensor.data_type());

    dl_tensor.shape =
        tensor.shape().is_no_dim() ? nullptr : new int64_t[dl_tensor.ndim];
    dl_tensor.strides =
        tensor.strides().is_no_dim() ? nullptr : new int64_t[dl_tensor.ndim];
    auto shape = tensor.shape();
    if (dl_tensor.shape) {
        for (int i = 0; i < dl_tensor.ndim; ++i) {
            dl_tensor.shape[i] = shape[i];
        }
    }
    if (dl_tensor.strides) {
        dl_tensor.strides[dl_tensor.ndim - 1] = 1;
        for (int i = dl_tensor.ndim - 2; i >= 0; --i) {
            dl_tensor.strides[i] =
                dl_tensor.shape[i + 1] * dl_tensor.strides[i + 1];
        }
    }
    DLManagedTensor *dl_managed_tensor = new DLManagedTensor();
    dl_managed_tensor->dl_tensor = dl_tensor;
    dl_managed_tensor->manager_ctx = nullptr;
    dl_managed_tensor->deleter = [](DLManagedTensor *self) {
        if (self->dl_tensor.shape) {
            delete[] self->dl_tensor.shape;
            self->dl_tensor.shape = nullptr;
        }
        if (self->dl_tensor.strides) {
            delete[] self->dl_tensor.strides;
            self->dl_tensor.strides = nullptr;
        }
    };
    return dl_managed_tensor;
}

void free_capsule(PyObject *capsule) {
    const char *name = PyCapsule_GetName(capsule);
    auto *dl_managed_tensor =
        static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule, name));
    if (dl_managed_tensor) {
        dl_managed_tensor->deleter(dl_managed_tensor);
        dl_managed_tensor = nullptr;
    }
}

py::capsule to_dlpack_capsule(ark::Executor &self, const ark::Tensor &tensor) {
    DLManagedTensor *dl_managed_tensor = to_dlpack(self, tensor);
    const char *capsule_name = "dltensor";
    PyObject *dl_capsule = PyCapsule_New(static_cast<void *>(dl_managed_tensor),
                                         capsule_name, free_capsule);
    return py::reinterpret_steal<py::capsule>(dl_capsule);
}

void register_executor(py::module &m) {
    py::class_<ark::Executor>(m, "_Executor")
        .def(py::init([](int device_id, uintptr_t stream,
                         const std::string &name, const std::string &plan) {
            return new ark::Executor(
                device_id, reinterpret_cast<ark::Stream>(stream), name, plan);
        }))
        .def("device_id", &ark::Executor::device_id)
        .def("stream",
             [](ark::Executor *self) {
                 return reinterpret_cast<uintptr_t>(self->stream());
             })
        .def("plan", &ark::Executor::plan)
        .def("compile", &ark::Executor::compile)
        .def("launch", &ark::Executor::launch, py::arg("max_spin_count") = -1)
        .def("run", &ark::Executor::run, py::arg("iter"))
        .def("wait", &ark::Executor::wait, py::arg("max_spin_count") = -1)
        .def("stop", &ark::Executor::stop, py::arg("max_spin_count") = -1)
        .def("barrier", &ark::Executor::barrier)
        .def("destroy", &ark::Executor::destroy)
        .def("destroyed", &ark::Executor::destroyed)
        .def("tensor_read",
             py::overload_cast<ark::Executor *, const ark::Tensor &, py::buffer,
                               uintptr_t>(&tensor_read),
             py::arg("tensor"), py::arg("data"), py::arg("stream"))
        .def("tensor_read",
             py::overload_cast<ark::Executor *, const ark::Tensor &, size_t,
                               size_t, uintptr_t, bool>(&tensor_read),
             py::arg("tensor"), py::arg("address"), py::arg("bytes"),
             py::arg("stream"), py::arg("is_d2d"))
        .def("tensor_write",
             py::overload_cast<ark::Executor *, const ark::Tensor &, py::buffer,
                               uintptr_t>(&tensor_write),
             py::arg("tensor"), py::arg("data"), py::arg("stream"))
        .def("tensor_write",
             py::overload_cast<ark::Executor *, const ark::Tensor &, size_t,
                               size_t, uintptr_t, bool>(&tensor_write),
             py::arg("tensor"), py::arg("address"), py::arg("bytes"),
             py::arg("stream"), py::arg("is_d2d"))
        .def("get_dl_tensor", &to_dlpack_capsule);
}
