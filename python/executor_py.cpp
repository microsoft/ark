// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <dlpack/dlpack.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/executor.hpp>
#include <ark/model.hpp>
#include <unordered_map>

#include "gpu/gpu_memory.hpp"
#include "logging.hpp"

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

static DLDataType to_dl_dtype(const ark::DataType &ark_dtype) {
    DLDataType dl_dtype;
    dl_dtype.lanes = 1;
    if (ark_dtype == ark::FP32) {
        dl_dtype.code = kDLFloat;
        dl_dtype.bits = 32;
    } else if (ark_dtype == ark::FP16) {
        dl_dtype.code = kDLFloat;
        dl_dtype.bits = 16;
    } else if (ark_dtype == ark::BF16) {
        dl_dtype.code = kDLBfloat;
        dl_dtype.bits = 16;
    } else if (ark_dtype == ark::INT32) {
        dl_dtype.code = kDLInt;
        dl_dtype.bits = 32;
    } else if (ark_dtype == ark::UINT32) {
        dl_dtype.code = kDLUInt;
        dl_dtype.bits = 32;
    } else if (ark_dtype == ark::INT8) {
        dl_dtype.code = kDLInt;
        dl_dtype.bits = 8;
    } else if (ark_dtype == ark::UINT8) {
        dl_dtype.code = kDLUInt;
        dl_dtype.bits = 8;
    } else if (ark_dtype == ark::BYTE) {
        dl_dtype.code = kDLUInt;
        dl_dtype.bits = 8;
    } else {
        ERR(ark::InternalError, "unexpected");
    }
    return dl_dtype;
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

namespace ark {

class SharedTensor {
   public:
    SharedTensor(Executor &exe, const Tensor &tensor);
    ~SharedTensor() = default;

    DLTensor dl_tensor() const;

   private:
    std::shared_ptr<GpuMemory> buffer_;
    void *data_;
    int device_id_;
    DataType dtype_;
    std::shared_ptr<std::vector<int64_t>> shape_;
    std::shared_ptr<std::vector<int64_t>> strides_;
    std::shared_ptr<std::vector<int64_t>> offsets_;
};

SharedTensor::SharedTensor(Executor &exe, const Tensor &tensor) {
    buffer_ = exe.buffer();
    data_ = reinterpret_cast<void *>(exe.tensor_address(tensor));
    device_id_ = exe.device_id();
    dtype_ = tensor.data_type();
    shape_ = std::make_shared<std::vector<int64_t>>(tensor.shape().vector());
    strides_ =
        std::make_shared<std::vector<int64_t>>(tensor.torch_strides().vector());
    offsets_ =
        std::make_shared<std::vector<int64_t>>(tensor.offsets().vector());
}

DLTensor SharedTensor::dl_tensor() const {
    DLTensor dl_tensor;
    dl_tensor.data = data_;
    size_t offset_in_elements = offsets_->empty() ? 0 : offsets_->at(0);
    dl_tensor.byte_offset = offset_in_elements * dtype_.bytes();
    dl_tensor.device.device_type = get_device_type();
    dl_tensor.device.device_id = device_id_;
    dl_tensor.ndim = static_cast<int32_t>(shape_->size());
    dl_tensor.dtype = to_dl_dtype(dtype_);
    dl_tensor.shape = shape_->data();
    dl_tensor.strides = strides_->data();
    return dl_tensor;
}

}  // namespace ark

static py::capsule tensor_to_dlpack(ark::Executor &self,
                                    const ark::Tensor &tensor) {
    auto shared_tensor = new ark::SharedTensor(self, tensor);
    DLManagedTensor *dl_managed_tensor = new DLManagedTensor();
    dl_managed_tensor->dl_tensor = shared_tensor->dl_tensor();
    dl_managed_tensor->manager_ctx = shared_tensor;
    dl_managed_tensor->deleter = [](DLManagedTensor *self) {
        if (self->manager_ctx) {
            delete static_cast<ark::SharedTensor *>(self->manager_ctx);
            self->manager_ctx = nullptr;
        }
    };
    const char *capsule_name = "dltensor";
    PyObject *dl_capsule = PyCapsule_New(
        static_cast<void *>(dl_managed_tensor), capsule_name,
        [](PyObject *capsule) {
            const char *name = PyCapsule_GetName(capsule);
            auto *dl_managed_tensor = static_cast<DLManagedTensor *>(
                PyCapsule_GetPointer(capsule, name));
            if (dl_managed_tensor) {
                dl_managed_tensor->deleter(dl_managed_tensor);
                dl_managed_tensor = nullptr;
            }
        });
    return py::reinterpret_steal<py::capsule>(dl_capsule);
}

void register_executor(py::module &m) {
    py::class_<ark::Executor>(m, "_Executor")
        .def(py::init<>())
        .def("device_id", &ark::Executor::device_id)
        .def("stream",
             [](ark::Executor *self) {
                 return reinterpret_cast<uintptr_t>(self->stream());
             })
        .def("plan", &ark::Executor::plan)
        .def("name", &ark::Executor::name)
        .def("compile", &ark::Executor::compile, py::arg("device_id"),
             py::arg("plan"), py::arg("name") = "executor")
        .def(
            "launch",
            [](ark::Executor *self,
               const std::unordered_map<ark::Tensor, uintptr_t>
                   &placeholder_data,
               uintptr_t stream, bool loop_mode, bool record) {
                std::unordered_map<ark::Tensor, void *> tensor_ptr_map;
                for (const auto &[tensor, addr] : placeholder_data) {
                    tensor_ptr_map[tensor] = reinterpret_cast<void *>(addr);
                }

                self->launch(tensor_ptr_map,
                             reinterpret_cast<ark::Stream>(stream), loop_mode,
                             record);
            },
            py::arg("placeholder_data") =
                std::unordered_map<ark::Tensor, void *>(),
            py::arg("stream") = 0, py::arg("loop_mode") = true,
            py::arg("record") = false)

        .def(
            "run",
            [](ark::Executor *self, int iter,
               const std::unordered_map<ark::Tensor, uintptr_t>
                   &placeholder_data) {
                std::unordered_map<ark::Tensor, void *> tensor_ptr_map;
                for (const auto &[tensor, addr] : placeholder_data) {
                    tensor_ptr_map[tensor] = reinterpret_cast<void *>(addr);
                }
                self->run(iter, tensor_ptr_map);
            },
            py::arg("iter"),
            py::arg("placeholder_data") =
                std::unordered_map<ark::Tensor, void *>())
        .def("wait", &ark::Executor::wait, py::arg("max_spin_count") = -1)
        .def("stop", &ark::Executor::stop, py::arg("max_spin_count") = -1)
        .def("barrier", &ark::Executor::barrier)
        .def("destroy", &ark::Executor::destroy)
        .def("destroyed", &ark::Executor::destroyed)
        .def(
            "tensor_address",
            [](ark::Executor *self, const ark::Tensor &tensor) {
                return reinterpret_cast<uintptr_t>(
                    self->tensor_address(tensor));
            },
            py::arg("tensor"))
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
        .def("tensor_to_dlpack", &tensor_to_dlpack);
}
