// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <dlpack/dlpack.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/executor.hpp>
#include <ark/model.hpp>
#include <iostream>
namespace py = pybind11;

static void tensor_write(ark::Executor *exe, const ark::Tensor &tensor,
                         py::buffer host_buffer) {
    py::buffer_info info = host_buffer.request();
    exe->tensor_write(tensor, reinterpret_cast<void *>(info.ptr),
                      info.size * info.itemsize);
}

static void tensor_write(ark::Executor *exe, const ark::Tensor &tensor,
                         size_t host_address, size_t bytes) {
    exe->tensor_write(tensor, reinterpret_cast<void *>(host_address), bytes);
}

static void tensor_read(ark::Executor *exe, const ark::Tensor &tensor,
                        py::buffer host_buffer) {
    py::buffer_info info = host_buffer.request();
    exe->tensor_read(tensor, reinterpret_cast<void *>(info.ptr),
                     info.size * info.itemsize);
}

DLManagedTensor *to_dlpack(ark::Executor &exe, const ark::Tensor &tensor) {
    DLManagedTensor *dl_tensor = exe.get_dl_tensor(tensor);
    return dl_tensor;
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
        .def(
            py::init<int, int, int, const std::string &, const std::string &>(),
            py::arg("rank"), py::arg("world_size"), py::arg("gpu_id"),
            py::arg("name"), py::arg("plan"))
        .def("compile", &ark::Executor::compile)
        .def("launch", &ark::Executor::launch, py::arg("max_spin_count") = -1)
        .def("run", &ark::Executor::run, py::arg("iter"))
        .def("wait", &ark::Executor::wait, py::arg("max_spin_count") = -1)
        .def("stop", &ark::Executor::stop, py::arg("max_spin_count") = -1)
        .def("barrier", &ark::Executor::barrier)
        .def("destroy", &ark::Executor::destroy)
        .def("destroyed", &ark::Executor::destroyed)
        .def("tensor_read", &tensor_read, py::arg("tensor"), py::arg("data"))
        .def(
            "tensor_write",
            py::overload_cast<ark::Executor *, const ark::Tensor &, py::buffer>(
                &tensor_write),
            py::arg("tensor"), py::arg("data"))
        .def("tensor_write",
             py::overload_cast<ark::Executor *, const ark::Tensor &, size_t,
                               size_t>(&tensor_write),
             py::arg("tensor"), py::arg("address"), py::arg("bytes"))
        .def("get_dl_tensor", &to_dlpack_capsule),
        py::arg("tensor");
}
