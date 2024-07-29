// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/executor.hpp>
#include <ark/model.hpp>

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

void register_executor(py::module &m) {
    py::class_<ark::Executor>(m, "_Executor")
        .def(py::init([](int device_id, uintptr_t stream,
                         const std::string &name, const std::string &plan,
                         bool loop_mode) {
            return new ark::Executor(device_id,
                                     reinterpret_cast<ark::Stream>(stream),
                                     name, plan, loop_mode);
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
        .def("add_plan", &ark::Executor::add_plan, py::arg("plan"));
}
