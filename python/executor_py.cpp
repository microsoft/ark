// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ark/executor.hpp>
#include <ark/model.hpp>

namespace py = pybind11;

static void tensor_write(ark::Executor *exe, const ark::Tensor &tensor,
                         py::buffer host_buffer) {
    py::buffer_info info = host_buffer.request();
    exe->tensor_write(tensor, reinterpret_cast<void *>(info.ptr),
                      info.size * info.itemsize);
}

static void tensor_read(ark::Executor *exe, const ark::Tensor &tensor,
                        py::buffer host_buffer) {
    py::buffer_info info = host_buffer.request();
    exe->tensor_read(tensor, reinterpret_cast<void *>(info.ptr),
                     info.size * info.itemsize);
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
        .def("tensor_write", &tensor_write, py::arg("tensor"), py::arg("data"));
}
