#include "ark.h"
#include <iostream>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

namespace py = pybind11;

// Data type for dimension.
typedef long long int DimType;

enum { DIMS_LEN = 4, NO_DIM = -1 };

PYBIND11_MODULE(ark, m)
{
    m.doc() = "pybind11 ark plugin"; // optional module docstring
    m.def("init", &ark::init, "A function that initializes the ark");

    m.def("srand", &ark::srand, py::arg("seed") = -1, "Sets the seed for the random number generator");  
    m.def("rand", &ark::rand, "Generates a random integer"); 
    m.attr("NO_DIM") = py::int_(static_cast<int>(ark::NO_DIM));  
    m.attr("DIMS_LEN") = py::int_(static_cast<int>(ark::DIMS_LEN)); 

    py::class_<ark::Dims>(m, "Dims")
        .def(py::init([](ark::DimType d0, ark::DimType d1, ark::DimType d2,
                         ark::DimType d3) {
                 return std::make_unique<ark::Dims>(d0, d1, d2, d3);
             }),
             py::arg_v("d0", static_cast<int>(ark::NO_DIM), "default value: NO_DIM"),
             py::arg_v("d1", static_cast<int>(ark::NO_DIM), "default value: NO_DIM"),
             py::arg_v("d2", static_cast<int>(ark::NO_DIM), "default value: NO_DIM"),
             py::arg_v("d3", static_cast<int>(ark::NO_DIM), "default value: NO_DIM"))
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
}
