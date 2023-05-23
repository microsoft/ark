#include <pybind11/pybind11.h>
#include "ark.h"
#include "ark_utils.h"
#include <iostream>

namespace py = pybind11;


PYBIND11_MODULE(ark, m) {
    m.doc() = "pybind11 ark plugin"; // optional module docstring

    m.def("init", &ark::init, "A function that prints add");  
  
}