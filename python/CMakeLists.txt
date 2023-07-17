# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.6.2
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(ark_py ${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp)
set_target_properties(ark_py PROPERTIES OUTPUT_NAME ark)
target_link_libraries(ark_py PRIVATE ark_static)

install(TARGETS ark_py LIBRARY DESTINATION ark)
