# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
if(Python3_FOUND)
  if(${Python3_VERSION} VERSION_LESS 3.8)
    message(FATAL_ERROR "Python version must be at least 3.8")
  endif()
endif()

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

file(GLOB_RECURSE BIND_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
pybind11_add_module(ark_py ${BIND_SOURCES})
set_target_properties(ark_py PROPERTIES OUTPUT_NAME _ark_core LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ark)
add_custom_command(TARGET ark_py POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/ark ${CMAKE_CURRENT_BINARY_DIR}/ark
)
target_link_libraries(ark_py PRIVATE ark_static)
target_include_directories(ark_py SYSTEM PRIVATE ${DLPACK_INCLUDE_DIRS})
target_include_directories(ark_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ark)

if(ARK_USE_CUDA)
    target_include_directories(ark_py SYSTEM PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )
endif()

if(ARK_USE_ROCM)
    target_include_directories(ark_py SYSTEM PRIVATE
        /opt/rocm/include
    )
endif()
