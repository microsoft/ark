# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set(ARK_MAJOR "0")
set(ARK_MINOR "5")
set(ARK_PATCH "0")

set(ARK_VERSION "${ARK_MAJOR}.${ARK_MINOR}.${ARK_PATCH}")
set(ARK_SOVERSION "${ARK_MAJOR}.${ARK_MINOR}")

cmake_minimum_required(VERSION 3.25)
enable_language(CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(ARK_USE_CUDA "Use NVIDIA/CUDA." OFF)
option(ARK_USE_ROCM "Use AMD/ROCm." OFF)
option(ARK_BYPASS_GPU_CHECK "Bypass GPU check." OFF)
option(ARK_BUILD_TESTS "Build unit tests." ON)
option(ARK_BUILD_PYTHON "Build Python module." ON)

if(ARK_BYPASS_GPU_CHECK)
    if(ARK_USE_CUDA)
        message("Bypassing GPU check: using NVIDIA/CUDA.")
        find_package(CUDAToolkit REQUIRED)
    elseif(ARK_USE_ROCM)
        message("Bypassing GPU check: using AMD/ROCm.")
        set(CMAKE_PREFIX_PATH "/opt/rocm;${CMAKE_PREFIX_PATH}")
        find_package(hip REQUIRED)
    else()
        message(FATAL_ERROR "Bypassing GPU check: neither NVIDIA/CUDA nor AMD/ROCm is specified.")
    endif()
else()
    # Detect GPUs
    include(CheckNvidiaGpu)
    include(CheckAmdGpu)
    if(NVIDIA_FOUND AND AMD_FOUND)
        message("Detected NVIDIA/CUDA and AMD/ROCm: prioritizing NVIDIA/CUDA.")
        set(ARK_USE_CUDA ON)
        set(ARK_USE_ROCM OFF)
    elseif(NVIDIA_FOUND)
        message("Detected NVIDIA/CUDA.")
        set(ARK_USE_CUDA ON)
        set(ARK_USE_ROCM OFF)
    elseif(AMD_FOUND)
        message("Detected AMD/ROCm.")
        set(ARK_USE_CUDA OFF)
        set(ARK_USE_ROCM ON)
    else()
        message(FATAL_ERROR "Neither NVIDIA/CUDA nor AMD/ROCm is found.")
    endif()
endif()

# Declare project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-deprecated-declarations")
if(ARK_USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra")
    project(ark LANGUAGES CXX CUDA)

    # CUDA 11 or higher is required
    if(CUDAToolkit_VERSION_MAJOR LESS 11)
        message(FATAL_ERROR "CUDA 11 or higher is required but detected ${CUDAToolkit_VERSION}")
    endif()

    # Set CUDA architectures
    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13)
        # CUDA 13+ dropped sm_60 and sm_70
        set(CMAKE_CUDA_ARCHITECTURES 80 90)
    elseif(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 12)
        set(CMAKE_CUDA_ARCHITECTURES 60 70 80 90)
    elseif(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 11)
        set(CMAKE_CUDA_ARCHITECTURES 60 70 80)
    endif()

    # CUDA 13+ moved CCCL headers into a cccl/ subdirectory.
    # Add it to the include path so third-party code (e.g. MSCCL++)
    # that includes <cuda/atomic> can still find the headers.
    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13)
        list(GET CUDAToolkit_INCLUDE_DIRS 0 _CUDA_INCLUDE_FIRST)
        set(CCCL_INCLUDE_DIR "${_CUDA_INCLUDE_FIRST}/cccl")
        if(EXISTS "${CCCL_INCLUDE_DIR}")
            include_directories(SYSTEM "${CCCL_INCLUDE_DIR}")
            message(STATUS "CUDA 13+: added CCCL include dir ${CCCL_INCLUDE_DIR}")
        else()
            message(WARNING "CUDA 13+: CCCL include dir not found at ${CCCL_INCLUDE_DIR}. Build may fail.")
        endif()
    endif()
else()  # ARK_USE_ROCM
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Wall -Wextra")
    project(ark LANGUAGES CXX HIP)
endif()

# Code coverage from https://github.com/codecov/example-cpp11-cmake
add_library(coverage_config INTERFACE)
if(CMAKE_BUILD_TYPE MATCHES "Debug" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Add required flags (GCC & LLVM/Clang)
    target_compile_options(coverage_config INTERFACE
        -O0        # no optimization
        -g         # generate debug info
        --coverage # sets all required flags
    )
    target_link_options(coverage_config INTERFACE --coverage)
endif()

# Find NUMA
include(FindNUMA)

# Third party libraries
add_subdirectory(third_party)

# ARK object
add_library(ark_obj OBJECT)
set_target_properties(ark_obj PROPERTIES
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE 1
    VERSION ${ARK_VERSION}
    SOVERSION ${ARK_SOVERSION}
)
add_dependencies(ark_obj ${TP_TARGETS})
target_link_libraries(ark_obj PUBLIC coverage_config)

# Build
add_custom_target(build)
add_dependencies(build ark_obj)
add_custom_command(TARGET build POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/ark/include ${CMAKE_CURRENT_BINARY_DIR}/include
)
add_custom_command(TARGET build POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${TP_KERNELS} ${CMAKE_CURRENT_BINARY_DIR}/include/kernels
)

# ARK shared library
add_library(ark SHARED)
set_target_properties(ark PROPERTIES
    VERSION ${ARK_VERSION}
    SOVERSION ${ARK_SOVERSION}
)
target_link_libraries(ark PUBLIC ark_obj)
add_dependencies(ark build)

# ARK static library
add_library(ark_static STATIC)
set_target_properties(ark_static PROPERTIES
    VERSION ${ARK_VERSION}
    SOVERSION ${ARK_SOVERSION}
)
target_link_libraries(ark_static PUBLIC ark_obj)
add_dependencies(ark_static build)

# ARK unit tests
include(CTest)
add_custom_target(ut)

# Details
add_subdirectory(ark)

if(ARK_BUILD_PYTHON)
    # Install Python module
    add_subdirectory(python)
    add_dependencies(ark_py build)
    install(TARGETS ark_py LIBRARY DESTINATION ark)
else()
    # Install libraries
    install(TARGETS ark ark_static
        LIBRARY DESTINATION ark/lib
        ARCHIVE DESTINATION ark/lib
    )
endif()

# Install header files
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include DESTINATION ark)

# Utils
include(${PROJECT_SOURCE_DIR}/cmake/Utils.cmake)
add_custom_target(lint_and_ut)
if(CLANG_FORMAT)
    add_dependencies(lint_and_ut cpplint-autofix ut)
endif()
