# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set(NVIDIA_FOUND "FALSE")

find_package(CUDAToolkit)

if(NOT CUDAToolkit_FOUND)
    return()
endif()

# Use sm_80 as minimum for the detection check.
# Must be a CACHE variable so cmake applies it before enable_language(CUDA)
# tests the compiler. Without CACHE, cmake 3.25+ may probe with a default
# architecture (e.g., compute_60) that newer CUDA toolkits have dropped.
set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "CUDA architectures for GPU detection" FORCE)
if(NOT CMAKE_CUDA_COMPILER)
    # In case the CUDA Toolkit directory is not in the PATH
    find_program(CUDA_COMPILER
                 NAMES nvcc
                 PATHS ${CUDAToolkit_BIN_DIR})
    if(NOT CUDA_COMPILER)
        message(WARNING "Could not find nvcc in ${CUDAToolkit_BIN_DIR}")
        unset(CMAKE_CUDA_ARCHITECTURES)
        return()
    endif()
    set(CMAKE_CUDA_COMPILER "${CUDA_COMPILER}")
endif()
enable_language(CUDA)

set(CHECK_SRC "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_nvidia_gpu.cu")

try_run(RUN_RESULT COMPILE_SUCCESS SOURCES ${CHECK_SRC})

if(COMPILE_SUCCESS AND RUN_RESULT EQUAL 0)
    set(NVIDIA_FOUND "TRUE")
elseif(COMPILE_SUCCESS)
    message(WARNING "CUDA compiler found but no NVIDIA GPU detected")
else()
    message(WARNING "CUDA compiler found but failed to compile a CUDA program")
    unset(CMAKE_CUDA_ARCHITECTURES)
    unset(CMAKE_CUDA_COMPILER)
endif()
