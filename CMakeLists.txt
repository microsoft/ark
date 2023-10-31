# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set(ARK_MAJOR "0")
set(ARK_MINOR "3")
set(ARK_PATCH "0")

set(ARK_VERSION "${ARK_MAJOR}.${ARK_MINOR}.${ARK_PATCH}")
set(ARK_SOVERSION "${ARK_MAJOR}.${ARK_MINOR}")

option(USE_MSLL "Use USE_MSLL" OFF)

cmake_minimum_required(VERSION 3.25)
enable_language(CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(USE_CUDA "Use NVIDIA/CUDA." OFF)
option(USE_ROCM "Use AMD/ROCm." OFF)
option(BYPASS_GPU_CHECK "Bypass GPU check." OFF)

if(BYPASS_GPU_CHECK)
    if(USE_CUDA)
        message("Bypassing GPU check: using NVIDIA/CUDA.")
        find_package(CUDAToolkit REQUIRED)
    elseif(USE_ROCM)
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
        set(USE_CUDA ON)
        set(USE_ROCM OFF)
    elseif(NVIDIA_FOUND)
        message("Detected NVIDIA/CUDA.")
        set(USE_CUDA ON)
        set(USE_ROCM OFF)
    elseif(AMD_FOUND)
        message("Detected AMD/ROCm.")
        set(USE_CUDA OFF)
        set(USE_ROCM ON)
    else()
        message(FATAL_ERROR "Neither NVIDIA/CUDA nor AMD/ROCm is found.")
    endif()
endif()

# Declare project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-deprecated-declarations")
if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra")
    project(ark LANGUAGES CXX CUDA)

    # CUDA 11 or higher is required
    if(CUDAToolkit_VERSION_MAJOR LESS 11)
        message(FATAL_ERROR "CUDA 11 or higher is required but detected ${CUDAToolkit_VERSION}")
    endif()

    # Set CUDA architectures
    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 11)
        set(CMAKE_CUDA_ARCHITECTURES 60 70 80)
    endif()

    # Hopper architecture
    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 12)
        set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} 90)
    endif()
else()  # USE_ROCM
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Wall -Wextra")
    project(ark LANGUAGES CXX HIP)
endif()

# Find MSLL
if(USE_MSLL)
    include(${PROJECT_SOURCE_DIR}/cmake/FindMSLL.cmake)
    if (NOT MSLL_FOUND)
        message(FATAL_ERROR "MSLL not found")
    endif()
endif()

# Code coverage from https://github.com/codecov/example-cpp11-cmake
add_library(coverage_config INTERFACE)
# if(CMAKE_BUILD_TYPE MATCHES "Debug" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#     # Add required flags (GCC & LLVM/Clang)
#     target_compile_options(coverage_config INTERFACE
#         -O0        # no optimization
#         -g         # generate debug info
#         --coverage # sets all required flags
#     )
#     target_link_options(coverage_config INTERFACE --coverage)
# endif()

# Find ibverbs
include(FindIBVerbs)

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

if(BUILD_PYTHON)
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
add_dependencies(lint_and_ut cpplint-autofix ut)
