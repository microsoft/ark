# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set(ARK_MAJOR "0")
set(ARK_MINOR "2")
set(ARK_PATCH "0")

set(ARK_VERSION "${ARK_MAJOR}.${ARK_MINOR}.${ARK_PATCH}")
set(ARK_SOVERSION "${ARK_MAJOR}.${ARK_MINOR}")

option(CODE_COVERAGE "Enable coverage reporting" OFF)
option(USE_KAHYPAR "Use KaHyPar for scheduling" OFF)

cmake_minimum_required(VERSION 3.25)
project(ark LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra")
set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Code coverage from https://github.com/codecov/example-cpp11-cmake
add_library(coverage_config INTERFACE)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Add required flags (GCC & LLVM/Clang)
    target_compile_options(coverage_config INTERFACE
        -O0        # no optimization
        -g         # generate debug info
        --coverage # sets all required flags
    )
    target_link_options(coverage_config INTERFACE --coverage)
endif()

# Find ibverbs
include(${PROJECT_SOURCE_DIR}/cmake/FindIBVerbs.cmake)

# Find NUMA
include(${PROJECT_SOURCE_DIR}/cmake/FindNUMA.cmake)

# Find CUDAToolkit
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    if(CUDAToolkit_VERSION_MAJOR LESS 11)
        message(FATAL_ERROR "CUDA 11 or higher is required but detected ${CUDAToolkit_VERSION}")
    endif()

    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 11)
        set(CMAKE_CUDA_ARCHITECTURES 70 80)
    endif()

    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 12)
        set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} 70 80 90)
    endif()
endif()

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
add_dependencies(ark_obj tp-cutlass)
target_link_libraries(ark_obj PUBLIC coverage_config)

# Build
add_custom_target(build)
add_dependencies(build ark_obj)
add_custom_command(TARGET build POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/ark/include ${BUILD_DIR}/include
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/third_party/cutlass/include ${BUILD_DIR}/include/kernels
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

# Add header files to library targets
file(GLOB_RECURSE ARK_HEADERS CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/ark/include/*.h)
file(GLOB_RECURSE CUTLASS_HEADERS CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/third_party/cutlass/include/*.h
    ${CMAKE_SOURCE_DIR}/third_party/cutlass/include/*.hpp
    ${CMAKE_SOURCE_DIR}/third_party/cutlass/include/*.inl)
target_sources(ark PUBLIC
    FILE_SET install_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/ark/include
    FILES ${ARK_HEADERS}
)
target_sources(ark_static PUBLIC
    FILE_SET install_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/ark/include
    FILES ${ARK_HEADERS}
)
target_sources(ark PUBLIC
    FILE_SET install_cutlass_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/third_party/cutlass/include
    FILES ${CUTLASS_HEADERS}
)
target_sources(ark_static PUBLIC
    FILE_SET install_cutlass_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/third_party/cutlass/include
    FILES ${CUTLASS_HEADERS}
)

# Install libraries
install(TARGETS ark ark_static
    LIBRARY DESTINATION ark/lib
    ARCHIVE DESTINATION ark/lib
    FILE_SET install_headers DESTINATION ark/include
    FILE_SET install_cutlass_headers DESTINATION ark/include/kernels
)

# Install Python module
if(BUILD_PYTHON)
    add_subdirectory(python)
endif()

# Utils
include(${PROJECT_SOURCE_DIR}/cmake/Utils.cmake)
