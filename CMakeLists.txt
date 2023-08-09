# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set(ARK_MAJOR "0")
set(ARK_MINOR "1")
set(ARK_PATCH "0")

set(ARK_VERSION "${ARK_MAJOR}.${ARK_MINOR}.${ARK_PATCH}")
set(ARK_SOVERSION "${ARK_MAJOR}.${ARK_MINOR}")

option(USE_KAHYPAR "Use KaHyPar for scheduling" OFF)
option(USE_MSCCLPP "Use MSCCL++" ON)

cmake_minimum_required(VERSION 3.25)
project(ark LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Find ibverbs
include(${PROJECT_SOURCE_DIR}/cmake/FindIBVerbs.cmake)

# Find NUMA
include(${PROJECT_SOURCE_DIR}/cmake/FindNUMA.cmake)

# Find MSCCL++
if(USE_MSCCLPP)
    include(${PROJECT_SOURCE_DIR}/cmake/FindMSCCLPP.cmake)
endif()

# Find CUDAToolkit
find_package(CUDAToolkit REQUIRED)

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
add_dependencies(ark_obj tp-cutlass-patch)

# Build
add_custom_target(build)
add_dependencies(build ark_obj tp-cutlass)
add_custom_command(TARGET build POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/ark/include ${BUILD_DIR}/include
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
