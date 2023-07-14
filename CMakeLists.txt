# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set(ARK_MAJOR "0")
set(ARK_MINOR "1")
set(ARK_PATCH "0")

set(ARK_VERSION "${ARK_MAJOR}.${ARK_MINOR}.${ARK_PATCH}")
set(ARK_SOVERSION "${ARK_MAJOR}.${ARK_MINOR}")

option(USE_KAHYPAR "Use KaHyPar for scheduling" OFF)

cmake_minimum_required(VERSION 3.25)
project(ark LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

include(${PROJECT_SOURCE_DIR}/cmake/FindIBVerbs.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindNUMA.cmake)

# Third party libraries
add_subdirectory(third_party)

# Find CUDAToolkit
find_package(CUDAToolkit REQUIRED)

# ARK object & unit tests
add_library(ark_obj OBJECT)
add_dependencies(ark_obj tp-cutlass-patch)
add_subdirectory(ark)

# ARK shared library
add_library(ark SHARED)
target_link_libraries(ark PUBLIC ark_obj)
set_target_properties(ark PROPERTIES
    VERSION ${ARK_VERSION}
    SOVERSION ${ARK_SOVERSION}
)

# ARK static library
add_library(ark_static STATIC)
target_link_libraries(ark_static PUBLIC ark_obj)
set_target_properties(ark_static PROPERTIES
    VERSION ${ARK_VERSION}
    SOVERSION ${ARK_SOVERSION}
)

# Install
install(TARGETS ark ark_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
# if(NOT SKBUILD)
#     install(TARGETS ark FILE_SET HEADERS)
# endif()

