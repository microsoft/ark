# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cc)
file(GLOB_RECURSE UT_SOURCES CONFIGURE_DEPENDS *_test.cc)
file(GLOB_RECURSE UT_COMMON_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/unittest/*.cc)
list(REMOVE_ITEM SOURCES ${UT_SOURCES} ${UT_COMMON_SOURCES})
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS *.h)
file(GLOB_RECURSE UT_HEADERS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/unittest/*.h)
file(GLOB_RECURSE PUBLIC_HEADERS CONFIGURE_DEPENDS include/ark*.h)
file(GLOB_RECURSE KERNEL_HEADERS CONFIGURE_DEPENDS include/kernels/*.h)
list(REMOVE_ITEM HEADERS ${UT_HEADERS} ${PUBLIC_HEADERS} ${KERNEL_HEADERS})
if (NOT USE_KAHYPAR)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/sched/sched/sched_kahypar.cc)
    list(REMOVE_ITEM UT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/kahypar_test.cc)
    list(REMOVE_ITEM HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/kahypar.h)
endif()

set(COMMON_LIBS CUDA::cuda_driver CUDA::nvml ARK::numa ARK::ibverbs pthread rt)

# ARK object
set_target_properties(ark_obj PROPERTIES
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE 1
    VERSION ${ARK_VERSION}
    SOVERSION ${ARK_SOVERSION})
target_sources(ark_obj PRIVATE ${SOURCES})
target_include_directories(ark_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(ark_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ark_obj SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
    ${PROJECT_SOURCE_DIR}/third_party/gpudma/module
    ${PROJECT_SOURCE_DIR}/third_party/json
    ${CUDAToolkit_INCLUDE_DIRS}
    ${IBVERBS_INCLUDE_DIRS}
    ${NUMA_INCLUDE_DIRS})
target_link_libraries(ark_obj PRIVATE ${COMMON_LIBS})

# ARK unit tests
foreach(ut_source IN ITEMS ${UT_SOURCES})
    get_filename_component(exe_name ${ut_source} NAME)
    add_executable(${exe_name} ${ut_source} ${UT_COMMON_SOURCES})
    target_link_libraries(${exe_name} PRIVATE ark_obj ${COMMON_LIBS})
    target_include_directories(${exe_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_include_directories(${exe_name} SYSTEM PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party/json
        ${CUDAToolkit_INCLUDE_DIRS}
        ${IBVERBS_INCLUDE_DIRS}
        ${NUMA_INCLUDE_DIRS})
    add_test(NAME ${exe_name} COMMAND ARK_ROOT=${CMAKE_BINARY_DIR} ${exe_name})
endforeach()
