# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cc)
file(GLOB_RECURSE UT_SOURCES CONFIGURE_DEPENDS *_test.cc)
file(GLOB_RECURSE UT_COMMON_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/unittest/*.cc)
list(REMOVE_ITEM SOURCES ${UT_SOURCES} ${UT_COMMON_SOURCES})
file(GLOB_RECURSE INTERFACE_HEADERS CONFIGURE_DEPENDS include/ark*.h)
file(GLOB_RECURSE KERNEL_HEADERS CONFIGURE_DEPENDS include/kernels/*.h)
file(GLOB_RECURSE CUTLASS_HEADERS CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/third_party/cutlass/include/*.h)
if (NOT USE_KAHYPAR)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/sched/sched/sched_kahypar.cc)
    list(REMOVE_ITEM UT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/kahypar_test.cc)
endif()

set(COMMON_LIBS ARK::numa ARK::ibverbs pthread rt)
if(CUDAToolkit_FOUND)
    list(APPEND COMMON_LIBS CUDA::cuda_driver CUDA::nvml)
elseif(ROCM_FOUND)
    list(APPEND COMMON_LIBS ${ROCR_LIBRARY} ${ROCT_LIBRARY})
endif()

# ARK object
target_include_directories(ark_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(ark_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ark_obj SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/gpudma/module
    ${PROJECT_SOURCE_DIR}/third_party/json
    ${IBVERBS_INCLUDE_DIRS}
    ${NUMA_INCLUDE_DIRS}
)
if(CUDAToolkit_FOUND)
    target_include_directories(ark_obj SYSTEM PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
        ${CUDAToolkit_INCLUDE_DIRS}
    )
elseif(ROCM_FOUND)
    target_include_directories(ark_obj SYSTEM PRIVATE
        ${ROCM_INCLUDE_DIRS}
        /opt/rocm/include
    )
endif()
target_sources(ark_obj PRIVATE ${SOURCES})
target_link_libraries(ark_obj PRIVATE ${COMMON_LIBS})
if(ROCM_FOUND)
    # define ROCM macro to enable ROCM-specific code
    target_compile_definitions(ark_obj PRIVATE ARK_ROCM)
endif()

# Add header files to library targets
target_sources(ark PUBLIC
    FILE_SET install_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES ${INTERFACE_HEADERS} ${KERNEL_HEADERS}
)
target_sources(ark_static PUBLIC
    FILE_SET install_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES ${INTERFACE_HEADERS} ${KERNEL_HEADERS}
)
target_sources(ark PUBLIC
    FILE_SET install_cutlass_headers
    TYPE HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
    FILES ${CUTLASS_HEADERS}
)
target_sources(ark_static PUBLIC
    FILE_SET install_cutlass_headers
    TYPE HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
    FILES ${CUTLASS_HEADERS}
)

# ARK unit tests
foreach(ut_source IN ITEMS ${UT_SOURCES})
    get_filename_component(exe_name ${ut_source} NAME)
    add_executable(${exe_name} ${ut_source} ${UT_COMMON_SOURCES})
    add_dependencies(${exe_name} build)
    set_target_properties(${exe_name} PROPERTIES EXCLUDE_FROM_ALL TRUE)
    target_link_libraries(${exe_name} PRIVATE ark_obj ${COMMON_LIBS})
    target_include_directories(${exe_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_include_directories(${exe_name} SYSTEM PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party/json
        ${CUDAToolkit_INCLUDE_DIRS}
        ${IBVERBS_INCLUDE_DIRS}
        ${NUMA_INCLUDE_DIRS}
    )
    add_test(NAME ${exe_name}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND ${exe_name}
    )
    add_dependencies(ut ${exe_name})
endforeach()
