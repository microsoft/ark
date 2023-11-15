# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cc)
file(GLOB_RECURSE UT_SOURCES CONFIGURE_DEPENDS *_test.cc *_test.cu)
file(GLOB_RECURSE UT_COMMON_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/unittest/*.cc)
list(REMOVE_ITEM SOURCES ${UT_SOURCES} ${UT_COMMON_SOURCES})

if(USE_ROCM)
    file(GLOB_RECURSE CU_SOURCES CONFIGURE_DEPENDS *.cu)
    set_source_files_properties(${CU_SOURCES} PROPERTIES LANGUAGE CXX)
endif()

set(COMMON_LIBS ARK::numa ARK::ibverbs pthread rt)
if (USE_MSLL)
    set(COMMON_LIBS ${COMMON_LIBS} ARK::msll)
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

if(USE_CUDA)
    list(APPEND COMMON_LIBS CUDA::cuda_driver)
    target_include_directories(ark_obj SYSTEM PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_compile_definitions(ark_obj PUBLIC ARK_CUDA)
endif()

if(USE_ROCM)
    list(APPEND COMMON_LIBS hip::host)
    target_include_directories(ark_obj SYSTEM PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
        /opt/rocm/include
    )
    target_compile_definitions(ark_obj PUBLIC ARK_ROCM)
endif()

if (USE_MSLL)
    target_include_directories(ark_obj SYSTEM PRIVATE
        ${MSLL_INCLUDE_DIRS}
    )
    target_compile_definitions(ark_obj PRIVATE ARK_USE_MSLL=1)
    target_compile_definitions(ark_obj PRIVATE ARK_MSLL_INCLUDE_DIR=${MSLL_INCLUDE_DIRS})
endif()

target_sources(ark_obj PRIVATE ${SOURCES})
target_link_libraries(ark_obj PRIVATE ${COMMON_LIBS})

# ARK unit tests
foreach(ut_source IN ITEMS ${UT_SOURCES})
    get_filename_component(exe_name ${ut_source} NAME_WE)
    add_executable(${exe_name} ${ut_source} ${UT_COMMON_SOURCES})
    add_dependencies(${exe_name} build)
    set_target_properties(${exe_name} PROPERTIES EXCLUDE_FROM_ALL TRUE)
    target_include_directories(${exe_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_include_directories(${exe_name} SYSTEM PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party/json
        ${IBVERBS_INCLUDE_DIRS}
        ${NUMA_INCLUDE_DIRS}
    )

    if(USE_CUDA)
        target_link_libraries(${exe_name} PRIVATE ark_obj ${COMMON_LIBS} CUDA::cudart CUDA::cublas)
        target_include_directories(${exe_name} SYSTEM PRIVATE
            ${CUDAToolkit_INCLUDE_DIRS}
        )
    else()
        find_package(rocblas REQUIRED)
        target_link_libraries(${exe_name} PRIVATE ark_obj ${COMMON_LIBS} roc::rocblas)
        target_include_directories(${exe_name} SYSTEM PRIVATE
            /opt/rocm/include
        )
    endif()

    add_test(NAME ${exe_name}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND ${exe_name}
    )
    set_tests_properties(${exe_name} PROPERTIES TIMEOUT 3600)
    add_dependencies(ut ${exe_name})
endforeach()
