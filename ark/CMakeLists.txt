# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cpp)
file(GLOB_RECURSE UT_SOURCES CONFIGURE_DEPENDS *_test.cpp)
file(GLOB_RECURSE UT_COMMON_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/unittest/*.cpp)
list(REMOVE_ITEM SOURCES ${UT_SOURCES} ${UT_COMMON_SOURCES})

if(ARK_USE_ROCM)
    file(GLOB_RECURSE CU_SOURCES CONFIGURE_DEPENDS *.cu)
    set_source_files_properties(${CU_SOURCES} PROPERTIES LANGUAGE CXX)
endif()

set(COMMON_LIBS ARK::numa pthread rt)

# ARK object
target_include_directories(ark_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(ark_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ark_obj SYSTEM PRIVATE
    ${DLPACK_INCLUDE_DIRS}
    ${JSON_INCLUDE_DIRS}
    ${MSCCLPP_INCLUDE_DIRS}
    ${NUMA_INCLUDE_DIRS}
)

if(ARK_USE_CUDA)
    list(APPEND COMMON_LIBS CUDA::cuda_driver)
    target_include_directories(ark_obj SYSTEM PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_compile_definitions(ark_obj PUBLIC ARK_CUDA)
endif()

if(ARK_USE_ROCM)
    list(APPEND COMMON_LIBS hip::host)
    target_include_directories(ark_obj SYSTEM PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
        /opt/rocm/include
    )
    target_compile_definitions(ark_obj PUBLIC ARK_ROCM)
endif()

target_sources(ark_obj PRIVATE ${SOURCES})
target_link_libraries(ark_obj PUBLIC mscclpp_static PRIVATE ${COMMON_LIBS})

# ARK unit tests
if(ARK_BUILD_TESTS)
    foreach(ut_source IN ITEMS ${UT_SOURCES})
        get_filename_component(exe_name ${ut_source} NAME_WE)
        add_executable(${exe_name} ${ut_source} ${UT_COMMON_SOURCES})
        add_dependencies(${exe_name} build)
        set_target_properties(${exe_name} PROPERTIES EXCLUDE_FROM_ALL TRUE)
        target_include_directories(${exe_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_include_directories(${exe_name} SYSTEM PRIVATE
            ${JSON_INCLUDE_DIRS}
            ${NUMA_INCLUDE_DIRS}
        )

        if(ARK_USE_CUDA)
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

    # Tests that call op_test() and require ARK_UT_CORRECTNESS=1 for real validation.
    # Keep in sync: grep -l 'op_test(' ark/ops/*_test.cpp
    set(CORRECTNESS_TESTS
        ops_all_reduce_test
        ops_arithmetic_test
        ops_cast_test
        ops_copy_test
        ops_embedding_test
        ops_layernorm_test
        ops_math_test
        ops_matmul_test
        ops_reduce_test
        ops_rope_test
        ops_scalar_test
        ops_softmax_test
        ops_transpose_test
    )
    set_tests_properties(${CORRECTNESS_TESTS} PROPERTIES LABELS "correctness")
endif()
