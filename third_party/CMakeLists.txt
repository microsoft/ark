# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set(CUTLASS_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include PARENT_SCOPE)
set(CK_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel/include PARENT_SCOPE)
set(GPUDMA_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/gpudma/module PARENT_SCOPE)
set(JSON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/json PARENT_SCOPE)

include(FetchContent)

if(USE_CUDA)
    # Configure GPUDMA
    FetchContent_Declare(
        gpudma
        GIT_REPOSITORY https://github.com/karakozov/gpudma
        GIT_TAG 7ecec64
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gpudma
    )
    FetchContent_MakeAvailable(gpudma)

    # Build GPUDMA
    add_custom_target(tp-gpudma
        COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && make gpudma
    )

    # Insert gpumem module
    add_custom_target(gpumem
        COMMENT "Inserting gpumem module..."
        COMMAND insmod ${PROJECT_SOURCE_DIR}/third_party/gpudma/module/gpumem.ko
        COMMAND chmod 666 /dev/gpumem
    )
    add_dependencies(gpumem tp-gpudma)

    # Configure CUTLASS
    FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass
        GIT_TAG v3.2.1
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cutlass
    )
    FetchContent_MakeAvailable(cutlass)

    # Patch CUTLASS
    add_custom_target(tp-cutlass
        COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/cutlass && patch -sN -p1 --reject-file=/dev/null < ${CMAKE_CURRENT_SOURCE_DIR}/patches/cutlass/cutlass.patch || true
    )

    set(TP_TARGETS tp-cutlass PARENT_SCOPE)
    set(TP_KERNELS ${CUTLASS_INCLUDE_DIRS} PARENT_SCOPE)
endif()


if(USE_ROCM)
    # Configure CK
    FetchContent_Declare(
        ck
        GIT_REPOSITORY https://github.com/ROCmSoftwarePlatform/composable_kernel.git
        GIT_TAG rocm-5.7.1
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel
    )
    FetchContent_MakeAvailable(ck)

    # Patch CK
    add_custom_target(tp-ck
        COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel && patch -sN -p1 --reject-file=/dev/null < ${CMAKE_CURRENT_SOURCE_DIR}/patches/composable_kernel/ck.patch || true
    )

    set(TP_TARGETS tp-ck PARENT_SCOPE)
    set(TP_KERNELS ${CK_INCLUDE_DIRS} PARENT_SCOPE)
endif()

# Clean up
add_custom_target(tp-clean
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && make clean
)
