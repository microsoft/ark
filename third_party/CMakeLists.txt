# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set(MSCCLPP_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/mscclpp/include)
set(CUTLASS_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include)
set(CK_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel/include)
set(GPUDMA_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/gpudma/module)
set(JSON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/json)

include(FetchContent)

# MSCCL++
FetchContent_Declare(
    mscclpp
    GIT_REPOSITORY https://github.com/microsoft/mscclpp
    GIT_TAG 5a9998b
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mscclpp
)
FetchContent_MakeAvailable(mscclpp)

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
    FetchContent_GetProperties(cutlass)
    if (NOT cutlass_POPULATED)
        FetchContent_Populate(cutlass)
    endif()

    # Patch CUTLASS
    add_custom_target(tp-cutlass
        COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/cutlass && git reset --hard && patch -sN -p1 --reject-file=/dev/null < ${CMAKE_CURRENT_SOURCE_DIR}/patches/cutlass/cutlass.patch || true
    )

    set(TP_TARGETS tp-cutlass PARENT_SCOPE)
    set(TP_KERNELS ${MSCCLPP_INCLUDE_DIRS} ${CUTLASS_INCLUDE_DIRS} PARENT_SCOPE)
endif()


if(USE_ROCM)
    # Configure CK
    FetchContent_Declare(
        ck
        GIT_REPOSITORY https://github.com/ROCmSoftwarePlatform/composable_kernel.git
        GIT_TAG rocm-5.7.1
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel
    )
    FetchContent_GetProperties(ck)
    if (NOT ck_POPULATED)
        FetchContent_Populate(ck)
    endif()

    # Patch CK
    add_custom_target(tp-ck
        COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel && git reset --hard && patch -sN -p1 --reject-file=/dev/null < ${CMAKE_CURRENT_SOURCE_DIR}/patches/composable_kernel/ck.patch || true
    )

    set(TP_TARGETS tp-ck PARENT_SCOPE)
    set(TP_KERNELS ${MSCCLPP_INCLUDE_DIRS} ${CK_INCLUDE_DIRS} PARENT_SCOPE)
endif()

# Clean up
add_custom_target(tp-clean
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && make clean
)
