# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set(MSCCLPP_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/mscclpp/include)
set(CUTLASS_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include)
set(CK_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel/include)

include(FetchContent)

# MSCCL++
FetchContent_Declare(
    mscclpp
    GIT_REPOSITORY https://github.com/microsoft/mscclpp
    GIT_TAG v0.5.0
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mscclpp
)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(INSTALL_PREFIX "ark")
FetchContent_GetProperties(mscclpp)
if (NOT mscclpp_POPULATED)
    FetchContent_Populate(mscclpp)
    add_subdirectory(${mscclpp_SOURCE_DIR} ${mscclpp_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# JSON
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG v3.11.3
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/json
)
FetchContent_GetProperties(json)
if (NOT json_POPULATED)
    FetchContent_Populate(json)
endif()
set(JSON_INCLUDE_DIRS ${json_SOURCE_DIR}/include PARENT_SCOPE)

if(USE_CUDA)
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
        COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/cutlass && git reset --hard && patch -sN -p1 --reject-file=/dev/null < ${CMAKE_CURRENT_SOURCE_DIR}/patches/cutlass.patch || true
    )

    set(TP_TARGETS tp-cutlass PARENT_SCOPE)
    set(TP_KERNELS ${MSCCLPP_INCLUDE_DIRS} ${CUTLASS_INCLUDE_DIRS} PARENT_SCOPE)
endif()


if(USE_ROCM)
    # Configure CK
    FetchContent_Declare(
        ck
        GIT_REPOSITORY https://github.com/ROCmSoftwarePlatform/composable_kernel.git
        GIT_TAG rocm-6.0.0
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel
    )
    FetchContent_GetProperties(ck)
    if (NOT ck_POPULATED)
        FetchContent_Populate(ck)
    endif()

    # Patch CK
    add_custom_target(tp-ck
        COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel && git reset --hard && patch -sN -p1 --reject-file=/dev/null < ${CMAKE_CURRENT_SOURCE_DIR}/patches/composable_kernel.patch || true
    )

    set(TP_TARGETS tp-ck PARENT_SCOPE)
    set(TP_KERNELS ${MSCCLPP_INCLUDE_DIRS} ${CK_INCLUDE_DIRS} PARENT_SCOPE)
endif()

# TODO: clean up
add_custom_target(tp-clean)
