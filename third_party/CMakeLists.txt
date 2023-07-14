# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set(CUTLASS_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include)
set(GPUDMA_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/gpudma/module)
set(JSON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/json)

add_custom_target(tp-update
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && git submodule update --init --recursive)

add_custom_target(tp-cutlass-patch
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && make cutlass)
add_dependencies(tp-cutlass-patch tp-update)

add_custom_target(tp-gpudma
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && make gpudma)
add_dependencies(tp-gpudma tp-update)

add_custom_target(tp-clean
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && make clean)
