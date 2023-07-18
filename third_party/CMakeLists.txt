# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set(CUTLASS_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include)
set(GPUDMA_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/gpudma/module)
set(JSON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/json)

# Update submodules
add_custom_target(tp-update
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && git submodule update --init --recursive
)

# Patch CUTLASS
add_custom_target(tp-cutlass-patch
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && make cutlass
)
add_dependencies(tp-cutlass-patch tp-update)

# Copy CUTLASS header files to the build directory
add_custom_target(tp-cutlass
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include ${BUILD_DIR}/include/kernels
)

# Build GPUDMA
add_custom_target(tp-gpudma
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && make gpudma
)
add_dependencies(tp-gpudma tp-update)

# Clean up
add_custom_target(tp-clean
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && make clean
)
