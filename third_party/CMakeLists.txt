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

# Install CUTLASS header files
add_library(tp-cutlass INTERFACE)
add_dependencies(tp-cutlass tp-cutlass-patch)
file(GLOB_RECURSE CUTLASS_HEADERS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include/*.h)
target_sources(tp-cutlass PUBLIC FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
    FILES ${CUTLASS_HEADERS}
)
install(TARGETS tp-cutlass FILE_SET HEADERS DESTINATION include/kernels)

# Build GPUDMA
add_custom_target(tp-gpudma
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && make gpudma
)
add_dependencies(tp-gpudma tp-update)

# Clean up
add_custom_target(tp-clean
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && make clean
)
