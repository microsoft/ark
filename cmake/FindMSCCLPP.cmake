# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Find the MSCCL++ libraries
#
# The following variables are optionally searched for defaults
#  MSLL_ROOT_DIR: Base directory where all msll components are found
#  MSLL_INCLUDE_DIR: Directory where msll headers are found
#  MSLL_LIB_DIR: Directory where msll libraries are found

# The following are set after configuration is done:
#  MSLL_FOUND
#  MSLL_INCLUDE_DIRS
#  MSLL_LIBRARIES

# An imported target ARK::msll is created if the library is found.

find_path(MSLL_INCLUDE_DIRS
    NAMES msll/core.hpp
    HINTS
    ${MSLL_INCLUDE_DIR}
    ${MSLL_ROOT_DIR}
    ${MSLL_ROOT_DIR}/include
    /usr/local/msll/include
)

find_library(MSLL_LIBRARIES
    NAMES msll
    HINTS
    ${MSLL_LIB_DIR}
    ${MSLL_ROOT_DIR}
    ${MSLL_ROOT_DIR}/lib
    /usr/local/msll/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MSLL DEFAULT_MSG MSLL_INCLUDE_DIRS MSLL_LIBRARIES)
mark_as_advanced(MSLL_INCLUDE_DIR MSLL_LIBRARIES)

if(MSLL_FOUND)
    if(NOT TARGET ARK::msll)
        add_library(ARK::msll UNKNOWN IMPORTED)
    endif()
    set_target_properties(ARK::msll PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${MSLL_INCLUDE_DIR}"
        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
        IMPORTED_LOCATION "${MSLL_LIBRARIES}"
    )
endif()
