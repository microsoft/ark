# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Find the MSCCL++ libraries
#
# The following variables are optionally searched for defaults
#  MSCCLPP_ROOT_DIR: Base directory where all mscclpp components are found
#  MSCCLPP_INCLUDE_DIR: Directory where mscclpp headers are found
#  MSCCLPP_LIB_DIR: Directory where mscclpp libraries are found

# The following are set after configuration is done:
#  MSCCLPP_FOUND
#  MSCCLPP_INCLUDE_DIRS
#  MSCCLPP_LIBRARIES

# An imported target ARK::mscclpp is created if the library is found.

find_path(MSCCLPP_INCLUDE_DIRS
    NAMES mscclpp/core.hpp
    HINTS
    ${MSCCLPP_INCLUDE_DIR}
    ${MSCCLPP_ROOT_DIR}
    ${MSCCLPP_ROOT_DIR}/include
    /usr/local/mscclpp/include
)

find_library(MSCCLPP_LIBRARIES
    NAMES mscclpp
    HINTS
    ${MSCCLPP_LIB_DIR}
    ${MSCCLPP_ROOT_DIR}
    ${MSCCLPP_ROOT_DIR}/lib
    /usr/local/mscclpp/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MSCCLPP DEFAULT_MSG MSCCLPP_INCLUDE_DIRS MSCCLPP_LIBRARIES)
mark_as_advanced(MSCCLPP_INCLUDE_DIR MSCCLPP_LIBRARIES)

if(MSCCLPP_FOUND)
    if(NOT TARGET ARK::mscclpp)
        add_library(ARK::mscclpp UNKNOWN IMPORTED)
    endif()
    set_target_properties(ARK::mscclpp PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${MSCCLPP_INCLUDE_DIR}"
        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
        IMPORTED_LOCATION "${MSCCLPP_LIBRARIES}"
    )
endif()
