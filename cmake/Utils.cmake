# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# git-clang-format
find_program(GIT_CLANG_FORMAT git-clang-format)
if(GIT_CLANG_FORMAT)
    message(STATUS "Found git-clang-format: ${GIT_CLANG_FORMAT}")
    set(FIND_DIRS
        ${PROJECT_SOURCE_DIR}/ark
        ${PROJECT_SOURCE_DIR}/python
        ${PROJECT_SOURCE_DIR}/examples
    )
    add_custom_target(cpplint
        COMMAND ${GIT_CLANG_FORMAT} --style=file --diff || true
    )
    add_custom_target(cpplint-autofix
        COMMAND ${GIT_CLANG_FORMAT} --style=file || true
    )
else()
    message(STATUS "git-clang-format not found.")
endif()

# black
find_program(BLACK black)
if(BLACK)
    add_custom_target(pylint
        COMMAND python3 -m black --check --config ${PROJECT_SOURCE_DIR}/pyproject.toml ${PROJECT_SOURCE_DIR}
    )
    add_custom_target(pylint-autofix
        COMMAND python3 -m black --config ${PROJECT_SOURCE_DIR}/pyproject.toml ${PROJECT_SOURCE_DIR}
    )
else()
    message(STATUS "black not found.")
endif()

# lcov
find_program(LCOV lcov)
if(LCOV)
    message(STATUS "Found lcov: ${LCOV}")
    add_custom_target(lcov
        COMMAND ${LCOV} --directory . --capture --output-file coverage.info
        COMMAND ${LCOV} --remove coverage.info
            '/usr/*'
            '/tmp/*'
            '*/third_party/*'
            '*/ark/*_test.*'
            '*/examples/*'
            '*/python/*'
            '*/ark/unittest/unittest_utils.cc'
            --output-file coverage.info
        COMMAND ${LCOV} --list coverage.info
    )
else()
    message(STATUS "lcov not found.")
endif()
