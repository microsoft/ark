# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# clang-format
find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
    message(STATUS "Found clang-format: ${CLANG_FORMAT}")
    set(FIND_DIRS
        ${PROJECT_SOURCE_DIR}/ark
        ${PROJECT_SOURCE_DIR}/python
        ${PROJECT_SOURCE_DIR}/examples
    )
    add_custom_target(cpplint
        COMMAND ${CLANG_FORMAT} -style=file --dry-run `find ${FIND_DIRS} -type f -name *.h -o -name *.hpp -o -name *.c -o -name *.cc -o -name *.cpp -o -name *.cu`
    )
    add_custom_target(cpplint-autofix
        COMMAND ${CLANG_FORMAT} -style=file -i `find ${FIND_DIRS} -type f -name *.h -o -name *.hpp -o -name *.c -o -name *.cc -o -name *.cpp -o -name *.cu`
    )
else()
    message(STATUS "clang-format not found.")
endif()

# black
find_program(BLACK black)
if(BLACK)
    add_custom_target(pylint
        COMMAND python3.8 -m black --check --config ${PROJECT_SOURCE_DIR}/pyproject.toml ${PROJECT_SOURCE_DIR}
    )
    add_custom_target(pylint-autofix
        COMMAND python3.8 -m black --config ${PROJECT_SOURCE_DIR}/pyproject.toml ${PROJECT_SOURCE_DIR}
    )
else()
    message(STATUS "black not found.")
endif()

# Insert gpumem module
add_custom_target(gpumem
    COMMENT "Inserting gpumem module..."
    COMMAND insmod ${PROJECT_SOURCE_DIR}/third_party/gpudma/module/gpumem.ko
    COMMAND chmod 666 /dev/gpumem
)
add_dependencies(gpumem tp-gpudma)

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
