# RESIZING test-subdirectory
cmake_minimum_required(VERSION 3.16)

set(RESIZING_TEST_SOURCE
    ${CMAKE_SOURCE_DIR}/test/main.cpp
    ${CMAKE_SOURCE_DIR}/test/test_image.cpp
)

add_executable(resizing_test
    ${RESIZING_TEST_SOURCE}
)
target_include_directories(resizing_test
    PUBLIC
        ${EIGEN3_INCLUDE_DIR}  
)

target_compile_options(resizing_test PUBLIC ${RESIZING_CXX_FLAGS})

target_link_libraries(resizing_test
    gtest
    gtest_main
    gmock
    pthread
)

