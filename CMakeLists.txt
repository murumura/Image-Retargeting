# LibWire top-level CMakeLists.txt

cmake_minimum_required (VERSION 3.16)

project (RESIZING LANGUAGES CXX C)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS ON)
set (CMAKE_INCLUDE_CURRENT_DIR ON)

#############################################
# Include files
#############################################

find_package (Eigen3 REQUIRED NO_MODULE)
if(EIGEN3_FOUND)
    message(STATUS "--    success. find_package() succeeded with Eigen3 at ${EIGEN3_INCLUDE_DIR}")
    include_directories(${EIGEN3_INCLUDE_DIR})
endif(EIGEN3_FOUND)

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/image/
    ${CMAKE_SOURCE_DIR}/src/numerical
    ${CMAKE_SOURCE_DIR}/src/stb
)

###########################################################################
# Compile options
###########################################################################
list(APPEND RESIZING_CXX_FLAGS "-std=c++17")
# Turn of eigen tensor warning message (https://stackoverflow.com/questions/54526502/compiler-warnings-with-eigen-tensor-example)
list(APPEND RESIZING_CXX_FLAGS "-isystem /usr/include/eigen3/unsupported/Eigen/CXX11/src/Tensor")
list(APPEND RESIZING_CXX_FLAGS "-w")
###########################################################################
# Sources/Headers
###########################################################################
set (RESIZING_HEADER
    # image headers
    ${CMAKE_SOURCE_DIR}/src/image/image.h
    ${CMAKE_SOURCE_DIR}/src/image/imageIO.h
    ${CMAKE_SOURCE_DIR}/src/image/padding_op.h
    ${CMAKE_SOURCE_DIR}/src/image/filter.h
    ${CMAKE_SOURCE_DIR}/src/image/colorspace_op.h
    ${CMAKE_SOURCE_DIR}/src/image/utils.h
    ${CMAKE_SOURCE_DIR}/src/image/segmentation.h
    ${CMAKE_SOURCE_DIR}/src/image/wrapping.h
    ${CMAKE_SOURCE_DIR}/src/image/saliency.h
    # numerical headers
    ${CMAKE_SOURCE_DIR}/src/numerical/types.h
)

set (RESIZING_SOURCE
    ${CMAKE_SOURCE_DIR}/src/resize.cpp
)

add_executable(patch_based_resizing
    ${RESIZING_HEADER}
    ${RESIZING_SOURCE}
)

target_include_directories(patch_based_resizing
PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/image/
    ${CMAKE_SOURCE_DIR}/src/numerical
    ${CMAKE_SOURCE_DIR}/src/stb
    ${EIGEN3_INCLUDE_DIR}    
)

target_compile_options(patch_based_resizing  PUBLIC  ${RESIZING_CXX_FLAGS})

add_subdirectory(test)
