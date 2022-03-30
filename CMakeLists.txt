# top-level CMakeLists.txt
cmake_minimum_required (VERSION 3.16)

project (RESIZING LANGUAGES CXX C CUDA)

###############################################################################
# C++ compiler setup
###############################################################################

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS ON)
set (CMAKE_INCLUDE_CURRENT_DIR ON)

######### Options ###########################
option( USE_CUDA    "Use CUDA to speed up certain parts of the code."  OFF )
#############################################

###############################################################################
# CUDA decision && CUDA compiler setup
###############################################################################
if (USE_CUDA)
    enable_language(CUDA)
    message( STATUS ">> --------------  USING CUDA --------------" )
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS ON)
    set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
endif()


######### CUDA decisions ####################
if (USE_CUDA)
    find_package(CUDA REQUIRED)
endif()

if(CUDA_FOUND)
    list(APPEND RESIZING_DEFS "RESIZING_USE_CUDA")
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    message(STATUS "--    success. find_package() succeeded with CUDA at ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    message(STATUS "CUDA found (library: ${CUDA_LIBRARIES} include: ${CUDA_INCLUDE_DIRS})")
endif()

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
list(APPEND RESIZING_CXX_FLAGS "-w")

###########################################################################
# Sources/Headers
###########################################################################
set (RESIZING_HEADER
    # image headers/sources
    ${CMAKE_SOURCE_DIR}/src/image/image.h
    ${CMAKE_SOURCE_DIR}/src/image/imageIO.h
    ${CMAKE_SOURCE_DIR}/src/image/padding_op.h
    ${CMAKE_SOURCE_DIR}/src/image/filter.h
    ${CMAKE_SOURCE_DIR}/src/image/colorspace_op.h
    ${CMAKE_SOURCE_DIR}/src/image/utils.h
    ${CMAKE_SOURCE_DIR}/src/image/segmentation.h
    ${CMAKE_SOURCE_DIR}/src/image/wrapping.h
    ${CMAKE_SOURCE_DIR}/src/image/saliency.h
    ${CMAKE_SOURCE_DIR}/src/image/pool.h
    ${CMAKE_SOURCE_DIR}/src/image/compute_saliency.h
    ${CMAKE_SOURCE_DIR}/src/image/compute_saliency.cpp
    ${CMAKE_SOURCE_DIR}/src/image/resizing_op.h
    ${CMAKE_SOURCE_DIR}/src/image/perspective_transform_op.h
    ${CMAKE_SOURCE_DIR}/src/image/homography.h
    # geometry header/sources
    ${CMAKE_SOURCE_DIR}/src/geometry/common.h
    ${CMAKE_SOURCE_DIR}/src/geometry/quad_mesh.h
    # numerical headers/sources
    ${CMAKE_SOURCE_DIR}/src/numerical/types.h
    ${CMAKE_SOURCE_DIR}/src/numerical/cg_solver.h
)

if(CUDA_FOUND)
    list(APPEND RESIZING_HEADER  ${CMAKE_SOURCE_DIR}/src/image/compute_saliency.cu)
endif()
set (RESIZING_SOURCE
    ${CMAKE_SOURCE_DIR}/src/retargeting.cpp
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

target_link_libraries(patch_based_resizing
    pthread
    ${CUDA_LIBRARIES}
)

if(CUDA_FOUND)
    target_link_libraries(patch_based_resizing
        ${CUDA_LIBRARIES}
    )
endif()

target_compile_options(patch_based_resizing  PUBLIC  ${RESIZING_CXX_FLAGS})
target_compile_definitions (patch_based_resizing PUBLIC ${RESIZING_DEFS})
add_subdirectory(test)
