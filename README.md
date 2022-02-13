# Image-Retargeting
Patch Based Image Warping for Content Aware Retargeting

A C++ implementation of [Patch Based Image Warping for Content Aware Retargeting (2013)](http://graphics.csie.ncku.edu.tw/Tony/papers/IEEE_Multimedia_resizing_2013_Feb.pdf), including the segmentation algorithm ([Efficient Graph-Based Image Segmentation(2004)](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf)) and saliency map detection algorithm ([Context-aware saliency detection(2010)](https://ieeexplore.ieee.org/document/6112774)) described in paper. 

The goal is to keep it less third-party library dependency and clean. 
# Requirements

- CMake
- GTest (optional ! only needed when you wish to run test suite)
- Eigen3
- Compliant C++17 compiler
  - The library is sytematically tested on following compilers 

Compiler | Version
---------|--------
GCC      | 9.3.0
clang    | 13.0.0

For image processing we only requires ```Eigen3``` to be installed on your system.
```bash
git clone -b '3.4' --single-branch --depth 1 https://gitlab.com/libeigen/eigen.git
cd eigen
mkdir build  
cd build 
cmake .. 
make install
```
Make sure Eigen3 can be found by your build system.

## Run test suite

You need to additionally install GTest to run test-suite.
```bash
apt-get install libgtest-dev -y
```
Make sure GTest can be found by your build system.

Use the following commands from the project's top-most directory to run the test suite.
```bash
cd Image-Retargeting
make test
make run-test
```