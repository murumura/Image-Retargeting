# Image-Retargeting
Patch Based Image Warping for Content Aware Retargeting

A C++ implementation of [Patch Based Image Warping for Content Aware Retargeting (2013)](http://graphics.csie.ncku.edu.tw/Tony/papers/IEEE_Multimedia_resizing_2013_Feb.pdf), including the segmentation algorithm ([Efficient Graph-Based Image Segmentation(2004)](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf)) and saliency map detection algorithm ([Context-aware saliency detection(2010)](https://ieeexplore.ieee.org/document/6112774)) described in paper. 

The goal is to keep it less third-party library dependency and clean. 
# How to RUN
```bash
cd Image-Retargeting
# ----- Default compile setting -----
make
# ----- Or compile with CUDA support -----
make CUDA=1
# ----- Execute with defualt parameters-----
/build/default/default/patch_based_resizing
```
(see below for detail execution arguments)
### Argument specification for execution
In order to easily adjust the relevant parameters of the program, the user can adjust the parameters according to the following format through the argument list.

All parameters have default values, once a parameter is ignored, the default value is used, see `src/retargeting.cpp` to learn more about parameter usage and its default values.

```bash
./build/default/default/patch_based_resizing \
  --InputImage ./datasets/butterfly.png \
  --Sigma 0.5 \
  --SegmentK 500.0 \
  --MinSize 100 \
  --MergePercent 0.0001 \
  --MergeColorDist 30.0 \
  --SaveSegment true \
  --DistC 3 \
  --SimilarK 64 \
  --NumScale 3 \
  --ScaleU 6 \
  --SaveSaliency true \
  --SaveScaledSaliency true \
  --newH 200 \
  --newW 300 \
  --Alpha 0.8 \
  --QuadSize 20 \
  --WeightDST 1.0 \
  --WeightDLT 1.0 \
  --WeightDOR 0.2
```

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

- CUDA toolkit (optional but highly recommend, otherwise it will take ~3hr to generate saliance map using solely multithread)

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
## Install CUDA Toolkit
Download and install the CUDA Toolkit (11.4 on my computer) for your corresponding platform. For system requirements and installation instructions of cuda toolkit, please refer to the [Linux Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/), and the [Windows Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

Make sure the environment variable `CUDA_PATH` is set to the CUDA Toolkit install directory.

Also Make sure NVCC and cuda-toolkit can be found by your build system.

*Since MacOS no longer support CUDA library since CUDA Toolkit 11.6. Therefore, mac users cannot use cuda to accelerate the calculation of saliance map. In the future, I plan to use metal as an alternative to cuda for mac users. The currently released version has only been fully tested for linux.*
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

# Result

original-grid                       |  200 x 500                                   | 300 x 200
:-------------------------:         |:-------------------------:|                :-------------------------:
![](./results/input-grid-girl.png)  |  ![](./results/result-girl-200-500.png) | ![](./results/result-girl-300-200.png)
![](./results/input-grid-butterfly.png)  |  ![](./results/result-butterfly-200-500.png) | ![](./results/result-butterfly-300-200.png)

# Acknowledgement
Thank [zyu-tien](https://github.com/zyu-tien) for helping me debugging and giving me helpful advices while developing this project.
