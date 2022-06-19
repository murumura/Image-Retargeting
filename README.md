# Image-Retargeting
Patch Based Image Warping for Content Aware Retargeting

A C++ implementation of [Patch Based Image Warping for Content Aware Retargeting (2013)](http://graphics.csie.ncku.edu.tw/Tony/papers/IEEE_Multimedia_resizing_2013_Feb.pdf), including the segmentation algorithm ([Efficient Graph-Based Image Segmentation(2004)](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf)) and saliency map detection algorithm ([Context-aware saliency detection(2010)](https://ieeexplore.ieee.org/document/6112774)) described in paper. 

Instead of relying on third-party image libraries such as openCV, **you only need to install this tensor library** -- [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page). I wrote the operations required to implement these papers myself,the goal is to keep it less third-party library dependency and clean. 

I wrote a cuda kernel for computing saliency to speed up the computation (from 2 min to 2 seconds), this is one of the improvements I made compared to the original implementation.

# How to RUN
```bash
cd Image-Retargeting
# ----- Default compile setting -----
make
# ----- Or compile with CUDA support -----
make CUDA=1
# ----- Execute with defualt parameters-----
./build/default/default/patch_based_resizing
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
  --MergeColorDist 20.0 \
  --SaveSegment true \
  --DistC 3 \
  --SimilarK 64 \
  --NumScale 3 \
  --PatchSize 7 \
  --SaveSaliency true \
  --SaveScaledSaliency true \
  --newH 300 \
  --newW 200 \
  --Alpha 0.8 \
  --QuadSize 10 \
  --WeightDST 3.0 \
  --WeightDLT 1.2 \
  --WeightDOR 3.0
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

- CUDA toolkit (optional but highly recommend, otherwise it will take ~2min to generate saliance map using solely multithread)

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
| original-grid                              | segmentation                                 | significance                                 | saliency                                 | 200 x 500                                      | 300 x 200                                      |
|--------------------------------------------|----------------------------------------------|----------------------------------------------|------------------------------------------|------------------------------------------------|------------------------------------------------|
| ![](./results/input-grid-girl.png)         | ![](./results/girl-segmentation.png)         | ![](./results/girl-significance.png)         | ![](./results/girl-saliency.png)         | ![](./results/result-girl-200-500.png)         | ![](./results/result-girl-300-200.png)         |
| ![](./results/input-grid-butterfly.png)    | ![](./results/butterfly-segmentation.png)    | ![](./results/butterfly-significance.png)    | ![](./results/butterfly-saliency.png)    | ![](./results/result-butterfly-200-500.png)    | ![](./results/result-butterfly-300-200.png)    |
| ![](./results/input-grid-Unazukin.png)     | ![](./results/Unazukin-segmentation.png)     | ![](./results/Unazukin-significance.png)     | ![](./results/Unazukin-saliency.png)     | ![](./results/result-Unazukin-200-500.png)     | ![](./results/result-Unazukin-300-200.png)     |
| ![](./results/input-grid-Sanfrancisco.png) | ![](./results/Sanfrancisco-segmentation.png) | ![](./results/Sanfrancisco-significance.png) | ![](./results/Sanfrancisco-saliency.png) | ![](./results/result-Sanfrancisco-200-500.png) | ![](./results/result-Sanfrancisco-300-200.png) |
| ![](./results/input-grid-painting2.png)    | ![](./results/painting2-segmentation.png)    | ![](./results/painting2-significance.png)    | ![](./results/painting2-saliency.png)    | ![](./results/result-painting2-200-500.png)    | ![](./results/result-painting2-300-200.png)    |
| ![](./results/input-grid-eagle.png)        | ![](./results/eagle-segmentation.png)        | ![](./results/eagle-significance.png)        | ![](./results/eagle-saliency.png)        | ![](./results/result-eagle-200-500.png)        | ![](./results/result-eagle-300-200.png)        |
| ![](./results/input-grid-child.png)        | ![](./results/child-segmentation.png)        | ![](./results/child-significance.png)        | ![](./results/child-saliency.png)        | ![](./results/result-child-200-500.png)        | ![](./results/result-child-300-200.png)        |
| ![](./results/input-grid-greek_wine.png)   | ![](./results/greek_wine-segmentation.png)   | ![](./results/greek_wine-significance.png)   | ![](./results/greek_wine-saliency.png)   | ![](./results/result-greek_wine-200-500.png)   | ![](./results/result-greek_wine-300-200.png)   |


# Acknowledgement
Thank [zyu-tien](https://github.com/zyu-tien) for helping me debugging and giving me helpful advices while developing this project.

All the images used in the project could be downloaded from https://people.csail.mit.edu/mrub/retargetme/download.html.