# Image-Retargeting
Patch Based Image Warping for Content Aware Retargeting

A C++ implementation of [Patch Based Image Warping for Content Aware Retargeting (2013)](http://graphics.csie.ncku.edu.tw/Tony/papers/IEEE_Multimedia_resizing_2013_Feb.pdf), including the segmentation algorithm ([Efficient Graph-Based Image Segmentation(2004)](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf)) and saliency map detection algorithm ([Context-aware saliency detection(2010)](https://ieeexplore.ieee.org/document/6112774)) described in paper. 

The goal is to keep it less third-party library dependency and clean. 

# Requirements
Only requires ```Eigen3``` to be installed on your system.
In Ubuntu based systems you can simply install these dependencies using apt-get.

```bash
apt-get install libeigen3-dev
```
Make sure Eigen3 can be found by your build system.

## Usage
[todo]