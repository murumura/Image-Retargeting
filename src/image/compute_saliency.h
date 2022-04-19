#ifndef COMPUTE_SALIENCY_H
#define COMPUTE_SALIENCY_H
#include <image/image.h>
namespace Image {

    float calcSaliencyValueCpu(
        const Eigen::Tensor<float, 3, Eigen::RowMajor>& singleScalePatch,
        const Eigen::Tensor<float, 4, Eigen::RowMajor>& multiScalePatch,
        const Eigen::Tensor<int, 3, Eigen::RowMajor>& indices,
        const int H, const int W,
        const int calcR,
        const int calcC,
        const int distC,
        const int K);

    void calcSaliencyValueCuda(
        const Eigen::Tensor<float, 3, Eigen::RowMajor>& singleScalePatch,
        const Eigen::Tensor<float, 4, Eigen::RowMajor>& multiScalePatches,
        const Eigen::Tensor<int, 3, Eigen::RowMajor>& indices,
        Eigen::Tensor<float, 3, Eigen::RowMajor>& salienceMap,
        const int distC,
        const int K, const int H, const int W);

} // namespace Image
#endif
