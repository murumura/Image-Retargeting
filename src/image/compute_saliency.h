#ifndef COMPUTE_SALIENCY_H
#define COMPUTE_SALIENCY_H
#include <image/image.h>
namespace Image {

    float calcSaliencyValueCpu(
        const Eigen::Tensor<float, 3, Eigen::RowMajor>& singleScalePatch,
        const Eigen::Tensor<float, 3, Eigen::RowMajor>& multiScalePatch,
        int calcR,
        int calcC,
        int distC,
        int K);

    void calcSaliencyValueCuda(
        const Eigen::Tensor<float, 3, Eigen::RowMajor>& singleScalePatch,
        const Eigen::Tensor<float, 3, Eigen::RowMajor>& multiScalePatch,
        Eigen::Tensor<float, 3, Eigen::RowMajor>& salienceMap,
        int distC,
        int K);

} // namespace Image
#endif
