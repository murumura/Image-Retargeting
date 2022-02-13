#ifndef COMPUTE_SALIENCY_H
#define COMPUTE_SALIENCY_H
#include <functional>
#include <image/image.h>
namespace Image {

    float calcSaliencyValueCPU(
        const Eigen::Tensor<float, 3, Eigen::RowMajor>& imgSrcLAB,
        int calcR,
        int calcC,
        int distC,
        int K);

    float calcSaliencyValueCUDA(
        const Eigen::Tensor<float, 3, Eigen::RowMajor>& imgSrcLAB,
        int calcR,
        int calcC,
        int distC,
        int K);

} // namespace Image
#endif