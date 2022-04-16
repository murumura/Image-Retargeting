#ifndef SALIENCY_UTILS_H
#define SALIENCY_UTILS_H
#include <image/padding_op.h>
namespace Image {

    void getWindowedOutputSize(int64_t input_size, int64_t filter_size,
        int64_t dilation_rate, int64_t stride, PadMode padding_type,
        int64_t& output_size, int64_t& padding_before, int64_t& padding_after);

    std::tuple<Eigen::Tensor<float, 3, Eigen::RowMajor>, Eigen::Tensor<int, 3, Eigen::RowMajor>>
    extractImagePatches(
        const Eigen::Tensor<float, 3, Eigen::RowMajor>& input,
        const int patch_rows, const int patch_cols, const int stride_rows,
        const int stride_cols, const int rate_rows, const int rate_cols,
        const std::string& padding_type);

    Eigen::Tensor<float, 3, Eigen::RowMajor>
    extractPatchesByIndices(
        const Eigen::Tensor<float, 3, Eigen::RowMajor>& input,
        const Eigen::Tensor<int, 3, Eigen::RowMajor>& indices,
        const int patch_rows, const int patch_cols);

} // namespace Image
#endif
