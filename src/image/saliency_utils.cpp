#include <image/saliency_utils.h>
namespace Image {
    void getWindowedOutputSize(int64_t input_size, int64_t filter_size,
        int64_t dilation_rate, int64_t stride, PadMode padding_type,
        int64_t& output_size, int64_t& padding_before, int64_t& padding_after)
    {
        if (stride <= 0) {
            throw std::invalid_argument("Stride must be > 0, but got " + std::to_string(stride));
        }
        if (dilation_rate < 1) {
            throw std::invalid_argument("Dilation rate must be >= 1, but got " + std::to_string(dilation_rate));
        }

        int64_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;

        if (padding_type == PadMode::VALID) {
            output_size = (input_size - effective_filter_size + stride) / stride;
            padding_before = padding_after = 0;
        }
        else {
            output_size = (input_size + stride - 1) / stride;
            const int64_t padding_needed = std::max(int64_t{0}, (output_size - 1) * stride + effective_filter_size - input_size);
            // For odd values of total padding, add more padding at the 'left'
            // side of the given dimension.
            padding_after = padding_needed / 2;
            padding_before = padding_needed - padding_after;
        }

        // clang-format off
        if (output_size < 0) {
            throw std::invalid_argument(
                "Computed output size would be negative: " + std::to_string(output_size) + \
                " [input_size: " + std::to_string(input_size) +
                ", effective_filter_size: "+ std::to_string(effective_filter_size) +
                ", stride: "+ std::to_string(stride) + "]");
        }
        // clang-format on
    }

    std::tuple<Eigen::Tensor<float, 3, Eigen::RowMajor>, Eigen::Tensor<int, 3, Eigen::RowMajor>>
    extractImagePatches(
        const Eigen::Tensor<float, 3, Eigen::RowMajor>& input,
        const int patch_rows, const int patch_cols, const int stride_rows,
        const int stride_cols, const int rate_rows, const int rate_cols,
        const std::string& padding_type)
    {
        const int in_rows = input.dimension(0);
        const int in_cols = input.dimension(1);
        const int depth = input.dimension(2);

        const int ksize_rows_eff = patch_rows + (patch_rows - 1) * (rate_rows - 1);
        const int ksize_cols_eff = patch_cols + (patch_cols - 1) * (rate_cols - 1);

        int64_t out_rows = 0, out_cols = 0;
        int64_t pad_right = 0, pad_left = 0, pad_top = 0, pad_bottom = 0;

        PadMode padding_mode = stringToPadMode(padding_type);

        getWindowedOutputSize(in_rows, ksize_rows_eff, 1 /*dilation_rate*/, stride_rows, padding_mode, out_rows, pad_top, pad_bottom);
        getWindowedOutputSize(in_cols, ksize_cols_eff, 1 /*dilation_rate*/, stride_cols, padding_mode, out_cols, pad_left, pad_right);

        const std::array<Eigen::Index, 4> patches_shape = {out_rows, out_cols, patch_rows * patch_cols, depth};

        PaddingImageOp padding_op = PaddingImageOp<float>(padding_type);
        Eigen::Tensor<float, 3, Eigen::RowMajor> padded_input;

        // Padding input image for patches extraction
        padding_op(input, padded_input, pad_top, pad_bottom, pad_left, pad_right, static_cast<float>(0.0) /*padding value for constant padding*/);
        Eigen::array<int, 1> reduction_axis = {2};
        Eigen::Tensor<float, 3, Eigen::RowMajor> patches = padded_input
                                                               .extract_image_patches(patch_cols, patch_rows, stride_cols,
                                                                   stride_rows, rate_cols, rate_rows,
                                                                   Eigen::PaddingType::PADDING_VALID)
                                                               .reshape(patches_shape)
                                                               .mean(reduction_axis);

        Eigen::Tensor<int, 3, Eigen::RowMajor> indices(out_rows, out_cols, 2);

        for (int row = 0; row < out_rows; row++)
            for (int col = 0; col < out_cols; col++) {
                const int r_coord = row * stride_rows;
                const int c_coord = col * stride_cols;
                indices(row, col, 0) = std::min(r_coord, in_rows - 1);
                indices(row, col, 1) = std::min(c_coord, in_cols - 1);
            }
        return {patches, indices};
    }

    Eigen::Tensor<float, 3, Eigen::RowMajor>
    extractPatchesByIndices(
        const Eigen::Tensor<float, 3, Eigen::RowMajor>& input,
        const Eigen::Tensor<int, 3, Eigen::RowMajor>& indices,
        const int patch_rows, const int patch_cols)
    {
        const int in_rows = input.dimension(0);
        const int in_cols = input.dimension(1);
        const int depth = input.dimension(2);
        const int out_rows = indices.dimension(0);
        const int out_cols = indices.dimension(1);

        Eigen::Tensor<float, 3, Eigen::RowMajor> patches(out_rows, out_cols, depth);
        patches.setZero();
        for (int p_row = 0; p_row < out_rows; p_row++)
            for (int p_col = 0; p_col < out_cols; p_col++) {
                const int r_coord = indices(p_row, p_col, 0);
                const int c_coord = indices(p_row, p_col, 1);
                int n = 0;
                const int r_right_offset = patch_rows / 2;
                const int r_left_offset = patch_rows - r_right_offset;
                const int c_right_offset = patch_cols / 2;
                const int c_left_offset = patch_cols - c_right_offset;
                const int r_start = r_coord + r_left_offset;
                const int c_start = c_coord + c_left_offset;
                for (int r = r_start - r_left_offset; r < (r_start + r_right_offset); ++r) {
                    if (r < 0 || r >= in_rows)
                        continue;
                    for (int c = c_start - c_left_offset; c < (c_start + c_right_offset); ++c) {
                        if (c < 0 || c >= in_cols)
                            continue;

                        for (int d = 0; d < depth; ++d) {
                            patches(p_row, p_col, d) += input(r, c, d);
                        }
                        ++n;
                    }
                }
                for (int d = 0; d < depth; ++d) {
                    patches(p_row, p_col, d) /= n;
                }
            }
        return patches;
    }

} // namespace Image
