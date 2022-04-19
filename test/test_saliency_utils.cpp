#include <gtest/gtest.h>
#include <image/saliency_utils.h>
#include <iostream>
using namespace Image;
using testing::Eq;

TEST(ExtractImagePatch, Basic)
{
    int input_depth = 3;
    int input_rows = 250;
    int input_cols = 170;
    int ksize = 7;
    int stride = 7;
    Eigen::Tensor<float, 3, Eigen::RowMajor> tensor(input_rows, input_cols, input_depth);
    // Initializes tensor with incrementing numbers.
    for (int i = 0; i < tensor.size(); ++i) {
        tensor.data()[i] = i + 1;
    }
    auto [patches, indices] = extractImagePatches(tensor, ksize, ksize, ksize / 2 + 1, ksize / 2 + 1, 1, 1, "reflect");

    int64_t out_rows = 0, out_cols = 0;
    int64_t pad_right = 0, pad_left = 0, pad_top = 0, pad_bottom = 0;
    PadMode padding_mode = stringToPadMode("reflect");
    const int ksize_rows_eff = ksize + (ksize - 1) * (1 - 1);
    const int ksize_cols_eff = ksize + (ksize - 1) * (1 - 1);
    getWindowedOutputSize(input_rows, ksize_rows_eff, 1 /*dilation_rate*/, ksize / 2, padding_mode, out_rows, pad_top, pad_bottom);
    getWindowedOutputSize(input_cols, ksize_cols_eff, 1 /*dilation_rate*/, ksize / 2, padding_mode, out_cols, pad_left, pad_right);
    PaddingImageOp padding_op = PaddingImageOp<float>("reflect");
    Eigen::Tensor<float, 3, Eigen::RowMajor> padded_input;

    // Padding input image for patches extraction
    padding_op(tensor, padded_input, pad_top, pad_bottom, pad_left, pad_right, static_cast<float>(0.0) /*padding value for constant padding*/);
    auto p77 = extractPatchesByIndices(padded_input, indices, 7, 7);
    EXPECT_EQ(p77.dimension(0), indices.dimension(0));
    EXPECT_EQ(p77.dimension(1), indices.dimension(1));

    for (int row = 0; row < p77.dimension(0); row++)
        for (int col = 0; col < p77.dimension(1); col++) {
            for (int d = 0; d < input_depth; d++)
                EXPECT_EQ(p77(row, col, d), patches(row, col, d));
        }
}