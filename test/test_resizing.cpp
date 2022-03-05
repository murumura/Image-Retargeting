#include <gtest/gtest.h>
#include <image/resizing_op.h>
#include <iostream>
using namespace Image;
using testing::Eq;

template <typename T>
void FillValues(T* flat, std::size_t size, std::initializer_list<T> vals)
{
    if (size > 0) {
        std::size_t i = 0;
        for (auto itr = vals.begin(); itr != vals.end(); ++itr, ++i) {
            flat[i] = T(*itr);
        }
    }
}

// This is the straight forward unoptimized implementation of resize bilinear
// We use this to confirm that the optimized version is exactly identical.
void BatchResizeBilinearBaseline(const Eigen::Tensor<float, 4, Eigen::RowMajor>& images,
    Eigen::Tensor<float, 4, Eigen::RowMajor>& output,
    bool half_pixel_centers_ = false)
{
    const int batch = images.dimension(0);
    const int64_t in_height = images.dimension(1);
    const int64_t in_width = images.dimension(2);
    const int channels = images.dimension(3);

    ASSERT_EQ(batch, output.dimension(0));
    ASSERT_EQ(channels, output.dimension(3));

    const int64_t out_height = output.dimension(1);
    const int64_t out_width = output.dimension(2);

    const float height_scale = in_height / static_cast<float>(out_height);
    const float width_scale = in_width / static_cast<float>(out_width);

    for (int b = 0; b < batch; ++b) {
        for (int64_t y = 0; y < out_height; ++y) {
            const float in_y = half_pixel_centers_
                ? (static_cast<float>(y) + 0.5f) * height_scale - 0.5f
                : y * height_scale;
            const int64_t top_y_index = std::max(static_cast<int64_t>(floorf(in_y)),
                static_cast<int64_t>(0));
            const int64_t bottom_y_index = std::min(static_cast<int64_t>(ceilf(in_y)), in_height - 1);
            const float y_lerp = in_y - std::floor(in_y);
            for (int64_t x = 0; x < out_width; ++x) {
                const float in_x = half_pixel_centers_
                    ? (static_cast<float>(x) + 0.5f) * width_scale - 0.5f
                    : x * width_scale;
                const int64_t left_x_index = std::max(
                    static_cast<int64_t>(floorf(in_x)), static_cast<int64_t>(0));
                const int64_t right_x_index = std::min(static_cast<int64_t>(ceilf(in_x)), in_width - 1);
                const float x_lerp = in_x - std::floor(in_x);
                for (int c = 0; c < channels; ++c) {
                    const float top_left = images(b, top_y_index, left_x_index, c);
                    const float top_right = images(b, top_y_index, right_x_index, c);
                    const float bottom_left = images(b, bottom_y_index, left_x_index, c);
                    const float bottom_right = images(b, bottom_y_index, right_x_index, c);
                    const float top = top_left + (top_right - top_left) * x_lerp;
                    const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
                    output(b, y, x, c) = top + (bottom - top) * y_lerp;
                }
            }
        }
    }
}

void NoBatchResizeBilinearBaseline(
    const Eigen::Tensor<float, 3, Eigen::RowMajor>& image,
    Eigen::Tensor<float, 3, Eigen::RowMajor>& output,
    bool half_pixel_centers_ = false)
{
    const int H = image.dimension(0);
    const int W = image.dimension(1);
    const int C = image.dimension(2);
    const int outH = output.dimension(0);
    const int outW = output.dimension(1);
    const int outC = output.dimension(2);
    Eigen::Tensor<float, 4, Eigen::RowMajor> batchifyIn = image.reshape(Eigen::array<Index, 4>{1, H, W, C});

    Eigen::Tensor<float, 4, Eigen::RowMajor> batchifyOut = output.reshape(Eigen::array<Index, 4>{1, outH, outW, outC});

    BatchResizeBilinearBaseline(batchifyIn, batchifyOut, half_pixel_centers_);
    // remove dummy batch
    output = batchifyOut.reshape(Eigen::array<Index, 3>{outH, outW, outC});
}

void TestBatchBilinaerResize(
    int batch_size, int input_width, int input_height,
    int channels, int output_width, int output_height,
    bool half_pixel_centers_ = false)
{
    Eigen::Tensor<float, 4, Eigen::RowMajor> input(batch_size, input_height, input_width, channels);
    input.setRandom();
    Eigen::Tensor<float, 4, Eigen::RowMajor> expected(batch_size, output_height, output_width, channels);
    BatchResizeBilinearBaseline(input, expected);
    Eigen::Tensor<float, 4, Eigen::RowMajor> output(batch_size, output_height, output_width, channels);
    const float height_scale = input_height / static_cast<float>(output_height);
    const float width_scale = input_width / static_cast<float>(output_width);
    Image::Functor::ResizeBilinear<float>()(input, height_scale, width_scale, half_pixel_centers_, output);

    for (int b = 0; b < batch_size; ++b)
        for (int r = 0; r < output_height; r++)
            for (int c = 0; c < output_width; c++)
                for (int d = 0; d < channels; d++)
                    EXPECT_NEAR(expected(b, r, c, d), output(b, r, c, d), 1e-6);
}

void TestNoBatchBilinaerResize(
    int input_width, int input_height,
    int channels, int output_width, int output_height,
    bool half_pixel_centers_ = false)
{
    Eigen::Tensor<float, 3, Eigen::RowMajor> input(input_height, input_width, channels);
    input.setRandom();
    Eigen::Tensor<float, 3, Eigen::RowMajor> expected(output_height, output_width, channels);
    NoBatchResizeBilinearBaseline(input, expected);

    Eigen::Tensor<float, 3, Eigen::RowMajor> output(output_height, output_width, channels);
    const float height_scale = input_height / static_cast<float>(output_height);
    const float width_scale = input_width / static_cast<float>(output_width);
    Image::Functor::ResizeBilinear<float>()(input, height_scale, width_scale, half_pixel_centers_, output);

    for (int r = 0; r < output_height; r++)
        for (int c = 0; c < output_width; c++)
            for (int d = 0; d < channels; d++)
                EXPECT_NEAR(expected(r, c, d), output(r, c, d), 1e-6);
}

void BatchBilinearRandomTests(int channels, bool half_pixel_centers_ = false)
{
    for (int batch_size : {1, 2, 5}) {
        for (int in_h : {2, 4, 7, 20, 165}) {
            for (int in_w : {1, 3, 5, 8, 100, 233}) {
                for (int target_height : {1, 2, 3, 50, 113}) {
                    for (int target_width : {target_height, target_height / 2 + 1}) {
                        TestBatchBilinaerResize(batch_size, in_w, in_h, channels, target_width, target_height, half_pixel_centers_);
                    }
                }
            }
        }
    }
}

void NoBatchBilinearRandomTests(int channels, bool half_pixel_centers_ = false)
{
    for (int in_h : {2, 4, 7, 20, 165}) {
        for (int in_w : {1, 3, 5, 8, 100, 233}) {
            for (int target_height : {1, 2, 3, 50, 113}) {
                for (int target_width : {target_height, target_height / 2 + 1}) {
                    TestNoBatchBilinaerResize(in_w, in_h, channels, target_width, target_height, half_pixel_centers_);
                }
            }
        }
    }
}
TEST(ResizeNearestNeighbor, TestNearestNeighborAlignCorners4x4To3x3)
{
    // Input:
    //  1,  2,  3,  4
    //  5,  6,  7,  8
    //  9, 10, 11, 12
    // 13, 14, 15, 16
    Eigen::Tensor<float, 3, Eigen::RowMajor> input(4, 4, 1);
    FillValues<float>(input.data(), input.size(),
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    Eigen::Tensor<float, 3, Eigen::RowMajor> expected(3, 3, 1);
    Eigen::Tensor<float, 3, Eigen::RowMajor> output(3, 3, 1);
    const float height_scale = 4 / static_cast<float>(3);
    const float width_scale = 4 / static_cast<float>(3);
    FillValues<float>(expected.data(), expected.size(),
        {1, 2, 4,
            5, 6, 8,
            13, 14, 16});

    Image::Functor::ResizeNearestNeighbor<float>()(input, height_scale, width_scale, false, true, output);
    for (int r = 0; r < output.dimension(0); r++)
        for (int c = 0; c < output.dimension(1); c++)
            EXPECT_NEAR(expected(r, c, 0), output(r, c, 0), 1e-6);
}

TEST(ResizeNearestNeighborHalfPixelCentersOpTest, TestNearest3x3To2x2)
{
    // Input:
    //  1, 2, 3
    //  4, 5, 6
    //  7, 8, 9
    Eigen::Tensor<float, 3, Eigen::RowMajor> input(3, 3, 1);
    FillValues<float>(input.data(), input.size(),
        {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Eigen::Tensor<float, 3, Eigen::RowMajor> expected(2, 2, 1);
    Eigen::Tensor<float, 3, Eigen::RowMajor> output(2, 2, 1);
    const float height_scale = 3 / static_cast<float>(2);
    const float width_scale = 3 / static_cast<float>(2);
    FillValues<float>(expected.data(), expected.size(),
        {1, 3, 7, 9});

    Image::Functor::ResizeNearestNeighbor<float>()(input, height_scale, width_scale, true, false, output);
    for (int r = 0; r < output.dimension(0); r++)
        for (int c = 0; c < output.dimension(1); c++)
            EXPECT_NEAR(expected(r, c, 0), output(r, c, 0), 1e-6);
}

TEST(ResizeBilinear, TestBilinear2x2To1x1WithoutHalfPixelCenters)
{
    Eigen::Tensor<float, 3, Eigen::RowMajor> input(2, 2, 3);
    input.setValues(
        {{{5.0, 4.0, 6.0},
            {30.0, 221.0, 243.0},
            {42.0, 69.0, 84.0},
            {73.0, 112.0, 212.0}}});
    Eigen::Tensor<float, 3, Eigen::RowMajor> output(1, 1, 3);
    const float height_scale = 2 / static_cast<float>(1);
    const float width_scale = 2 / static_cast<float>(1);
    Image::Functor::ResizeBilinear<float>()(input, height_scale, width_scale, false, output);
    EXPECT_NEAR(output(0, 0, 0), input(0, 0, 0), 1e-6);
    EXPECT_NEAR(output(0, 0, 1), input(0, 0, 1), 1e-6);
    EXPECT_NEAR(output(0, 0, 2), input(0, 0, 2), 1e-6);
}

TEST(ResizeBilinear, TestBilinear2x2To1x1WithHalfPixelCenters)
{
    Eigen::Tensor<float, 3, Eigen::RowMajor> input(2, 2, 3);
    input.setValues(
        {{{5.0, 4.0, 6.0},
            {30.0, 221.0, 243.0},
            {42.0, 69.0, 84.0},
            {73.0, 112.0, 212.0}}});
    Eigen::Tensor<float, 3, Eigen::RowMajor> output(1, 1, 3);
    const float height_scale = 2 / static_cast<float>(1);
    const float width_scale = 2 / static_cast<float>(1);
    Image::Functor::ResizeBilinear<float>()(input, height_scale, width_scale, true, output);
    EXPECT_NEAR(output(0, 0, 0), 37.5, 1e-6);
    EXPECT_NEAR(output(0, 0, 1), 101.5, 1e-6);
    EXPECT_NEAR(output(0, 0, 2), 136.25, 1e-6);
}

TEST(ResizeBilinearOpTest, TestBilinear2x2To3x3WithoutHalfPixelCenters)
{
    // Input:
    //  1, 2
    //  3, 4
    Eigen::Tensor<float, 3, Eigen::RowMajor> input(2, 2, 1);
    input(0, 0, 0) = 1.0;
    input(0, 1, 0) = 2.0;
    input(1, 0, 0) = 3.0;
    input(1, 1, 0) = 4.0;
    Eigen::Tensor<float, 3, Eigen::RowMajor> expected(3, 3, 1);
    // clang-format off
    expected.setValues(
            {{
                {1}, 
                {5.0f / 3}, 
                {2},
                {7.0f / 3}, 
                {3}, 
                {10.0f / 3},
                {3}, 
                {11.0f / 3}, 
                {4}
            }}
    );
    // clang-format on
    const float height_scale = 2 / static_cast<float>(3);
    const float width_scale = 2 / static_cast<float>(3);
    Eigen::Tensor<float, 3, Eigen::RowMajor> output(3, 3, 1);
    Image::Functor::ResizeBilinear<float>()(input, height_scale, width_scale, false, output);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            EXPECT_NEAR(expected(r, c, 0), output(r, c, 0), 1e-6);
}

TEST(ResizeBilinearOpTest, TestBilinear2x2To3x3WithHalfPixelCenters)
{
    // Input:
    //  1, 2
    //  3, 4
    Eigen::Tensor<float, 3, Eigen::RowMajor> input(2, 2, 1);
    input(0, 0, 0) = 1.0;
    input(0, 1, 0) = 2.0;
    input(1, 0, 0) = 3.0;
    input(1, 1, 0) = 4.0;
    Eigen::Tensor<float, 3, Eigen::RowMajor> expected(3, 3, 1);
    // clang-format off
    expected.setValues(
            {{
                {1}, 
                {1.5}, 
                {2},
                {2}, 
                {2.5}, 
                {3},
                {3}, 
                {3.5}, 
                {4}
            }}
    );
    // clang-format on
    const float height_scale = 2 / static_cast<float>(3);
    const float width_scale = 2 / static_cast<float>(3);
    Eigen::Tensor<float, 3, Eigen::RowMajor> output(3, 3, 1);
    Image::Functor::ResizeBilinear<float>()(input, height_scale, width_scale, true, output);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            EXPECT_NEAR(expected(r, c, 0), output(r, c, 0), 1e-6);
}

TEST(ResizeBilinear, TestNoBatch1Channel)
{
    NoBatchBilinearRandomTests(1);
}

TEST(ResizeBilinear, TestNoBatch3Channel)
{
    NoBatchBilinearRandomTests(3);
}

TEST(ResizeBilinear, TestNoBatch4Channel)
{
    NoBatchBilinearRandomTests(4);
}

TEST(ResizeBilinear, TestBatch1Channel)
{
    BatchBilinearRandomTests(1);
}

TEST(ResizeBilinear, TestBatch3Channel)
{
    BatchBilinearRandomTests(3);
}

TEST(ResizeBilinear, TestBatch4Channel)
{
    BatchBilinearRandomTests(4);
}

TEST(ResizeNearestNeighbor, TestNearest2x2To1x1)
{
    // Input:
    //  1, 2
    //  3, 4
    Eigen::Tensor<float, 3, Eigen::RowMajor> input(2, 2, 1);
    input(0, 0, 0) = 1.0;
    input(0, 1, 0) = 2.0;
    input(1, 0, 0) = 3.0;
    input(1, 1, 0) = 4.0;
    Eigen::Tensor<float, 3, Eigen::RowMajor> output(1, 1, 1);
    Eigen::Tensor<float, 3, Eigen::RowMajor> expected(1, 1, 1);
    const float height_scale = 2 / static_cast<float>(1);
    const float width_scale = 2 / static_cast<float>(1);
    // clang-format off
    expected.setValues(
            {{
                {1}
            }}
    );
    // clang-format off
    Image::Functor::ResizeNearestNeighbor<float>()(input, height_scale, width_scale, false, false, output);
    EXPECT_NEAR(output(0, 0, 0), 1.0, 1e-6);
}

TEST(ResizeNearestNeighbor, TestNearest2x2To3x3) {
  // Input:
  //  1, 2
  //  3, 4
    Eigen::Tensor<float, 3, Eigen::RowMajor> input(2, 2, 1);
    input(0, 0, 0) = 1.0;
    input(0, 1, 0) = 2.0;
    input(1, 0, 0) = 3.0;
    input(1, 1, 0) = 4.0;
    Eigen::Tensor<float, 3, Eigen::RowMajor> output(3, 3, 1);
    Eigen::Tensor<float, 3, Eigen::RowMajor> expected(3, 3, 1);
    const float height_scale = 2 / static_cast<float>(3);
    const float width_scale = 2 / static_cast<float>(3);
  // clang-format off
    expected.setValues(
            {{
                {1}, 
                {1}, 
                {2},
                {1}, 
                {1}, 
                {2},
                {3}, 
                {3}, 
                {4}
            }}
    );
    // clang-format on
    Image::Functor::ResizeNearestNeighbor<float>()(input, height_scale, width_scale, false, false, output);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            EXPECT_NEAR(expected(r, c, 0), output(r, c, 0), 1e-6);
}

TEST(ResizeNearestNeighbor, TestNearestAlignCorners2x2To3x3)
{
    // Input:
    //  1, 2
    //  3, 4
    Eigen::Tensor<float, 3, Eigen::RowMajor> input(2, 2, 1);
    input(0, 0, 0) = 1.0;
    input(0, 1, 0) = 2.0;
    input(1, 0, 0) = 3.0;
    input(1, 1, 0) = 4.0;
    Eigen::Tensor<float, 3, Eigen::RowMajor> output(3, 3, 1);
    Eigen::Tensor<float, 3, Eigen::RowMajor> expected(3, 3, 1);
    const float height_scale = 2 / static_cast<float>(3);
    const float width_scale = 2 / static_cast<float>(3);
    // clang-format off
    expected.setValues(
            {{
                {1}, 
                {2}, 
                {2},
                {3}, 
                {4}, 
                {4},
                {3}, 
                {4}, 
                {4}
            }}
    );
    // clang-format on
    Image::Functor::ResizeNearestNeighbor<float>()(input, height_scale, width_scale, false, true, output);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            EXPECT_NEAR(expected(r, c, 0), output(r, c, 0), 1e-6);
}

TEST(ResizeNearestNeighbor, TestNearest3x3To2x2)
{
    // Input:
    //  1, 2, 3
    //  4, 5, 6
    //  7, 8, 9
    Eigen::Tensor<float, 3, Eigen::RowMajor> input(3, 3, 1);
    input.setValues(
        {{{1},
            {2},
            {3},
            {4},
            {5},
            {6},
            {7},
            {8},
            {9}}});
    Eigen::Tensor<float, 3, Eigen::RowMajor> output(2, 2, 1);
    Eigen::Tensor<float, 3, Eigen::RowMajor> expected(2, 2, 1);
    const float height_scale = 3 / static_cast<float>(2);
    const float width_scale = 3 / static_cast<float>(2);
    expected.setValues(
        {{
            {1},
            {2},
            {4},
            {5},
        }});
    Image::Functor::ResizeNearestNeighbor<float>()(input, height_scale, width_scale, false, false, output);
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 2; c++)
            EXPECT_NEAR(expected(r, c, 0), output(r, c, 0), 1e-6);
}

TEST(ResizeNearestNeighbor, TestNearest2x2To2x5)
{
    // Input:
    //  1, 2
    //  3, 4
    Eigen::Tensor<float, 3, Eigen::RowMajor> input(2, 2, 1);
    input(0, 0, 0) = 1.0;
    input(0, 1, 0) = 2.0;
    input(1, 0, 0) = 3.0;
    input(1, 1, 0) = 4.0;
    Eigen::Tensor<float, 3, Eigen::RowMajor> expected(2, 5, 1);
    Eigen::Tensor<float, 3, Eigen::RowMajor> output(2, 5, 1);
    expected(0, 0, 0) = 1.0;
    expected(0, 1, 0) = 1.0;
    expected(0, 2, 0) = 1.0;
    expected(0, 3, 0) = 2.0;
    expected(0, 4, 0) = 2.0;
    expected(1, 0, 0) = 3.0;
    expected(1, 1, 0) = 3.0;
    expected(1, 2, 0) = 3.0;
    expected(1, 3, 0) = 4.0;
    expected(1, 4, 0) = 4.0;

    // clang-format on
    const float height_scale = 2 / static_cast<float>(2);
    const float width_scale = 2 / static_cast<float>(5);
    Image::Functor::ResizeNearestNeighbor<float>()(input, height_scale, width_scale, false, false, output);
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 5; c++)
            EXPECT_NEAR(expected(r, c, 0), output(r, c, 0), 1e-6);
}

TEST(ResizeNearestNeighbor, TestNearestNeighbor4x4To3x3)
{
    // Input:
    //  1,  2,  3,  4
    //  5,  6,  7,  8
    //  9, 10, 11, 12
    // 13, 14, 15, 16
    Eigen::Tensor<float, 3, Eigen::RowMajor> input(4, 4, 1);
    FillValues<float>(input.data(), input.size(),
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    Eigen::Tensor<float, 3, Eigen::RowMajor> expected(3, 3, 1);
    Eigen::Tensor<float, 3, Eigen::RowMajor> output(3, 3, 1);
    const float height_scale = 4 / static_cast<float>(3);
    const float width_scale = 4 / static_cast<float>(3);
    FillValues<float>(expected.data(), expected.size(),
        {1, 2, 3,
            5, 6, 7,
            9, 10, 11});

    Image::Functor::ResizeNearestNeighbor<float>()(input, height_scale, width_scale, false, false, output);
    for (int r = 0; r < output.dimension(0); r++)
        for (int c = 0; c < output.dimension(1); c++)
            EXPECT_NEAR(expected(r, c, 0), output(r, c, 0), 1e-6);
}

TEST(ResizeNearestNeighborOp, TestNearestNeighbor4x4To3x3)
{
    // Input:
    //  1,  2,  3,  4
    //  5,  6,  7,  8
    //  9, 10, 11, 12
    // 13, 14, 15, 16
    Eigen::Tensor<float, 3, Eigen::RowMajor> input(4, 4, 1);
    FillValues<float>(input.data(), input.size(),
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    Eigen::Tensor<float, 3, Eigen::RowMajor> expected(3, 3, 1);
    Eigen::Tensor<float, 3, Eigen::RowMajor> output(3, 3, 1);
    FillValues<float>(expected.data(), expected.size(),
        {1, 2, 3,
            5, 6, 7,
            9, 10, 11});
    ResizingImageOp resizing_op = ResizingImageOp<float>("nearest_neighbor", false, false);
    resizing_op(input, output);
    for (int r = 0; r < output.dimension(0); r++)
        for (int c = 0; c < output.dimension(1); c++)
            EXPECT_NEAR(expected(r, c, 0), output(r, c, 0), 1e-6);
}
