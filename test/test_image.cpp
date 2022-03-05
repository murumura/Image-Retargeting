#include <cstdlib>
#include <gtest/gtest.h>
#include <image/colorspace_op.h>
#include <image/filter.h>
#include <image/image.h>
#include <image/imageIO.h>
#include <image/resizing_op.h>
#include <iostream>
using namespace Image;
using testing::Eq;

template <typename data>
data randomValue(int maximum)
{
    if (maximum == 0)
        return 0;
    else
        return static_cast<data>(static_cast<uint32_t>(rand()) % maximum);
}

template <typename data>
data randomValue(data minimum, int maximum)
{
    if (maximum == 0)
        return 0;
    else {
        data value = static_cast<data>(rand() % maximum);
        if (value < minimum)
            value = minimum;
        return value;
    }
}
TEST(Image, interpolationNN)
{
    Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> RGB = loadPNG<uint8_t>("./test/test_image/blur.png", 3);
    Eigen::Tensor<float, 3, Eigen::RowMajor> interpolatedRGB(RGB.dimension(0) * 1.5, RGB.dimension(1) * 0.75, RGB.dimension(2));
    const float height_scale = 1 / 1.5;
    const float width_scale = 1 / 0.75;
    Image::Functor::ResizeNearestNeighbor<float>()(RGB.cast<float>(), height_scale, width_scale, true, true, interpolatedRGB);
    savePNG<uint8_t, 3>("./interpolatedRGB", interpolatedRGB.cast<uint8_t>());
    EXPECT_EQ(0, remove("./interpolatedRGB.png"));
}

TEST(Image, Shape)
{
    uint32_t runIter = 16;
    for (uint32_t i = 0; i < runIter; ++i) {
        const int width = randomValue<int>(2048);
        const int height = randomValue<int>(2048);
        const std::size_t channelCount = std::rand() % 4 + 1;
        ImageTemplate<int, 3> image;
        image.resize(static_cast<Eigen::Index>(height), static_cast<Eigen::Index>(width), static_cast<Eigen::Index>(channelCount));
        EXPECT_EQ(image.dimension(0), height);
        EXPECT_EQ(image.dimension(1), width);
        EXPECT_EQ(image.dimension(2), channelCount);
    }
}

TEST(Image, loadPngByReference)
{
    Uint8Image lenaImg;
    loadPNG("./test/test_image/lena.png", 1, lenaImg);
    EXPECT_EQ(lenaImg.dimension(0), 512);
    EXPECT_EQ(lenaImg.dimension(1), 512);
    EXPECT_EQ(lenaImg.dimension(2), 1);
}

TEST(Image, loadPngByReturn)
{
    auto lenaImg = loadPNG<uint8_t>("./test/test_image/lena.png", 1);
    EXPECT_EQ(lenaImg.dimension(0), 512);
    EXPECT_EQ(lenaImg.dimension(1), 512);
    EXPECT_EQ(lenaImg.dimension(2), 1);
}

TEST(Image, savePng)
{
    // fill with white
    Uint8Image whiteImg;
    std::size_t height = 40;
    std::size_t width = 40;
    std::size_t channelCount = 3;
    whiteImg.resize(static_cast<Eigen::Index>(height), static_cast<Eigen::Index>(width), static_cast<Eigen::Index>(channelCount));
    whiteImg.setConstant(255);
    savePNG("./white", whiteImg);

    Uint8Image loadWhite = loadPNG<uint8_t>("./white.png", channelCount);
    EXPECT_EQ(loadWhite.dimension(0), width);
    EXPECT_EQ(loadWhite.dimension(1), height);
    EXPECT_EQ(loadWhite.dimension(2), channelCount);
    for (Eigen::Index depth = 0; depth < channelCount; depth++)
        for (Eigen::Index col = 0; col < width; col++)
            for (Eigen::Index row = 0; row < height; row++)
                EXPECT_EQ(loadWhite(row, col, depth), whiteImg(row, col, depth));
    EXPECT_EQ(0, remove("./white.png"));
}

TEST(Imgae, foreach)
{
    Uint8Image gray(4, 4, 1);
    gray.setConstant(1);
    int sum{0};
    forEachPixel(gray, [&sum](int x) {
        sum += x;
    });
    EXPECT_EQ(sum, 16);
}

TEST(Image, rgb_to_gray)
{
    Eigen::Tensor<int, 3, Eigen::RowMajor> rgb(2, 2, 3);
    rgb.setConstant(0);
    rgb.setRandom();
    Eigen::Tensor<int, 3, Eigen::RowMajor> gray;
    Image::Functor::RGBToGray<int>()(rgb, gray);
    EXPECT_EQ(gray.dimension(0), 2);
    EXPECT_EQ(gray.dimension(1), 2);
    EXPECT_EQ(gray.dimension(2), 1);
}

TEST(Image, rgb_to_hsv)
{
    Eigen::Tensor<float, 3, Eigen::RowMajor> rgb(2, 2, 3);
    Eigen::Tensor<float, 3, Eigen::RowMajor> hsv(2, 2, 3);
    Eigen::Tensor<float, 3, Eigen::RowMajor> rgb_from_hsv(2, 2, 3);
    rgb.setRandom();
    Image::Functor::RGBToHSV<float>()(rgb, hsv);
    EXPECT_EQ(hsv.dimension(0), 2);
    EXPECT_EQ(hsv.dimension(1), 2);
    EXPECT_EQ(hsv.dimension(2), 3);

    Image::Functor::HSVToRGB<float>()(hsv, rgb_from_hsv);
    EXPECT_EQ(rgb_from_hsv.dimension(0), 2);
    EXPECT_EQ(rgb_from_hsv.dimension(1), 2);
    EXPECT_EQ(rgb_from_hsv.dimension(2), 3);
    for (Index r = 0; r < rgb_from_hsv.dimension(0); r++)
        for (Index c = 0; c < rgb_from_hsv.dimension(1); c++)
            for (Index d = 0; d < rgb_from_hsv.dimension(2); d++)
                EXPECT_NEAR(rgb_from_hsv(r, c, d), rgb(r, c, d), 1e-6);
}

TEST(Image, rgb_to_xyz)
{
    Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> lenaRGB = loadPNG<uint8_t>("./test/test_image/lena256.png", 3);
    Eigen::Tensor<float, 3, Eigen::RowMajor> lenaXYZ;
    Image::Functor::RGBToXYZ<float>()(lenaRGB.cast<float>(), lenaXYZ);

    savePNG<uint8_t, 3>("./lenaXYZ", lenaXYZ.cast<uint8_t>());
    EXPECT_EQ(0, remove("./lenaXYZ.png"));
}

TEST(Image, rgb_to_cie)
{
    Eigen::Tensor<float, 3, Eigen::RowMajor> rgb(2, 2, 3);
    Eigen::Tensor<float, 3, Eigen::RowMajor> cie(2, 2, 3);
    Eigen::Tensor<float, 3, Eigen::RowMajor> cieExp(2, 2, 3);
    rgb.setValues({{{255.0, 255.0, 255.0},
        {30.0, 221.0, 243.0},
        {42.0, 69.0, 84.0},
        {73.0, 112.0, 212.0}}});

    cieExp.setValues({{{100.00, 0.00, 0.00},
        {80.99, -35.53, -23.09},
        {27.81, -5.55, -12.15},
        {49.20, 18.44, -55.67}}});

    Image::Functor::RGBToCIE<float>()(rgb, cie);

    for (int r = 0; r < cie.dimension(0); r++)
        for (int c = 0; c < cie.dimension(1); c++)
            for (int d = 0; d < cie.dimension(2); d++)
                EXPECT_NEAR(cie(r, c, d), cieExp(r, c, d), 1e-2);
}

TEST(Image, Panda_to_gray)
{
    Uint8Image pandaRGB = loadPNG<uint8_t>("./test/test_image/panda.png", 3);
    EXPECT_EQ(pandaRGB.dimension(0), 800);
    EXPECT_EQ(pandaRGB.dimension(1), 600);
    EXPECT_EQ(pandaRGB.dimension(2), 3);
    Uint8Image pandaGray;
    Image::Functor::RGBToGray<uint8_t>()(pandaRGB, pandaGray);

    EXPECT_EQ(pandaGray.dimension(0), 800);
    EXPECT_EQ(pandaGray.dimension(1), 600);
    EXPECT_EQ(pandaGray.dimension(2), 1);

    savePNG("./PandaGray", pandaGray);
    EXPECT_EQ(0, remove("./PandaGray.png"));
}

TEST(Image, Lena_to_gray)
{
    Uint8Image lenaRGB = loadPNG<uint8_t>("./test/test_image/lena256.png", 3);
    EXPECT_EQ(lenaRGB.dimension(0), 256);
    EXPECT_EQ(lenaRGB.dimension(1), 256);
    EXPECT_EQ(lenaRGB.dimension(2), 3);
    Uint8Image lenaGray;
    Image::Functor::RGBToGray<uint8_t>()(lenaRGB, lenaGray);
    EXPECT_EQ(lenaGray.dimension(0), 256);
    EXPECT_EQ(lenaGray.dimension(1), 256);
    EXPECT_EQ(lenaGray.dimension(2), 1);

    savePNG("./lenaGray", lenaGray);
    EXPECT_EQ(0, remove("./lenaGray.png"));
}

TEST(Image, image_padder)
{
    const int H = 800;
    const int W = 600;
    const int D = 3;
    const int pL = 18;
    const int pR = 27;
    const int pT = 16;
    const int pD = 7;
    const int outH = H + pD + pT;
    const int outW = W + pL + pR;
    auto paddingOp = PaddingImageOp<uint8_t>("reflect");
    Uint8Image pandaRGB = loadPNG<uint8_t>("./test/test_image/panda.png", D);
    EXPECT_EQ(pandaRGB.dimension(0), 800);
    EXPECT_EQ(pandaRGB.dimension(1), 600);
    EXPECT_EQ(pandaRGB.dimension(2), 3);

    Uint8Image paddedPandaRGB;
    paddingOp(pandaRGB, paddedPandaRGB, std::make_tuple(pT, pD), std::make_tuple(pL, pR));
    savePNG("./paddedPandaRGB", paddedPandaRGB);
    EXPECT_EQ(0, remove("./paddedPandaRGB.png"));
}

TEST(Image, random_filter)
{
    const int H = 800;
    const int W = 600;
    const int D = 3;
    Uint8Image pandaRGB = loadPNG<uint8_t>("./test/test_image/panda.png", D);

    Index KsizeW = 5;
    Index KsizeH = 5;
    Index Knum = 1;
    Eigen::Tensor<float, 3, Eigen::RowMajor> kernel(KsizeH, KsizeW, Knum);
    kernel.setRandom();

    auto KPandaRGB = imageConvolution(pandaRGB, kernel, "reflect");
    EXPECT_EQ(KPandaRGB.dimension(0), H);
    EXPECT_EQ(KPandaRGB.dimension(1), W);
    EXPECT_EQ(KPandaRGB.dimension(2), D);
    savePNG("./KPandaRGB", KPandaRGB);
    EXPECT_EQ(0, remove("./KPandaRGB.png"));
}

TEST(Image, convolution)
{
    uint32_t runIter = 36;
    const int HWmin = 50;
    const int HWmax = 500;
    const int Kmax = 10;
    const int Kmin = 1;
    for (uint32_t i = 0; i < runIter; ++i) {
        const int H = randomValue<int>(HWmin, HWmax);
        const int W = randomValue<int>(HWmin, HWmax);
        const int D = randomValue<int>(1, 3);
        const int kH = randomValue<int>(Kmin, Kmax);
        const int kW = randomValue<int>(Kmin, Kmax);
        Eigen::Tensor<int, 3, Eigen::RowMajor> rgb(H, W, D);
        rgb.setRandom();
        Eigen::Tensor<float, 3, Eigen::RowMajor> kernel(kH, kW, 1);
        kernel.setRandom();
        auto output = imageConvolution(rgb, kernel, "reflect");
        EXPECT_EQ(output.dimension(0), H);
        EXPECT_EQ(output.dimension(1), W);
        EXPECT_EQ(output.dimension(2), D);
    }

    for (uint32_t i = 0; i < runIter; ++i) {
        const int H = randomValue<int>(HWmin, HWmax);
        const int W = randomValue<int>(HWmin, HWmax);
        const int D = randomValue<int>(1, 3);
        const int kH = randomValue<int>(Kmin, Kmax);
        const int kW = randomValue<int>(Kmin, Kmax);
        Eigen::Tensor<int, 3, Eigen::RowMajor> rgb(H, W, D);
        rgb.setRandom();
        Eigen::Tensor<float, 3, Eigen::RowMajor> kernel(kH, kW, 1);
        kernel.setRandom();
        auto output = imageConvolution(rgb, kernel, "symmetric");
        EXPECT_EQ(output.dimension(0), H);
        EXPECT_EQ(output.dimension(1), W);
        EXPECT_EQ(output.dimension(2), D);
    }

    for (uint32_t i = 0; i < runIter; ++i) {
        const int H = randomValue<int>(HWmin, HWmax);
        const int W = randomValue<int>(HWmin, HWmax);
        const int D = randomValue<int>(1, 3);
        const int kH = randomValue<int>(Kmin, Kmax);
        const int kW = randomValue<int>(Kmin, Kmax);
        Eigen::Tensor<int, 3, Eigen::RowMajor> rgb(H, W, D);
        rgb.setRandom();
        Eigen::Tensor<float, 3, Eigen::RowMajor> kernel(kH, kW, 1);
        kernel.setRandom();
        auto output = imageConvolution(rgb, kernel, "constant");
        EXPECT_EQ(output.dimension(0), H);
        EXPECT_EQ(output.dimension(1), W);
        EXPECT_EQ(output.dimension(2), D);
    }

    for (uint32_t i = 0; i < runIter; ++i) {
        const int H = randomValue<int>(HWmin, HWmax);
        const int W = randomValue<int>(HWmin, HWmax);
        const int D = randomValue<int>(1, 3);
        const int kH = randomValue<int>(Kmin, Kmax);
        const int kW = randomValue<int>(Kmin, Kmax);
        Eigen::Tensor<int, 3, Eigen::RowMajor> rgb(H, W, D);
        rgb.setRandom();
        Eigen::Tensor<float, 3, Eigen::RowMajor> kernel(kH, kW, 1);
        kernel.setRandom();
        auto output = imageConvolution(rgb, kernel, "edge");
        EXPECT_EQ(output.dimension(0), H);
        EXPECT_EQ(output.dimension(1), W);
        EXPECT_EQ(output.dimension(2), D);
    }
}

TEST(Image, gaussian_kernel)
{
    const int H = 40;
    const int W = 40;
    const int D = 3;
    auto gaussianKernel = create("gaussian", "reflect");
    Eigen::Tensor<int, 3, Eigen::RowMajor> rgb(H, W, D);
    rgb.setRandom();
    auto random = imageConvolution(rgb, gaussianKernel->getKernel(), gaussianKernel->getPaddingMode());
    EXPECT_EQ(random.dimension(0), H);
    EXPECT_EQ(random.dimension(1), W);
    EXPECT_EQ(random.dimension(2), D);
    savePNG("./Random", random);
    EXPECT_EQ(0, remove("./Random.png"));
}

TEST(Image, gaussian_kernel_lena)
{
    Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> RGB = loadPNG<uint8_t>("./test/test_image/blur.png", 3);
    Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> blurRGB;
    GaussianBlur(RGB, blurRGB, "reflect", 0.707, 1.5);
    savePNG("./blurRGB", blurRGB);
    EXPECT_EQ(0, remove("./blurRGB.png"));
}
