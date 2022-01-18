#include <cstdlib>
#include <gmock/gmock.h>
#include <image/image.h>
#include <image/imageIO.h>
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

TEST(Image, Shape)
{
    uint32_t runIter = 16;
    for (uint32_t i = 0; i < runIter; ++i) {
        const int width = randomValue<int>(2048);
        const int height = randomValue<int>(2048);
        const std::size_t channelCount = std::rand() % 4 + 1;
        ImageTemplate<int, 3> image;
        image.resize(static_cast<Eigen::Index>(height), static_cast<Eigen::Index>(width), static_cast<Eigen::Index>(channelCount));
        ASSERT_EQ(image.dimension(0), height);
        ASSERT_EQ(image.dimension(1), width);
        ASSERT_EQ(image.dimension(2), channelCount);
    }
}

TEST(Image, loadPngByReference)
{
    Uint8Image lenaImg;
    loadPNG("./test/test_image/lena.png", 1, lenaImg);
    ASSERT_EQ(lenaImg.dimension(0), 512);
    ASSERT_EQ(lenaImg.dimension(1), 512);
    ASSERT_EQ(lenaImg.dimension(2), 1);
}

TEST(Image, loadPngByReturn)
{
    auto lenaImg = loadPNG<uint8_t>("./test/test_image/lena.png", 1);
    ASSERT_EQ(lenaImg.dimension(0), 512);
    ASSERT_EQ(lenaImg.dimension(1), 512);
    ASSERT_EQ(lenaImg.dimension(2), 1);
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
    ASSERT_EQ(loadWhite.dimension(0), width);
    ASSERT_EQ(loadWhite.dimension(1), height);
    ASSERT_EQ(loadWhite.dimension(2), channelCount);
    for (Eigen::Index depth = 0; depth < channelCount; depth++)
        for (Eigen::Index col = 0; col < width; col++)
            for (Eigen::Index row = 0; row < height; row++)
                ASSERT_EQ(loadWhite(row, col, depth), whiteImg(row, col, depth));
    ASSERT_EQ(0, remove("./white.png"));
}

TEST(Imgae, foreach)
{
    Uint8Image gray(4, 4, 1);
    gray.setConstant(1);
    int sum{0};
    forEachPixel(gray, [&sum](int x) {
        sum += x;
    });
    ASSERT_EQ(sum, 16);
}

TEST(Image, rgb_to_gray)
{
    Eigen::Tensor<int, 3, Eigen::RowMajor> rgb(2, 2, 3);
    rgb(0, 0, 0) = 255;
    rgb(0, 1, 0) = 255;
    rgb(1, 0, 0) = 255;
    rgb(1, 1, 0) = 255;
    rgb(0, 0, 1) = 128;
    rgb(0, 1, 1) = 128;
    rgb(1, 0, 1) = 128;
    rgb(1, 1, 1) = 128;
    rgb(0, 0, 2) = 0;
    rgb(0, 1, 2) = 0;
    rgb(1, 0, 2) = 0;
    rgb(1, 1, 2) = 0;
    // Eigen::Tensor<int, 3> gray = rgb.customOp(rgbToGray<int>());
    rgbToGrayFunctor<int> functor;
    Eigen::Tensor<int, 3, Eigen::RowMajor> gray = functor(rgb);
    ASSERT_EQ(gray.dimension(0), 2);
    ASSERT_EQ(gray.dimension(1), 2);
    ASSERT_EQ(gray.dimension(2), 1);
}

TEST(Image, Panda)
{
    Uint8Image pandaRGB = loadPNG<uint8_t>("./test/test_image/panda.png", 3);
    ASSERT_EQ(pandaRGB.dimension(0), 800);
    ASSERT_EQ(pandaRGB.dimension(1), 600);
    ASSERT_EQ(pandaRGB.dimension(2), 3);

    rgbToGrayFunctor<uint8_t> functor;
    Uint8Image pandaGray = functor(pandaRGB);

    ASSERT_EQ(pandaGray.dimension(0), 800);
    ASSERT_EQ(pandaGray.dimension(1), 600);
    ASSERT_EQ(pandaGray.dimension(2), 1);

    savePNG("./lenaPanda", pandaGray);
    ASSERT_EQ(0, remove("./lenaPanda.png"));
}

TEST(Image, Lena)
{
    Uint8Image lenaRGB = loadPNG<uint8_t>("./test/test_image/lena256.png", 3);
    ASSERT_EQ(lenaRGB.dimension(0), 256);
    ASSERT_EQ(lenaRGB.dimension(1), 256);
    ASSERT_EQ(lenaRGB.dimension(2), 3);

    rgbToGrayFunctor<uint8_t> functor;
    Uint8Image lenaGray = functor(lenaRGB);
    ASSERT_EQ(lenaGray.dimension(0), 256);
    ASSERT_EQ(lenaGray.dimension(1), 256);
    ASSERT_EQ(lenaGray.dimension(2), 1);

    savePNG("./lenaGray", lenaGray);
    ASSERT_EQ(0, remove("./lenaGray.png"));
}