#include <cstdlib>
#include <gmock/gmock.h>
#include <image/image.h>
#include <image/imageIO.h>
#include <image/pad_op.h>
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
    rgb.setConstant(0);
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
    //Eigen::Tensor<int, 3, Eigen::RowMajor> gray = rgb.customOp(rgbToGray<int>());
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

TEST(Image, const_padding)
{
    Eigen::Tensor<int, 3, Eigen::RowMajor> rgb(2, 2, 3);
    rgb.setConstant(10);
    Eigen::array<std::pair<int, int>, 3> paddings;
    paddings[0] = std::make_pair(0, 1);
    paddings[1] = std::make_pair(2, 3);
    paddings[2] = std::make_pair(0, 0);
    Eigen::Tensor<int, 3, Eigen::RowMajor> padded = rgb.pad(paddings);
    ASSERT_EQ(padded.dimension(0), 2 + 1);
    ASSERT_EQ(padded.dimension(1), 2 + 2 + 3);
    ASSERT_EQ(padded.dimension(2), 3);
}

TEST(Image, image_const_padding)
{
    const int H = 41;
    const int W = 7;
    const int D = 3;
    const int pH = 1;
    const int pW = 2;
    const int padVal = -2;
    Eigen::Tensor<int, 3, Eigen::RowMajor> rgb(H, W, D);
    rgb.setConstant(10);
    Eigen::Tensor<int, 3, Eigen::RowMajor> padded = rgb.customOp(padConstant<int>(padVal, pH, pW));
    ASSERT_EQ(padded.dimension(0), H + pH * 2);
    ASSERT_EQ(padded.dimension(1), W + pW * 2);
    ASSERT_EQ(padded.dimension(2), D);
}

TEST(Image, reflect_padding)
{
    const int H = 41;
    const int W = 7;
    const int D = 3;
    Eigen::Tensor<int, 3, Eigen::RowMajor> rgb(H, W, D);
    rgb.setRandom();
    int pH = 3;
    int pW = 4;
    Eigen::Tensor<int, 3, Eigen::RowMajor> paddedRGB = rgb.customOp(padReflect<int>(pH, pW));

    ASSERT_EQ(paddedRGB.dimension(0), H + 2 * pH);
    ASSERT_EQ(paddedRGB.dimension(1), W + 2 * pW);
    ASSERT_EQ(paddedRGB.dimension(2), D);
    const int newH = H + 2 * pH;
    const int newW = W + 2 * pW;
    // check padding equal
    Eigen::Tensor<int, 3, Eigen::RowMajor> padU = \ 
        paddedRGB.slice(Eigen::array<Index, D>{0, pW, 0}, Eigen::array<Index, D>{pH, W, D})
                                                      .reverse(Eigen::array<Index, D>{true, false, false});

    Eigen::Tensor<int, 3, Eigen::RowMajor> padUGt = rgb.slice(Eigen::array<Index, D>{1, 0, 0}, Eigen::array<Index, D>{pH, W, D});

    ASSERT_EQ(padUGt.dimension(0), padU.dimension(0));
    ASSERT_EQ(padUGt.dimension(1), padU.dimension(1));
    ASSERT_EQ(padUGt.dimension(2), padU.dimension(2));

    for (Index r = 0; r < padU.dimension(0); r++)
        for (Index c = 0; c < padU.dimension(1); c++)
            for (Index d = 0; d < padU.dimension(2); d++)
                ASSERT_EQ(padU(r, c, d), padUGt(r, c, d));

    Eigen::Tensor<int, 3, Eigen::RowMajor> padD = \ 
        paddedRGB.slice(Eigen::array<Index, D>{newH - pH, pW, 0}, Eigen::array<Index, D>{pH, W, D})
                                                      .reverse(Eigen::array<Index, D>{true, false, false});

    Eigen::Tensor<int, 3, Eigen::RowMajor> padDGt = rgb.slice(Eigen::array<Index, D>{H - pH - 1, 0, 0}, Eigen::array<Index, D>{pH, W, D});

    ASSERT_EQ(padDGt.dimension(0), padD.dimension(0));
    ASSERT_EQ(padDGt.dimension(1), padD.dimension(1));
    ASSERT_EQ(padDGt.dimension(2), padD.dimension(2));

    for (Index r = 0; r < padD.dimension(0); r++)
        for (Index c = 0; c < padD.dimension(1); c++)
            for (Index d = 0; d < padD.dimension(2); d++)
                ASSERT_EQ(padD(r, c, d), padDGt(r, c, d));

    Eigen::Tensor<int, 3, Eigen::RowMajor> padL = \ 
        paddedRGB.slice(Eigen::array<Index, D>{pH, 0, 0}, Eigen::array<Index, D>{H, pW, D})
                                                      .reverse(Eigen::array<Index, D>{false, true, false});

    Eigen::Tensor<int, 3, Eigen::RowMajor> padLGt = rgb.slice(Eigen::array<Index, D>{0, 1, 0}, Eigen::array<Index, D>{H, pW, D});

    ASSERT_EQ(padLGt.dimension(0), padL.dimension(0));
    ASSERT_EQ(padLGt.dimension(1), padL.dimension(1));
    ASSERT_EQ(padLGt.dimension(2), padL.dimension(2));

    for (Index r = 0; r < padL.dimension(0); r++)
        for (Index c = 0; c < padL.dimension(1); c++)
            for (Index d = 0; d < padL.dimension(2); d++)
                ASSERT_EQ(padL(r, c, d), padLGt(r, c, d));

    Eigen::Tensor<int, 3, Eigen::RowMajor> padR = \ 
        paddedRGB.slice(Eigen::array<Index, D>{pH, newW - pW, 0}, Eigen::array<Index, D>{H, pW, D})
                                                      .reverse(Eigen::array<Index, D>{false, true, false});

    Eigen::Tensor<int, 3, Eigen::RowMajor> padRGt = rgb.slice(Eigen::array<Index, D>{0, W - pW - 1, 0}, Eigen::array<Index, D>{H, pW, D});

    ASSERT_EQ(padRGt.dimension(0), padR.dimension(0));
    ASSERT_EQ(padRGt.dimension(1), padR.dimension(1));
    ASSERT_EQ(padRGt.dimension(2), padR.dimension(2));

    for (Index r = 0; r < padR.dimension(0); r++)
        for (Index c = 0; c < padR.dimension(1); c++)
            for (Index d = 0; d < padR.dimension(2); d++)
                ASSERT_EQ(padR(r, c, d), padRGt(r, c, d));
}
