#include <cstdlib>
#include <gtest/gtest.h>
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
    rgbToGrayFunctor<int> functor;
    Eigen::Tensor<int, 3, Eigen::RowMajor> gray = functor(rgb);
    EXPECT_EQ(gray.dimension(0), 2);
    EXPECT_EQ(gray.dimension(1), 2);
    EXPECT_EQ(gray.dimension(2), 1);
}

TEST(Image, Panda)
{
    Uint8Image pandaRGB = loadPNG<uint8_t>("./test/test_image/panda.png", 3);
    EXPECT_EQ(pandaRGB.dimension(0), 800);
    EXPECT_EQ(pandaRGB.dimension(1), 600);
    EXPECT_EQ(pandaRGB.dimension(2), 3);

    rgbToGrayFunctor<uint8_t> functor;
    Uint8Image pandaGray = functor(pandaRGB);

    EXPECT_EQ(pandaGray.dimension(0), 800);
    EXPECT_EQ(pandaGray.dimension(1), 600);
    EXPECT_EQ(pandaGray.dimension(2), 1);

    savePNG("./lenaPanda", pandaGray);
    EXPECT_EQ(0, remove("./lenaPanda.png"));
}

TEST(Image, Lena)
{
    Uint8Image lenaRGB = loadPNG<uint8_t>("./test/test_image/lena256.png", 3);
    EXPECT_EQ(lenaRGB.dimension(0), 256);
    EXPECT_EQ(lenaRGB.dimension(1), 256);
    EXPECT_EQ(lenaRGB.dimension(2), 3);

    rgbToGrayFunctor<uint8_t> functor;
    Uint8Image lenaGray = functor(lenaRGB);
    EXPECT_EQ(lenaGray.dimension(0), 256);
    EXPECT_EQ(lenaGray.dimension(1), 256);
    EXPECT_EQ(lenaGray.dimension(2), 1);

    savePNG("./lenaGray", lenaGray);
    EXPECT_EQ(0, remove("./lenaGray.png"));
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
    Eigen::Tensor<int, 3, Eigen::RowMajor> paddedRGB = rgb.customOp(padConstant<int>(pH, pW, padVal));

    EXPECT_EQ(paddedRGB.dimension(0), H + pH * 2);
    EXPECT_EQ(paddedRGB.dimension(1), W + pW * 2);
    EXPECT_EQ(paddedRGB.dimension(2), D);
    const int newH = H + 2 * pH;
    const int newW = W + 2 * pW;
    // check padding equal
    Eigen::Tensor<int, 3, Eigen::RowMajor> padU = \ 
        paddedRGB.slice(Eigen::array<Index, D>{0, 0, 0}, Eigen::array<Index, D>{pH, newW, D});

    EXPECT_EQ(pH, padU.dimension(0));
    EXPECT_EQ(newW, padU.dimension(1));
    EXPECT_EQ(D, padU.dimension(2));

    for (Index r = 0; r < padU.dimension(0); r++)
        for (Index c = 0; c < padU.dimension(1); c++)
            for (Index d = 0; d < padU.dimension(2); d++)
                EXPECT_EQ(padU(r, c, d), padVal);

    Eigen::Tensor<int, 3, Eigen::RowMajor> padD = \ 
        paddedRGB.slice(Eigen::array<Index, D>{newH - pH, 0, 0}, Eigen::array<Index, D>{pH, newW, D});

    EXPECT_EQ(pH, padD.dimension(0));
    EXPECT_EQ(newW, padD.dimension(1));
    EXPECT_EQ(D, padD.dimension(2));

    for (Index r = 0; r < padD.dimension(0); r++)
        for (Index c = 0; c < padD.dimension(1); c++)
            for (Index d = 0; d < padD.dimension(2); d++)
                EXPECT_EQ(padD(r, c, d), padVal);

    Eigen::Tensor<int, 3, Eigen::RowMajor> padL = \ 
        paddedRGB.slice(Eigen::array<Index, D>{0, 0, 0}, Eigen::array<Index, D>{newH, pW, D});

    EXPECT_EQ(newH, padL.dimension(0));
    EXPECT_EQ(pW, padL.dimension(1));
    EXPECT_EQ(D, padL.dimension(2));

    for (Index r = 0; r < padL.dimension(0); r++)
        for (Index c = 0; c < padL.dimension(1); c++)
            for (Index d = 0; d < padL.dimension(2); d++)
                EXPECT_EQ(padL(r, c, d), padVal);

    Eigen::Tensor<int, 3, Eigen::RowMajor> padR = \ 
        paddedRGB.slice(Eigen::array<Index, D>{0, newW - pW, 0}, Eigen::array<Index, D>{newH, pW, D});

    EXPECT_EQ(newH, padR.dimension(0));
    EXPECT_EQ(pW, padR.dimension(1));
    EXPECT_EQ(D, padR.dimension(2));

    for (Index r = 0; r < padR.dimension(0); r++)
        for (Index c = 0; c < padR.dimension(1); c++)
            for (Index d = 0; d < padR.dimension(2); d++)
                EXPECT_EQ(padR(r, c, d), padVal);
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

    EXPECT_EQ(paddedRGB.dimension(0), H + 2 * pH);
    EXPECT_EQ(paddedRGB.dimension(1), W + 2 * pW);
    EXPECT_EQ(paddedRGB.dimension(2), D);

    const int newH = H + 2 * pH;
    const int newW = W + 2 * pW;

    // check padding equal
    Eigen::Tensor<int, 3, Eigen::RowMajor> padU = \ 
        paddedRGB.slice(Eigen::array<Index, D>{0, pW, 0}, Eigen::array<Index, D>{pH, W, D})
                                                      .reverse(Eigen::array<Index, D>{true, false, false});

    Eigen::Tensor<int, 3, Eigen::RowMajor> padUGt = rgb.slice(Eigen::array<Index, D>{1, 0, 0}, Eigen::array<Index, D>{pH, W, D});

    EXPECT_EQ(padUGt.dimension(0), padU.dimension(0));
    EXPECT_EQ(padUGt.dimension(1), padU.dimension(1));
    EXPECT_EQ(padUGt.dimension(2), padU.dimension(2));

    for (Index r = 0; r < padU.dimension(0); r++)
        for (Index c = 0; c < padU.dimension(1); c++)
            for (Index d = 0; d < padU.dimension(2); d++)
                EXPECT_EQ(padU(r, c, d), padUGt(r, c, d));

    Eigen::Tensor<int, 3, Eigen::RowMajor> padD = \ 
        paddedRGB.slice(Eigen::array<Index, D>{newH - pH, pW, 0}, Eigen::array<Index, D>{pH, W, D})
                                                      .reverse(Eigen::array<Index, D>{true, false, false});

    Eigen::Tensor<int, 3, Eigen::RowMajor> padDGt = rgb.slice(Eigen::array<Index, D>{H - pH - 1, 0, 0}, Eigen::array<Index, D>{pH, W, D});

    EXPECT_EQ(padDGt.dimension(0), padD.dimension(0));
    EXPECT_EQ(padDGt.dimension(1), padD.dimension(1));
    EXPECT_EQ(padDGt.dimension(2), padD.dimension(2));

    for (Index r = 0; r < padD.dimension(0); r++)
        for (Index c = 0; c < padD.dimension(1); c++)
            for (Index d = 0; d < padD.dimension(2); d++)
                EXPECT_EQ(padD(r, c, d), padDGt(r, c, d));

    Eigen::Tensor<int, 3, Eigen::RowMajor> padL = \ 
        paddedRGB.slice(Eigen::array<Index, D>{pH, 0, 0}, Eigen::array<Index, D>{H, pW, D})
                                                      .reverse(Eigen::array<Index, D>{false, true, false});

    Eigen::Tensor<int, 3, Eigen::RowMajor> padLGt = rgb.slice(Eigen::array<Index, D>{0, 1, 0}, Eigen::array<Index, D>{H, pW, D});

    EXPECT_EQ(padLGt.dimension(0), padL.dimension(0));
    EXPECT_EQ(padLGt.dimension(1), padL.dimension(1));
    EXPECT_EQ(padLGt.dimension(2), padL.dimension(2));

    for (Index r = 0; r < padL.dimension(0); r++)
        for (Index c = 0; c < padL.dimension(1); c++)
            for (Index d = 0; d < padL.dimension(2); d++)
                EXPECT_EQ(padL(r, c, d), padLGt(r, c, d));

    Eigen::Tensor<int, 3, Eigen::RowMajor> padR = \ 
        paddedRGB.slice(Eigen::array<Index, D>{pH, newW - pW, 0}, Eigen::array<Index, D>{H, pW, D})
                                                      .reverse(Eigen::array<Index, D>{false, true, false});

    Eigen::Tensor<int, 3, Eigen::RowMajor> padRGt = rgb.slice(Eigen::array<Index, D>{0, W - pW - 1, 0}, Eigen::array<Index, D>{H, pW, D});

    EXPECT_EQ(padRGt.dimension(0), padR.dimension(0));
    EXPECT_EQ(padRGt.dimension(1), padR.dimension(1));
    EXPECT_EQ(padRGt.dimension(2), padR.dimension(2));

    for (Index r = 0; r < padR.dimension(0); r++)
        for (Index c = 0; c < padR.dimension(1); c++)
            for (Index d = 0; d < padR.dimension(2); d++)
                EXPECT_EQ(padR(r, c, d), padRGt(r, c, d));
}

TEST(Image, edge_padding)
{
    const int H = 41;
    const int W = 14;
    const int D = 3;
    Eigen::Tensor<int, 3, Eigen::RowMajor> rgb(H, W, D);
    rgb.setRandom();
    int pH = 3;
    int pW = 2;
    Eigen::Tensor<int, 3, Eigen::RowMajor> paddedRGB = rgb.customOp(padEdge<int>(pH, pW));

    EXPECT_EQ(paddedRGB.dimension(0), H + 2 * pH);
    EXPECT_EQ(paddedRGB.dimension(1), W + 2 * pW);
    EXPECT_EQ(paddedRGB.dimension(2), D);

    const int newH = H + 2 * pH;
    const int newW = W + 2 * pW;

    // check padding equal
    Eigen::Tensor<int, 3, Eigen::RowMajor> padU = \ 
        paddedRGB.slice(Eigen::array<Index, D>{0, pW, 0}, Eigen::array<Index, D>{pH, W, D});

    Eigen::Tensor<int, 3, Eigen::RowMajor> padUGt = rgb.slice(Eigen::array<Index, D>{0, 0, 0}, Eigen::array<Index, D>{1, W, D})
                                                        .broadcast(Eigen::array<Index, 3>{pH, 1, 1});

    EXPECT_EQ(padUGt.dimension(0), padU.dimension(0));
    EXPECT_EQ(padUGt.dimension(1), padU.dimension(1));
    EXPECT_EQ(padUGt.dimension(2), padU.dimension(2));

    for (Index r = 0; r < padU.dimension(0); r++)
        for (Index c = 0; c < padU.dimension(1); c++)
            for (Index d = 0; d < padU.dimension(2); d++)
                EXPECT_EQ(padU(r, c, d), padUGt(r, c, d));

    Eigen::Tensor<int, 3, Eigen::RowMajor> padD = \ 
        paddedRGB.slice(Eigen::array<Index, D>{newH - pH, pW, 0}, Eigen::array<Index, D>{pH, W, D});

    Eigen::Tensor<int, 3, Eigen::RowMajor> padDGt = rgb.slice(Eigen::array<Index, D>{H - 1, 0, 0}, Eigen::array<Index, D>{1, W, D})
                                                        .broadcast(Eigen::array<Index, 3>{pH, 1, 1});

    EXPECT_EQ(padDGt.dimension(0), padD.dimension(0));
    EXPECT_EQ(padDGt.dimension(1), padD.dimension(1));
    EXPECT_EQ(padDGt.dimension(2), padD.dimension(2));

    for (Index r = 0; r < padD.dimension(0); r++)
        for (Index c = 0; c < padD.dimension(1); c++)
            for (Index d = 0; d < padD.dimension(2); d++)
                EXPECT_EQ(padD(r, c, d), padDGt(r, c, d));

    Eigen::Tensor<int, 3, Eigen::RowMajor> padL = \ 
        paddedRGB.slice(Eigen::array<Index, D>{pH, 0, 0}, Eigen::array<Index, D>{H, pW, D});

    Eigen::Tensor<int, 3, Eigen::RowMajor> padLGt = rgb.slice(Eigen::array<Index, D>{0, 0, 0}, Eigen::array<Index, D>{H, 1, D})
                                                        .broadcast(Eigen::array<Index, 3>{1, pW, 1});

    EXPECT_EQ(padLGt.dimension(0), padL.dimension(0));
    EXPECT_EQ(padLGt.dimension(1), padL.dimension(1));
    EXPECT_EQ(padLGt.dimension(2), padL.dimension(2));

    for (Index r = 0; r < padL.dimension(0); r++)
        for (Index c = 0; c < padL.dimension(1); c++)
            for (Index d = 0; d < padL.dimension(2); d++)
                EXPECT_EQ(padL(r, c, d), padLGt(r, c, d));

    Eigen::Tensor<int, 3, Eigen::RowMajor> padR = \ 
        paddedRGB.slice(Eigen::array<Index, D>{pH, newW - pW, 0}, Eigen::array<Index, D>{H, pW, D});

    Eigen::Tensor<int, 3, Eigen::RowMajor> padRGt = rgb.slice(Eigen::array<Index, D>{0, W - 1, 0}, Eigen::array<Index, D>{H, 1, D})
                                                        .broadcast(Eigen::array<Index, 3>{1, pW, 1});

    EXPECT_EQ(padRGt.dimension(0), padR.dimension(0));
    EXPECT_EQ(padRGt.dimension(1), padR.dimension(1));
    EXPECT_EQ(padRGt.dimension(2), padR.dimension(2));

    for (Index r = 0; r < padR.dimension(0); r++)
        for (Index c = 0; c < padR.dimension(1); c++)
            for (Index d = 0; d < padR.dimension(2); d++)
                EXPECT_EQ(padR(r, c, d), padRGt(r, c, d));
}

TEST(Image, padding_constant_functor)
{
    const int H = 41;
    const int W = 14;
    const int D = 3;
    Eigen::Tensor<int, 3, Eigen::RowMajor> rgb(H, W, D);
    rgb.setRandom();
    int pH = 3;
    int pW = 2;
    const int padVal = -2;
    PadImageOp padOp = PadImageOp<int, PadMode::CONSTANT>(pH, pW, padVal);
    Eigen::Tensor<int, 3, Eigen::RowMajor> paddedRGB = padOp(rgb);
    EXPECT_EQ(paddedRGB.dimension(0), H + pH * 2);
    EXPECT_EQ(paddedRGB.dimension(1), W + pW * 2);
    EXPECT_EQ(paddedRGB.dimension(2), D);
    Eigen::Tensor<int, 3, Eigen::RowMajor> constantPaddedRGB = rgb.customOp(padConstant<int>(pH, pW, padVal));
    EXPECT_EQ(constantPaddedRGB.dimension(0), H + pH * 2);
    EXPECT_EQ(constantPaddedRGB.dimension(1), W + pW * 2);
    EXPECT_EQ(constantPaddedRGB.dimension(2), D);
    const int newH = H + 2 * pH;
    const int newW = W + 2 * pW;
    for (Index r = 0; r < paddedRGB.dimension(0); r++)
        for (Index c = 0; c < paddedRGB.dimension(1); c++)
            for (Index d = 0; d < paddedRGB.dimension(2); d++)
                EXPECT_EQ(paddedRGB(r, c, d), constantPaddedRGB(r, c, d));
}

TEST(Image, padding_refect_functor)
{
    const int H = 41;
    const int W = 14;
    const int D = 3;
    Eigen::Tensor<int, 3, Eigen::RowMajor> rgb(H, W, D);
    rgb.setRandom();
    int pH = 3;
    int pW = 2;
    PadImageOp padOp = PadImageOp<int, PadMode::REFLECT>(pH, pW);
    Eigen::Tensor<int, 3, Eigen::RowMajor> paddedRGB = padOp(rgb);
    EXPECT_EQ(paddedRGB.dimension(0), H + pH * 2);
    EXPECT_EQ(paddedRGB.dimension(1), W + pW * 2);
    EXPECT_EQ(paddedRGB.dimension(2), D);
    Eigen::Tensor<int, 3, Eigen::RowMajor> reflectPaddedRGB = rgb.customOp(padReflect<int>(pH, pW));
    EXPECT_EQ(reflectPaddedRGB.dimension(0), H + pH * 2);
    EXPECT_EQ(reflectPaddedRGB.dimension(1), W + pW * 2);
    EXPECT_EQ(reflectPaddedRGB.dimension(2), D);
    const int newH = H + 2 * pH;
    const int newW = W + 2 * pW;
    for (Index r = 0; r < paddedRGB.dimension(0); r++)
        for (Index c = 0; c < paddedRGB.dimension(1); c++)
            for (Index d = 0; d < paddedRGB.dimension(2); d++)
                EXPECT_EQ(paddedRGB(r, c, d), reflectPaddedRGB(r, c, d));
}

TEST(Image, padding_edge_functor)
{
    const int H = 41;
    const int W = 14;
    const int D = 3;
    Eigen::Tensor<int, 3, Eigen::RowMajor> rgb(H, W, D);
    rgb.setRandom();
    int pH = 3;
    int pW = 2;
    PadImageOp padOp = PadImageOp<int, PadMode::EDGE>(pH, pW);
    Eigen::Tensor<int, 3, Eigen::RowMajor> paddedRGB = padOp(rgb);
    EXPECT_EQ(paddedRGB.dimension(0), H + pH * 2);
    EXPECT_EQ(paddedRGB.dimension(1), W + pW * 2);
    EXPECT_EQ(paddedRGB.dimension(2), D);
    Eigen::Tensor<int, 3, Eigen::RowMajor> edgePaddedRGB = rgb.customOp(padEdge<int>(pH, pW));
    EXPECT_EQ(edgePaddedRGB.dimension(0), H + pH * 2);
    EXPECT_EQ(edgePaddedRGB.dimension(1), W + pW * 2);
    EXPECT_EQ(edgePaddedRGB.dimension(2), D);
    const int newH = H + 2 * pH;
    const int newW = W + 2 * pW;
    for (Index r = 0; r < paddedRGB.dimension(0); r++)
        for (Index c = 0; c < paddedRGB.dimension(1); c++)
            for (Index d = 0; d < paddedRGB.dimension(2); d++)
                EXPECT_EQ(paddedRGB(r, c, d), edgePaddedRGB(r, c, d));
}

TEST(Image, padding_constant_with_image)
{
    Uint8Image pandaRGB = loadPNG<uint8_t>("./test/test_image/panda.png", 3);
    EXPECT_EQ(pandaRGB.dimension(0), 800);
    EXPECT_EQ(pandaRGB.dimension(1), 600);
    EXPECT_EQ(pandaRGB.dimension(2), 3);

    int pH = 100;
    int pW = 10;
    const int padVal = 0;
    PadImageOp padOp = PadImageOp<uint8_t, PadMode::CONSTANT>(pH, pW, padVal);
    Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> paddedPanda = padOp(pandaRGB);

    EXPECT_EQ(paddedPanda.dimension(0), 800 + pH * 2);
    EXPECT_EQ(paddedPanda.dimension(1), 600 + pW * 2);
    EXPECT_EQ(paddedPanda.dimension(2), 3);

    savePNG("./paddedPanda", paddedPanda);

    // NOTE: uncomment following line to see the result!
    EXPECT_EQ(0, remove("./paddedPanda.png"));
}

TEST(Image, padding_reflect_with_image)
{
    Uint8Image pandaRGB = loadPNG<uint8_t>("./test/test_image/panda.png", 3);
    EXPECT_EQ(pandaRGB.dimension(0), 800);
    EXPECT_EQ(pandaRGB.dimension(1), 600);
    EXPECT_EQ(pandaRGB.dimension(2), 3);

    int pH = 100;
    int pW = 10;
    const int padVal = 0;
    PadImageOp padOp = PadImageOp<uint8_t, PadMode::REFLECT>(pH, pW);
    Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> paddedPanda = padOp(pandaRGB);

    EXPECT_EQ(paddedPanda.dimension(0), 800 + pH * 2);
    EXPECT_EQ(paddedPanda.dimension(1), 600 + pW * 2);
    EXPECT_EQ(paddedPanda.dimension(2), 3);

    savePNG("./paddedPanda", paddedPanda);

    // NOTE: uncomment following line to see the result!
    EXPECT_EQ(0, remove("./paddedPanda.png"));
}

TEST(Image, padding_edge_with_image)
{
    Uint8Image pandaRGB = loadPNG<uint8_t>("./test/test_image/panda.png", 3);
    EXPECT_EQ(pandaRGB.dimension(0), 800);
    EXPECT_EQ(pandaRGB.dimension(1), 600);
    EXPECT_EQ(pandaRGB.dimension(2), 3);

    int pH = 100;
    int pW = 50;
    const int padVal = 0;
    PadImageOp padOp = PadImageOp<uint8_t, PadMode::EDGE>(pH, pW);
    Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> paddedPanda = padOp(pandaRGB);

    EXPECT_EQ(paddedPanda.dimension(0), 800 + pH * 2);
    EXPECT_EQ(paddedPanda.dimension(1), 600 + pW * 2);
    EXPECT_EQ(paddedPanda.dimension(2), 3);

    savePNG("./paddedPanda", paddedPanda);

    // NOTE: uncomment following line to see the result!
    EXPECT_EQ(0, remove("./paddedPanda.png"));
}
