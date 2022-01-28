#include <gtest/gtest.h>
#include <image/padding_op.h>
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

TEST(ImagePadding, constant_padding_random)
{
    uint32_t runIter = 36;
    const int HWmin = 5;
    const int HWmax = 50;
    const int padValMax = 100;
    for (uint32_t i = 0; i < runIter; ++i) {
        const int H = randomValue<int>(HWmin, HWmax);
        const int W = randomValue<int>(HWmin, HWmax);
        const int D = randomValue<int>(1, 3);
        const int pL = randomValue<int>(W);
        const int pR = randomValue<int>(W);
        const int pT = randomValue<int>(H);
        const int pD = randomValue<int>(H);
        const int padVal = randomValue<int>(0, padValMax);
        Eigen::Tensor<int, 3, Eigen::RowMajor> rgb(H, W, D);
        rgb.setRandom();
        const int outH = H + pD + pT;
        const int outW = W + pL + pR;
        Eigen::Tensor<int, 3, Eigen::RowMajor> output;
        output.resize(outH, outW, D);
        Functor::ConstantPad<int>()(rgb, output, pT, pD, pL, pR, padVal);

        // check padding equal
        if (pT > 0) {
            Eigen::Tensor<int, 3, Eigen::RowMajor> padT = \ 
                    output.slice(Eigen::array<Index, 3>{0, 0, 0}, Eigen::array<Index, 3>{pT, outW, D});
            for (Index r = 0; r < padT.dimension(0); r++)
                for (Index c = 0; c < padT.dimension(1); c++)
                    for (Index d = 0; d < padT.dimension(2); d++)
                        EXPECT_EQ(padT(r, c, d), padVal);
        }

        if (pD > 0) {
            Eigen::Tensor<int, 3, Eigen::RowMajor> padD = output.slice(Eigen::array<Index, 3>{outH - pD, 0, 0}, Eigen::array<Index, 3>{pD, outW, D});

            for (Index r = 0; r < padD.dimension(0); r++)
                for (Index c = 0; c < padD.dimension(1); c++)
                    for (Index d = 0; d < padD.dimension(2); d++)
                        EXPECT_EQ(padD(r, c, d), padVal);
        }

        if (pL > 0) {
            Eigen::Tensor<int, 3, Eigen::RowMajor> padL = \ 
                output.slice(Eigen::array<Index, 3>{0, 0, 0}, Eigen::array<Index, 3>{outH, pL, D});

            for (Index r = 0; r < padL.dimension(0); r++)
                for (Index c = 0; c < padL.dimension(1); c++)
                    for (Index d = 0; d < padL.dimension(2); d++)
                        EXPECT_EQ(padL(r, c, d), padVal);
        }

        if (pR > 0) {
            Eigen::Tensor<int, 3, Eigen::RowMajor> padR = \ 
            output.slice(Eigen::array<Index, 3>{0, outW - pR, 0}, Eigen::array<Index, 3>{outH, pR, D});

            for (Index r = 0; r < padR.dimension(0); r++)
                for (Index c = 0; c < padR.dimension(1); c++)
                    for (Index d = 0; d < padR.dimension(2); d++)
                        EXPECT_EQ(padR(r, c, d), padVal);
        }
    }
}

TEST(ImagePadding, mirror_padding_random)
{
    uint32_t runIter = 36;
    const int HWmin = 5;
    const int HWmax = 50;

    for (uint32_t i = 0; i < runIter; ++i) {
        const int H = randomValue<int>(HWmin, HWmax);
        const int W = randomValue<int>(HWmin, HWmax);
        const int D = randomValue<int>(1, 3);
        const int offset = randomValue<int>(0, 1);
        const int pL = randomValue<int>(W);
        const int pR = randomValue<int>(W);
        const int pT = randomValue<int>(H);
        const int pD = randomValue<int>(H);

        Eigen::Tensor<int, 3, Eigen::RowMajor> rgb(H, W, D);
        rgb.setRandom();
        const int outH = H + pD + pT;
        const int outW = W + pL + pR;
        Eigen::Tensor<int, 3, Eigen::RowMajor> output;
        output.resize(outH, outW, D);
        Functor::MirrorPad<int>()(rgb, output, pT, pD, pL, pR, offset);
        // check padding equal
        if (pT > 0) {
            Eigen::Tensor<int, 3, Eigen::RowMajor> padT = output.slice(Eigen::array<Index, 3>{0, pL, 0}, Eigen::array<Index, 3>{pT, W, D})
                                                              .reverse(Eigen::array<Index, 3>{true, false, false});

            Eigen::Tensor<int, 3, Eigen::RowMajor> padTGt = rgb.slice(Eigen::array<Index, 3>{offset, 0, 0}, Eigen::array<Index, 3>{pT, W, D});

            EXPECT_EQ(padTGt.dimension(0), padT.dimension(0));
            EXPECT_EQ(padTGt.dimension(1), padT.dimension(1));
            EXPECT_EQ(padTGt.dimension(2), padT.dimension(2));

            for (Index r = 0; r < padT.dimension(0); r++)
                for (Index c = 0; c < padT.dimension(1); c++)
                    for (Index d = 0; d < padT.dimension(2); d++)
                        EXPECT_EQ(padT(r, c, d), padTGt(r, c, d));
        }

        if (pD > 0) {
            Eigen::Tensor<int, 3, Eigen::RowMajor> padD = output.slice(Eigen::array<Index, 3>{outH - pD, pL, 0}, Eigen::array<Index, 3>{pD, W, D})
                                                              .reverse(Eigen::array<Index, 3>{true, false, false});

            Eigen::Tensor<int, 3, Eigen::RowMajor> padDGt = rgb.slice(Eigen::array<Index, 3>{H - pD - offset, 0, 0}, Eigen::array<Index, 3>{pD, W, D});

            EXPECT_EQ(padDGt.dimension(0), padD.dimension(0));
            EXPECT_EQ(padDGt.dimension(1), padD.dimension(1));
            EXPECT_EQ(padDGt.dimension(2), padD.dimension(2));

            for (Index r = 0; r < padD.dimension(0); r++)
                for (Index c = 0; c < padD.dimension(1); c++)
                    for (Index d = 0; d < padD.dimension(2); d++)
                        EXPECT_EQ(padD(r, c, d), padDGt(r, c, d));
        }

        if (pL > 0) {
            Eigen::Tensor<int, 3, Eigen::RowMajor> padL = \ 
                output.slice(Eigen::array<Index, 3>{pT, 0, 0}, Eigen::array<Index, 3>{H, pL, D})
                                                              .reverse(Eigen::array<Index, 3>{false, true, false});

            Eigen::Tensor<int, 3, Eigen::RowMajor> padLGt = rgb.slice(Eigen::array<Index, 3>{0, offset, 0}, Eigen::array<Index, 3>{H, pL, D});

            EXPECT_EQ(padLGt.dimension(0), padL.dimension(0));
            EXPECT_EQ(padLGt.dimension(1), padL.dimension(1));
            EXPECT_EQ(padLGt.dimension(2), padL.dimension(2));

            for (Index r = 0; r < padL.dimension(0); r++)
                for (Index c = 0; c < padL.dimension(1); c++)
                    for (Index d = 0; d < padL.dimension(2); d++)
                        EXPECT_EQ(padL(r, c, d), padLGt(r, c, d));
        }

        if (pR > 0) {
            Eigen::Tensor<int, 3, Eigen::RowMajor> padR = \ 
            output.slice(Eigen::array<Index, 3>{pT, outW - pR, 0}, Eigen::array<Index, 3>{H, pR, D})
                                                              .reverse(Eigen::array<Index, 3>{false, true, false});

            Eigen::Tensor<int, 3, Eigen::RowMajor> padRGt = rgb.slice(Eigen::array<Index, 3>{0, W - pR - offset, 0}, Eigen::array<Index, 3>{H, pR, D});

            EXPECT_EQ(padRGt.dimension(0), padR.dimension(0));
            EXPECT_EQ(padRGt.dimension(1), padR.dimension(1));
            EXPECT_EQ(padRGt.dimension(2), padR.dimension(2));

            for (Index r = 0; r < padR.dimension(0); r++)
                for (Index c = 0; c < padR.dimension(1); c++)
                    for (Index d = 0; d < padR.dimension(2); d++)
                        EXPECT_EQ(padR(r, c, d), padRGt(r, c, d));
        }
    }
}

TEST(ImagePadding, edge_padding_random)
{
    uint32_t runIter = 36;
    const int HWmin = 5;
    const int HWmax = 50;

    for (uint32_t i = 0; i < runIter; ++i) {
        const int H = randomValue<int>(HWmin, HWmax);
        const int W = randomValue<int>(HWmin, HWmax);
        const int D = randomValue<int>(1, 3);
        const int pL = randomValue<int>(W);
        const int pR = randomValue<int>(W);
        const int pT = randomValue<int>(H);
        const int pD = randomValue<int>(H);

        Eigen::Tensor<int, 3, Eigen::RowMajor> rgb(H, W, D);
        rgb.setRandom();
        const int outH = H + pD + pT;
        const int outW = W + pL + pR;
        Eigen::Tensor<int, 3, Eigen::RowMajor> output;
        output.resize(outH, outW, D);
        Functor::EdgePad<int>()(rgb, output, pT, pD, pL, pR);
        // check padding equal
        if (pT > 0) {
            Eigen::Tensor<int, 3, Eigen::RowMajor> padT = output.slice(Eigen::array<Index, 3>{0, pL, 0}, Eigen::array<Index, 3>{pT, W, D});

            Eigen::Tensor<int, 3, Eigen::RowMajor> padTGt = rgb.slice(Eigen::array<Index, 3>{0, 0, 0}, Eigen::array<Index, 3>{1, W, D})
                                                                .broadcast(Eigen::array<Index, 3>{pT, 1, 1});
            EXPECT_EQ(padTGt.dimension(0), padT.dimension(0));
            EXPECT_EQ(padTGt.dimension(1), padT.dimension(1));
            EXPECT_EQ(padTGt.dimension(2), padT.dimension(2));

            for (Index r = 0; r < padT.dimension(0); r++)
                for (Index c = 0; c < padT.dimension(1); c++)
                    for (Index d = 0; d < padT.dimension(2); d++)
                        EXPECT_EQ(padT(r, c, d), padTGt(r, c, d));
        }

        if (pD > 0) {
            Eigen::Tensor<int, 3, Eigen::RowMajor> padD = output.slice(Eigen::array<Index, 3>{outH - pD, pL, 0}, Eigen::array<Index, 3>{pD, W, D});

            Eigen::Tensor<int, 3, Eigen::RowMajor> padDGt = rgb.slice(Eigen::array<Index, 3>{H - 1, 0, 0}, Eigen::array<Index, 3>{1, W, D})
                                                                .broadcast(Eigen::array<Index, 3>{pD, 1, 1});

            EXPECT_EQ(padDGt.dimension(0), padD.dimension(0));
            EXPECT_EQ(padDGt.dimension(1), padD.dimension(1));
            EXPECT_EQ(padDGt.dimension(2), padD.dimension(2));

            for (Index r = 0; r < padD.dimension(0); r++)
                for (Index c = 0; c < padD.dimension(1); c++)
                    for (Index d = 0; d < padD.dimension(2); d++)
                        EXPECT_EQ(padD(r, c, d), padDGt(r, c, d));
        }

        if (pL > 0) {
            Eigen::Tensor<int, 3, Eigen::RowMajor> padL = \ 
                output.slice(Eigen::array<Index, 3>{pT, 0, 0}, Eigen::array<Index, 3>{H, pL, D});

            Eigen::Tensor<int, 3, Eigen::RowMajor> padLGt = rgb.slice(Eigen::array<Index, 3>{0, 0, 0}, Eigen::array<Index, 3>{H, 1, D})
                                                                .broadcast(Eigen::array<Index, 3>{1, pL, 1});

            EXPECT_EQ(padLGt.dimension(0), padL.dimension(0));
            EXPECT_EQ(padLGt.dimension(1), padL.dimension(1));
            EXPECT_EQ(padLGt.dimension(2), padL.dimension(2));

            for (Index r = 0; r < padL.dimension(0); r++)
                for (Index c = 0; c < padL.dimension(1); c++)
                    for (Index d = 0; d < padL.dimension(2); d++)
                        EXPECT_EQ(padL(r, c, d), padLGt(r, c, d));
        }

        if (pR > 0) {
            Eigen::Tensor<int, 3, Eigen::RowMajor> padR = \ 
            output.slice(Eigen::array<Index, 3>{pT, outW - pR, 0}, Eigen::array<Index, 3>{H, pR, D});

            Eigen::Tensor<int, 3, Eigen::RowMajor> padRGt = rgb.slice(Eigen::array<Index, 3>{0, W - 1, 0}, Eigen::array<Index, 3>{H, 1, D})
                                                                .broadcast(Eigen::array<Index, 3>{1, pR, 1});

            EXPECT_EQ(padRGt.dimension(0), padR.dimension(0));
            EXPECT_EQ(padRGt.dimension(1), padR.dimension(1));
            EXPECT_EQ(padRGt.dimension(2), padR.dimension(2));

            for (Index r = 0; r < padR.dimension(0); r++)
                for (Index c = 0; c < padR.dimension(1); c++)
                    for (Index d = 0; d < padR.dimension(2); d++)
                        EXPECT_EQ(padR(r, c, d), padRGt(r, c, d));
        }
    }
}
