#ifndef PADOP_H
#define PADOP_H
#include <stdexcept>
#include <string>
#include <variant>
namespace Image {

    enum class PadMode {
        CONSTANT = 0, // pad constant values, with string "constant"
        REFLECT = 1, // pads with reflect values, with string "reflect"
        SYMMETRIC = 2,
        EDGE = 3, // pads with the edge values, with string "edge"
    };

    /* C C |3 1 2| C C */
    template <typename Scalar>
    struct padConstant {
    private:
        Scalar padValue;
        int padWidth, padHeight;

    public:
        padConstant() : padValue(0), padWidth{0}, padHeight{0} {}

        padConstant(int padHeight_, int padWidth_, Scalar value = 0) : padValue(value), padWidth{padWidth_}, padHeight{padHeight_} {}

        ImageDsizes dimensions(const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& image) const
        {
            ImageDsizes dims = image.dimensions();
            dims[0] = image.dimension(0) + padHeight * 2;
            dims[1] = image.dimension(1) + padWidth * 2;
            dims[2] = image.dimension(2);
            return dims;
        }

        template <typename Output, typename Device = Eigen::DefaultDevice>
        void eval(
            const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& image,
            Output& output,
            const Device& device) const
        {
            auto heightPadding = std::make_pair(padHeight, padHeight);
            auto widthPadding = std::make_pair(padWidth, padWidth);
            auto channelPadding = std::make_pair(0, 0);
            Eigen::array<std::pair<int, int>, 3> padding{heightPadding, widthPadding, channelPadding};
            output.device(device) = image.template pad(padding, padValue);
        }
    };

    /* 2 1 |3 1 2| 1 3 */
    template <typename Scalar>
    struct padReflect {
    private:
        int padWidth, padHeight;
        Scalar padValue;

    public:
        padReflect() : padValue{0}, padWidth{0}, padHeight{0} {}

        padReflect(int padHeight_, int padWidth_, Scalar value = 0) : padWidth{padWidth_}, padHeight{padHeight_}, padValue{value} {}

        ImageDsizes dimensions(const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& image) const
        {
            ImageDsizes dims = image.dimensions();
            dims[0] = image.dimension(0) + padHeight * 2;
            dims[1] = image.dimension(1) + padWidth * 2;
            dims[2] = image.dimension(2);

            // TODO: Handle the case when padding height is greater than image height
            assert(padHeight < image.dimension(0));
            assert(padWidth < image.dimension(1));
            return dims;
        }

        template <typename Output, typename Device = Eigen::DefaultDevice>
        void eval(
            const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& image,
            Output& output,
            const Device& device) const
        {
            const Index height = image.dimension(0);
            const Index width = image.dimension(1);
            const Index channelNum = image.dimension(2);

            // clang-format off
            Eigen::array<Index, 3> offsetL = {0, 1, 0};
            Eigen::array<Index, 3> extentRL = {height, padWidth, channelNum};

            Eigen::Tensor<Scalar, 3, Eigen::RowMajor> paddingL = \
                image.template slice(offsetL, extentRL).eval()
                     .template reverse(Eigen::array<bool, 3>{false, true, false}).eval();

            Eigen::array<Index, 3> offsetR = {0, width - padWidth - 1, 0};
            Eigen::Tensor<Scalar, 3, Eigen::RowMajor> paddingR = \
                image.template slice(offsetR, extentRL).eval()
                     .template reverse(Eigen::array<bool, 3>{false, true, false}).eval();

            // intermediate result we need later
            Eigen::Tensor<Scalar, 3, Eigen::RowMajor> temp = \
                paddingL.template concatenate(image, 1).eval()
                        .template concatenate(paddingR, 1).eval();

            Eigen::array<Index, 3> offsetU = {1, 0, 0};
            Eigen::array<Index, 3> extentUD = {padHeight, width + 2 * padWidth, channelNum};

            Eigen::Tensor<Scalar, 3, Eigen::RowMajor> paddingU = \
                temp.template slice(offsetU, extentUD).eval()
                    .template reverse(Eigen::array<bool, 3>{true, false, false}).eval();

            Eigen::array<Index, 3> offsetD = {height - padHeight - 1, 0, 0};

            Eigen::Tensor<Scalar, 3, Eigen::RowMajor> paddingD = \
                temp.template slice(offsetD, extentUD).eval()
                    .template reverse(Eigen::array<bool, 3>{true, false, false}).eval();

            output.device(device) = \
                paddingU.template concatenate(temp, 0).eval()
                        .template concatenate(paddingD, 0);
            // clang-format on
        }
    };

    /* 3 3 |3 1 2| 2 2 */
    template <typename Scalar>
    struct padEdge {
    private:
        int padWidth, padHeight;
        Scalar padValue; ///< not used, for compatibility
    public:
        padEdge() : padValue{0}, padWidth{0}, padHeight{0} {}

        padEdge(int padHeight_, int padWidth_, Scalar value = 0) : padValue{value}, padWidth{padWidth_}, padHeight{padHeight_} {}

        ImageDsizes dimensions(const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& image) const
        {
            ImageDsizes dims = image.dimensions();
            dims[0] = image.dimension(0) + padHeight * 2;
            dims[1] = image.dimension(1) + padWidth * 2;
            dims[2] = image.dimension(2);
            return dims;
        }

        template <typename Output, typename Device = Eigen::DefaultDevice>
        void eval(
            const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& image,
            Output& output,
            const Device& device) const
        {

            const Index height = image.dimension(0);
            const Index width = image.dimension(1);
            const Index channelNum = image.dimension(2);

            // clang-format off
            Eigen::array<Index, 3> offsetL = {0, 0, 0};
            Eigen::array<Index, 3> extentRL = {height, 1, channelNum};

            Eigen::Tensor<Scalar, 3, Eigen::RowMajor> paddingL = \
                image.template slice(offsetL, extentRL).eval()
                     .template broadcast(Eigen::array<Index, 3>{1, padWidth, 1}).eval();

            Eigen::array<Index, 3> offsetR = {0, width - 1, 0};
            Eigen::Tensor<Scalar, 3, Eigen::RowMajor> paddingR = \
                image.template slice(offsetR, extentRL).eval()
                     .template broadcast(Eigen::array<Index, 3>{1, padWidth, 1}).eval();

            // intermediate result we need later
            Eigen::Tensor<Scalar, 3, Eigen::RowMajor> temp = \
                paddingL.template concatenate(image, 1).eval()
                        .template concatenate(paddingR, 1).eval();

            Eigen::array<Index, 3> offsetU = {0, 0, 0};
            Eigen::array<Index, 3> extentUD = {1, width + 2 * padWidth, channelNum};

            Eigen::Tensor<Scalar, 3, Eigen::RowMajor> paddingU = \
                temp.template slice(offsetU, extentUD).eval()
                    .template broadcast(Eigen::array<Index, 3>{padHeight, 1, 1}).eval();

            Eigen::array<Index, 3> offsetD = {height - 1, 0, 0};

            Eigen::Tensor<Scalar, 3, Eigen::RowMajor> paddingD = \
                temp.template slice(offsetD, extentUD).eval()
                    .template broadcast(Eigen::array<Index, 3>{padHeight, 1, 1}).eval();

            output.device(device) = \
                paddingU.template concatenate(temp, 0).eval()
                        .template concatenate(paddingD, 0);
            // clang-format on
        }
    };

    namespace Functor {

        // offset argument must be either 0 or 1. This controls whether the boundary
        // values are replicated (offset == 0)(symmetric) or not replicated (offset == 1)(reflect).
        template <typename T, typename Device = Eigen::DefaultDevice>
        struct MirrorPad {
            void operator()(
                const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& input,
                Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& output,
                const Eigen::Tensor<Index, 2, Eigen::RowMajor> paddings, int offset,
                const Device& device = Eigen::DefaultDevice())
            {

                Index pad_l = paddings(1, 0);
                Index pad_r = paddings(1, 1);
                Index pad_t = paddings(0, 0);
                Index pad_d = paddings(0, 1);

                Index outH = output.dimension(0);
                Index outW = output.dimension(1);
                Index inH = input.dimension(0);
                Index inW = input.dimension(1);

                Eigen::array<int32, 3> l_RhsOffsets = {pad_l + offset, pad_t, 0};
                Eigen::array<int32, 3> l_LhsOffsets = {pad_l, , 0};
                Eigen::array<int32, 3> rOffsets = {};
                Eigen::array<int32, 3> tdOffsets = {};
                Eigen::array<int32, 3> lrExtents ={0, inW, 3};
                Eigen::array<int32, 3> tdExtents ={};
                Eigen::array<bool, 3> lrReverse = {false, true, false};
                Eigen::array<bool, 3> tdReverse = {true, false, false};

                
                // populate input from top-left output
                output.template slice(
                    Eigen::array<Index, 3>{pad_t, pad_l, 0}, 
                    Eigne::array<Index, 3>{inH, inW, 3}
                ).device(device) = input;
                /*
                    padding strategy:
                    start from top-down
                        if reflect(offset=1):
                            (1) padding reverse of input from (offset, 0, 0) extend (pT, W, 3) to output's (0, pL, 0) extend (pT, W, 3)
                            (2) padding reverse of input from (H - pD - offset, 0, 0) extend (pD, W, 3) to output's (outH - pD, pL, 0) extend (pD, W, 3)
                            (3) [check pR > 0] padding reverse of output from (0, pL+offset, 0) extend (pL, outH, 3) to output's (0, 0, 0) extend (pR, outH, 3)
                            (4) [check pR > 0] padding reverse of output from (0, outW - 2 * pR - offset, 0) extend (pR, outH, 3) to output's (outH - pR, 0, 0) extend (pR 
               */
               
               
                
            }
        };
    } // namespace Functor

    template <typename Scalar, typename Device = Eigen::DefaultDevice>
    class PaddingImageOp final {
    public:
        explicit PaddingImageOp(std::string_view padMode, int offset_ = -1)
        {
            switch (padMode) {
            case "symmetric": {
                mode = PadMode::SYMMETRIC;
                offset = 0;
                break;
            }
            case "reflect": {
                mode = PadMode::REFLECT;
                offset = 1;
                break;
            }
            case "edge": {
                mode = PadMode::EDGE;
                offset = 0;
            }
            case "constant": {
                mode = PadMode::CONSTANT:
            }
            default:
                throw std::invalid_argument("Invalid padding mode" + padMode);
            }
        }

        ~PaddingImageOp() = default;

        void operator()(
            const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& input,
            Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& output,
            int pL, int pR, int pT, int pD,
            const Device& device = Eigen::DefaultDevice()) const
        {
            // Compute the shape of the output tensor, and allocate it.
            ImageDsizes outputShape;
            Eigen::Tensor<Index, 2, Eigen::RowMajor> paddings(2, 2);
            paddings(0, 0) = pL;
            paddings(0, 1) = pR;
            paddings(1, 0) = pT;
            paddings(1, 1) = pD;
            output.resize(input.dimensions()[0] + pL + pR, input.dimensions()[1] + pT + pD, 3);
            if (mode == PadMode::CONSTANT) {

            }
            else {
            }
        }

    private:
        int offset;
        PadMode mode;
    }
}; // namespace Image

} // namespace Image
#endif
