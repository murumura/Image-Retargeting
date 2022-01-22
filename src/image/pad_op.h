#include <stdexcept>
#include <string>

namespace Image {

    enum class PadMode {
        CONSTANT = 0, // pad constant values, with string "constant"
        REFLECT = 1, // pads with reflect values, with string "reflect"
        EDGE = 2, // pads with the edge values, with string "edge"
    };

    PadMode stringToPadMode(const std::string& mode)
    {
        if (mode == "constant") {
            return PadMode::CONSTANT;
        }
        else if (mode == "reflect") {
            return PadMode::REFLECT;
        }
        else if (mode == "edge") {
            return PadMode::EDGE;
        }
        else {
            throw std::invalid_argument("Unknown padding mode: " + mode);
        }
    }

    /* C C |3 1 2| C C */
    template <typename Scalar>
    struct padConstant {
    private:
        Scalar padValue;
        int padWidth, padHeight;

    public:
        padConstant() : padValue(0), padWidth{0}, padHeight{0} {}

        padConstant(Scalar value, int padHeight_, int padWidth_) : padValue(value), padWidth{padWidth_}, padHeight{padHeight_} {}

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

    public:
        padReflect() : padWidth{0}, padHeight{0} {}

        padReflect(int padHeight_, int padWidth_) : padWidth{padWidth_}, padHeight{padHeight_} {}

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

    public:
        padEdge() : padWidth{0}, padHeight{0} {}

        padEdge(int padHeight_, int padWidth_) : padWidth{padWidth_}, padHeight{padHeight_} {}

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

} // namespace Image
