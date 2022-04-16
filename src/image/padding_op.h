#ifndef PADOP_H
#define PADOP_H
#include <algorithm>
#include <cctype>
#include <image/image.h>
#include <image/utils.h>
#include <istream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <variant>
namespace Image {

    enum class PadMode {
        CONSTANT = 0, ///< pad constant values, with string "constant"
        REFLECT = 1, ///< pads with reflect values, with string "reflect" which not replicate edge value
        SYMMETRIC = 2, ///< pads with reflect values, with string "reflect" which replicate edge value
        EDGE = 3, ///<pads with the edge values, with string "edge"
        VALID = 4, ///< no padding perform
    };

    // Converts a string into the corresponding padding mode.
    // Invoke invalid argument exception if the string couldn't be converted.
    inline PadMode stringToPadMode(const std::string& mode)
    {
        const std::string lower_case = Utils::stringToLower(mode);
        if (lower_case == "constant") {
            return PadMode::CONSTANT;
        }
        else if (lower_case == "reflect") {
            return PadMode::REFLECT;
        }
        else if (lower_case == "symmetric") {
            return PadMode::SYMMETRIC;
        }
        else if (lower_case == "edge") {
            return PadMode::EDGE;
        }
        else if (lower_case == "valid") {
            return PadMode::VALID;
        }
        else {
            throw std::invalid_argument("Unknown padding mode: " + mode);
        }
    }

    namespace Functor {
        template <typename Scalar, typename Device = Eigen::DefaultDevice>
        struct ConstantPad {
            void operator()(
                const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& input,
                Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& output,
                Index pad_t, Index pad_d, Index pad_l, Index pad_r,
                Scalar padding_value = 0,
                const Device& device = Eigen::DefaultDevice()) const
            {
                Index outH = output.dimension(0);
                Index outW = output.dimension(1);
                Index inH = input.dimension(0);
                Index inW = input.dimension(1);
                Index dim = input.dimension(2);

                auto lr_paddings = std::make_pair(pad_l, pad_r);
                auto td_paddings = std::make_pair(pad_t, pad_d);
                auto channels_paddings = std::make_pair(0, 0);
                Eigen::array<std::pair<Index, Index>, 3> padding{td_paddings, lr_paddings, channels_paddings};
                output.device(device) = input.template pad(padding, padding_value);
            }
        };

        template <typename Scalar, typename Device = Eigen::DefaultDevice>
        struct EdgePad {
            void operator()(
                const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& input,
                Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& output,
                Index pad_t, Index pad_d, Index pad_l, Index pad_r,
                const Device& device = Eigen::DefaultDevice()) const
            {
                Index outH = output.dimension(0);
                Index outW = output.dimension(1);
                Index inH = input.dimension(0);
                Index inW = input.dimension(1);
                Index dim = input.dimension(2);

                Eigen::array<Index, 3> lr_rhsExtents = {inH, 1, dim};
                Eigen::array<Index, 3> td_rhsExtents = {1, outW, dim};

                Eigen::array<Index, 3> l_rhsOffsets = {pad_t, pad_l, 0};
                Eigen::array<Index, 3> l_lhsOffsets = {pad_t, 0, 0};
                Eigen::array<Index, 3> l_lhsExtents = {inH, pad_l, dim};
                Eigen::array<Index, 3> l_broadcast = {1, pad_l, 1};

                Eigen::array<Index, 3> r_rhsOffsets = {pad_t, outW - pad_r - 1, 0};
                Eigen::array<Index, 3> r_lhsOffsets = {pad_t, outW - pad_r, 0};
                Eigen::array<Index, 3> r_lhsExtents = {inH, pad_r, dim};
                Eigen::array<Index, 3> r_broadcast = {1, pad_r, 1};

                Eigen::array<Index, 3> t_rhsOffsets = {pad_t, 0, 0};
                Eigen::array<Index, 3> t_lhsOffsets = {0, 0, 0};
                Eigen::array<Index, 3> t_lhsExtents = {pad_t, outW, dim};
                Eigen::array<Index, 3> t_broadcast = {pad_t, 1, 1};

                Eigen::array<Index, 3> d_rhsOffsets = {outH - pad_d - 1, 0, 0};
                Eigen::array<Index, 3> d_lhsOffsets = {outH - pad_d, 0, 0};
                Eigen::array<Index, 3> d_lhsExtents = {pad_d, outW, dim};
                Eigen::array<Index, 3> d_broadcast = {pad_d, 1, 1};

                // populate input from top-left output
                output.template slice(Eigen::array<Index, 3>{pad_t, pad_l, 0}, Eigen::array<Index, 3>{inH, inW, dim}).device(device) = input;

                if (pad_l > 0) {
                    output.template slice(l_lhsOffsets, l_lhsExtents).device(device) = output.template slice(l_rhsOffsets, lr_rhsExtents)
                                                                                           .template broadcast(l_broadcast);
                }

                if (pad_r > 0) {
                    output.template slice(r_lhsOffsets, r_lhsExtents).device(device) = output.template slice(r_rhsOffsets, lr_rhsExtents)
                                                                                           .template broadcast(r_broadcast);
                }

                if (pad_t > 0) {
                    output.template slice(t_lhsOffsets, t_lhsExtents).device(device) = output.template slice(t_rhsOffsets, td_rhsExtents)
                                                                                           .template broadcast(t_broadcast);
                }

                if (pad_d > 0) {
                    output.template slice(d_lhsOffsets, d_lhsExtents).device(device) = output.template slice(d_rhsOffsets, td_rhsExtents)
                                                                                           .template broadcast(d_broadcast);
                }
            }
        };

        // offset argument must be either 0 or 1. This controls whether the boundary
        // values are replicated (offset == 0)(symmetric) or not replicated (offset == 1)(reflect).
        template <typename Scalar, typename Device = Eigen::DefaultDevice>
        struct MirrorPad {
            void operator()(
                const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& input,
                Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& output,
                Index pad_t, Index pad_d, Index pad_l, Index pad_r,
                Index offset = 1, const Device& device = Eigen::DefaultDevice()) const
            {
                Index outH = output.dimension(0);
                Index outW = output.dimension(1);
                Index inH = input.dimension(0);
                Index inW = input.dimension(1);
                Index dim = input.dimension(2);

                Eigen::array<Index, 3> l_rhsOffsets = {pad_t, pad_l + offset, 0};
                Eigen::array<Index, 3> l_lhsOffsets = {pad_t, 0, 0};
                Eigen::array<Index, 3> l_extents = {inH, pad_l, dim};

                Eigen::array<Index, 3> r_rhsOffsets = {pad_t, outW - 2 * pad_r - offset, 0};
                Eigen::array<Index, 3> r_lhsOffsets = {pad_t, outW - pad_r, 0};
                Eigen::array<Index, 3> r_extents = {inH, pad_r, dim};

                Eigen::array<Index, 3> t_rhsOffsets = {pad_t + offset, 0, 0};
                Eigen::array<Index, 3> t_lhsOffsets = {0, 0, 0};
                Eigen::array<Index, 3> t_extents = {pad_t, outW, dim};

                Eigen::array<Index, 3> d_rhsOffsets = {outH - 2 * pad_d - offset, 0, 0};
                Eigen::array<Index, 3> d_lhsOffsets = {outH - pad_d, 0, 0};
                Eigen::array<Index, 3> d_extents = {pad_d, outW, dim};

                Eigen::array<bool, 3> lr_reverse = {false, true, false};
                Eigen::array<bool, 3> td_reverse = {true, false, false};

                // populate input from top-left output
                output.template slice(Eigen::array<Index, 3>{pad_t, pad_l, 0}, Eigen::array<Index, 3>{inH, inW, dim}).device(device) = input;

                if (pad_l > 0) {
                    output.template slice(l_lhsOffsets, l_extents).device(device) = output.template slice(l_rhsOffsets, l_extents).template reverse(lr_reverse);
                }

                if (pad_r > 0) {
                    output.template slice(r_lhsOffsets, r_extents).device(device) = output.template slice(r_rhsOffsets, r_extents).template reverse(lr_reverse);
                }

                if (pad_t > 0) {
                    output.template slice(t_lhsOffsets, t_extents).device(device) = output.template slice(t_rhsOffsets, t_extents).template reverse(td_reverse);
                }

                if (pad_d > 0) {
                    output.template slice(d_lhsOffsets, d_extents).device(device) = output.template slice(d_rhsOffsets, d_extents).template reverse(td_reverse);
                }
            }
        };
    } // namespace Functor

    template <typename Scalar, typename Device = Eigen::DefaultDevice>
    class PaddingImageOp final {
    public:
        explicit PaddingImageOp(const std::string& padMode)
        {
            switch (stringToPadMode(padMode)) {
            case PadMode::SYMMETRIC: {
                mode = PadMode::SYMMETRIC;
                offset = 0;
                break;
            }
            case PadMode::REFLECT: {
                mode = PadMode::REFLECT;
                offset = 1;
                break;
            }
            case PadMode::EDGE: {
                mode = PadMode::EDGE;
                offset = 0;
                break;
            }
            case PadMode::CONSTANT: {
                mode = PadMode::CONSTANT;
                break;
            }
            case PadMode::VALID: {
                mode = PadMode::VALID;
                break;
            }
            default:
                throw std::invalid_argument("Invalid padding mode" + padMode);
            }
        }

        ~PaddingImageOp() = default;

        void operator()(
            const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& input,
            Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& output,
            int pT, int pD, int pL, int pR, Scalar padValue = 0,
            const Device& device = Eigen::DefaultDevice()) const
        {
            output.resize(input.dimensions()[0] + pL + pR, input.dimensions()[1] + pT + pD, input.dimensions()[2]);
            if (mode == PadMode::VALID)
                output = input;
            else if (mode == PadMode::CONSTANT)
                Functor::ConstantPad<Scalar>()(input, output, pL, pR, pT, pD, padValue);
            else if (mode == PadMode::REFLECT || mode == PadMode::SYMMETRIC)
                Functor::MirrorPad<Scalar>()(input, output, pL, pR, pT, pD, offset);
            else
                Functor::EdgePad<Scalar>()(input, output, pL, pR, pT, pD);
        }

        void operator()(
            const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& input,
            Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& output,
            const std::variant<std::tuple<Index, Index>, Index>& paddingTD,
            const std::variant<std::tuple<Index, Index>, Index>& paddingLR,
            Scalar padValue = 0,
            const Device& device = Eigen::DefaultDevice()) const
        {
            Index pad_l, pad_r, pad_t, pad_d;
            if (std::holds_alternative<std::tuple<Index, Index>>(paddingLR)) {
                std::tie(pad_l, pad_r) = std::get<std::tuple<Index, Index>>(paddingLR);
            }
            else {
                pad_l = pad_r = std::get<Index>(paddingLR);
            }

            if (std::holds_alternative<std::tuple<Index, Index>>(paddingTD)) {
                std::tie(pad_t, pad_d) = std::get<std::tuple<Index, Index>>(paddingTD);
            }
            else {
                pad_t = pad_d = std::get<Index>(paddingTD);
            }
            this->operator()(input, output, pad_t, pad_d, pad_l, pad_r, padValue);
        }

    private:
        int offset;
        PadMode mode;
    };
} // namespace Image

#endif