#ifndef COLORSPACE_H
#define COLORSPACE_H
#include <image/image.h>
#include <type_traits>
namespace Image {
    namespace Functor {
        template <typename Scalar, typename Device = Eigen::DefaultDevice,
            typename = std::enable_if_t<std::is_floating_point_v<Scalar>>>
        struct RGBToHSV {
            void operator()(
                const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& inputRGB,
                Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& outputHSV,
                const Device& device = Eigen::DefaultDevice()) const
            {
                auto H = outputHSV.template chip<2>(0);
                auto S = outputHSV.template chip<2>(1);
                auto V = outputHSV.template chip<2>(2);

                auto R = inputRGB.template chip<2>(0);
                auto G = inputRGB.template chip<2>(1);
                auto B = inputRGB.template chip<2>(2);

                Eigen::Tensor<Scalar, 2, Eigen::RowMajor> range(inputRGB.dimension(0), inputRGB.dimension(1));
                Eigen::IndexList<Eigen::type2index<2>> channel_axis;

                V.device(device) = inputRGB.maximum(channel_axis);

                range.device(device) = V - inputRGB.minimum(channel_axis);

                S.device(device) = (V > Scalar(0)).select(range / V, V.constant(Scalar(0)));

                auto norm = range.inverse() * (Scalar(1) / Scalar(6));
                // TODO: all these assignments are only necessary because a combined
                // expression is larger than kernel parameter space. A custom kernel is
                // probably in order.
                H.device(device) = (R == V).select(
                    norm * (G - B), (G == V).select(norm * (B - R) + Scalar(2) / Scalar(6), norm * (R - G) + Scalar(4) / Scalar(6)));
                H.device(device) = (range > Scalar(0)).select(H, H.constant(Scalar(0)));
                H.device(device) = (H < Scalar(0)).select(H + Scalar(1), H);
            }
        };

        template <typename Scalar, typename Device = Eigen::DefaultDevice>
        struct RGBToGray {
            void operator()(
                const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& inputRGB,
                Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& outputGray,
                const Device& device = Eigen::DefaultDevice()) const
            {
                Eigen::array<Index, 3> offset = {inputRGB.dimension(0), inputRGB.dimension(1), 1};
                // clang-format off
                outputGray = 
                      (0.2126f * inputRGB.template cast<float>().slice(Eigen::array<Index, 3>{0, 0, 0}, offset) \
                     + 0.7152f * inputRGB.template cast<float>().slice(Eigen::array<Index, 3>{0, 0, 1}, offset) \
                     + 0.0722f * inputRGB.template cast<float>().slice(Eigen::array<Index, 3>{0, 0, 2}, offset))
                     .template cast<Scalar>();
                // clang-format on
            }
        };

        template <typename Scalar, typename Device = Eigen::DefaultDevice,
            typename = std::enable_if_t<std::is_floating_point_v<Scalar>>>
        struct HSVToRGB {
            void operator()(
                const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& inputHSV,
                Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& outputRGB,
                const Device& device = Eigen::DefaultDevice())
            {
                auto H = inputHSV.template chip<2>(0);
                auto S = inputHSV.template chip<2>(1);
                auto V = inputHSV.template chip<2>(2);

                // TODO compute only the fractional part of H for robustness
                auto dh = H * Scalar(6);
                auto dr = ((dh - Scalar(3)).abs() - Scalar(1)).cwiseMax(Scalar(0)).cwiseMin(Scalar(1));
                auto dg = (-(dh - Scalar(2)).abs() + Scalar(2)).cwiseMax(Scalar(0)).cwiseMin(Scalar(1));
                auto db = (-(dh - Scalar(4)).abs() + Scalar(2)).cwiseMax(Scalar(0)).cwiseMin(Scalar(1));
                auto one_s = -S + Scalar(1);

                auto R = outputRGB.template chip<2>(0);
                auto G = outputRGB.template chip<2>(1);
                auto B = outputRGB.template chip<2>(2);

                R.device(device) = (one_s + S * dr) * V;
                G.device(device) = (one_s + S * dg) * V;
                B.device(device) = (one_s + S * db) * V;
            }
        };

    } // namespace Functor
} // namespace Image

#endif
