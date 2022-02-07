#ifndef COLORSPACE_H
#define COLORSPACE_H
#include <array>
#include <image/image.h>
#include <map>
#include <type_traits>
namespace Image {
    namespace Functor {
        // clang-format off
        static std::map<
            std::string, 
            std::map<std::string, std::array<float, 3>>
        >
        illuminants
        {
            {"A", 
                {
                    {"2", {1.098466069456375, 1, 0.3558228003436005}}, 
                    {"10",{1.111420406956693, 1, 0.3519978321919493}}
                }
            },
            {"D50", 
                {
                    {"2", {0.9642119944211994, 1, 0.8251882845188288}}, 
                    {"10",{0.9672062750333777, 1, 0.8142801513128616}}
                }
            },
            {"D55", 
                {
                    {"2", {0.956797052643698, 1, 0.9214805860173273}}, 
                    {"10",{0.9579665682254781, 1, 0.9092525159847462}}
                }
            },
            {"D65", 
                {
                    {"2", {0.95047, 1.0, 1.08883}}, 
                    {"10",{0.94809667673716, 1, 1.0730513595166162}}
                }
            },
            {"D75", 
                {
                    {"2", {0.9497220898840717, 1, 1.226393520724154}}, 
                    {"10",{0.9441713925645873, 1, 1.2064272211720228}}
                }
            },
            {"E", 
                {
                    {"2", {1.0, 1.0, 1.0}}, 
                    {"10",{1.0, 1.0, 1.0}}
                }
            }
        };
        // clang-format on

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

        template <typename Scalar, typename Device = Eigen::DefaultDevice,
            typename = std::enable_if_t<std::is_floating_point_v<Scalar>>>
        struct RGBToXYZ {
            void operator()(
                const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& inputRGB,
                Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& outputXYZ,
                const Device& device = Eigen::DefaultDevice())
            {
                const int H = inputRGB.dimension(0);
                const int W = inputRGB.dimension(1);
                const int C = inputRGB.dimension(2);
                outputXYZ.resize(H, W, C);
                Eigen::Tensor<float, 2, Eigen::RowMajor> kernel(3, 3);
                kernel.setValues(
                    {{0.412453, 0.357580, 0.180423},
                        {0.212671, 0.715160, 0.072169},
                        {0.019334, 0.119193, 0.950227}});
                Eigen::Tensor<float, 3, Eigen::RowMajor> normalizedRGB = inputRGB.template cast<float>() / float(255.0);
                Eigen::Tensor<float, 3, Eigen::RowMajor> value(H, W, C);

                value = (normalizedRGB > float(0.04045)).select(((normalizedRGB + float(0.055)) / float(1.055)).pow(2.4), normalizedRGB / float(12.92));

                Eigen::array<Eigen::IndexPair<int>, 1> transposed_product_dims = {Eigen::IndexPair<int>(2, 1)};
                outputXYZ.device(device) = value.contract(kernel, transposed_product_dims) * float(255.0);
            }
        };

        template <typename Scalar, typename Device = Eigen::DefaultDevice,
            typename = std::enable_if_t<std::is_floating_point_v<Scalar>>>
        struct RGBToCIE {
            void operator()(
                const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& inputRGB,
                Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& outputCIE,
                const std::string& lluminant = "D65", const std::string& observer = "2",
                const Device& device = Eigen::DefaultDevice())
            {
                const int H = inputRGB.dimension(0);
                const int W = inputRGB.dimension(1);
                const int C = inputRGB.dimension(2);
                outputCIE.resize(H, W, C);
                std::array<float, 3> coords = illuminants[lluminant][observer];
                Eigen::Tensor<float, 3, Eigen::RowMajor> XYZ = RGBToXYZ(inputRGB.template cast<float>() / float(255.0));

                Eigen::array<Index, 3> offset = {H, W, 1};
                XYZ.slice(Eigen::array<Index, 3>{0, 0, 0}, offset) = (XYZ.slice(Eigen::array<Index, 3>{0, 0, 0}, offset) / coords[0]);
                XYZ.slice(Eigen::array<Index, 3>{0, 0, 1}, offset) = (XYZ.slice(Eigen::array<Index, 3>{0, 0, 1}, offset) / coords[1]);
                XYZ.slice(Eigen::array<Index, 3>{0, 0, 2}, offset) = (XYZ.slice(Eigen::array<Index, 3>{0, 0, 2}, offset) / coords[2]);

                // clang-format off
                XYZ = (XYZ > Scalar(0.008856)).select(
                    XYZ.pow(1.0 / 3.0), 
                    XYZ * Scalar(7.787) + Scalar(16.0) / Scalar(116.0)
                );
                // clang-format on

                auto X = XYZ.chip<2>(0);
                auto Y = XYZ.chip<2>(1);
                auto Z = XYZ.chip<2>(2);

                auto L = outputCIE.template chip<2>(0);
                auto A = outputCIE.template chip<2>(1);
                auto B = outputCIE.template chip<2>(2);

                // The L channel values are in the range 0..100. a and b are in the range -127..127.
                L.device(device) = Y * Scalar(116.0) - Scalar(16.0);
                A.device(device) = (X - Y) * Scalar(500.0);
                B.device(device) = (Y - Z) * Scalar(200.0);
            }

            Eigen::Tensor<float, 3, Eigen::RowMajor>
            RGBToXYZ(const Eigen::Tensor<float, 3, Eigen::RowMajor>& normalizedRGB)
            {
                Eigen::Tensor<float, 2, Eigen::RowMajor> kernel(3, 3);
                kernel.setValues(
                    {{0.412453, 0.357580, 0.180423},
                        {0.212671, 0.715160, 0.072169},
                        {0.019334, 0.119193, 0.950227}});
                Eigen::Tensor<float, 3, Eigen::RowMajor> value(normalizedRGB.dimension(0), normalizedRGB.dimension(1), normalizedRGB.dimension(2));

                // clang-format off
                value = \
                    (normalizedRGB > float(0.04045)).select(
                        ((normalizedRGB + float(0.055)) / float(1.055)).pow(2.4), 
                        normalizedRGB / float(12.92)
                    );
                // clang-format on
                Eigen::array<Eigen::IndexPair<int>, 1> transposed_product_dims = {Eigen::IndexPair<int>(2, 1)};
                return value.contract(kernel, transposed_product_dims);
            }
        };
    } // namespace Functor
} // namespace Image

#endif
