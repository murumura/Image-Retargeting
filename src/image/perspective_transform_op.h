#ifndef PERSPECTIVE_TRANSFORM_OP_H
#define PERSPECTIVE_TRANSFORM_OP_H
#include <image/image.h>
#include <image/utils.h>
namespace Image {

    enum Interpolation {
        NEAREST,
        BILINEAR
    };

    enum Mode {
        FILL_REFLECT,
        FILL_WRAP,
        FILL_CONSTANT,
        FILL_NEAREST
    };

    inline Interpolation stringToInterpolation(const std::string& mode)
    {
        const std::string lower_case = Utils::stringToLower(mode);
        if (lower_case == "nearest") {
            return Interpolation::NEAREST;
        }
        else if (lower_case == "bilinear") {
            return Interpolation::BILINEAR;
        }
        else {
            throw std::invalid_argument("Unknown interpolation mode: " + mode);
        }
    }

    inline Mode stringToPadding(const std::string& mode)
    {
        const std::string lower_case = Utils::stringToLower(mode);
        if (lower_case == "reflect") {
            return Mode::FILL_REFLECT;
        }
        else if (lower_case == "wrap") {
            return Mode::FILL_WRAP;
        }
        else if (lower_case == "constant") {
            return Mode::FILL_CONSTANT;
        }
        else if (lower_case == "nearest") {
            return Mode::FILL_NEAREST;
        }
        else {
            throw std::invalid_argument("Unknown padding mode: " + mode);
        }
    }

    template <Mode M>
    struct MapCoordinate {
        float operator()(const float out_coord, const Eigen::DenseIndex len);
    };

    template <>
    struct MapCoordinate<Mode::FILL_REFLECT> {
        inline float operator()(
            const float out_coord,
            const Eigen::DenseIndex len)
        {
            // Reflect [abcd] to [dcba|abcd|dcba].
            float in_coord = out_coord;
            if (in_coord < 0) {
                if (len <= 1) {
                    in_coord = 0;
                }
                else {
                    const Eigen::DenseIndex sz2 = 2 * len;
                    if (in_coord < sz2) {
                        in_coord = sz2 * static_cast<Eigen::DenseIndex>(-in_coord / sz2) + in_coord;
                    }
                    in_coord = (in_coord < -len) ? in_coord + sz2 : -in_coord - 1;
                }
            }
            else if (in_coord > len - 1) {
                if (len <= 1) {
                    in_coord = 0;
                }
                else {
                    const Eigen::DenseIndex sz2 = 2 * len;
                    in_coord -= sz2 * static_cast<Eigen::DenseIndex>(in_coord / sz2);
                    if (in_coord >= len) {
                        in_coord = sz2 - in_coord - 1;
                    }
                }
            }
            // clamp is necessary because when out_coord = 3.5 and len = 4,
            // in_coord = 3.5 and will be rounded to 4 in nearest interpolation.
            return Eigen::internal::scalar_clamp_op<float>(0.0f, len - 1)(in_coord);
        }
    };

    template <>
    struct MapCoordinate<Mode::FILL_WRAP> {
        inline float operator()(const float out_coord,
            const Eigen::DenseIndex len)
        {
            // Wrap [abcd] to [abcd|abcd|abcd].
            float in_coord = out_coord;
            if (in_coord < 0) {
                if (len <= 1) {
                    in_coord = 0;
                }
                else {
                    const Eigen::DenseIndex sz = len - 1;
                    in_coord += len * (static_cast<Eigen::DenseIndex>(-in_coord / sz) + 1);
                }
            }
            else if (in_coord > len - 1) {
                if (len <= 1) {
                    in_coord = 0;
                }
                else {
                    const Eigen::DenseIndex sz = len - 1;
                    in_coord -= len * static_cast<Eigen::DenseIndex>(in_coord / sz);
                }
            }
            // clamp is necessary because when out_coord = -0.5 and len = 4,
            // in_coord = 3.5 and will be rounded to 4 in nearest interpolation.
            return Eigen::internal::scalar_clamp_op<float>(0.0f, len - 1)(in_coord);
        }
    };

    template <>
    struct MapCoordinate<Mode::FILL_CONSTANT> {
        inline float operator()(const float out_coord,
            const Eigen::DenseIndex len)
        {
            return out_coord;
        }
    };

    template <>
    struct MapCoordinate<Mode::FILL_NEAREST> {
        inline float operator()(const float out_coord,
            const Eigen::DenseIndex len)
        {
            return Eigen::internal::scalar_clamp_op<float>(0.0f, len - 1)(out_coord);
        }
    };

    namespace Functor {

        template <typename T, Mode M>
        class ProjectiveGenerator {
        private:
            typedef Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> TransformsType;
            const Interpolation interpolation_;
            const T fill_value_;
            TransformsType transforms_;
            Eigen::Tensor<T, 3, Eigen::RowMajor> input_;

        public:
            inline ProjectiveGenerator(
                const Eigen::Tensor<T, 3, Eigen::RowMajor>& input,
                const TransformsType& transforms,
                const Interpolation interpolation, const T fill_value)
                : input_(input),
                  transforms_(transforms),
                  interpolation_(interpolation),
                  fill_value_(fill_value) {}

            inline T
            operator()(const Eigen::array<Eigen::DenseIndex, 3>& coords) const
            {
                const int64_t output_y = coords[0];
                const int64_t output_x = coords[1];
                const float* transform = transforms_.data();

                float projection = transform[6] * output_x + transform[7] * output_y + 1.f;

                if (projection == 0) {
                    // Return the fill value for infinite coordinates,
                    // which are outside the input image
                    return fill_value_;
                }

                const float input_x = (transform[0] * output_x + transform[1] * output_y + transform[2]) / projection;
                const float input_y = (transform[3] * output_x + transform[4] * output_y + transform[5]) / projection;

                // Map out-of-boundary input coordinates to in-boundary based on fill_mode.
                auto map_functor = MapCoordinate<M>();
                const float x = map_functor(input_x, input_.dimension(1));
                const float y = map_functor(input_y, input_.dimension(0));

                const Eigen::DenseIndex channels = coords[2];

                switch (interpolation_) {
                case NEAREST:
                    return nearest_interpolation(y, x, channels, fill_value_);
                case BILINEAR:
                    return bilinear_interpolation(y, x, channels, fill_value_);
                }
                // Unreachable; ImageProjectiveTransform only uses INTERPOLATION_NEAREST
                // or INTERPOLATION_BILINEAR.
                return fill_value_;
            }

            inline T
            nearest_interpolation(const float y, const float x, const Eigen::DenseIndex channel, const T fill_value) const
            {
                return read_with_fill_value(Eigen::DenseIndex(std::round(y)), Eigen::DenseIndex(std::round(x)), channel, fill_value);
            }

            inline T
            bilinear_interpolation(const float y, const float x, const Eigen::DenseIndex channel, const T fill_value) const
            {
                const float y_floor = std::floor(y);
                const float x_floor = std::floor(x);
                const float y_ceil = y_floor + 1;
                const float x_ceil = x_floor + 1;
                // clang-format off
                // f(x, y_floor) = (x_ceil - x) / (x_ceil - x_floor) * f(x_floor, y_floor)
                //               + (x - x_floor) / (x_ceil - x_floor) * f(x_ceil, y_floor)
                const float value_yfloor = \
                    (x_ceil - x) * static_cast<float>(read_with_fill_value(Eigen::DenseIndex(y_floor), Eigen::DenseIndex(x_floor), channel, fill_value)) + \
                    (x - x_floor) * static_cast<float>(read_with_fill_value(Eigen::DenseIndex(y_floor), Eigen::DenseIndex(x_ceil), channel, fill_value));

                // f(x, y_ceil) = (x_ceil - x) / (x_ceil - x_floor) * f(x_floor, y_ceil)
                //              + (x - x_floor) / (x_ceil - x_floor) * f(x_ceil, y_ceil)
                const float value_yceil = \
                    (x_ceil - x) * static_cast<float>(read_with_fill_value(Eigen::DenseIndex(y_ceil), Eigen::DenseIndex(x_floor), channel, fill_value)) + \
                    (x - x_floor) * static_cast<float>(read_with_fill_value(Eigen::DenseIndex(y_ceil), Eigen::DenseIndex(x_ceil), channel, fill_value));

                // f(x, y) = (y_ceil - y) / (y_ceil - y_floor) * f(x, y_floor)
                //         + (y - y_floor) / (y_ceil - y_floor) * f(x, y_ceil)
                return T((y_ceil - y) * value_yfloor + (y - y_floor) * value_yceil);
                // clang-format on
            }

            inline T read_with_fill_value(
                const Eigen::DenseIndex y, const Eigen::DenseIndex x,
                const Eigen::DenseIndex channel, const T fill_value) const
            {
                // channel must be correct, because they are passed unchanged from
                // the input.
                return (0 <= y && y < input_.dimension(0) && 0 <= x && x < input_.dimension(1))
                    ? input_(Eigen::array<Eigen::DenseIndex, 3>{y, x, channel})
                    : fill_value;
            }
        };
    } // namespace Functor

    template <typename T, typename Device = Eigen::DefaultDevice>
    class ImageProjectiveTransformOp final {
    private:
        Interpolation interpolation;
        Mode fill_mode;
        typedef Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> TransformsType;

    public:
        explicit ImageProjectiveTransformOp(const std::string& interpolationMode, const std::string& fillMode)
        {
            interpolation = stringToInterpolation(interpolationMode);
            fill_mode = stringToPadding(fillMode);
        }

        void operator()(
            const Eigen::Tensor<T, 3, Eigen::RowMajor>& input,
            Eigen::Tensor<T, 3, Eigen::RowMajor>& output,
            const TransformsType& transform, const T fill_value = 0)
        {
            const int out_height = output.dimension(0);
            const int out_width = output.dimension(1);
            const int out_channel = output.dimension(2);
            
            switch (fill_mode) {
            case Mode::FILL_REFLECT:
                output = output.generate(Image::Functor::ProjectiveGenerator<T, Mode::FILL_REFLECT>(
                    input, transform, interpolation, fill_value));
                break;
            case Mode::FILL_WRAP:
                output = output.generate(Image::Functor::ProjectiveGenerator<T, Mode::FILL_WRAP>(
                    input, transform, interpolation, fill_value));
                break;
            case Mode::FILL_CONSTANT:
                output = output.generate(
                    Image::Functor::ProjectiveGenerator<T, Mode::FILL_CONSTANT>(
                        input, transform, interpolation, fill_value));
                break;
            case Mode::FILL_NEAREST:
                output = output.generate(Image::Functor::ProjectiveGenerator<T, Mode::FILL_NEAREST>(
                    input, transform, interpolation, fill_value));
                break;
            }
        }
    };

} // namespace Image
#endif