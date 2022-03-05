#ifndef RESIZING_OP
#define RESIZING_OP
#include <image/image.h>
#include <image/utils.h>
namespace Image {

    enum class InterpolationMode {
        NEAREST_NEIGHBOR = 0,
        BILINEAR = 1,
    };

    // Converts a string into the corresponding interpolation mode.
    // Invoke invalid argument exception if the string couldn't be converted.
    inline InterpolationMode stringToInterpolationMode(const std::string& mode)
    {
        const std::string lower_case = Utils::stringToLower(mode);
        if (lower_case == "bilinear") {
            return InterpolationMode::BILINEAR;
        }
        else if (lower_case == "nearest_neighbor") {
            return InterpolationMode::NEAREST_NEIGHBOR;
        }
        else {
            throw std::invalid_argument("Unknown interpolation mode: " + mode);
        }
    }

    // Compute the interpolation indices only once.
    struct CachedInterpolation {
        int lower; // Lower source index used in the interpolation
        int upper; // Upper source index used in the interpolation
        // 1-D linear interpolation scale (see:
        // https://en.wikipedia.org/wiki/Bilinear_interpolation)
        float lerp;
    };

    template <typename Scaler>
    inline void compute_interpolation_weights(const Scaler scaler,
        const int out_size,
        const int in_size,
        const float scale,
        CachedInterpolation* interpolation)
    {
        interpolation[out_size].lower = 0;
        interpolation[out_size].upper = 0;
        for (int i = out_size - 1; i >= 0; --i) {
            const float in = scaler(i, scale);
            const float in_f = std::floor(in);
            interpolation[i].lower = std::max(static_cast<int>(in_f), static_cast<int>(0));
            interpolation[i].upper = std::min(static_cast<int>(std::ceil(in)), in_size - 1);
            interpolation[i].lerp = in - in_f;
        }
    }

    /**
     * Computes the bilinear interpolation from the appropriate 4 float points
     * and the linear interpolation weights.
     */
    inline float compute_lerp(const float top_left, const float top_right,
        const float bottom_left, const float bottom_right,
        const float x_lerp, const float y_lerp)
    {
        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
        return top + (bottom - top) * y_lerp;
    }

    template <typename T>
    void ResizeLineChannels(const T* const ys_input_lower_ptr,
        const T* const ys_input_upper_ptr,
        const CachedInterpolation* const xs,
        const float ys_lerp, const int out_width,
        float* out_y, const int channels)
    {
        for (int x = 0; x < out_width; ++x) {
            const int xs_lower = xs[x].lower;
            const int xs_upper = xs[x].upper;
            const float xs_lerp = xs[x].lerp;

            for (int c = 0; c < channels; ++c) {
                const float top_left(ys_input_lower_ptr[xs_lower + c]);
                const float top_right(ys_input_lower_ptr[xs_upper + c]);
                const float bottom_left(ys_input_upper_ptr[xs_lower + c]);
                const float bottom_right(ys_input_upper_ptr[xs_upper + c]);

                out_y[x * channels + c] = compute_lerp(top_left, top_right, bottom_left,
                    bottom_right, xs_lerp, ys_lerp);
            }
        }
    }

    // batchify version
    template <typename T>
    void resizeImage(
        const Eigen::Tensor<T, 4, Eigen::RowMajor>& images,
        const int batch_size, const int in_height,
        const int in_width, const int out_height,
        const int out_width, const int channels,
        const std::vector<CachedInterpolation>& xs_vec,
        const std::vector<CachedInterpolation>& ys,
        Eigen::Tensor<T, 4, Eigen::RowMajor>& output)
    {
        const int in_row_size = in_width * channels;
        const int in_batch_num_values = in_height * in_row_size;
        const int out_row_size = out_width * channels;

        const T* input_b_ptr = images.data();
        const CachedInterpolation* xs = xs_vec.data();

        if (channels == 3) {
            float* output_y_ptr = output.data();
            for (int b = 0; b < batch_size; ++b) {
                for (int y = 0; y < out_height; ++y) {
                    const T* ys_input_lower_ptr = input_b_ptr + ys[y].lower * in_row_size;
                    const T* ys_input_upper_ptr = input_b_ptr + ys[y].upper * in_row_size;
                    ResizeLineChannels(ys_input_lower_ptr, ys_input_upper_ptr, xs,
                        ys[y].lerp, out_width, output_y_ptr, 3);
                    output_y_ptr += out_row_size;
                }
                input_b_ptr += in_batch_num_values;
            }
        }
        else {
            float* output_y_ptr = output.data();
            for (int b = 0; b < batch_size; ++b) {
                for (int y = 0; y < out_height; ++y) {
                    const T* ys_input_lower_ptr = input_b_ptr + ys[y].lower * in_row_size;
                    const T* ys_input_upper_ptr = input_b_ptr + ys[y].upper * in_row_size;

                    ResizeLineChannels(ys_input_lower_ptr, ys_input_upper_ptr, xs,
                        ys[y].lerp, out_width, output_y_ptr, channels);

                    output_y_ptr += out_row_size;
                }
                input_b_ptr += in_batch_num_values;
            }
        }
    }

    // CalculateResizeScale determines the float scaling factor.
    inline float CalculateResizeScale(int in_size, int out_size, bool align_corners)
    {
        return (align_corners && out_size > 1)
            ? (in_size - 1) / static_cast<float>(out_size - 1)
            : in_size / static_cast<float>(out_size);
    }

    // Half pixel scaler scales assuming that the pixel centers are at 0.5, i.e. the
    // floating point coordinates of the top,left pixel is 0.5,0.5.
    struct HalfPixelScaler {
        HalfPixelScaler(){};
        inline float operator()(const int x, const float scale) const
        {
            // Note that we subtract 0.5 from the return value, as the existing bilinear
            // sampling code etc assumes pixels are in the old coordinate system.
            return (static_cast<float>(x) + 0.5f) * scale - 0.5f;
        }
    };

    // Older incorrect scaling method that causes all resizes to have a slight
    // translation leading to inconsistent results. For example, a flip then a
    // resize gives different results then a resize then a flip.
    struct LegacyScaler {
        LegacyScaler(){};
        inline float operator()(const int x, const float scale) const
        {
            return static_cast<float>(x) * scale;
        }
    };

    struct HalfPixelScalerForNN {
        inline float operator()(const int x, const float scale) const
        {
            // All of the nearest neighbor code below immediately follows a call to this
            // function with a std::floor(), so instead of subtracting the 0.5 as we
            // do in HalfPixelScale, we leave it as is, as the std::floor does the
            // correct thing.
            return (static_cast<float>(x) + 0.5f) * scale;
        }
    };

    // Helper struct to convert a bool to the correct scaler type.
    template <bool half_pixel_centers>
    struct BoolToScaler {
    };

    template <>
    struct BoolToScaler<true> {
        typedef HalfPixelScalerForNN Scaler;
    };

    template <>
    struct BoolToScaler<false> {
        typedef LegacyScaler Scaler;
    };

    namespace Functor {

        template <typename Scalar, typename Device = Eigen::DefaultDevice>
        struct ResizeBilinear {
            void operator()(
                const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& image,
                const float height_scale, const float width_scale,
                const bool half_pixel_centers,
                Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& resized_image,
                const Device& device = Eigen::DefaultDevice())
            {
                const int H = image.dimension(0);
                const int W = image.dimension(1);
                const int C = image.dimension(2);
                const int outH = resized_image.dimension(0);
                const int outW = resized_image.dimension(1);
                const int outC = resized_image.dimension(2);
                Eigen::Tensor<Scalar, 4, Eigen::RowMajor> batchifyIn = image.reshape(Eigen::array<Index, 4>{1, H, W, C});

                Eigen::Tensor<Scalar, 4, Eigen::RowMajor> batchifyOut = resized_image.reshape(Eigen::array<Index, 4>{1, outH, outW, outC});

                this->operator()(batchifyIn, height_scale, width_scale, half_pixel_centers, batchifyOut, device);

                // remove dummy batch
                resized_image = batchifyOut.reshape(Eigen::array<Index, 3>{outH, outW, outC});
            }

            // batchify version
            void operator()(
                const Eigen::Tensor<Scalar, 4, Eigen::RowMajor>& images,
                const float height_scale, const float width_scale,
                const bool half_pixel_centers,
                Eigen::Tensor<Scalar, 4, Eigen::RowMajor>& resized_images,
                const Device& device = Eigen::DefaultDevice())
            {
                const int batch_size = images.dimension(0);
                const int in_height = images.dimension(1);
                const int in_width = images.dimension(2);
                const int channels = images.dimension(3);

                const int out_height = resized_images.dimension(1);
                const int out_width = resized_images.dimension(2);

                // Handle no-op resizes efficiently.
                if (out_height == in_height && out_width == in_width) {
                    resized_images = images.template cast<float>();
                    return;
                }

                std::vector<CachedInterpolation> ys(out_height + 1);
                std::vector<CachedInterpolation> xs(out_width + 1);

                // Compute the cached interpolation weights on the x and y dimensions.
                if (half_pixel_centers) {
                    compute_interpolation_weights(HalfPixelScaler(), out_height, in_height,
                        height_scale, ys.data());
                    compute_interpolation_weights(HalfPixelScaler(), out_width, in_width,
                        width_scale, xs.data());
                }
                else {
                    compute_interpolation_weights(LegacyScaler(), out_height, in_height,
                        height_scale, ys.data());
                    compute_interpolation_weights(LegacyScaler(), out_width, in_width,
                        width_scale, xs.data());
                }
                // Scale x interpolation weights to avoid a multiplication during iteration.
                for (int i = 0; i < xs.size(); ++i) {
                    xs[i].lower *= channels;
                    xs[i].upper *= channels;
                }

                resizeImage<Scalar>(images, batch_size, in_height, in_width, out_height,
                    out_width, channels, xs, ys, resized_images);
            }
        };

        template <typename Scalar, typename Device = Eigen::DefaultDevice>
        struct ResizeNearestNeighbor {
            void operator()(
                const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& image,
                const float height_scale, const float width_scale,
                const bool half_pixel_centers, const bool align_corners,
                Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& resized_image,
                const Device& device = Eigen::DefaultDevice())
            {
                const int H = image.dimension(0);
                const int W = image.dimension(1);
                const int C = image.dimension(2);
                const int outH = resized_image.dimension(0);
                const int outW = resized_image.dimension(1);
                const int outC = resized_image.dimension(2);
                Eigen::Tensor<Scalar, 4, Eigen::RowMajor> batchifyIn = image.reshape(Eigen::array<Index, 4>{1, H, W, C});

                Eigen::Tensor<Scalar, 4, Eigen::RowMajor> batchifyOut = resized_image.reshape(Eigen::array<Index, 4>{1, outH, outW, outC});

                this->operator()(batchifyIn, height_scale, width_scale, half_pixel_centers, align_corners, batchifyOut, device);

                // remove dummy batch
                resized_image = batchifyOut.reshape(Eigen::array<Index, 3>{outH, outW, outC});
            }

            // batchify version
            void operator()(
                const Eigen::Tensor<Scalar, 4, Eigen::RowMajor>& images,
                const float height_scale, const float width_scale,
                const bool half_pixel_centers, const bool align_corners,
                Eigen::Tensor<Scalar, 4, Eigen::RowMajor>& resized_images,
                const Device& device = Eigen::DefaultDevice())
            {
                const Eigen::Index batch_size = images.dimension(0);
                const Eigen::Index in_height = images.dimension(1);
                const Eigen::Index in_width = images.dimension(2);
                const Eigen::Index channels = images.dimension(3);
                const Eigen::Index out_height = resized_images.dimension(1);
                const Eigen::Index out_width = resized_images.dimension(2);
                resized_images.setZero();

                for (Eigen::Index b = 0; b < batch_size; ++b) {
                    for (Eigen::Index y = 0; y < out_height; ++y) {
                        Eigen::Index in_y, in_x;
                        if (half_pixel_centers) {
                            in_y = std::min(
                                (align_corners)
                                    ? static_cast<Eigen::Index>(std::roundf(HalfPixelScalerForNN()(y, height_scale)))
                                    : static_cast<Eigen::Index>(std::floor(HalfPixelScalerForNN()(y, height_scale))),
                                in_height - 1);
                            in_y = std::max(static_cast<Eigen::Index>(0), in_y);
                        }
                        else {
                            in_y = std::min(
                                (align_corners)
                                    ? static_cast<Eigen::Index>(std::roundf(LegacyScaler()(y, height_scale)))
                                    : static_cast<Eigen::Index>(std::floor(LegacyScaler()(y, height_scale))),
                                in_height - 1);
                        }

                        for (Eigen::Index x = 0; x < out_width; ++x) {
                            if (half_pixel_centers) {
                                in_x = std::min(
                                    (align_corners)
                                        ? static_cast<Eigen::Index>(std::roundf(HalfPixelScalerForNN()(x, width_scale)))
                                        : static_cast<Eigen::Index>(std::floor(HalfPixelScalerForNN()(x, width_scale))),
                                    in_width - 1);
                                in_x = std::max(static_cast<Eigen::Index>(0), in_x);
                            }
                            else {
                                in_x = std::min(
                                    (align_corners)
                                        ? static_cast<Eigen::Index>(std::roundf(LegacyScaler()(x, width_scale)))
                                        : static_cast<Eigen::Index>(std::floor(LegacyScaler()(x, width_scale))),
                                    in_width - 1);
                            }
                            std::copy_n(&images(b, in_y, in_x, 0), channels, &resized_images(b, y, x, 0));
                        }
                    }
                }
            }
        };

    } // namespace Functor

    /**
     * @brief   Resize images to size using nearest neighbor / bilinaer interpolation.
     *
     * @param[in] half_pixel_centers_  If true, assuming that the pixel centers are at 0.5.
     * @param[in] align_corners_  If true, the centers of the 4 corner pixels of the input and output tensors are aligned,
     *   preserving the values at the corner pixels.
     */
    template <typename Scalar, typename Device = Eigen::DefaultDevice>
    class ResizingImageOp {
    public:
        ResizingImageOp() = default;
        explicit ResizingImageOp(const std::string& interpolationMode,
            const bool half_pixel_centers_ = false,
            const bool align_corners_ = false)
        {
            mode = stringToInterpolationMode(interpolationMode);
            half_pixel_centers = half_pixel_centers_;
            align_corners = align_corners_;
        }
        template <typename Input, typename Output>
        void operator()(const Input& input, Output& output)
        {
            static_assert(Eigen::internal::traits<Input>::NumDimensions == 3 || Eigen::internal::traits<Input>::NumDimensions == 4,
                "Require Eigen Tensor of 3(single image)/4(batchify image) dimensions");
            const float height_scale = input.dimension(0) / static_cast<float>(output.dimension(0));
            const float width_scale = input.dimension(1) / static_cast<float>(output.dimension(1));
            if (mode == InterpolationMode::BILINEAR)
                Functor::ResizeBilinear<Scalar>()(input, height_scale, width_scale, half_pixel_centers, output);
            else if (mode == InterpolationMode::NEAREST_NEIGHBOR)
                Functor::ResizeNearestNeighbor<Scalar>()(input, height_scale, width_scale, half_pixel_centers, align_corners, output);
        }

    private:
        InterpolationMode mode;
        bool half_pixel_centers;
        bool align_corners;
    };

} // namespace Image
#endif
