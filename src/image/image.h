#ifndef IMAGE_H
#define IMAGE_H
#include <type_traits>
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>
#include <variant>

namespace Image {
    /// < Alias for image type
    template <typename TColorDepth, int Rank>
    using ImageTemplate = Eigen::Tensor<TColorDepth, Rank, Eigen::RowMajor>;

    using Uint8Image = Eigen::Tensor<uint8_t, 3, Eigen::RowMajor>;

    /// < Alias for pixel type
    using Uint8Pixel = Eigen::Tensor<uint8_t, 1, Eigen::RowMajor>;

    using Index = Eigen::DenseIndex;
    using ImageDsizes = Eigen::DSizes<Index, 3>;

    template <typename Scalar>
    struct rgbToGray {
        ImageDsizes dimensions(const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& rgbImage) const
        {
            ImageDsizes dims = rgbImage.dimensions();
            dims[0] = rgbImage.dimension(0);
            dims[1] = rgbImage.dimension(1);
            dims[2] = 1;
            return dims;
        }
        template <typename Output, typename Device>
        void eval(
            const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& rgbImage,
            Output& output,
            const Device& device) const
        {
            const Index height = rgbImage.dimension(0);
            const Index width = rgbImage.dimension(1);
            Eigen::array<Index, 3> offset = {height, width, 1};
            // clang-format off
            output =  (0.2126f * rgbImage.template cast<float>().slice(Eigen::array<Index, 3>{0, 0, 0}, offset) \
                     + 0.7152f * rgbImage.template cast<float>().slice(Eigen::array<Index, 3>{0, 0, 1}, offset) \
                     + 0.0722f * rgbImage.template cast<float>().slice(Eigen::array<Index, 3>{0, 0, 2}, offset)).template cast<Scalar>();
            // clang-format on
        }
    };

    // Functor used by rgbToGray to do the computations.
    template <typename Scalar>
    struct rgbToGrayFunctor {
        auto operator()(const Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& rgbImage)
        {
            return rgbImage.customOp(rgbToGray<Scalar>());
        }
    };

    template <typename TColorDepth, int Rank, typename Func>
    void forEachPixel(const Eigen::Tensor<TColorDepth, Rank, Eigen::RowMajor>& image, Func func)
    {
        for (Index d = 0; d < image.dimension(2); ++d)
            for (Index c = 0; c < image.dimension(1); ++c)
                for (Index r = 0; r < image.dimension(0); ++r)
                    func(image(r, c, d));
    }
} // namespace Image

#endif
