#ifndef IMAGE_H
#define IMAGE_H
#include <cassert>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>
namespace Image {
    /// < Alias for image type
    template <typename TColorDepth, int Rank>
    using ImageTemplate = Eigen::Tensor<TColorDepth, Rank, Eigen::RowMajor>;

    using Uint8Image = Eigen::Tensor<uint8_t, 3, Eigen::RowMajor>;

    /// < Alias for pixel type
    using Uint8Pixel = Eigen::Tensor<uint8_t, 1, Eigen::RowMajor>;

    using Index = Eigen::DenseIndex;
    using ImageDsizes = Eigen::DSizes<Index, 3>;

    template <typename TColorDepth, int Rank, typename Func>
    void forEachPixel(const Eigen::Tensor<TColorDepth, Rank, Eigen::RowMajor>& image, Func func)
    {
        for (Index d = 0; d < image.dimension(2); ++d)
            for (Index c = 0; c < image.dimension(1); ++c)
                for (Index r = 0; r < image.dimension(0); ++r)
                    func(image(r, c, d));
    }

    namespace Functor {

        template <typename SrcType, typename DstType, typename Device = Eigen::DefaultDevice>
        struct CastFunctor {
            void operator()(
                Eigen::Tensor<SrcType, 3> inTensor,
                Eigen::Tensor<DstType, 3> outTensor,
                const Device& device = Eigen::DefaultDevice()) const
            {
                outTensor.device(device) = inTensor.template cast<DstType>();
            }
        };

    } // namespace Functor

} // namespace Image

#endif
