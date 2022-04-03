#ifndef IMAGE_H
#define IMAGE_H
#include <cassert>
#include <string>

// These macros must be defined before eigen files are included.
#ifdef RESIZING_USE_CUDA
#define EIGEN_USE_GPU
#endif

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

    template <typename TColorDepth>
    void drawLine(Eigen::Tensor<TColorDepth, 3, Eigen::RowMajor>& image, int r1, int c1, int r2, int c2)
    {
        const int CHANNELS = image.dimension(2);
        const int H = image.dimension(0);
        const int W = image.dimension(1);
        // calcullate coefficients A,B,C of line
        // from equation Ax + By + C = 0
        int A = r2 - r1;
        int B = c1 - c2;
        float C = c2 * r1 - c1 * r2;
        float m;
        // make sure A is positive to utilize the functiom properly
        if (A < 0) {
            A = -A;
            B = -B;
            C = -C;
        }
        // calculate the slope of the line
        // check for division by zero
        if (B != 0)
            m = -A / B;
        // make sure you start drawing in the right direction
        std::tie(c1, c2) = std::minmax(c1, c2);
        std::tie(r1, r2) = std::minmax(r1, r2);

        std::function<float(float, float)> lineEquation = [&](float x, float y) {
            return A * x + B * y + C;
        };

        // vertical line
        if (B == 0) {
            for (int r_ = r1; r_ <= r2; r_++)
                for (int d = 0; d < CHANNELS; d++)
                    image(r_, c1, d) = 255.0;
        }
        else if (A == 0) {
            // horizontal line
            for (int c_ = c1; c_ <= c2; c_++)
                for (int d = 0; d < CHANNELS; d++)
                    image(r1, c_, d) = 255.0;
        }
        else if (0 < m < 1) { // slope between 0 and 1
            for (int c_ = c1; c_ <= c2; c_++) {
                for (int d = 0; d < CHANNELS; d++)
                    image(r1, c_, d) = 255.0;
                if (lineEquation(c_ + 1, (float)r1 + 0.5) > 0)
                    r1 = (r1 + 1 < H) ? r1 + 1 : r1;
            }
        }
        else if (m >= 1) { // slope greater than or equal to 1
            for (int r_ = r1; r_ <= r2; r_++) {
                for (int d = 0; d < CHANNELS; d++)
                    image(r_, c1, d) = 255.0;
                if (lineEquation((float)c1 + 0.5, r_ + 1) > 0)
                    c1 = (c1 + 1 < W) ? c1 + 1 : c1;
            }
        }
        else if (m <= -1) { // slope less then -1
            for (int r_ = r1; r_ <= r2; r_++) {
                for (int d = 0; d < CHANNELS; d++)
                    image(r_, c2, d) = 255.0;
                if (lineEquation((float)c2 - 0.5, r_ + 1) > 0)
                    c2 = (c2 - 1 >= 0) ? c2 - 1 : c2;
            }
        }
        else if (-1 < m < 0) { //slope between -1 and 0
            for (int c_ = c1; c_ <= c2; c_++) {
                for (int d = 0; d < CHANNELS; d++)
                    image(r2, c_, d) = 255.0;
                if (lineEquation(c_ + 1, (float)r2 - 0.5) > 0)
                    r2 = (r2 - 1 >= 0) ? r2 - 1 : r2;
            }
        }
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
