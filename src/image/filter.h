#ifndef FILTER_H
#define FILTER_H
#include <cmath>
#include <image/image.h>
#include <image/padding_op.h>
#include <image/utils.h>
namespace Image {
    template <typename InputScalar, typename Kernel>
    Eigen::Tensor<InputScalar, 3, Eigen::RowMajor>
    imageConvolution(
        const Eigen::Tensor<InputScalar, 3, Eigen::RowMajor>& image,
        const Kernel& kernel,
        const std::string& paddingMode,
        const InputScalar padValue = 0)
    {
        typedef typename Eigen::internal::traits<Kernel>::Scalar KScalar;

        // Number of filters to apply. This is the same as the output depth of the result
        const Index C = image.dimensions()[2];
        const Index inH = image.dimensions()[0];
        const Index inW = image.dimensions()[1];
        // Number of channels. This is the same as the input depth.
        const Index kC = kernel.dimensions()[2];
        const Index kW = kernel.dimensions()[1];
        const Index kH = kernel.dimensions()[0];

        const Index l_pad = std::ceil((kW - 1) * 0.5);
        const Index r_pad = (kW - 1) - l_pad;

        const Index t_pad = std::ceil((kH - 1) * 0.5);
        const Index d_pad = (kH - 1) - t_pad;

        const Index outH = inH + t_pad + d_pad;
        const Index outW = inW + l_pad + r_pad;

        Eigen::Tensor<KScalar, 3, Eigen::RowMajor> output(outH, outW, C);
        PaddingImageOp paddingOp = PaddingImageOp<KScalar>(paddingMode);

        // Padding output before convolution
        paddingOp(image.template cast<KScalar>(), output, t_pad, d_pad, l_pad, r_pad, static_cast<KScalar>(padValue));

        // Molds the output of the patch extraction code into a 2d tensor:
        // - the first dimension (dims[0]): everything else
        // - the second dimension (dims[1]): the patch values to be multiplied with the kernels
        Eigen::array<Index, 2> pre_contract_dims{inH * inW * C, kH * kW};

        Eigen::array<Index, 2> kernel_dims{kH * kW, kC};
        Eigen::array<Eigen::IndexPair<Index>, 1> contract_dims = {Eigen::IndexPair<Index>(1, 0)};
        Eigen::array<Index, 3> post_contract_dims{inH, inW, C};

        // 1. Use the Eigen method extract_image_patches to extract the perceptive fields of each kernel instance from the input tensor.
        // 2. Reshaping the extracted patches and your kernel tensor into 2D tensors.
        //  (This means that each kernel is a vertical column of the reshaped kernel tensor and each row of the reshaped image patches is each patch.)
        // 3. Perform a contraction which is actually a matrix multiplication of these two 2D tensors
        // 4. Reshape the result back into the correct dimensions to produce the output.
        // 5. Finally convert output back to input tensor's types before return.

        // Since We already doing padding on our own, just passing 'PADDING_VALID' and let the size shrink
        return output
            .template extract_image_patches(kH, kW, 1, 1, 1, 1, Eigen::PaddingType::PADDING_VALID)
            .template reshape(pre_contract_dims)
            .template contract(kernel.template reshape(kernel_dims), contract_dims)
            .template reshape(post_contract_dims)
            .template cast<InputScalar>();
    }

    // Defines functions for different types of sampling kernels.
    enum class KernelType {
        // Lanczos kernel with radius 1.  Aliases but does not ring.
        LanczosKernel,

        // Gaussian kernel with radius 3, sigma = 1.5 / 3.  Less commonly used.
        GaussianKernel,

        // Rectangle function.  Equivalent to "nearest" sampling when upscaling.
        // Has value 1 in interval (-0.5, 0.5), value 0.5 on edge, and 0 elsewhere.
        BoxKernel,

        // Hat/tent function with radius 1.  Equivalent to "bilinear" reconstruction
        // when upsampling.
        // Has value zero at -1.0 and 1.0.
        TriangleKernel,

        // Cubic interpolant of Keys.  Equivalent to Catmull-Rom kernel.  Reasonably
        // good quality and faster than Lanczos3Kernel.
        KeysCubicKernel,

        // Cubic non-interpolating scheme.  For synthetic images (especially those
        // lacking proper prefiltering), less ringing than Keys cubic kernel but less
        // sharp.
        MitchellCubicKernel,

        // Always insert new kernel types before this.
        KernelTypeEnd
    };

    // Converts a string into the corresponding kernel type.
    // Invoke invalid argument exception if the string couldn't be converted.
    inline KernelType stringToKernelType(const std::string& kernelType)
    {
        const std::string lower_case = stringToLower(kernelType);
        if (lower_case == "lanczos")
            return KernelType::LanczosKernel;
        else if (lower_case == "gaussian")
            return KernelType::GaussianKernel;
        else if (lower_case == "box")
            return KernelType::BoxKernel;
        else if (lower_case == "triangle")
            return KernelType::TriangleKernel;
        else if (lower_case == "keyscubic")
            return KernelType::KeysCubicKernel;
        else if (lower_case == "mitchellcubic")
            return KernelType::MitchellCubicKernel;
        else
            throw std::invalid_argument("Unknown kernel type: " + kernelType);
    }

    namespace Functor {

        // A function object for a Lanczos kernel.
        struct LanczosKernelFunc {
            // Pass 1 for Lanczos1 kernel, 3 for Lanczos3 etc.
            template <typename... Args>
            explicit LanczosKernelFunc(float _radius, Args&&... args) : radius(_radius)
            {
                Index kernelSize = static_cast<Index>(kSize());
                kernelTensor.resize(kernelSize, kernelSize, 1);
                for (Index d = 0; d < kernelTensor.dimension(2); ++d)
                    for (Index x = 0; x < kernelTensor.dimension(1); ++x)
                        for (Index y = 0; y < kernelTensor.dimension(0); ++y)
                            kernelTensor(y, x, d) = computeKernel(static_cast<float>(x), static_cast<float>(y));

                // normalize kernel such that sum of elements is one
                // if it is not normalized, the image becomes darker
                float kernelSum = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)kernelTensor.sum())(0);
                kernelTensor = kernelTensor.unaryExpr([kernelSum](float x) { return x / kernelSum; });
            }

            float computeKernel(float x) const
            {
                constexpr float kPI = 3.14159265359;
                x = std::abs(x);
                if (x > radius)
                    return 0.0;
                // Need to special case the limit case of sin(x) / x when x is zero.
                if (x <= 1e-3) {
                    return 1.0;
                }
                return radius * std::sin(kPI * x) * std::sin(kPI * x / radius) / (kPI * kPI * x * x);
            }

            float computeKernel(float x, float y) const
            {
                return computeKernel(x) * computeKernel(y);
            }

            auto operator()() const
            {
                return kernelTensor;
            }

            float kSize() const { return radius; }
            const float radius;
            Eigen::Tensor<float, 3, Eigen::RowMajor> kernelTensor;
        };

        struct GaussianKernelFunc {
            static constexpr float kRadiusMultiplier = 3.0f;
            // https://en.wikipedia.org/wiki/Gaussian_function
            // We use sigma = 0.5, as suggested on p. 4 of Ken Turkowski's "Filters
            // for Common Resampling Tasks" for kernels with a support of 3 pixels:
            // www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
            // This implies a radius of 1.5,
            template <typename... Args>
            explicit GaussianKernelFunc(float _radius = 1.5f, Args&&... args)
                : radius(_radius), sigmaX(radius / kRadiusMultiplier), sigmaY(sigmaX)
            {
                createKernel();
            }

            template <typename... Args>
            explicit GaussianKernelFunc(float sigma_, float _radius, Args&&... args)
                : radius(_radius), sigmaX(sigma_), sigmaY(sigma_)
            {
                createKernel();
            }

            template <typename... Args>
            explicit GaussianKernelFunc(float sigmaX_, float sigmaY_, float _radius, Args&&... args)
                : radius(_radius), sigmaX(sigmaX_), sigmaY(sigmaY_)
            {
                createKernel();
            }

            void createKernel()
            {
                Index kernelSize = static_cast<Index>(kSize());
                kernelTensor.resize(kernelSize, kernelSize, 1);
                for (Index d = 0; d < kernelTensor.dimension(2); ++d)
                    for (Index x = 0; x < kernelTensor.dimension(1); ++x)
                        for (Index y = 0; y < kernelTensor.dimension(0); ++y) {
                            kernelTensor(y, x, d) = computeKernel(
                                std::ceil(static_cast<float>(x) - radius),
                                std::ceil(static_cast<float>(y) - radius));
                        }
                // normalize kernel such that sum of elements is one
                float kernelSum = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)kernelTensor.sum())(0);
                kernelTensor = kernelTensor.unaryExpr([kernelSum](float x) { return x / kernelSum; });
            }

            float computeKernel(float x, float y) const
            {
                y = std::abs(y);
                x = std::abs(x);
                if (x >= radius)
                    x = 0.0;
                if (y >= radius)
                    y = 0.0;
                return std::exp(-(x * x + y * y) / (2.0 * sigmaX * sigmaY));
            }

            float computeKernel(float x) const
            {
                x = std::abs(x);
                if (x >= radius)
                    return 0.0;
                return std::exp(-x * x / (2.0 * sigmaX * sigmaX));
            }

            auto operator()() const { return kernelTensor; }
            float kSize() const { return radius * 2; }
            const float radius;
            const float sigmaX, sigmaY; // Gaussian standard deviation
            Eigen::Tensor<float, 3, Eigen::RowMajor> kernelTensor;
        };

        template <typename... Args>
        inline LanczosKernelFunc createLanczosKernel(float radius = 1.0f, Args&&... args)
        {
            return LanczosKernelFunc(radius, std::forward<Args>(args)...);
        }

        template <typename... Args>
        inline GaussianKernelFunc createGaussianKernel(float radius = 1.5f, Args&&... args)
        {
            return GaussianKernelFunc(std::forward<Args>(args)...);
        }

    } // namespace Functor

    class Kernel {
    public:
        virtual ~Kernel() {}
        virtual Eigen::Tensor<float, 3, Eigen::RowMajor> getKernel() const = 0;
        virtual float kSize() const = 0;
        virtual std::string getPaddingMode() const = 0;
        virtual float getPaddingValue() const = 0;
    };

    // Wraps sampling kernel in a common interface.
    template <typename KType>
    class TypedKernel : public Kernel {
    public:
        explicit TypedKernel(const KType& kernel, std::string paddingMode = "reflect", float paddingValue = 0)
            : kernel_(kernel), paddingMode_(paddingMode), paddingValue_(paddingValue) {}

        Eigen::Tensor<float, 3, Eigen::RowMajor> getKernel() const override { return kernel_(); }

        float kSize() const override { return kernel_.kSize(); }

        std::string getPaddingMode() const { return paddingMode_; }

        float getPaddingValue() const { return paddingValue_; }

        const KType kernel_;

        std::string paddingMode_{"reflect"};

        float paddingValue_{};
    };

    template <typename KType>
    std::unique_ptr<const Kernel> createKernel(
        const KType& kernel,
        const std::string& paddingMode,
        float paddingValue)
    {
        return Utils::makeUnique<TypedKernel<KType>>(kernel, paddingMode, paddingValue);
    }

    template <typename... Args>
    std::unique_ptr<const Kernel> create(const std::string& kernelTypeStr, const std::string& paddingMode = "reflect", float paddingValue = 0, Args&&... args)
    {
        KernelType kernelType = stringToKernelType(kernelTypeStr);
        switch (kernelType) {
        case KernelType::LanczosKernel:
            return createKernel(Functor::createLanczosKernel(std::forward<Args>(args)...), paddingMode, paddingValue);
        case KernelType::GaussianKernel:
            return createKernel(Functor::createGaussianKernel(std::forward<Args>(args)...), paddingMode, paddingValue);
        default:
            return nullptr;
        }
    }

    template <typename T>
    void GaussianBlur(
        const Eigen::Tensor<T, 3, Eigen::RowMajor>& src,
        Eigen::Tensor<T, 3, Eigen::RowMajor>& dst,
        const std::string& paddingMode = "reflect",
        float sigma = 0.5,
        float radius = 1.5,
        float paddingValue = 0,
        bool seperateConv = true)
    {
        // create gaussian filter
        auto gaussianKernel = create("gaussian", paddingMode, paddingValue, sigma, radius);
        const Index C = src.dimension(2);
        const Index H = src.dimension(0);
        const Index W = src.dimension(1);
        dst.resize(H, W, C);
        if (seperateConv && C > 1) {
            for (Index i = 0; i < C; i++) {
                Eigen::array<Index, 3> offset = {0, 0, i};
                Eigen::array<Index, 3> extent = {H, W, 1};
                dst.slice(offset, extent) = imageConvolution<T>(
                    src.slice(offset, extent).eval(), gaussianKernel->getKernel(), gaussianKernel->getPaddingMode(), gaussianKernel->getPaddingValue());
            }
        }
        else
            dst = imageConvolution<T>(src, gaussianKernel->getKernel(), gaussianKernel->getPaddingMode(), gaussianKernel->getPaddingValue());
    }

} // namespace Image
#endif
