#ifndef FILTER_H
#define FILTER_H
#include <image/image.h>
#include <image/padding_op.h>
namespace Image {
    template <typename InputScalar, typename Kernel, PadMode Mode>
    Eigen::Tensor<InputScalar, 3, Eigen::RowMajor>
    imageConvolution(
        const Eigen::Tensor<InputScalar, 3, Eigen::RowMajor>& image,
        const Kernel& kernel,
        const InputScalar padValue = 0)
    {
        typedef typename Eigen::internal::traits<Kernel>::Scalar KernelScalar;
        const Index numDims = 3; ///< image should have 3 dimensions
        const Index imageHeight = image.dimensions()[0];
        const Index imageWidth = image.dimensions()[1];
        const Index imageChannels = image.dimensions()[2];
        // Number of filter to apply
        const Index kernelFilters = imageChannels;
        const Index kernelWidth = kernel.dimensions()[1];
        const Index kernelHeight = kernel.dimensions()[0];

        const Index paddingWidth = (kernelWidth - 1) * 0.5;
        const Index paddingHeight = (kernelHeight - 1) * 0.5;

        PadImageOp paddingOp = PadImageOp<InputScalar, Mode>(paddingHeight, paddingWidth, padValue);
        Eigen::Tensor<InputScalar, 3, Eigen::RowMajor> paddedImage = paddingOp(image);

        Eigen::array<Eigen::IndexPair<Index>, 1> contractDims;
        contractDims[0] = Eigen::IndexPair<Index>(1, 0);
        // Molds the output of patch extraction code into a 2d tensor:
        // the second dimensions (dims[1]) the patch value to be multiplied with the kernels
        // the first dimensions (dims[0]) everything else
        Eigen::DSizes<Index, 2> preContractDims;
        preContractDims[1] = kernelFilters * kernelWidth * kernelHeight;
        preContractDims[0] = imageHeight * imageWidth * numDims;

        Eigen::DSizes<Index, 2> postContractDims;
        postContractDims[0] = imageHeight;
        postContractDims[1] = imageWidth;
        postContractDims[2] = numDims;

        Eigen::DSizes<Index, 2> kernelDims;
        kernelDims[0] = kernelFilters * kernelWidth * kernelHeight;
        kernelDims[1] = kernelFilters;
    }

} // namespace Image
#endif
