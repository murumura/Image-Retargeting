#ifndef WRAPPING_H
#define WRAPPING_H
#include <image/image.h>
namespace Image {

    class Patch {
    public:
        Eigen::Tensor<float, 3, Eigen::RowMajor> patchColor;
        Eigen::Tensor<float, 3, Eigen::RowMajor> significanceColor;
        int segmentId;
        unsigned int size;
        double saliencyValue;

        Patch()
            : saliencyValue{-1.0}, size{0.0}, segmentId{-1}
        {
            patchColor.setZero();
            significanceColor.setZero();
        }

        Patch(unsigned int segmentId_, unsigned int size_)
            : segmentId{segmentId_}, size{size_}, saliencyValue{-1}
        {
            patchColor.setZero();
            significanceColor.setZero();
        }

        void setSaliencyValue(int saliencyValue_)
        {
            saliencyValue = saliencyValue_;
        }

        void setPatchColor(const Eigen::Tensor<float, 3, Eigen::RowMajor>& patchColor_)
        {
            patchColor = patchColor_;
        }

        void setSignificanceColor(const Eigen::Tensor<float, 3, Eigen::RowMajor>& significanceColor_)
        {
            significanceColor = significanceColor_;
        }
    };

    class Wrapping {
    public:
        void reconstructImage()
        {
        }

    private:
    };

} // namespace Image

#endif
