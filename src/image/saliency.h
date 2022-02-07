#ifndef SALIENCY_H
#define SALIENCY_H
#include <image/colorspace_op.h>
#include <image/image.h>
#include <memory>
namespace Image {

    class ContextAwareSaliency {
    private:
        template <typename T>
        void calcDistance(
            const Eigen::Tensor<T, 3, Eigen::RowMajor>& imgSrc)
        {
        }

    public:
        template <typename T>
        void
        processImage(
            const Eigen::Tensor<T, 3, Eigen::RowMajor>& imgSrc,
            Eigen::Tensor<T, 3, Eigen::RowMajor>& imgSaliency)
        {
            imgSaliency.resize(imgSrc.dimension(0), imgSrc.dimension(1), 1);
            Eigen::Tensor<float, 3, Eigen::RowMajor> imgLab;
            Image::Functor::RGBToCIE<float>()(imgSrc.template cast<float>(), imgLab);
        }
    };

    std::shared_ptr<ContextAwareSaliency> createContextAwareSaliency()
    {
        std::shared_ptr<ContextAwareSaliency> caSaliency = std::make_shared<ContextAwareSaliency>();
        return caSaliency;
    }

} // namespace Image

#endif
