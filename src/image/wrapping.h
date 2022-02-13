#ifndef WRAPPING_H
#define WRAPPING_H
#include <image/image.h>
#include <iostream>
#include <vector>
namespace Image {

    class Patch {
    public:
        Eigen::Tensor<float, 3, Eigen::RowMajor> patchColor;
        Eigen::Tensor<float, 3, Eigen::RowMajor> significanceColor;
        int segmentId;
        unsigned int size;
        float saliencyValue;

        Patch()
            : saliencyValue{-1.0}, size{0.0}, segmentId{-1}
        {
            patchColor.resize(1, 1, 3);
            significanceColor.resize(1, 1, 3);
            patchColor.setZero();
            significanceColor.setZero();
        }

        Patch(unsigned int segmentId_, unsigned int size_)
            : segmentId{segmentId_}, size{size_}, saliencyValue{-1.0}
        {
            patchColor.resize(1, 1, 3);
            significanceColor.resize(1, 1, 3);
            patchColor.setZero();
            significanceColor.setZero();
        }

        void setPatchColor(const Eigen::Tensor<float, 3, Eigen::RowMajor>& patchColor_)
        {
            patchColor = patchColor_;
        }

        void setSignificanceColor(const Eigen::Tensor<float, 3, Eigen::RowMajor>& significanceColor_)
        {
            significanceColor = significanceColor_;
        }

        bool operator<(const Patch& p) const
        {
            return segmentId < p.segmentId;
        }
    };

    class Wrapping {
    public:
        void reconstructImage()
        {
        }

        static void assignSignificance(
            const Eigen::Tensor<uint8_t, 3, Eigen::RowMajor>& saliencyMap,
            const Eigen::Tensor<int, 3, Eigen::RowMajor>& segMapping,
            Eigen::Tensor<float, 3, Eigen::RowMajor>& significanceMap,
            std::vector<Image::Patch>& patches)
        {
            const int H = segMapping.dimension(0);
            const int W = segMapping.dimension(1);

            for (int row = 0; row < H; row++)
                for (int col = 0; col < W; col++) {
                    int segId = segMapping(row, col, 0);

                    std::vector<Patch>::iterator patchItr = std::find_if(patches.begin(), patches.end(),
                        [&segId](const Patch& patch) { return patch.segmentId == segId; });

                    /*
                     each segmented patch is assigned a significance value by averaging the saliency values
                     of pixels within this patch
                   */
                    float patchSize = (float)patches[segId].size;
                    patchItr->saliencyValue += ((float)(saliencyMap(row, col, 0))) / patchSize;
                }

            auto comparison = [](const Patch& patchA, const Patch& patchB) {
                return patchA.saliencyValue < patchB.saliencyValue;
            };

            // Normalized saliency value
            float maxSaliencyValue = (*(std::max_element(patches.begin(), patches.end(), comparison))).saliencyValue;
            float minSaliencyValue = (*(std::min_element(patches.begin(), patches.end(), comparison))).saliencyValue;
            std::for_each(patches.begin(), patches.end(),
                [&maxSaliencyValue, &minSaliencyValue](Patch& p) {
                    p.saliencyValue = (p.saliencyValue - minSaliencyValue) / (maxSaliencyValue - minSaliencyValue);
                });
        }

    private:
    };

} // namespace Image

#endif
