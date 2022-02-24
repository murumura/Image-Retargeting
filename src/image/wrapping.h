#ifndef WRAPPING_H
#define WRAPPING_H
#include <image/image.h>
#include <iostream>
#include <vector>
namespace Image {
    const int r = 0;
    const int g = 1;
    const int b = 2;

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

        bool operator<(const Patch& p) const
        {
            return segmentId < p.segmentId;
        }
    };

    class QuadMesh {
    public:
        std::vector<Eigen::Vector2d> vertices;
        std::vector<Eigen::Vector2d> edges;
    };

    class Wrapping {
    public:
        void reconstructImage()
        {
        }

        static Eigen::Tensor<float, 3, Eigen::RowMajor> applyColorMap(
            const Eigen::Tensor<uint8_t, 3, Eigen::RowMajor>& saliencyMap)
        {
            const int H = saliencyMap.dimension(0);
            const int W = saliencyMap.dimension(1);
            static const int nColorMapping = 5;
            static std::array<float, nColorMapping> rLookUpTable = {255.0, 255.0, 255.0, 0.0, 0.0};
            static std::array<float, nColorMapping> gLookUpTable = {0.0, 125.0, 255.0, 255.0, 0.0};
            static std::array<float, nColorMapping> bLookUpTable = {0.0, 0.0, 0.0, 0.0, 255.0};
            float step = std::ceil(360.0 / nColorMapping);
            Eigen::Tensor<float, 3, Eigen::RowMajor> rgb(H, W, 3);
            for (int row = 0; row < H; row++)
                for (int col = 0; col < W; col++) {
                    float degree = 360 - 360.0 * saliencyMap(row, col, 0) / 255.0;
                    int idx = (degree / step);
                    if (idx < 0)
                        idx = 0;
                    else if (idx >= nColorMapping)
                        idx = nColorMapping - 1;
                    rgb(row, col, r) = rLookUpTable[idx];
                    rgb(row, col, g) = gLookUpTable[idx];
                    rgb(row, col, b) = bLookUpTable[idx];
                }
            return rgb;
        }

        static void assignSignificance(
            const Eigen::Tensor<uint8_t, 3, Eigen::RowMajor>& saliencyMap,
            const Eigen::Tensor<int, 3, Eigen::RowMajor>& segMapping,
            Eigen::Tensor<float, 3, Eigen::RowMajor>& significanceMap,
            std::vector<Image::Patch>& patches)
        {
            const int H = segMapping.dimension(0);
            const int W = segMapping.dimension(1);
            significanceMap.resize(H, W, 3);
            significanceMap.setZero();
            // Create rgb-saliance map for visualization
            Eigen::Tensor<float, 3, Eigen::RowMajor> saliencyMapRGB = Wrapping::applyColorMap(saliencyMap);
            savePNG<uint8_t, 3>("./saliencyMapRGB", saliencyMapRGB.cast<uint8_t>());

            for (int row = 0; row < H; row++)
                for (int col = 0; col < W; col++) {
                    int segId = segMapping(row, col, 0);

                    std::vector<Patch>::iterator patchItr = std::find_if(patches.begin(), patches.end(),
                        [&segId](const Patch& patch) { return patch.segmentId == segId; });

                    // each segmented patch is assigned a significance value by averaging the saliency values
                    // of pixels within this patch
                    float patchSize = (float)patchItr->size;
                    patchItr->saliencyValue += ((float)(saliencyMap(row, col, 0)) / patchSize);

                    // assign significance rgb color to each segmented patch
                    Eigen::array<Index, 3> offset = {row, col, 0};
                    Eigen::array<Index, 3> extent = {1, 1, 3};
                    patchItr->significanceColor += saliencyMapRGB.slice(offset, extent).cast<float>() / patchSize;
                }

            auto comparison = [](const Patch& patchA, const Patch& patchB) {
                return patchA.saliencyValue < patchB.saliencyValue;
            };

            // Normalized saliency value of each patch to 0 - 1
            float maxSaliencyValue = (*(std::max_element(patches.begin(), patches.end(), comparison))).saliencyValue;
            float minSaliencyValue = (*(std::min_element(patches.begin(), patches.end(), comparison))).saliencyValue;
            std::for_each(patches.begin(), patches.end(),
                [&maxSaliencyValue, &minSaliencyValue](Patch& p) {
                    p.saliencyValue = (p.saliencyValue - minSaliencyValue) / (maxSaliencyValue - minSaliencyValue);
                });

            // Merge segementation and saliance value to create significance map
            for (int row = 0; row < H; row++)
                for (int col = 0; col < W; col++) {
                    int segId = segMapping(row, col, 0);

                    std::vector<Patch>::iterator patchItr = std::find_if(patches.begin(), patches.end(),
                        [&segId](const Patch& patch) { return patch.segmentId == segId; });
                    Eigen::array<Index, 3> offset = {row, col, 0};
                    Eigen::array<Index, 3> extent = {1, 1, 3};
                    significanceMap.slice(offset, extent) = patchItr->significanceColor;
                }
        }

    private:
    };

} // namespace Image

#endif
