#ifndef SALIENCY_H
#define SALIENCY_H
#include <cmath>
#include <image/colorspace_op.h>
#include <image/compute_saliency.h>
#include <image/imageIO.h>
#include <image/pool.h>
#include <image/resizing_op.h>
#include <image/wrapping.h>
#include <iostream>
#include <memory>
#include <saliency_utils.h>
#include <vector>
namespace Image {

    static float Rq[3] = {1.0, 0.5, 0.25};
    static float R[4] = {1.0, 0.8, 0.5, 0.3};
    class ContextAwareSaliency {
    private:
        int K;
        int distC;
        int nScale;
        int origPatchSize;
        bool saveScaledResults;
        std::vector<float> patchScales;
        std::vector<float> resolutions;

        ResizingImageOp<float> resizingOp;

        void computeSalienceValueParallel(
            const Eigen::Tensor<float, 3, Eigen::RowMajor>& singleScalePatches,
            const Eigen::Tensor<float, 4, Eigen::RowMajor>& multiScalePatches,
            const Eigen::Tensor<int, 3, Eigen::RowMajor>& patchesIndices,
            const int H, const int W,
            Eigen::Tensor<float, 3, Eigen::RowMajor>& S)
        {
            const int pH = singleScalePatches.dimension(0);
            const int pW = singleScalePatches.dimension(1);

#ifdef RESIZING_USE_CUDA
            calcSaliencyValueCuda(singleScalePatches, multiScalePatches, patchesIndices, S, distC, K, H, W);
#else
            const uint32_t workerSize = 16;
            CustomThreadPool pool(workerSize);
            uint32_t numTasks = pH;
            // calculate each row in parallelForLoop
            pool.parallelForLoop(
                0, pH, [this, &S, &singleScalePatches, &multiScalePatches, &patchesIndices, &pW, &H, &W](const int& start, const int& end) {
                    for (int r = start; r < end; r++)
                        for (int c = 0; c < pW; c++)
                            S(r, c, 0) = calcSaliencyValueCpu(
                                singleScalePatches, multiScalePatches, patchesIndices, H, W, r, c, distC, K);
                },
                numTasks);
#endif
        }

        std::tuple<Eigen::Tensor<float, 3, Eigen::RowMajor>, Eigen::Tensor<int, 3, Eigen::RowMajor>>
        createPatchMap(
            const Eigen::Tensor<float, 3, Eigen::RowMajor>& imgSrcLAB, const int u)
        {
            auto [patches, indices] = extractImagePatches(imgSrcLAB, u, u, std::ceil(u / 2), std::ceil(u / 2), 1, 1, "symmetric");
            return {patches, indices};
        }

        Eigen::Tensor<float, 3, Eigen::RowMajor>
        createSalienceMap(
            const Eigen::Tensor<float, 3, Eigen::RowMajor>& imgSrcLAB,
            const Eigen::Tensor<float, 3, Eigen::RowMajor>& singleScalePatches,
            const Eigen::Tensor<int, 3, Eigen::RowMajor>& patchesIndices,
            const std::vector<int>& multiPatchSizes, const int origH, const int origW)
        {
            const int B = multiPatchSizes.size();
            const int H = imgSrcLAB.dimension(0);
            const int W = imgSrcLAB.dimension(1);
            const int C = imgSrcLAB.dimension(2);
            const int pH = singleScalePatches.dimension(0);
            const int pW = singleScalePatches.dimension(1);

            Eigen::Tensor<float, 4, Eigen::RowMajor> multiScalePatches(B, pH, pW, C);

            for (int i = 0; i < multiPatchSizes.size(); i++) {
                Eigen::array<Index, 4> offset = {i, 0, 0, 0};
                Eigen::array<Index, 4> extent = {1, pH, pW, C};
                multiScalePatches.slice(offset, extent)
                    = extractPatchesByIndices(imgSrcLAB, patchesIndices, multiPatchSizes[i], multiPatchSizes[i]).reshape(Eigen::array<Index, 4>{1, pH, pW, C});
            }

            Eigen::Tensor<float, 3, Eigen::RowMajor> salienceMap(pH, pW, 1);
            computeSalienceValueParallel(singleScalePatches, multiScalePatches, patchesIndices, H, W, salienceMap);

            // Interpolated back to original image size.
            Eigen::Tensor<float, 3, Eigen::RowMajor> salienceMapOrigSize(origH, origW, 1);
            resizingOp(salienceMap, salienceMapOrigSize);

            // The saliency map S_i^r at each scale is normalized to the range [0,1]
            normalizeSaliency(salienceMapOrigSize);

            return salienceMapOrigSize;
        }

        void normalizeSaliency(Eigen::Tensor<float, 3, Eigen::RowMajor>& S)
        {
            float minimum = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)S.minimum())(0);
            float maximum = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)S.maximum())(0);
            S = ((S - minimum) / (maximum - minimum)).eval();
        }

        void normalizeLAB(Eigen::Tensor<float, 3, Eigen::RowMajor>& lab)
        {
            auto L = lab.chip<2>(0);
            auto A = lab.chip<2>(1);
            auto B = lab.chip<2>(2);

            const float l_min = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)L.minimum())(0);
            const float a_min = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)A.minimum())(0);
            const float b_min = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)B.minimum())(0);

            const float l_max = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)L.maximum())(0);
            const float a_max = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)A.maximum())(0);
            const float b_max = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)B.maximum())(0);

            L = (L - l_min) / (l_max - l_min);
            A = (A - a_min) / (a_max - a_min);
            B = (B - b_min) / (b_max - b_min);
        }

        void optimization(Eigen::Tensor<float, 3, Eigen::RowMajor>& S, float threshold = 0.8)
        {
            const int H = S.dimension(0);
            const int W = S.dimension(1);

            std::vector<std::pair<int, int>> attendedAreas;

            // (1) record important part information
            for (int row = 0; row < H; ++row) {
                for (int col = 0; col < W; ++col) {
                    if (S(row, col, 0) > threshold)
                        attendedAreas.emplace_back(std::make_pair(row, col));
                }
            }

            // (2) optimization: each pixel outside the attended areas is weighted
            // according to its euclidean distance to the closest attended
            // pixel
            std::vector<std::tuple<int, int, float>> attendedDists;
            float minDist = 2e5;
            float maxDist = 2e-5;

            if (!attendedAreas.empty()) {
                for (int row = 0; row < H; ++row) {
                    for (int col = 0; col < W; ++col) {
                        float value = S(row, col, 0);
                        if (value > threshold)
                            continue;
                        float dist = 2e5;
                        for (auto p : attendedAreas) {
                            float dRow = (p.first - row + 0.0);
                            float dCol = (p.second - col + 0.0);
                            float _dist = sqrt(dRow * dRow + dCol * dCol);
                            // minimum distance to attended area
                            if (dist > _dist)
                                dist = _dist;
                        }
                        attendedDists.emplace_back(std::make_tuple(row, col, dist));
                        if (dist > maxDist)
                            maxDist = dist;
                        if (dist < minDist)
                            minDist = dist;
                    }
                }
            }

            for (auto& tup : attendedDists) {
                int row = std::get<0>(tup);
                int col = std::get<1>(tup);
                float d_foci = (std::get<2>(tup) - minDist) / (maxDist - minDist);
                S(row, col, 0) = S(row, col, 0) * (1 - d_foci);
            }

            savePNG<uint8_t, 3>("./optimization" + std::to_string(threshold), (S * 255.0f).cast<uint8_t>());
        }

    public:
        explicit ContextAwareSaliency() : K{64}, distC{3}, nScale{3}
        {
            resizingOp = ResizingImageOp<float>("bilinear", true, true);
        }

        void setK(int K_)
        {
            K = K_;
        }

        void setC(int distC_)
        {
            distC = distC_;
        }

        void setNumScale(int nScale_)
        {
            nScale = nScale_;
            for (int i = 0; i < nScale; i++)
                patchScales.push_back(Rq[i]);
            for (int i = 0; i < 4; i++)
                resolutions.push_back(R[i]);
        }

        void setorigPatchSize(int origPatchSize_)
        {
            origPatchSize = origPatchSize_;
        }

        void setSaveScaledResults(bool saveScaledResults_)
        {
            saveScaledResults = saveScaledResults;
        }

        template <typename T>
        void
        processImage(
            const Eigen::Tensor<T, 3, Eigen::RowMajor>& imgSrc,
            Eigen::Tensor<T, 3, Eigen::RowMajor>& imgSaliency)
        {
            const int origH = imgSrc.dimension(0);
            const int origW = imgSrc.dimension(1);
            imgSaliency.resize(origH, origW, 1);
            imgSaliency.setZero();

            Eigen::Tensor<float, 3, Eigen::RowMajor> S(imgSrc.dimension(0), imgSrc.dimension(1), 1);
            S.setZero();

            Eigen::Tensor<float, 3, Eigen::RowMajor> imgLab;

            // Convert input image to CIE**LAB** color space
            Image::Functor::RGBToCIE<float>()(imgSrc.template cast<float>(), imgLab);

            // Scale all the images to the same size of 250 pixels (largest dimension).
            float scaleRatio = 250.0 / float(std::max(imgSrc.dimension(0), imgSrc.dimension(1)));

            Eigen::Tensor<float, 3, Eigen::RowMajor> imgLab250(imgLab.dimension(0) * scaleRatio, imgLab.dimension(1) * scaleRatio, imgLab.dimension(2));
            resizingOp(imgLab, imgLab250);

            const int M = resolutions.size();
            std::vector<int> patchSizes;

            const int imgLab250_H = imgLab250.dimension(0);
            const int imgLab250_W = imgLab250.dimension(1);

            Eigen::Tensor<float, 3, Eigen::RowMajor> labEachScale;

            Eigen::Tensor<float, 3, Eigen::RowMajor> S_250(imgLab250_H, imgLab250_W, 1);
            S_250.setZero();

            normalizeLAB(imgLab250);

            // Create image patch of scale r within multiple scale R = {100%, 80%, 50%, 30%} and calculate their saliance
            for (int i = 0; i < M; i++) {
                std::cout << "Generating Saliance map at scale level=" << i << " ....";

                const int eachScaleH = imgLab250_H * resolutions[i];
                const int eachScaleW = imgLab250_W * resolutions[i];

                const int origPatchSize_ = origPatchSize * resolutions[i];
                patchSizes.clear();
                for (int r = 0; r < nScale; r++) {
                    int pSize = std::max((int)std::ceil(patchScales[r] * origPatchSize_), (int)std::ceil(0.2 * origPatchSize));
                    patchSizes.push_back(pSize);
                }

                labEachScale.resize(eachScaleH, eachScaleW, 3);
                resizingOp(imgLab250, labEachScale);

                // Create image patch of each scale R = {1.0r, 0.5r, 0.25r}
                auto [singleScalePatches, indices] = createPatchMap(labEachScale, origPatchSize_);

                Eigen::Tensor<float, 3, Eigen::RowMajor> S_i
                    = createSalienceMap(labEachScale, singleScalePatches, indices, patchSizes, imgLab250_H, imgLab250_W);

                std::cout << "  Done" << std::endl;
                if (saveScaledResults)
                    savePNG<uint8_t, 3>("./scale-saliency" + std::to_string(i), (S_i * 255.0f).cast<uint8_t>());

                // Avaerage final saliance map by total itertions
                S_250 += (S_i / (float)(M));
            }
            // Including the immediate context by S_i = \bar{S_i}(1−d_foci(i)).
            optimization(S_250, float(0.8));

            resizingOp(S_250, S);

            imgSaliency = (S * 255.0f).cast<T>();
        }
    };

    std::shared_ptr<ContextAwareSaliency> createContextAwareSaliency(int distC, int K, int nScale, int origPatchSize, bool saveScaledResults = false)
    {
        std::shared_ptr<ContextAwareSaliency> caSaliency = std::make_shared<ContextAwareSaliency>();
        caSaliency->setK(K);
        caSaliency->setC(distC);
        caSaliency->setorigPatchSize(origPatchSize);
        caSaliency->setNumScale(nScale);
        caSaliency->setSaveScaledResults(saveScaledResults);
        return caSaliency;
    }

} // namespace Image

#endif
