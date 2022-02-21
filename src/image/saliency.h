#ifndef SALIENCY_H
#define SALIENCY_H
#include <cmath>
#include <image/colorspace_op.h>
#include <image/compute_saliency.h>
#include <image/imageIO.h>
#include <image/pool.h>
#include <image/wrapping.h>
#include <iostream>
#include <memory>
#include <vector>

namespace Image {

    static float scaleTable[6] = {1.0, 0.5, 0.25, 0.1, 0.2, 0.1};

    class ContextAwareSaliency {
    private:
        int K;
        int distC;
        int nScale;
        int scaleU;
        bool saveScaledResults;
        std::vector<float> scalePercents;

        void computeSalienceValueParallel(
            const Eigen::Tensor<float, 3, Eigen::RowMajor>& singleScalePatch,
            const Eigen::Tensor<float, 3, Eigen::RowMajor>& multiScalePatch,
            const int H, const int W,
            Eigen::Tensor<float, 3, Eigen::RowMajor>& salienceMap)
        {

#ifdef RESIZING_USE_CUDA
            return calcSaliencyValueCuda(singleScalePatch, multiScalePatch, salienceMap, distC, K);
#else
            const uint32_t workerSize = 16;
            CustomThreadPool pool(workerSize);
            uint32_t numTasks = H;

            pool.parallelForLoop(
                0, H, [this, &salienceMap, &singleScalePatch, &multiScalePatch, &W](const int& start, const int& end) {
                    for (int r = start; r < end; r++)
                        for (int c = 0; c < W; c++)
                            salienceMap(r, c, 0) = calcSaliencyValueCpu(singleScalePatch, multiScalePatch, r, c, distC, K);
                },
                numTasks);
#endif
        }

        Eigen::Tensor<float, 3, Eigen::RowMajor>
        createPatchMap(
            const Eigen::Tensor<float, 3, Eigen::RowMajor>& imgSrcLAB,
            int u, bool normalize = true)
        {
            const int H = imgSrcLAB.dimension(0);
            const int W = imgSrcLAB.dimension(1);
            const int C = imgSrcLAB.dimension(2);
            Eigen::Tensor<float, 3, Eigen::RowMajor> imgSrcLABPatch = imgSrcLAB;

            // represent each patch by the pixel surrounding it
            for (int row = 0; row < H; ++row) {
                for (int col = 0; col < W; ++col) {
                    int n = 0;
                    float l = 0, a = 0, b = 0;
                    for (int r = row - u; r <= row + u; ++r) {
                        if (r < 0 || r >= H)
                            continue;
                        for (int c = col - u; c <= col + u; ++c) {
                            if (c < 0 || c >= W)
                                continue;
                            ++n;
                            l += imgSrcLABPatch(r, c, 0);
                            a += imgSrcLABPatch(r, c, 1);
                            b += imgSrcLABPatch(r, c, 2);
                        }
                    }

                    imgSrcLABPatch(row, col, 0) = l / n;
                    imgSrcLABPatch(row, col, 1) = a / n;
                    imgSrcLABPatch(row, col, 2) = b / n;
                }
            }

            return imgSrcLABPatch;
        }

        Eigen::Tensor<float, 3, Eigen::RowMajor>
        createSalienceMap(
            const Eigen::Tensor<float, 3, Eigen::RowMajor>& imgSrcLAB,
            const Eigen::Tensor<float, 3, Eigen::RowMajor>& singleScalePatch,
            int multiScale)
        {
            const int H = imgSrcLAB.dimension(0);
            const int W = imgSrcLAB.dimension(1);
            const int C = imgSrcLAB.dimension(2);
            Eigen::Tensor<float, 3, Eigen::RowMajor> multiScalePatch = createPatchMap(imgSrcLAB, multiScale);

            Eigen::Tensor<float, 3, Eigen::RowMajor> salienceMap(H, W, 1);

            computeSalienceValueParallel(singleScalePatch, multiScalePatch, H, W, salienceMap);

            float minimum = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)salienceMap.minimum())(0);
            float maximum = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)salienceMap.maximum())(0);
            salienceMap = ((salienceMap - minimum) / (maximum)).eval();

            return salienceMap;
        }

        void supressBackground(Eigen::Tensor<float, 3, Eigen::RowMajor>& S)
        {
            float minimum = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)S.minimum())(0);
            float maximum = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)S.maximum())(0);
            S = ((S - minimum) / (maximum)).eval();
        }

        void optimization(Eigen::Tensor<float, 3, Eigen::RowMajor>& S, float threshold = 0.8)
        {
            const int H = S.dimension(0);
            const int W = S.dimension(1);

            // optimization
            // (1) normalize accroding to maximum value
            supressBackground(S);

            std::vector<std::pair<std::pair<int, int>, float>> importants;

            // (2) record important part information
            for (int row = 0; row < H; ++row) {
                for (int col = 0; col < W; ++col) {
                    if (S(row, col, 0) > threshold)
                        importants.emplace_back(std::pair<int, int>(row, col), S(row, col, 0));
                }
            }

            // (3) optimization: each pixel outside the attended areas is weighted
            // according to its euclidean distance to the closest attended
            // pixel
            std::vector<std::tuple<int, int, float>> attendedDists;
            float minDist = 2e5;
            float maxDist = 2e-5;
            if (!importants.empty()) {
                for (int row = 0; row < H; ++row) {
                    for (int col = 0; col < W; ++col) {
                        float value = S(row, col, 0);
                        if (value > threshold)
                            continue;
                        float dist = 2e5;
                        for (auto p : importants) {
                            float dRow = (p.first.first - row + 0.0);
                            float dCol = (p.first.second - col + 0.0);
                            float _dist = sqrt(dRow * dRow + dCol * dCol);
                            //minimum distance to attended area
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
                scalePercents.push_back(scaleTable[i]);
        }

        void setScaleU(int scaleU_)
        {
            scaleU = scaleU_;
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
            imgSaliency.resize(imgSrc.dimension(0), imgSrc.dimension(1), 1);
            imgSaliency.setZero();

            Eigen::Tensor<float, 3, Eigen::RowMajor> S(imgSrc.dimension(0), imgSrc.dimension(1), 1);
            S.setZero();

            Eigen::Tensor<float, 3, Eigen::RowMajor> imgLab;
            Image::Functor::RGBToCIE<float>()(imgSrc.template cast<float>(), imgLab);

            Eigen::Tensor<float, 3, Eigen::RowMajor> singleScalePatch = createPatchMap(imgLab, scaleU);
            for (int r = 0; r < nScale; r++) {
                int u = std::ceil(scalePercents[r] * scaleU);
                std::cout << "Generating at scale U = " << u << std::endl;
                Eigen::Tensor<float, 3, Eigen::RowMajor> scaledMap = createSalienceMap(imgLab, singleScalePatch, u);
                optimization(scaledMap, float(0.8));
                if (saveScaledResults)
                    savePNG<uint8_t, 3>("./scale" + std::to_string(u), (scaledMap * 255.0f).cast<uint8_t>());
                S += (scaledMap / (float)(nScale));
            }

            if (saveScaledResults)
                savePNG<uint8_t, 3>("./scale", (S * 255.0f).cast<uint8_t>());

            optimization(S, float(0.8));
            imgSaliency = (S * 255.0f).cast<T>();
        }
    };

    std::shared_ptr<ContextAwareSaliency> createContextAwareSaliency(int distC, int K, int nScale, int scaleU, bool saveScaledResults = false)
    {
        std::shared_ptr<ContextAwareSaliency> caSaliency = std::make_shared<ContextAwareSaliency>();
        caSaliency->setK(K);
        caSaliency->setC(distC);
        caSaliency->setNumScale(nScale);
        caSaliency->setScaleU(scaleU);
        caSaliency->setSaveScaledResults(saveScaledResults);
        return caSaliency;
    }

} // namespace Image

#endif
