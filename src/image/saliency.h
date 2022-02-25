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

    static float Rq[3] = {1.0, 0.5, 0.25};
    static float R[5] = {1.0, 0.8, 0.5, 0.3, 0.2};
    class ContextAwareSaliency {
    private:
        int K;
        int distC;
        int nScale;
        int scaleU;
        bool saveScaledResults;
        std::vector<float> scalePercents;
        std::vector<float> scaleUSets;

        void computeSalienceValueParallel(
            const Eigen::Tensor<float, 3, Eigen::RowMajor>& singleScalePatch,
            const Eigen::Tensor<float, 4, Eigen::RowMajor>& multiScalePatch,
            Eigen::Tensor<float, 3, Eigen::RowMajor>& salienceMap)
        {
            const int H = multiScalePatch.dimension(1);
            const int W = multiScalePatch.dimension(2);
#ifdef RESIZING_USE_CUDA
            return calcSaliencyValueCuda(singleScalePatch, multiScalePatch, salienceMap, distC, K);
#else
            const uint32_t workerSize = 16;
            CustomThreadPool pool(workerSize);
            uint32_t numTasks = H;
            // calculate each row in parallelForLoop
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
            const std::vector<int>& multiScales)
        {
            const int B = multiScales.size();
            const int H = imgSrcLAB.dimension(0);
            const int W = imgSrcLAB.dimension(1);
            const int C = imgSrcLAB.dimension(2);
            Eigen::Tensor<float, 4, Eigen::RowMajor> multiScalePatches(B, H, W, C);

            for (int i = 0; i < multiScales.size(); i++) {
                Eigen::array<Index, 4> offset = {i, 0, 0, 0};
                Eigen::array<Index, 4> extent = {1, H, W, C};
                multiScalePatches.slice(offset, extent)
                    = createPatchMap(imgSrcLAB, multiScales[i]).reshape(Eigen::array<Index, 4>{1, H, W, C});
            }

            Eigen::Tensor<float, 3, Eigen::RowMajor> salienceMap(H, W, 1);

            computeSalienceValueParallel(singleScalePatch, multiScalePatches, salienceMap);

            // The saliency map S_i^r at each scale is normalized to the range [0,1]
            normalizeSaliency(salienceMap);
            return salienceMap;
        }

        void supressBackground(Eigen::Tensor<float, 3, Eigen::RowMajor>& S)
        {
            float minimum = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)S.minimum())(0);
            float maximum = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)S.maximum())(0);
            S = ((S - minimum) / (maximum)).eval();
        }

        void normalizeSaliency(Eigen::Tensor<float, 3, Eigen::RowMajor>& S)
        {
            float minimum = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)S.minimum())(0);
            float maximum = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)S.maximum())(0);
            S = ((S - minimum) / (maximum - minimum)).eval();
        }

        void optimization(Eigen::Tensor<float, 3, Eigen::RowMajor>& S, float threshold = 0.8)
        {
            const int H = S.dimension(0);
            const int W = S.dimension(1);

            // optimization
            // (1) normalize accroding to maximum value
            supressBackground(S);

            std::vector<std::pair<int, int>> attendedAreas;

            // (2) record important part information
            for (int row = 0; row < H; ++row) {
                for (int col = 0; col < W; ++col) {
                    if (S(row, col, 0) > threshold)
                        attendedAreas.emplace_back(std::make_pair(row, col));
                }
            }

            // (3) optimization: each pixel outside the attended areas is weighted
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
                scalePercents.push_back(Rq[i]);
            for (int i = 0; i < 4; i++)
                scaleUSets.push_back(R[i]);
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

            // Convert input image to CIE**LAB** color space
            Image::Functor::RGBToCIE<float>()(imgSrc.template cast<float>(), imgLab);
            std::vector<int> u;
            for (int i = 0; i < 4; i++) {
                int scaleU_ = scaleUSets[i] * scaleU;
                u.clear();
                // Create image patch of each scale R = {1.0r, 0.8r, 0.5r, 0.3r, ...}
                Eigen::Tensor<float, 3, Eigen::RowMajor> singleScalePatch = createPatchMap(imgLab, scaleU_);

                for (int r = 0; r < nScale; r++) {
                    // Calculate scale value for multi-scale saliency enhancement
                    // The smallest scale allowed in Rq is 20% of the original image scale.
                    int u_ = std::ceil(scalePercents[r] * scaleU_) > std::ceil(0.2 * scaleU) ? std::ceil(scalePercents[r] * scaleU_) : std::ceil(0.2 * scaleU);
                    u.push_back(u_);
                }

                std::cout << "Generating Saliance map at Scale=" << scaleU_ << std::endl;
                // Create image patch of scale r within multiple scale R = {100%,80%,50%,30%} and calculate their saliance
                Eigen::Tensor<float, 3, Eigen::RowMajor> S_i = createSalienceMap(imgLab, singleScalePatch, u);

                // Including the immediate context by S_i = \bar{S_i}(1−d_foci(i)).
                optimization(S_i, float(0.8));

                if (saveScaledResults)
                    savePNG<uint8_t, 3>("./scale-saliency" + std::to_string(scaleU_), (S_i * 255.0f).cast<uint8_t>());

                // Avaerage final saliance map by total itertions
                S += (S_i / (float)(4));
            }
           
            if (saveScaledResults)
                savePNG<uint8_t, 3>("./scale", (S * 255.0f).cast<uint8_t>());
           
            imgSaliency = (S * 255.0f).cast<T>();
        }
    };

    std::shared_ptr<ContextAwareSaliency> createContextAwareSaliency(int distC, int K, int nScale, int scaleU, bool saveScaledResults = false)
    {
        std::shared_ptr<ContextAwareSaliency> caSaliency = std::make_shared<ContextAwareSaliency>();
        caSaliency->setK(K);
        caSaliency->setC(distC);
        caSaliency->setScaleU(scaleU);
        caSaliency->setNumScale(nScale);
        caSaliency->setSaveScaledResults(saveScaledResults);
        return caSaliency;
    }

} // namespace Image

#endif
