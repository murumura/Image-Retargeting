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

    class ContextAwareSaliency {
    private:
        int K;
        int distC;
        int nScale;
        int scaleU;
        bool saveScaledResults;

        void computeSalienceValueParallel(
            const Eigen::Tensor<float, 3, Eigen::RowMajor>& imgSrcLAB,
            const int H, const int W,
            Eigen::Tensor<float, 3, Eigen::RowMajor>& salienceMap)
        {
            const uint32_t workerSize = 16;
            CustomThreadPool pool(workerSize);
            uint32_t numTasks = H;

            pool.parallelForLoop(
                0, H, [this, &salienceMap, &imgSrcLAB, &W](const int& start, const int& end) {
                    for (int r = start; r < end; r++)
                        for (int c = 0; c < W; c++)
                            salienceMap(r, c, 0) = calcSaliencyValueCPU(imgSrcLAB, r, c, distC, K);
                },
                numTasks);
        }

        Eigen::Tensor<float, 3, Eigen::RowMajor>
        createSalienceMap(
            const Eigen::Tensor<float, 3, Eigen::RowMajor>& imgSrcLAB,
            int u)
        {
            const int H = imgSrcLAB.dimension(0);
            const int W = imgSrcLAB.dimension(1);
            const int C = imgSrcLAB.dimension(2);
            Eigen::Tensor<float, 3, Eigen::RowMajor> imgSrcLABClone = imgSrcLAB;
            Eigen::Tensor<float, 3, Eigen::RowMajor> salienceMap(H, W, 1);
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
                            l += imgSrcLABClone(r, c, 0);
                            a += imgSrcLABClone(r, c, 1);
                            b += imgSrcLABClone(r, c, 2);
                        }
                    }
                    imgSrcLABClone(row, col, 0) = l / n;
                    imgSrcLABClone(row, col, 1) = a / n;
                    imgSrcLABClone(row, col, 2) = b / n;
                }
            }

            computeSalienceValueParallel(imgSrcLABClone, H, W, salienceMap);

            float maximum = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)salienceMap.maximum())(0);
            salienceMap = (salienceMap / maximum).eval() * float(255.0);
            return salienceMap;
        }

        template <typename T>
        void optimization(Eigen::Tensor<T, 3, Eigen::RowMajor>& imgSaliency, float threshold = 204)
        {
            const int H = imgSaliency.dimension(0);
            const int W = imgSaliency.dimension(1);
            float maximum = ((Eigen::Tensor<float, 0, Eigen::RowMajor>)imgSaliency.template cast<float>().template maximum())(0);

            // optimization
            // (1) normalize accroding to maximum value
            Eigen::Tensor<float, 3, Eigen::RowMajor> S = imgSaliency.template cast<float>();
            S = (S / maximum) * float(255.0);

            std::vector<std::pair<std::pair<int, int>, float>> importants;

            // (2) record important part information
            for (int row = 0; row < H; ++row) {
                for (int col = 0; col < W; ++col) {
                    if (S(row, col, 0) > threshold)
                        importants.emplace_back(std::pair<int, int>(row, col), S(row, col, 0));
                }
            }

            // (3) optimization
            if (!importants.empty()) {
                for (int row = 0; row < H; ++row) {
                    for (int col = 0; col < W; ++col) {
                        float value = S(row, col, 0);
                        if (value > threshold)
                            continue;
                        float dis = 1;
                        for (auto p : importants) {
                            double dRow = (p.first.first - row + 0.0) / H;
                            double dCol = (p.first.second - col + 0.0) / W;
                            double _dis = sqrt(dRow * dRow + dCol * dCol);
                            if (_dis < dis)
                                dis = _dis;
                        }
                        S(row, col, 0) = value * (1.0 - dis);
                    }
                }
            }
            imgSaliency = S.cast<T>();
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

            Eigen::Tensor<float, 3, Eigen::RowMajor> imgLab;
            Image::Functor::RGBToCIE<float>()(imgSrc.template cast<float>(), imgLab);

            for (int u = scaleU; u != 0; u /= 2) {
                std::cout << "Generating at scale U = " << u << std::endl;
                Eigen::Tensor<float, 3, Eigen::RowMajor> scaledMap = createSalienceMap(imgLab, u);
                if (saveScaledResults)
                    savePNG<uint8_t, 3>("./scale" + std::to_string(u), scaledMap.cast<uint8_t>());
                imgSaliency += (scaledMap / (float)(scaleU - 1)).cast<T>();
            }

            if (saveScaledResults)
                savePNG<uint8_t, 3>("./scale", imgSaliency);

            optimization<T>(imgSaliency, float(204));
        }
    };

    std::shared_ptr<ContextAwareSaliency> createContextAwareSaliency(int K, int distC, int nScale, int scaleU, bool saveScaledResults = false)
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
