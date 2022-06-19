#include <image/compute_saliency.h>
#include <queue>
namespace Image {

    inline float sigmoidCpu(float val, float alpha, float beta = 0.f)
    {
        return 1.f / (1.f + std::exp(-(val - beta) / alpha));
    }

    float calcSaliencyValueCpu(
        const Eigen::Tensor<float, 3, Eigen::RowMajor>& singleScalePatch,
        const Eigen::Tensor<float, 4, Eigen::RowMajor>& multiScalePatch,
        const Eigen::Tensor<int, 3, Eigen::RowMajor>& indices,
        const int H, const int W,
        const int calcR, const int calcC, const int distC, const int K)
    {
        const int B = multiScalePatch.dimension(0);
        const int pH = singleScalePatch.dimension(0);
        const int pW = singleScalePatch.dimension(1);
        const int C = singleScalePatch.dimension(2);
        const int L = std::max(H, W);

        std::function<float(int, int, int, int, int)> calcColorDist = [&](int b, int r1, int c1, int r2, int c2) {
            Eigen::array<Index, 3> offset1 = {r1, c1, 0};
            Eigen::array<Index, 4> offset2 = {b, r2, c2, 0};
            Eigen::array<Index, 3> extent1 = {1, 1, C};
            Eigen::array<Index, 4> extent2 = {1, 1, 1, C};
            Eigen::Tensor<float, 3, Eigen::RowMajor> C1 = singleScalePatch.slice(offset1, extent1);
            Eigen::Tensor<float, 3, Eigen::RowMajor> C2 = multiScalePatch.slice(offset2, extent2).reshape(Eigen::array<Index, 3>{1, 1, C});

            float colorDist = std::sqrt(((Eigen::Tensor<float, 0, Eigen::RowMajor>)(C1 - C2).square().sum().eval())(0));
            return colorDist;
        };

        std::function<float(int, int, int, int)> calcPosDist = [&](int r1, int c1, int r2, int c2) {
            float dRow = (r1 - r2 + 1e-2);
            float dCol = (c1 - c2 + 1e-2);
            return std::sqrt(dRow * dRow + dCol * dCol) / L;
        };

        float minColorDist = 2e5;
        float maxColorDist = -2e5;

        auto cmp = [](float left, float right) {
            return left < right;
        };

        const int pixelR = indices(calcR, calcC, 0);
        const int pixelC = indices(calcR, calcC, 1);

        std::priority_queue<float, std::vector<float>, decltype(cmp)> minK(cmp);

        for (int b = 0; b < B; b++)
            for (int p_row = 0; p_row < pH; p_row++) {
                for (int p_col = 0; p_col < pW; p_col++) {
                    const int r = indices(p_row, p_col, 0);
                    const int c = indices(p_row, p_col, 1);
                    if (calcR == p_row && calcC == p_col)
                        continue;
                    float colorDist = sigmoidCpu((calcColorDist(b, calcR, calcC, p_row, p_col)), 0.1, 0.0);
                    float posDist = calcPosDist(pixelR, pixelC, r, c);
                    float dist = colorDist / (1 + distC * posDist);
                    if (minK.size() < K)
                        minK.push(dist);
                    else if (dist < minK.top()) {
                        minK.pop();
                        minK.push(dist);
                    }
                }
            }

        float sum = 0;
        int n = 0;
        for (n = 0; n < K && n < minK.size(); ++n) {
            sum += minK.top();
            minK.pop();
        }
        return 1 - std::exp(-sum / n);
    }
} // namespace Image
