#include <image/compute_saliency.h>
namespace Image {
    float calcSaliencyValueCpu(
        const Eigen::Tensor<float, 3, Eigen::RowMajor>& singleScalePatch,
        const Eigen::Tensor<float, 3, Eigen::RowMajor>& multiScalePatch,
        int calcR, int calcC, int distC, int K)
    {
        const int H = multiScalePatch.dimension(0);
        const int W = multiScalePatch.dimension(1);
        const int C = multiScalePatch.dimension(2);
        const int L = std::max(H, W);
        std::function<float(int, int, int, int)> calcColorDist = [&](int r1, int c1, int r2, int c2) {
            Eigen::array<Index, 3> offset1 = {r1, c1, 0};
            Eigen::array<Index, 3> offset2 = {r2, c2, 0};
            Eigen::array<Index, 3> extent = {1, 1, C};
            Eigen::Tensor<float, 3, Eigen::RowMajor> C1 = singleScalePatch.slice(offset1, extent);
            Eigen::Tensor<float, 3, Eigen::RowMajor> C2 = multiScalePatch.slice(offset2, extent);

            float colorDist = std::sqrt(
                (
                    (Eigen::Tensor<float, 0, Eigen::RowMajor>)(C1 - C2).square().sum().eval())(0));
            return colorDist;
        };

        std::function<float(int, int, int, int)> calcPosDist = [&](int r1, int c1, int r2, int c2) {
            float dRow = (r1 - r2 + 0.0) / L;
            float dCol = (c1 - c2 + 0.0) / L;
            return std::sqrt(dRow * dRow + dCol * dCol);
        };

        std::vector<float> diffs;

        for (int r = 0; r < H; r++) {
            for (int c = 0; c < W; c++) {
                float dist = calcColorDist(calcR, calcC, r, c) / (1 + distC * calcPosDist(calcR, calcC, r, c));
                diffs.push_back(dist);
            }
        }

        std::sort(diffs.begin(), diffs.end());
        float sum = 0;
        int n = 0;
        for (n = 0; n <= K && n < diffs.size(); ++n)
            sum += diffs[n];
        return 1 - std::exp(-sum / n);
    }
} // namespace Image
