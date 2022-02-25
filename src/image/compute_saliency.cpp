#include <image/compute_saliency.h>
#include <queue>
namespace Image {
    float calcSaliencyValueCpu(
        const Eigen::Tensor<float, 3, Eigen::RowMajor>& singleScalePatch,
        const Eigen::Tensor<float, 4, Eigen::RowMajor>& multiScalePatch,
        int calcR, int calcC, int distC, int K)
    {
        const int B = multiScalePatch.dimension(0);
        const int H = multiScalePatch.dimension(1);
        const int W = multiScalePatch.dimension(2);
        const int C = multiScalePatch.dimension(3);
        const int L = std::max(H, W);
        std::function<float(int, int, int, int, int)> calcColorDist = [&](int b, int r1, int c1, int r2, int c2) {
            Eigen::array<Index, 3> offset1 = {r1, c1, 0};
            Eigen::array<Index, 4> offset2 = {b, r2, c2, 0};
            Eigen::array<Index, 3> extent1 = {1, 1, C};
            Eigen::array<Index, 4> extent2 = {1, 1, 1, C};
            Eigen::Tensor<float, 3, Eigen::RowMajor> C1 = singleScalePatch.slice(offset1, extent1);
            Eigen::Tensor<float, 3, Eigen::RowMajor> C2 = multiScalePatch.slice(offset2, extent2).reshape(Eigen::array<Index, 3>{1, 1, C});

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
        float minColorDist = 2e5;
        float maxColorDist = -2e5;

        auto cmp = [](std::tuple<float, float, float> left, std::tuple<float, float, float> right) {
            return std::get<0>(left) < std::get<0>(right);
        };

        std::priority_queue<
            std::tuple<float, float, float>,
            std::vector<std::tuple<float, float, float>>,
            decltype(cmp)>
            minK(cmp);

        for (int b = 0; b < B; b++)
            for (int r = 0; r < H; r++) {
                for (int c = 0; c < W; c++) {
                    float colorDist = calcColorDist(b, calcR, calcC, r, c);
                    float posDist = calcPosDist(calcR, calcC, r, c);
                    float dist = colorDist / (1 + distC * posDist);
                    if (minK.size() < K)
                        minK.push(std::make_tuple(dist, colorDist, posDist));
                    else if (dist < std::get<0>(minK.top())) {
                        minK.pop();
                        minK.push(std::make_tuple(dist, colorDist, posDist));
                    }
                    if (colorDist > maxColorDist)
                        maxColorDist = colorDist;
                    if (colorDist < minColorDist)
                        minColorDist = colorDist;
                }
            }

        float sum = 0;
        int n = 0;
        for (n = 0; n < K && n < minK.size(); ++n) {
            auto tup = minK.top();
            minK.pop();
            // The distance dcolor is the Euclidean distance between
            // vectorized patches in CIE L*a*b color space normalized to the range
            // [0, 1].
            float dColor = (std::get<1>(tup) - minColorDist) / (maxColorDist - minColorDist);
            sum += dColor / (1 + distC * std::get<2>(tup));
        }
        return 1 - std::exp(-sum / n);
    }
} // namespace Image
