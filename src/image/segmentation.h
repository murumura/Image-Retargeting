#ifndef SEGMENTATION_H
#define SEGMENTATION_H
#include <cstdlib>
#include <functional>
#include <image/filter.h>
#include <image/image.h>
#include <image/imageIO.h>
#include <image/wrapping.h>
#include <iostream>
#include <map>
#include <memory>
#include <stack>
#include <utility>
#include <vector>
namespace Image {
    // Helpers

    template <typename T>
    Eigen::Tensor<T, 3, Eigen::RowMajor>
    randomColor(int segId, int C)
    {
        srand(segId);
        Eigen::Tensor<T, 3, Eigen::RowMajor> randomColor;
        randomColor.resize(1, 1, C);
        for (int d = 0; d < C; d++)
            randomColor(0, 0, d) = (T)rand();
        return randomColor;
    }

    // Represent an edge between two pixels
    class Edge {
    public:
        int from;
        int to;
        std::pair<int, int> fromUV;
        std::pair<int, int> toUV;
        float weight;

        Edge(int from_, int to_, float weight_, int fromU, int fromV, int toU, int toV)
            : from{from_}, to{to_}, weight{weight_}, fromUV{fromU, fromV}, toUV{toU, toV} {}

        bool operator<(const Edge& e) const
        {
            return weight < e.weight;
        }
    };

    // A point in the sets of points
    class PointSetElement {
    public:
        int p;
        int size;

        PointSetElement() {}

        PointSetElement(int p_)
        {
            p = p_;
            size = 1;
        }
    };

    // An object to manage set of points, who can be fusionned
    class PointSet {
    public:
        PointSet(int numElements_)
        {
            numElements = numElements_;
            for (int i = 0; i < numElements; i++) {
                mapping.emplace_back(PointSetElement(i));
            }
        }

        int numElements;

        // Return the main point of the point's set
        int getBasePoint(int p)
        {
            int baseP = p;

            while (baseP != mapping[baseP].p) {
                baseP = mapping[baseP].p;
            }

            // Save mapping for faster acces later
            mapping[p].p = baseP;

            return baseP;
        }

        // Join two sets of points, based on their main point
        void joinPoints(int pA, int pB)
        {
            // Always target smaller set, to avoid redirection in getBasePoint
            if (mapping[pA].size < mapping[pB].size)
                std::swap(pA, pB);

            mapping[pB].p = pA;
            mapping[pA].size += mapping[pB].size;

            numElements--;
        }

        // Return the set size of a set (based on the main point)
        int size(unsigned int p) { return mapping[p].size; }

    private:
        std::vector<PointSetElement> mapping;
    };

    class GraphSegmentation {
    public:
        explicit GraphSegmentation()
            : sigma{0.5f}, k{300.0f}, minSize{100}, pixelInPatch{-1.0f}, adjColorDist{-1.0f}
        {
            name = "GraphSegmentation";
        }

        ~GraphSegmentation(){};

        void setSigma(float sigma_)
        {
            if (sigma_ <= 0) {
                sigma_ = 0.001;
            }
            sigma = sigma_;
        }
        float getSigma() { return sigma; }

        void setK(float k_) { k = k_; }
        float getK() { return k; }

        void setMinSize(int minSize_) { minSize = minSize_; }
        int getMinSize() { return minSize; }

        void setPixelInPatch(float pixelInPatch_) { pixelInPatch = pixelInPatch_; }

        void setAdjColorDist(float adjColorDist_) { adjColorDist = adjColorDist_; }

    private:
        float sigma;
        float k;
        int minSize;
        std::string name;
        // threshold of patch area of entire image to be merged
        float pixelInPatch;
        // color similarity of adjacent area to be merged
        float adjColorDist;

        // Pre-filter the image
        template <typename T>
        void smooth(
            const Eigen::Tensor<T, 3, Eigen::RowMajor>& img,
            Eigen::Tensor<T, 3, Eigen::RowMajor>& imgFiltered)
        {
            ///< smoothing image using 3x3 kernel
            GaussianBlur(img, imgFiltered, "reflect", sigma, 1.5f);
        }

        // Build the graph between each pixels
        template <typename T>
        void buildGraph(
            std::vector<Edge>& edges,
            const Eigen::Tensor<T, 3, Eigen::RowMajor>& imgFiltered)
        {
            int height = imgFiltered.dimension(0);
            int width = imgFiltered.dimension(1);
            int numChannels = imgFiltered.dimension(2);
            Eigen::Tensor<float, 3, Eigen::RowMajor> imgFilteredFloat = imgFiltered.template cast<float>();

            for (int r = 0; r < height; r++) {
                for (int c = 0; c < width; c++) {
                    // Take the right, left, top and down pixel
                    for (int delta = -1; delta <= 1; delta += 2) {
                        for (int delta_c = 0, delta_r = 1; delta_c <= 1; delta_c++ || delta_r--) {
                            int r2 = r + delta * delta_r;
                            int c2 = c + delta * delta_c;

                            if (r2 >= 0 && r2 < height && c2 >= 0 && c2 < width) {

                                float diffSquare = 0;

                                for (int d = 0; d < numChannels; ++d) {
                                    float diff = imgFilteredFloat(r, c, d) - imgFilteredFloat(r2, c2, d);
                                    diffSquare += (diff * diff);
                                }

                                float diff = std::sqrt(diffSquare);
                                int from = r * width + c;
                                int to = r2 * width + c2;

                                edges.emplace_back(Edge(from, to, diff, r, c, r2, c2));
                            }
                        }
                    }
                }
            }
        }

        // Segment the graph
        template <typename T>
        void segmentGraph(
            std::vector<Edge>& edges,
            const Eigen::Tensor<T, 3, Eigen::RowMajor>& imgFiltered,
            std::shared_ptr<PointSet>& es)
        {
            int height = imgFiltered.dimension(0);
            int width = imgFiltered.dimension(1);
            int totalPoints = (int)(height * width);

            // Sort edges
            std::sort(edges.begin(), edges.end());
            // Create a set with all point (by default mapped to themselves)
            es = std::make_shared<PointSet>(height * width);

            // Thresholds
            std::vector<float> thresholds(totalPoints, k);

            for (int i = 0; i < edges.size(); i++) {

                int pA = es->getBasePoint(edges[i].from);
                int pB = es->getBasePoint(edges[i].to);

                if (pA != pB) {
                    if (edges[i].weight <= thresholds[pA] && edges[i].weight <= thresholds[pB]) {
                        es->joinPoints(pA, pB);
                        pA = es->getBasePoint(pA);
                        thresholds[pA] = edges[i].weight + k / es->size(pA);
                        edges[i].weight = 0;
                    }
                }
            }
        }

        // Remove areas too small
        void filterSmallAreas(
            const std::vector<Edge>& edges,
            std::shared_ptr<PointSet>& es)
        {
            for (int i = 0; i < edges.size(); i++) {

                if (edges[i].weight > 0) {

                    int pA = es->getBasePoint(edges[i].from);
                    int pB = es->getBasePoint(edges[i].to);

                    if (pA != pB && (es->size(pA) < minSize || es->size(pB) < minSize)) {
                        es->joinPoints(pA, pB);
                    }
                }
            }
        }

        // Map the segemented graph to a image with uniques, sequentials ids
        void finalMapping(
            std::shared_ptr<PointSet>& es,
            Eigen::Tensor<int, 3, Eigen::RowMajor>& segMapping)
        {
            int height = segMapping.dimension(0);
            int width = segMapping.dimension(1);
            int maximumSize = (int)(height * width);
            int lastId = 0;
            std::vector<int> mappedId(maximumSize, -1);
            segMapping.setZero();
            for (int i = 0; i < height; i++) {

                for (int j = 0; j < width; j++) {

                    int point = es->getBasePoint(i * width + j);

                    if (mappedId[point] == -1) {
                        mappedId[point] = lastId;
                        lastId++;
                    }

                    segMapping(i, j, 0) = mappedId[point];
                }
            }
        }

        void additionalMerge(
            Eigen::Tensor<int, 3, Eigen::RowMajor>& segMapping,
            std::map<int, Eigen::Tensor<float, 3, Eigen::RowMajor>>& patchAvgColors,
            std::map<int, int>& patchCounts,
            int& nSegment)
        {
            const int H = segMapping.dimension(0);
            const int W = segMapping.dimension(1);
            const int patchMinSize = pixelInPatch * H * W;

            std::function<float(int, int)> calcColorDist = [&](int segIdA, int segIdB) {
                float colorDist = 0;
                // clang-format off
                    colorDist = std::sqrt(
                        (
                            (Eigen::Tensor<float, 0, Eigen::RowMajor>)(patchAvgColors[segIdA] - patchAvgColors[segIdB]).square().sum().eval()
                        )(0)
                    );
                // clang-format on
                return colorDist;
            };

            std::function<void(int, int)> changeMapping = [&](int fromID, int toID) {
                for (int r = 0; r < H; r++)
                    for (int c = 0; c < W; c++)
                        if (segMapping(r, c, 0) == fromID)
                            segMapping(r, c, 0) = toID;
            };

            std::function<bool(int, int)> isInside = [&](int row, int col) {
                return row >= 0 && row < H && col >= 0 && col < W;
            };

            std::function<bool(int, int, int, int)>
                mergeable = [&](int r1, int c1, int r2, int c2) {
                    if (!isInside(r1, c1) || !isInside(r2, c2))
                        return false;
                    int segIdA = segMapping(r1, c1, 0);
                    int segIdB = segMapping(r2, c2, 0);

                    if (segIdA == segIdB)
                        return false;
                    bool areaMerge = std::min(patchCounts[segIdA], patchCounts[segIdB]) < patchMinSize;
                    bool similarColor = calcColorDist(segIdA, segIdB) <= adjColorDist;
                    return (areaMerge || similarColor);
                };

            std::function<void(int, int, int, int)> merge = [&](int r1, int c1, int r2, int c2) {
                int segIdpA = segMapping(r1, c1, 0);
                int segIdpB = segMapping(r2, c2, 0);
                if (patchCounts[segIdpA] < patchCounts[segIdpB])
                    std::swap(segIdpA, segIdpB);
                float ratioA = float(patchCounts[segIdpA]) / float(patchCounts[segIdpA] + patchCounts[segIdpB]);
                float ratioB = float(patchCounts[segIdpB]) / float(patchCounts[segIdpA] + patchCounts[segIdpB]);
                patchAvgColors[segIdpA] = patchAvgColors[segIdpA] * ratioA + patchAvgColors[segIdpB] * ratioB;
                patchCounts[segIdpA] += patchCounts[segIdpB];
                patchCounts.erase(segIdpB);
                patchAvgColors.erase(segIdpB);
                changeMapping(segIdpB, segIdpA);
                nSegment--;
            };

            std::stack<std::tuple<int, int>> stk;

            Eigen::Tensor<bool, 2, Eigen::RowMajor> visited(H, W);
            visited.setConstant(false);

            std::function<void(int row, int col)> dfs = [&](int row, int col) {
                stk.push(std::make_tuple(row, col));
                while (!stk.empty()) {
                    int r1 = std::get<0>(stk.top());
                    int c1 = std::get<1>(stk.top());
                    stk.pop();
                    if (visited(r1, c1) || !isInside(r1, c1))
                        continue;

                    for (int delta_r = -1; delta_r <= 1; delta_r++)
                        for (int delta_c = -1; delta_c <= 1; delta_c++) {
                            int r2 = r1 + delta_r;
                            int c2 = c1 + delta_c;
                            if (r2 == r1 && c2 == c1)
                                continue;
                            if (mergeable(r1, c1, r2, c2)) {
                                merge(r1, c1, r2, c2);
                                stk.push(std::make_tuple(r2, c2));
                            }
                        }
                }
                visited(row, col) = true;
            };

            for (int row = 0; row < H; ++row)
                for (int col = 0; col < W; col++)
                    dfs(row, col);
        }

        template <typename T>
        void
        outputSegResult(
            const Eigen::Tensor<T, 3, Eigen::RowMajor>& imgSrc,
            Eigen::Tensor<int, 3, Eigen::RowMajor>& segMapping,
            std::vector<Image::Patch>& patches,
            Eigen::Tensor<T, 3, Eigen::RowMajor>& imgDst)
        {
            imgDst.setZero();
            const int C = imgSrc.dimension(2);
            const int H = imgSrc.dimension(0);
            const int W = imgSrc.dimension(1);
            int nSegment = ((Eigen::Tensor<int, 0, Eigen::RowMajor>)segMapping.maximum())(0) + 1;
            std::cout << "Num Segments: " << nSegment << std::endl;

            // random color mapping as specified in original paper
            // Efficient Graph-Based Image Segmentation(2004)](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf)
            Eigen::Tensor<T, 3, Eigen::RowMajor> origSegments(H, W, C);
            for (int r = 0; r < H; r++)
                for (int c = 0; c < W; c++) {
                    int segId = segMapping(r, c, 0);
                    Eigen::array<Index, 3> offset = {r, c, 0};
                    Eigen::array<Index, 3> extent = {1, 1, C};
                    origSegments.template slice(offset, extent) = randomColor<T>(segId, C);
                }

            savePNG<uint8_t, 3>("./original-segmentation" + std::to_string(k) + "-" + std::to_string(minSize), origSegments.template cast<uint8_t>());

            if (pixelInPatch > 0 && adjColorDist > 0) {
                Eigen::Tensor<float, 3, Eigen::RowMajor> emptyColor(1, 1, C);
                emptyColor.setZero();

                // store aggregate colors of each segment (patch)
                std::map<int, Eigen::Tensor<float, 3, Eigen::RowMajor>> patchColors;

                // store average colors of each segment (patch)
                std::map<int, Eigen::Tensor<float, 3, Eigen::RowMajor>> patchAvgColors;

                // store pixel numbers of each segment (patch)
                std::map<int, int> patchCounts;

                // Each patch is assigned an average color to roughly represent this patch
                for (int r = 0; r < H; r++) {
                    for (int c = 0; c < W; c++) {
                        int segId = segMapping(r, c, 0);
                        Eigen::array<Index, 3> offset = {r, c, 0};
                        Eigen::array<Index, 3> extent = {1, 1, C};
                        if (!patchColors.count(segId)) {
                            patchColors.insert({segId, emptyColor});
                            patchCounts.insert({segId, 0});
                        }
                        patchColors[segId] += imgSrc.template slice(offset, extent).template cast<float>().eval();
                        patchCounts[segId]++;
                    }
                }

                // Calculate average color of each patch
                for (int segId = 0; segId < nSegment; segId++) {
                    patchAvgColors.insert({segId, (patchColors[segId] / (float)patchCounts[segId]).eval()});
                }

                // additional merge
                additionalMerge(segMapping, patchAvgColors, patchCounts, nSegment);

                std::cout << "Num Segments: " << nSegment << std::endl;
                // mapping result to output image
                for (int r = 0; r < H; r++)
                    for (int c = 0; c < W; c++) {
                        int segId = segMapping(r, c, 0);
                        Eigen::array<Index, 3> offset = {r, c, 0};
                        Eigen::array<Index, 3> extent = {1, 1, C};
                        imgDst.template slice(offset, extent) = patchAvgColors[segId].template cast<T>();
                    }

                // populate patches before return
                for (const auto& [segId, patchSize] : patchCounts) {
                    patches.emplace_back(Patch{segId, patchSize});
                    patches.back().setPatchColor(patchAvgColors[segId]);
                }
                std::cout << "Num patches: " << patches.size() << std::endl;
            }
        }

    public:
        template <typename T>
        void
        processImage(
            const Eigen::Tensor<T, 3, Eigen::RowMajor>& imgSrc,
            std::vector<Image::Patch>& patches,
            Eigen::Tensor<int, 3, Eigen::RowMajor>& segMapping,
            Eigen::Tensor<T, 3, Eigen::RowMajor>& imgDst)
        {
            // For storing final segment results
            imgDst.resize(imgSrc.dimension(0), imgSrc.dimension(1), imgSrc.dimension(2));

            // Filter graph
            Eigen::Tensor<T, 3, Eigen::RowMajor> imgFiltered;
            smooth<T>(imgSrc, imgFiltered);

            // Build graph
            std::vector<Edge> edges;
            buildGraph<T>(edges, imgFiltered);

            // Pointer to edge information of paired pixels
            std::shared_ptr<PointSet> es = nullptr;

            // Segment graph
            segmentGraph<T>(edges, imgFiltered, es);

            // Remove small areas
            filterSmallAreas(edges, es);

            // Map segmentation to gray scalar image for later output
            finalMapping(es, segMapping);

            // Output
            outputSegResult<T>(imgSrc, segMapping, patches, imgDst);
        }
    };

    std::shared_ptr<GraphSegmentation> createGraphSegmentation(
        float sigma, float k, int minSize, float pixelInPatch = -1.0f, float adjColorDist = -1.0f)
    {

        std::shared_ptr<GraphSegmentation> graphSeg = std::make_shared<GraphSegmentation>();

        graphSeg->setSigma(sigma);
        graphSeg->setK(k);
        graphSeg->setMinSize(minSize);
        graphSeg->setPixelInPatch(pixelInPatch);
        graphSeg->setAdjColorDist(adjColorDist);

        return graphSeg;
    }

} // namespace Image

#endif
