#ifndef SEGMENTATION_H
#define SEGMENTATION_H
#include <image/filter.h>
#include <image/image.h>
#include <memory>
#include <vector>
namespace Image {
    // Helpers

    // Represent an edge between two pixels
    class Edge {
    public:
        int from;
        int to;
        float weight;
        Edge(int from_, int to_, float weight_) : from{from_}, to{to_}, weight{weight_} {}
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
            mapping.resize(numElements);
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
            : sigma{0.5f}, k{300.0f}, minSize{100}, pixelInPatch{-1.0f}, adjColorDist{20.0f}
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
        float pixelInPatch;
        float adjColorDist;

        // Pre-filter the image
        template <typename T>
        void smooth(
            const Eigen::Tensor<T, 3, Eigen::RowMajor>& img,
            Eigen::Tensor<T, 3, Eigen::RowMajor>& imgFiltered)
        {
            ///< smoothing image using 3x3 kernel
            GaussianBlur(img, imgFiltered, "reflect", sigma);
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
            Eigen::Tensor<T, 3, Eigen::RowMajor> imgFilteredFloat = imgFiltered.template cast<float>();
            for (int r = 0; r < height; r++) {
                for (int c = 0; c < width; c++) {
                    //Take the right, left, top and down pixel
                    for (int delta = -1; delta <= 1; delta += 2) {
                        for (int delta_c = 0, delta_r = 1; delta_c <= 1; delta_c++ || delta_r--) {
                            int r2 = r + delta * delta_r;
                            int c2 = c + delta * delta_c;
                            if (r2 >= 0 && r2 < height && c2 >= 0 && c2 < width) {

                                float diffSquare = 0;

                                for (int d = 0; d < numChannels; ++d) {
                                    float diff = imgFilteredFloat(r, c, d) - imgFilteredFloat(r2, c2, d);
                                    diffSquare += diff * diff;
                                }

                                float diff = std::sqrt(diffSquare);
                                int from = r * width + c;
                                int to = r2 * width + c2;
                                edges.emplace_back(Edge(diff, from, to));
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
            std::vector<float> thresholds(k, totalPoints);

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
        template <typename T>
        void finalMapping(
            std::shared_ptr<PointSet>& es,
            Eigen::Tensor<T, 3, Eigen::RowMajor>& output)
        {
            int height = output.dimension(0);
            int width = output.dimension(1);
            int maximumSize = (int)(height * width);
            int lastId = 0;
            std::vector<int> mappedId(maximumSize, -1);
            output.setZero();
            for (int i = 0; i < height; i++) {

                for (int j = 0; j < width; j++) {

                    int point = es->getBasePoint(i * width + j);

                    if (mappedId[point] == -1) {
                        mappedId[point] = lastId;
                        lastId++;
                    }

                    output(i, j, 0) = mappedId[point];
                }
            }
        }

        template <typename T>
        void mergePatch()
        {
        }

        template <typename T>
        void
        outputSegResult(
            const Eigen::Tensor<T, 3, Eigen::RowMajor>& segMapping,
            Eigen::Tensor<T, 3, Eigen::RowMajor>& imgDst)
        {
        }

    public:
        template <typename T>
        void processImage(
            const Eigen::Tensor<T, 3, Eigen::RowMajor>& imgSrc,
            Eigen::Tensor<T, 3, Eigen::RowMajor>& imgDst)
        {
            // For storing final segment results
            imgDst.resize(imgSrc.dimension(0), imgSrc.dimension(1), 3);

            Eigen::Tensor<T, 3, Eigen::RowMajor> segMapping(imgSrc.dimension(0), imgSrc.dimension(1), 1);
            segMapping.set(0);

            // Filter graph
            Eigen::Tensor<T, 3, Eigen::RowMajor> imgFiltered;
            smooth<T>(imgSrc, imgFiltered);

            // Build graph
            std::vector<Edge> edges;
            buildGraph<T>(edges, imgFiltered);

            // Segment graph
            std::shared_ptr<PointSet> es = nullptr;

            segmentGraph<T>(edges, imgFiltered, es);

            // Remove small areas
            filterSmallAreas(edges, es);

            // Map segmentation to gray scalar image for later output
            finalMapping(es, segMapping);

            //Output
            outputSegResult(segMapping, imgDst);
        }
    };

    std::shared_ptr<GraphSegmentation> createGraphSegmentation(
        float sigma, float k, int minSize, float pixelInPatch = -1.0f, float adjColorDist=20.0f)
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
