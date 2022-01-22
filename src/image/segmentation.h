#ifndef SEGMENTATION_H
#define SEGMENTATION_H
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
        explicit GraphSegmentation() : sigma{0.5}, k{300}, minSize{100}
        {
            name = "GraphSegmentation";
        }

        ~GraphSegmentation(){};

        void setSigma(double sigma_)
        {
            if (sigma_ <= 0) {
                sigma_ = 0.001;
            }
            sigma = sigma_;
        }
        double getSigma() { return sigma; }

        void setK(float k_) { k = k_; }
        float getK() { return k; }

        void setMinSize(int minSize_) { minSize = minSize_; }
        int getMinSize() { return minSize; }

        template <typename TColorDepth, int Rank>
        void processImage(
            const Eigen::Tensor<TColorDepth, Rank, Eigen::RowMajor>& imgSrc,
            Eigen::Tensor<TColorDepth, Rank, Eigen::RowMajor>& imgDst);

    private:
        double sigma;
        float k;
        int minSize;
        std::string name;

        // Pre-filter the image
        template <typename TColorDepth, int Rank>
        void smooth(
            const Eigen::Tensor<TColorDepth, Rank, Eigen::RowMajor>& img,
            Eigen::Tensor<TColorDepth, Rank, Eigen::RowMajor, Eigen::RowMajor>& imgFiltered);

        // Build the graph between each pixels
        template <typename TColorDepth, int Rank>
        void buildGraph(
            std::vector<Edge>& edges,
            const Eigen::Tensor<TColorDepth, Rank, Eigen::RowMajor>& imgFiltered);

        // Segment the graph
        template <typename TColorDepth, int Rank>
        void segmentGraph(
            std::vector<Edge>& edges,
            const Eigen::Tensor<TColorDepth, Rank, Eigen::RowMajor>& imgFiltered,
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
        template <typename TColorDepth, int Rank>
        void finalMapping(
            std::shared_ptr<PointSet>& es,
            Eigen::Tensor<TColorDepth, Rank, Eigen::RowMajor>& output)
        {
            int height = imgFiltered.dimension(0);
            int width = imgFiltered.dimension(1);
            int maximumSize = (int)(height * width);

            int lastId = 0;
            std::vector<int> mappedId(maximumSize, -1);

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
    };

    std::shared_ptr<GraphSegmentation> createGraphSegmentation(double sigma, float k, int minSize)
    {

        std::shared_ptr<GraphSegmentation> graphSeg = makePtr<GraphSegmentation>();

        graphSeg->setSigma(sigma);
        graphSeg->setK(k);
        graphSeg->setMinSize(minSize);

        return graphSeg;
    }

} // namespace Image

#endif
