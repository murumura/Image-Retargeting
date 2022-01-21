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

            mapping = new PointSetElement[numElements];

            for (int i = 0; i < numElements; i++) {
                mapping[i] = PointSetElement(i);
            }
        }
        ~PointSet()
        {
            delete[] mapping;
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
        PointSetElement* mapping;
    };

    class GraphSegmentation {
    public:
        explicit GraphSegmentation() : sigma{0.5}, k{300}, minSize{100}
        {
            name = "GraphSegmentation";
        }

        ~GraphSegmentation(){};

        template <typename TColorDepth, int Rank>
        void processImage(
            const Eigen::Tensor<TColorDepth, Rank, Eigen::RowMajor>& imgSrc,
            Eigen::Tensor<TColorDepth, Rank, Eigen::RowMajor>& imgDst);

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
            const std::vector<Edge>& edges,
            const Eigen::Tensor<TColorDepth, Rank, Eigen::RowMajor>& imgFiltered,
            std::vector<PointSet>& es);

        // Remove areas too small
        void filterSmallAreas(
            const std::vector<Edge>& edges,
            std::vector<PointSet>& es);

        // Map the segemented graph to a image with uniques, sequentials ids
        template <typename TColorDepth, int Rank>
        void finalMapping(
            const std::vector<PointSet>& es,
            Eigen::Tensor<TColorDepth, Rank, Eigen::RowMajor>& output);
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
