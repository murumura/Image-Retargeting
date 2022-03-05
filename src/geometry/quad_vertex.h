#ifndef QUAD_VERTEX_H
#define QUAD_VERTEX_H
#include <geometry/common.h>

namespace Geometry {
    class QuadEdge;
    class QuadFacet;

    class QuadVertex {
    public:
        using EdgePtr = std::shared_ptr<QuadEdge>;
        using FacePtr = std::shared_ptr<QuadFacet>;
        using VertexPtr = std::shared_ptr<QuadFacet>;
        using EdgeList = std::list<EdgePtr>;
        using FacetList = std::list<FacePtr>;
        using VertexVector = std::vector<VertexPtr>;

        QuadVertex(const Eigen::Vector2f& uv_) : uv(uv_) {}

        QuadVertex(float u, float v) : uv(u, v) {}

        const Eigen::Vector2f coordinate() const { return uv; }

        EdgeList& getEdgesLists() { return edgesLists; }
        const EdgeList& getEdgesLists() const { return edgesLists; }

        const QuadEdge& getEdges(size_t i) const
        {
            assert(!(i < 0 || i >= edgesLists.size()));
            auto iter = edgesLists.begin();
            std::advance(iter, i);
            assert(*iter);
            return *(*iter);
        }

        const FacetList& getFacetLists() const { return facetsLists; }

        const QuadFacet& getFacets(size_t i) const
        {
            assert(!(i < 0 || i >= facetsLists.size()));
            auto iter = facetsLists.begin();
            std::advance(iter, i);
            assert(*iter);
            return *(*iter);
        }

        const VertexVector& getVerticesVector() const { return vertsVector; }
        VertexVector& getVerticesVector() { return vertsVector; }

        const VertexPtr& adjacentVertex(std::size_t i) const { return vertsVector[i]; }
        VertexPtr& adjacentVertex(std::size_t i) { return vertsVector[i]; }

        /*------------------------------------------------------------------
        | Add / remove adjacent simplices 
        ------------------------------------------------------------------*/
        void addEdge(EdgePtr& e) { edgesLists.push_back(e); }
        void removeEdge(EdgePtr& e) { edgesLists.remove(e); }

        void addFacet(FacePtr& t) { facetsLists.push_back(t); }
        void removeFacet(FacePtr& t) { facetsLists.remove(t); }

        /*------------------------------------------------------------------
        | Functions for adjacency checks
        ------------------------------------------------------------------*/
        bool isAdjacent(const EdgePtr& q)
        {
            for (auto e : edgesLists)
                if (*q == *e)
                    return true;
            return false;
        }

        bool isAdjacent(const FacePtr& q)
        {
            for (auto f : facetsLists)
                if (*q == *f)
                    return true;
            return false;
        }

    private:
        Eigen::Vector2f uv;
        EdgeList edgesLists;
        FacetList facetsLists;
        VertexVector vertsVector;
    };
}

#endif