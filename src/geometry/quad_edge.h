#ifndef QUAD_EDGE_H
#define QUAD_EDGE_H
#include <geometry/common.h>
#include <geometry/quad_vertex.h>
namespace Geometry {
    
    /*********************************************************************
    * A simple edge class that define connection between two vertices
    * The edge is defined by two vertices v1 and v2.             
    *            
    *     v1 ---------- v2
    *
    *********************************************************************/
   class QuadFacet;

    class QuadEdge {
    public:
        using EdgeList = std::list<std::shared_ptr<QuadEdge>>;

        QuadEdge(QuadVertex& v_l, QuadVertex& v_r, EdgeList& edgelist_)
            : vertex_l{&v_l}, vertex_r{&v_r}, edgelist{&edgelist_}
        {
            uv = 0.5 * (vertex_l->coordinate() + vertex_r->coordinate());

            vertex_l->addEdge(std::make_shared<QuadEdge>(*this));
            vertex_r->addEdge(std::make_shared<QuadEdge>(*this));
        }

        EdgeList& edgelist() { return edgelist; }
        const EdgeList& edgelist() const { return edgelist; }
        const Eigen::Vector2f coordinate() const { return uv; }

        const std::shared_ptr<QuadVertex>& v_l() const { return vertex_l; };
        const std::shared_ptr<QuadVertex>& v_r() const { return vertex_r; };
        std::shared_ptr<QuadVertex>& v_l() { return vertex_l; };
        std::shared_ptr<QuadVertex>& v_r() { return vertex_r; };

        const std::shared_ptr<QuadFacet> facet_l() const { return face_l; }
        const std::shared_ptr<QuadFacet> facet_r() const { return face_r; }
        std::shared_ptr<QuadFacet>& facet_l() { return face_l; }
        std::shared_ptr<QuadFacet>& facet_r() { return face_r; }

        void setFacet_l(std::shared_ptr<QuadFacet>& f) { face_l = f; }
        void setFacet_r(std::shared_ptr<QuadFacet>& f) { face_r = f; }

        /***********************************************************
        * Edge equality operator 
        ***********************************************************/
        bool operator==(const QuadEdge& e)
        {
            return this->coordinate() == e.coordinate();
        }

        bool operator!=(const QuadEdge& e)
        {
            return !(*this == e);
        }

    private:
        Eigen::Vector2f uv;
        std::shared_ptr<QuadVertex> vertex_l{nullptr};
        std::shared_ptr<QuadVertex> vertex_r{nullptr};
        EdgeList edgelist;

        std::shared_ptr<QuadFacet> face_l{nullptr};
        std::shared_ptr<QuadFacet> face_r{nullptr};
    };
}

#endif