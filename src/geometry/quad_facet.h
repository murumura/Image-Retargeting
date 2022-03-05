#ifndef QUAD_FACET_H
#define QUAD_FACET_H
#include <geometry/common.h>

namespace Geometry {
  
    class QuadFacet;
    class QuadVertex;
    /*********************************************************************
    * This class defines any two dimensional facet
    *********************************************************************/
    class QuadFacet {
    public:
        QuadFacet() = default;

        std::size_t nVertices() const { return 4; }

        virtual const QuadVertex& vertex(std::size_t i) const = 0;
        virtual QuadVertex& vertex(std::size_t i) = 0;

        virtual int index() const
        {
            return faceIndex;
        }

        virtual void setIndex(int i)
        {
            faceIndex = i;
        }

        /*------------------------------------------------------------------
        | Functions to return indices of vertices, edges...
        ------------------------------------------------------------------*/
        virtual int getVertexIndex(const QuadVertex& v1) const = 0;
        virtual int getEdgeIndex(const QuadVertex& v1, const QuadVertex& v2) const = 0;

        /*------------------------------------------------------------------
        | Set Facet neighbor 
        ------------------------------------------------------------------*/
        virtual void setNeighborFacet(std::size_t i, QuadFacet& f) = 0;

        bool operator==(const QuadFacet& f)
        {
            return this->index() == f.index();
        }

        bool operator!=(const QuadFacet& f)
        {
            return !(*this == f);
        }

    private:
        int faceIndex;
    };
}

#endif