#ifndef QUAD_MESH_H
#define QUAD_MESH_H
#include <algorithm>
#include <geometry/common.h>
#include <iostream>
namespace Geometry {

    enum class LocationType {
        TopBoundary,
        BottomBoundary,
        LeftBoundary,
        RightBoundary,
        Regular
    };

    struct MeshVert {
        MeshVert() : uv{-1.0, -1.0}, locType{LocationType::Regular, LocationType::Regular} {}

        MeshVert(const Eigen::Vector2f& uv_,
            const LocationType& u_loc = LocationType::Regular, const LocationType& v_loc = LocationType::Regular)
            : uv{uv_}, locType{u_loc, v_loc} {}

        MeshVert(const float u, const float v,
            const LocationType& u_loc = LocationType::Regular, const LocationType& v_loc = LocationType::Regular)
            : uv{u, v}, locType{u_loc, v_loc} {}

        Eigen::Vector2f uv;
        std::pair<LocationType, LocationType> locType;

        bool operator==(const MeshVert& other)
        {
            return this->uv == other.uv;
        }

        bool operator!=(const MeshVert& other)
        {
            return !(*this == other);
        }

        bool operator<(const MeshVert& other)
        {
            if (uv(1) == other.uv(1))
                return uv(0) < other.uv(0);
            return uv(1) < other.uv(1);
        }
    };

    struct MeshEdge {
        MeshEdge() = default;
        MeshEdge(const std::shared_ptr<MeshVert>& v0, const std::shared_ptr<MeshVert>& v1)
            : centroid(-1.0, -1.0)
        {
            if (v0 && v1) {
                v[0] = (*v0.get() < *v1.get()) ? v0 : v1;
                v[1] = (*v0.get() < *v1.get()) ? v1 : v0;
                computeCentroid();
            }
        }

        std::shared_ptr<MeshVert> v[2];
        Eigen::Vector2f centroid;

        bool contains(const MeshVert& v1, const MeshVert& v2) const
        {
            return (*v[0].get() == v1 && *v[1].get() == v2) || (*v[0].get() == v2 && *v[1].get() == v1);
        }

        Eigen::Vector2f
        getDstDltTerm(const std::shared_ptr<MeshEdge>& reprEdge,
            const std::size_t H, const std::size_t W, const std::size_t newH, const std::size_t newW) const
        {
            Eigen::Vector2f e = v[0]->uv - v[1]->uv;
            Eigen::Vector2f c = reprEdge->v[0]->uv - reprEdge->v[1]->uv;
            Eigen::Matrix2f m{
                {c(0), c(1)},
                {-c(1), c(0)}};
            Eigen::Vector2f s_r = m.inverse() * e;
            Eigen::Matrix2f T{
                {s_r(0), s_r(1)},
                {-s_r(1), s_r(0)}};
            Eigen::Vector2f cTrans = T * c;
            Eigen::Vector2f eTrans = T * e;
            float DstTerm = (eTrans - T * cTrans).squaredNorm();

            Eigen::Matrix2f L{
                {newH / H, 0},
                {0, newW / W}};

            float DltTerm = (eTrans - L * T * cTrans).squaredNorm();
            return {DstTerm, DltTerm};
        }

        void computeCentroid()
        {
            centroid = 0.5 * (v[0]->uv + v[1]->uv);
        }

        bool operator==(const MeshEdge& other)
        {
            return this->centroid == other.centroid
                && *(this->v[0].get()) == *(other.v[0].get())
                && *(this->v[1].get()) == *(other.v[1].get());
        }

        bool operator!=(const MeshEdge& other)
        {
            return !(*this == other);
        }
    };

    class PatchMesh {
    public:
        static std::shared_ptr<PatchMesh> createPatchMesh()
        {
        }

        PatchMesh(
            const std::vector<Eigen::Vector2f>& vertices_uv,
            const std::vector<std::pair<LocationType, LocationType>>& loc_types)
        {
            nEdges = vertices_uv.size() / 2;
            constexpr int offset = 2;

            for (int i = 0; i < vertices_uv.size(); i += offset) {
                edges.emplace_back(
                    std::make_shared<MeshEdge>(
                        std::make_shared<MeshVert>(vertices_uv[i], loc_types[i].first, loc_types[i].second), ///< v0
                        std::make_shared<MeshVert>(vertices_uv[i + 1], loc_types[i + 1].first, loc_types[i + 1].second) ///< v1
                        ));
            }
            computeCentroid();
        }

        [[nodiscard]] std::shared_ptr<MeshEdge>
        getCentralEdge()
        {
            float minDist = 2e5;
            int minIdx = 0;
            for (int k = 0; k < edges.size(); k++) {
                float dist = (edges[k]->centroid - centroid).squaredNorm();
                if (dist < minDist) {
                    minDist = dist;
                    minIdx = k;
                }
            }
            return edges[minIdx];
        }

        void computeCentroid()
        {
            for (int i = 0; i < nEdges; i++)
                centroid += (edges[i]->centroid / nEdges);
        }

        Eigen::Vector2f centroid{0, 0};
        std::size_t nEdges{0};
        std::vector<std::shared_ptr<MeshEdge>> edges;
    };

} // namespace Geometry
#endif
