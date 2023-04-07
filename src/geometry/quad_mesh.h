#ifndef QUAD_MESH_H
#define QUAD_MESH_H
#include <algorithm>
#include <geometry/common.h>
#include <image/image.h>
#include <iostream>
namespace Geometry {

    struct MeshVert {
        MeshVert() : uv{-1.0, -1.0}, index{-1} {}

        MeshVert(const Eigen::Vector2f& uv_, const int index_)
            : uv{uv_}, index{index_} {}

        MeshVert(const float u, const float v, const int index_)
            : uv{u, v}, index{index_} {}

        Eigen::Vector2f uv; ///< u for row index and v for column index in pixel coordinates
        int index; ///< vertex index for mapping

        bool operator==(const MeshVert& other)
        {
            return this->uv == other.uv && this->index == other.index;
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

        Eigen::Vector2i deserialize()
        {
            int v0Index = v[0]->index;
            int v1Index = v[1]->index;
            return {v0Index, v1Index};
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
        PatchMesh(
            const std::vector<Eigen::Vector2f>& vertices_uv, const std::vector<int>& indices)
        {
            nEdges = vertices_uv.size() / 2;
            constexpr int offset = 2;
            for (int i = 0; i < vertices_uv.size(); i += offset) {
                edges.emplace_back(
                    std::make_shared<MeshEdge>(
                        std::make_shared<MeshVert>(vertices_uv[i], indices[i]), ///< v0
                        std::make_shared<MeshVert>(vertices_uv[i + 1], indices[i + 1]) ///< v1
                        ));
            }
            computeCentroid();
        }

        [[nodiscard]] std::shared_ptr<MeshEdge>
        getCentralEdge() const
        {
            float minDist = 2e5;
            int minIdx = 0;
            if (edges.empty())
                return nullptr;
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

        void drawOnCanvas(Eigen::Tensor<float, 3, Eigen::RowMajor>& canvas)
        {
            const int C = canvas.dimension(2);
            const int H = canvas.dimension(0);
            const int W = canvas.dimension(1);
            // plot dot on canvas
            for (int i = 0; i < edges.size(); i++) {
                Eigen::Vector2f v_l = edges[i]->v[0]->uv;
                Eigen::Vector2f v_r = edges[i]->v[1]->uv;

                int r1 = v_l(0);
                int c1 = v_l(1);
                int r2 = v_r(0);
                int c2 = v_r(1);
                r1 = std::min(r1, H - 1);
                c1 = std::min(c1, W - 1);
                r2 = std::min(r2, H - 1);
                c2 = std::min(c2, W - 1);

                // store discrete point between vertices
                std::vector<std::pair<int, int>> rc_lines;
                for (int r_ = std::min(r1, r2); r_ <= std::max(r1, r2); r_++)
                    for (int c_ = std::min(c1, c2); c_ <= std::max(c1, c2); c_++)
                        rc_lines.push_back(std::make_pair(r_, c_));

                for (int d = 0; d < C; d++) {
                    canvas(r1, c1, d) = 255.0;
                    canvas(r2, c2, d) = 255.0;
                    for (auto& p : rc_lines)
                        canvas(p.first, p.second, d) = 255.0;
                }
            }
        }

        Eigen::Vector2f centroid{0, 0};
        std::size_t nEdges{0};
        std::vector<std::shared_ptr<MeshEdge>> edges;
    };

} // namespace Geometry
#endif
