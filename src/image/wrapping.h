#ifndef WRAPPING_H
#define WRAPPING_H
#include <cmath>
#include <geometry/quad_mesh.h>
#include <image/image.h>
#include <image/utils.h>
#include <iostream>
#include <list>
#include <numerical/cg_solver.h>
#include <image/perspective_transform_op.h>
#include <vector>
namespace Image {
    const int r = 0;
    const int g = 1;
    const int b = 2;

    class Patch {
    public:
        Eigen::Tensor<float, 3, Eigen::RowMajor> patchColor;
        Eigen::Tensor<float, 3, Eigen::RowMajor> significanceColor;
        int segmentId;
        unsigned int size;
        float saliencyValue;
        std::shared_ptr<Geometry::MeshEdge> reprEdge{nullptr};
        std::shared_ptr<Geometry::PatchMesh> patchMesh{nullptr};
        Eigen::Vector2f centroid{-1, -1};

        Patch()
            : saliencyValue{-1.0}, size{0.0}, segmentId{-1}
        {
            patchColor.resize(1, 1, 3);
            significanceColor.resize(1, 1, 3);
            patchColor.setZero();
            significanceColor.setZero();
        }

        Patch(unsigned int segmentId_, unsigned int size_)
            : segmentId{segmentId_}, size{size_}, saliencyValue{-1.0}
        {
            patchColor.resize(1, 1, 3);
            significanceColor.resize(1, 1, 3);
            patchColor.setZero();
            significanceColor.setZero();
        }

        void setPatchColor(const Eigen::Tensor<float, 3, Eigen::RowMajor>& patchColor_)
        {
            patchColor = patchColor_;
        }

        void setPatchMesh(const std::vector<Eigen::Vector2f>& vertices_uv)
        {
            patchMesh = std::make_shared<Geometry::PatchMesh>(vertices_uv);
            computeCentroid();
            computeReprEdge();
        }

        void computeCentroid()
        {
            if (patchMesh)
                centroid = patchMesh->centroid;
        }

        void computeReprEdge()
        {
            if (patchMesh)
                reprEdge = patchMesh->getCentralEdge();
        }

        void drawOnCanvas(
            Eigen::Tensor<float, 3, Eigen::RowMajor>& canvas, float quadHeight, float quadWidth) const
        {
            patchMesh->drawOnCanvas(canvas, quadHeight, quadWidth);
        }

        bool operator<(const Patch& p) const
        {
            return segmentId < p.segmentId;
        }
    };

    struct CachedCoordMapping {
        Eigen::Vector2i pixel_coord; ///< pixel coordinate of each quad vertices in serialized form
        int patch_index; ///< corresponding patch index of stored segment Id
        Eigen::Vector2i deformed_uv_coord; ///< deformed uv coordinate
    };

    class Wrapping {
    public:
        explicit Wrapping(std::size_t targetHeight_, std::size_t targetWidth_, float alpha_, float quadSize_, float weightDST_, float weightDLT_, float weightDOR_)
            : alpha{alpha_}, targetHeight{targetHeight_}, targetWidth{targetWidth_}, quadSize{quadSize_}, weightDLT{weightDLT_}, weightDST{weightDST_}, weightDOR{weightDOR_}
        {
        }

        void buildMeshGrid(
            const Eigen::Tensor<int, 3, Eigen::RowMajor>& segMapping,
            std::vector<Image::Patch>& patches)
        {
            const int origH = segMapping.dimension(0);
            const int origW = segMapping.dimension(1);
            meshCols = origW / quadSize + 1;
            meshRows = origH / quadSize + 1;

            quadWidth = (float)(origW) / (meshCols - 1);
            quadHeight = (float)(origH) / (meshRows - 1);

            std::function<Eigen::Vector2i(int, int)> coordTransform = [&](int mesh_row, int mesh_col) {
                int pixel_row = mesh_row * quadHeight;
                int pixel_col = mesh_col * quadWidth;
                // boundary case
                if (mesh_col == meshCols - 1)
                    pixel_col--;
                if (mesh_row == meshRows - 1)
                    pixel_row--;
                pixel_row = std::min(pixel_row, origH - 1);
                pixel_col = std::min(pixel_col, origW - 1);
                return Eigen::Vector2i{std::max(pixel_row, 0), std::max(pixel_col, 0)};
            };

            std::function<int(int)> findPatchIndex = [&](int segId) {
                for (int i = 0; i < patches.size(); i++)
                    if (patches[i].segmentId == segId)
                        return i;
                return -1;
            };
            nVertices = meshRows * meshCols;
            cache_mappings.reserve(nVertices);

            for (int row = 0; row < meshRows; row++)
                for (int col = 0; col < meshCols; col++) {
                    int index = row * meshCols + col;
                    cache_mappings[index].pixel_coord = coordTransform(row, col);
                    int segId = segMapping(cache_mappings[index].pixel_coord(0), cache_mappings[index].pixel_coord(1), 0);
                    cache_mappings[index].patch_index = findPatchIndex(segId);
                }

            std::vector<std::vector<Eigen::Vector2f>> vertices_uvs(patches.size());

            for (int row = 0; row < meshRows - 1; row++)
                for (int col = 0; col < meshCols - 1; col++) {
                    int v = row * meshCols + col;

                    // iterate vertices of each quad in CCW order
                    std::array<int, 4> quad_uv = {
                        v, // top-left corner
                        v + meshCols, // botton-left corner
                        v + meshCols + 1, // botton-right corner
                        v + 1 // top-right corner
                    };

                    std::array<std::pair<int, int>, 4> uv_mapping = {
                        std::make_pair(row, col),
                        std::make_pair(row + 1, col),
                        std::make_pair(row + 1, col + 1),
                        std::make_pair(row, col + 1)};

                    for (int i = 0; i < 4; i++) {
                        int pidx1 = cache_mappings[quad_uv[i]].patch_index;
                        int pidx2 = cache_mappings[quad_uv[(i + 1) == 4 ? 0 : (i + 1)]].patch_index;
                        int r1, c1, r2, c2;
                        std::tie(r1, c1) = uv_mapping[i];
                        std::tie(r2, c2) = uv_mapping[(i + 1) == 4 ? 0 : (i + 1)];
                        vertices_uvs[pidx1].insert(vertices_uvs[pidx1].end(), {Eigen::Vector2f{r1, c1}, Eigen::Vector2f{r2, c2}});
                    }
                    nQuads++;
                }

            std::cout << "Number of MeshCols(top/bottom vertices) " << meshCols << std::endl;
            std::cout << "Number of MeshRows(left/right vertices) " << meshRows << std::endl;
            std::cout << "Number of Quads: " << nQuads << std::endl;
            std::cout << "Number of Vertices: " << nVertices << std::endl;

            // Maintain edge list of each patch
            for (int i = 0; i < patches.size(); i++)
                patches[i].setPatchMesh(vertices_uvs[i]);
        }

        template <typename T>
        void drawMeshGrid(
            const Eigen::Tensor<T, 3, Eigen::RowMajor>& canvas,
            const std::string filename,
            std::vector<Image::Patch>& patches)
        {
            Eigen::Tensor<float, 3, Eigen::RowMajor> C = canvas.template cast<float>();
            for (int i = 0; i < patches.size(); i++)
                patches[i].drawOnCanvas(C, quadHeight, quadWidth);
            savePNG<uint8_t, 3>(filename, C.cast<uint8_t>());
        }

        void buildAndSolveConstraint(std::vector<Image::Patch>& patches, int origH, int origW)
        {
            nVbottom = nVtop = meshCols;
            nVleft = nVright = meshRows;
            const int rows = 16 * nQuads /*8(DST) + 8(DLT)*/ + 4 * nQuads /*DOR*/ + nVtop + nVbottom + nVleft + nVright /*boundary condition*/;

            const int columns = nVertices * 2;
            Numerical::CGSolver solver(rows, columns);
            const float high_ratio = static_cast<float>(targetHeight) / origH;
            const float width_ratio = static_cast<float>(targetWidth) / origW;

            int rowIdx = 0;
            int rhsRowIdx = 0;
            Eigen::VectorXd rhs(rows);

            for (int i = 0; i < patches.size(); i++) {
                float s_i = patches[i].saliencyValue;
                const std::vector<std::shared_ptr<Geometry::MeshEdge>> edgesList = patches[i].patchMesh->edges;
                if (edgesList.empty())
                    continue;

                const std::shared_ptr<Geometry::MeshEdge> repr_edge = patches[i].reprEdge;
                Eigen::Vector2f c = repr_edge->v[1]->uv - repr_edge->v[0]->uv;

                Eigen::Matrix2f M{
                    {c(1), c(0)},
                    {c(0), -c(1)},
                };

                Eigen::Matrix2f M_inv;

                if (M.determinant() > 0)
                    M_inv = M.inverse();
                else
                    M_inv = M.completeOrthogonalDecomposition().pseudoInverse();

                const Eigen::Vector2i repr_vertices = repr_edge->deserialize(meshCols);
                const int c1X = repr_vertices(0);
                const int c1Y = repr_vertices(0) + nVertices;
                const int c2X = repr_vertices(1);
                const int c2Y = repr_vertices(1) + nVertices;

                for (int j = 0; j < edgesList.size(); j++) {
                    const Eigen::Vector2f e = edgesList[j]->v[1]->uv - edgesList[j]->v[0]->uv;
                    const Eigen::Vector2f s_r = M_inv * e;
                    const float s = s_r(0);
                    const float r = s_r(1);

                    const Eigen::Vector2i vertices = edgesList[j]->deserialize(meshCols);
                    const int vaX = vertices(0);
                    const int vaY = vertices(0) + nVertices;
                    const int vbX = vertices(1);
                    const int vbY = vertices(1) + nVertices;

                    /*-- Set up patch transformation constraint --*/
                    /*---DOR---*/
                    // s(vax - vbx) + r(vay - vby) - s(c1x - c2x) - r(c1y - c2y)
                    solver.addSysElement(rowIdx, vaX, alpha * s_i * s * weightDST);
                    solver.addSysElement(rowIdx, vbX, -alpha * s_i * s * weightDST);
                    solver.addSysElement(rowIdx, vaY, alpha * s_i * r * weightDST);
                    solver.addSysElement(rowIdx, vbY, -alpha * s_i * r * weightDST);
                    solver.addSysElement(rowIdx, c1X, -alpha * s_i * s * weightDST);
                    solver.addSysElement(rowIdx, c2X, alpha * s_i * s * weightDST);
                    solver.addSysElement(rowIdx, c1Y, -alpha * s_i * r * weightDST);
                    solver.addSysElement(rowIdx, c2Y, alpha * s_i * r * weightDST);
                    rhs(rhsRowIdx) = 0;
                    rowIdx++;
                    rhsRowIdx++;
                    // -r(vax - vbx) + s(vay - vby)  + r(c1x - c2x) - s(c1y - c2y)
                    solver.addSysElement(rowIdx, vaX, -alpha * s_i * r * weightDST);
                    solver.addSysElement(rowIdx, vbX, alpha * s_i * r * weightDST);
                    solver.addSysElement(rowIdx, vaY, alpha * s_i * s * weightDST);
                    solver.addSysElement(rowIdx, vbY, -alpha * s_i * s * weightDST);
                    solver.addSysElement(rowIdx, c1X, alpha * s_i * r * weightDST);
                    solver.addSysElement(rowIdx, c2X, -alpha * s_i * r * weightDST);
                    solver.addSysElement(rowIdx, c1Y, -alpha * s_i * s * weightDST);
                    solver.addSysElement(rowIdx, c2Y, alpha * s_i * s * weightDST);
                    rhs(rhsRowIdx) = 0;
                    rowIdx++;
                    rhsRowIdx++;

                    /*---DLT---*/
                    // s * (vax - vbx) + r * (vay - vby) - m'/m * s * (c1x - c2x) - m'/m * r * (c1y - c2y)
                    solver.addSysElement(rowIdx, vaX, (1 - alpha) * (1 - s_i) * s * weightDLT);
                    solver.addSysElement(rowIdx, vbX, -(1 - alpha) * (1 - s_i) * s * weightDLT);
                    solver.addSysElement(rowIdx, vaY, (1 - alpha) * (1 - s_i) * r * weightDLT);
                    solver.addSysElement(rowIdx, vbY, -(1 - alpha) * (1 - s_i) * r * weightDLT);
                    solver.addSysElement(rowIdx, c1X, -high_ratio * (1 - alpha) * (1 - s_i) * s * weightDLT);
                    solver.addSysElement(rowIdx, c2X, high_ratio * (1 - alpha) * (1 - s_i) * s * weightDLT);
                    solver.addSysElement(rowIdx, c1Y, -high_ratio * (1 - alpha) * (1 - s_i) * r * weightDLT);
                    solver.addSysElement(rowIdx, c2Y, high_ratio * (1 - alpha) * (1 - s_i) * r * weightDLT);
                    rhs(rhsRowIdx) = 0;
                    rowIdx++;
                    rhsRowIdx++;

                    // -r(vax - vbx) + s(vay - vby) + n'/n * r * (c1x - c2x) - n'/n * s * (c1y - c2y)
                    solver.addSysElement(rowIdx, vaX, -(1 - alpha) * (1 - s_i) * r * weightDLT);
                    solver.addSysElement(rowIdx, vbX, (1 - alpha) * (1 - s_i) * r * weightDLT);
                    solver.addSysElement(rowIdx, vaY, (1 - alpha) * (1 - s_i) * s * weightDLT);
                    solver.addSysElement(rowIdx, vbY, -(1 - alpha) * (1 - s_i) * s * weightDLT);
                    solver.addSysElement(rowIdx, c1X, width_ratio * (1 - alpha) * (1 - s_i) * r * weightDLT);
                    solver.addSysElement(rowIdx, c2X, -width_ratio * (1 - alpha) * (1 - s_i) * r * weightDLT);
                    solver.addSysElement(rowIdx, c1Y, -width_ratio * (1 - alpha) * (1 - s_i) * s * weightDLT);
                    solver.addSysElement(rowIdx, c2Y, width_ratio * (1 - alpha) * (1 - s_i) * s * weightDLT);
                    rhs(rhsRowIdx) = 0;
                    rowIdx++;
                    rhsRowIdx++;
                }
            }

            // Set up grid orientation constraint
            for (int row = 0; row < meshRows - 1; row++) {
                for (int col = 0; col < meshCols - 1; col++) {
                    int vertices = row * meshCols + col;
                    //iterate in CCW order
                    const int vax = vertices;
                    const int vay = vertices + nVertices;
                    const int vbx = vertices + meshCols;
                    const int vby = vertices + meshCols + nVertices;
                    const int vcx = vertices + meshCols + 1;
                    const int vcy = vertices + meshCols + 1 + nVertices;
                    const int vdx = vertices + 1;
                    const int vdy = vertices + 1 + nVertices;
                    solver.addSysElement(rowIdx, vay, weightDOR);
                    solver.addSysElement(rowIdx++, vby, -weightDOR);
                    rhs(rhsRowIdx++) = 0;
                    solver.addSysElement(rowIdx, vdy, weightDOR);
                    solver.addSysElement(rowIdx++, vcy, -weightDOR);
                    rhs(rhsRowIdx++) = 0;
                    solver.addSysElement(rowIdx, vax, weightDOR);
                    solver.addSysElement(rowIdx++, vdx, -weightDOR);
                    rhs(rhsRowIdx++) = 0;
                    solver.addSysElement(rowIdx, vbx, weightDOR);
                    solver.addSysElement(rowIdx++, vcx, -weightDOR);
                    rhs(rhsRowIdx++) = 0;
                }
            }

            constexpr double HARD_CONSTRAINT = 300.0;

            // Set up boundary condition
            for (int row = 0; row < meshRows; row++) {
                int left_bound_vx = row * meshCols;
                int left_bound_vy = row * meshCols + nVertices;
                int right_bound_vx = row * meshCols + meshCols - 1;
                int right_bound_vy = row * meshCols + meshCols - 1 + nVertices;

                solver.addSysElement(rowIdx++, left_bound_vx, HARD_CONSTRAINT);
                rhs(rhsRowIdx++) = 1e-12;

                solver.addSysElement(rowIdx++, right_bound_vx, HARD_CONSTRAINT);
                rhs(rhsRowIdx++) = HARD_CONSTRAINT * (targetWidth - 1);
            }

            for (int col = 0; col < meshCols; col++) {
                int top_bound_vx = col * meshRows;
                int top_bound_vy = col * meshRows + nVertices;
                int bottom_bound_vx = col * meshRows + meshRows - 1;
                int bottom_bound_vy = col * meshRows + meshRows - 1 + nVertices;

                solver.addSysElement(rowIdx++, top_bound_vy, HARD_CONSTRAINT);
                rhs(rhsRowIdx++) = 1e-12;

                solver.addSysElement(rowIdx++, bottom_bound_vy, HARD_CONSTRAINT);
                rhs(rhsRowIdx++) = HARD_CONSTRAINT * (targetHeight - 1);
            }

            solver.setStatus(Numerical::SolverStatus::RhsStable);
            Eigen::VectorXd Vp(columns); ///< deformed vertices to be solved
            solver.solve(Vp, rhs);
            std::cout << Vp << std::endl;
            // Record deformed vertice cooridinate
            for (int row = 0; row < meshRows; row++)
                for (int col = 0; col < meshCols; col++) {
                    int index = row * meshCols + col;
                    cache_mappings[index].deformed_uv_coord(0) = std::round(Vp(index));
                    cache_mappings[index].deformed_uv_coord(1) = std::round(Vp(index + nVertices));
                }
        }

        template <typename T>
        void wrapEachQuad(Eigen::Tensor<T, 3, Eigen::RowMajor>& resizedImage)
        {
            ImageProjectiveTransformOp<double> transformOp{"nearest", "reflect"};
            for (int row = 0; row < meshRows - 1; row++) {
                for (int col = 0; col < meshCols - 1; col++) {
                    int vertices = row * meshCols + col;
                    //iterate in CCW order
                    const int vax = vertices;
                    const int vay = vertices + nVertices;
                    const int vbx = vertices + meshCols;
                    const int vby = vertices + meshCols + nVertices;
                    const int vcx = vertices + meshCols + 1;
                    const int vcy = vertices + meshCols + 1 + nVertices;
                    const int vdx = vertices + 1;
                    const int vdy = vertices + 1 + nVertices;
                }
            }
        }

        template <typename T>
        void reconstructImage(
            const Eigen::Tensor<T, 3, Eigen::RowMajor>& input,
            const Eigen::Tensor<int, 3, Eigen::RowMajor>& segMapping,
            std::vector<Image::Patch>& patches,
            Eigen::Tensor<T, 3, Eigen::RowMajor>& resizedImage)
        {
            const int origH = segMapping.dimension(0);
            const int origW = segMapping.dimension(1);
            resizedImage.resize(targetHeight, targetWidth, 3);

            // Convert image to float type
            Eigen::Tensor<float, 3, Eigen::RowMajor> retargetImgFloat = resizedImage.template cast<float>();

            // Build mesh grid & setup patch mesh
            buildMeshGrid(segMapping, patches);

            // Plot quad mesh upon original image
            drawMeshGrid<T>(input, "InputGrid", patches);

            // Solve constraint for later resizing
            buildAndSolveConstraint(patches, origH, origW);
        }

        static Eigen::Tensor<float, 3, Eigen::RowMajor> applyColorMap(
            const Eigen::Tensor<uint8_t, 3, Eigen::RowMajor>& saliencyMap)
        {
            const int H = saliencyMap.dimension(0);
            const int W = saliencyMap.dimension(1);
            static const int nColorMapping = 5;
            static std::array<float, nColorMapping> rLookUpTable = {255.0, 255.0, 255.0, 0.0, 0.0};
            static std::array<float, nColorMapping> gLookUpTable = {0.0, 125.0, 255.0, 255.0, 0.0};
            static std::array<float, nColorMapping> bLookUpTable = {0.0, 0.0, 0.0, 0.0, 255.0};
            float step = std::ceil(360.0 / nColorMapping);
            Eigen::Tensor<float, 3, Eigen::RowMajor> rgb(H, W, 3);
            for (int row = 0; row < H; row++)
                for (int col = 0; col < W; col++) {
                    float degree = 360 - 360.0 * saliencyMap(row, col, 0) / 255.0;
                    int idx = (degree / step);
                    if (idx < 0)
                        idx = 0;
                    else if (idx >= nColorMapping)
                        idx = nColorMapping - 1;
                    rgb(row, col, r) = rLookUpTable[idx];
                    rgb(row, col, g) = gLookUpTable[idx];
                    rgb(row, col, b) = bLookUpTable[idx];
                }
            return rgb;
        }

        static void assignSignificance(
            const Eigen::Tensor<uint8_t, 3, Eigen::RowMajor>& saliencyMap,
            const Eigen::Tensor<int, 3, Eigen::RowMajor>& segMapping,
            Eigen::Tensor<float, 3, Eigen::RowMajor>& significanceMap,
            std::vector<Image::Patch>& patches)
        {
            const int H = segMapping.dimension(0);
            const int W = segMapping.dimension(1);
            significanceMap.resize(H, W, 3);
            significanceMap.setZero();
            // Create rgb-saliance map for visualization
            Eigen::Tensor<float, 3, Eigen::RowMajor> saliencyMapRGB = Wrapping::applyColorMap(saliencyMap);
            savePNG<uint8_t, 3>("./saliencyMapRGB", saliencyMapRGB.cast<uint8_t>());

            for (int row = 0; row < H; row++)
                for (int col = 0; col < W; col++) {
                    int segId = segMapping(row, col, 0);

                    std::vector<Patch>::iterator patchItr = std::find_if(patches.begin(), patches.end(),
                        [&segId](const Patch& patch) { return patch.segmentId == segId; });

                    // each segmented patch is assigned a significance value by averaging the saliency values
                    // of pixels within this patch
                    float patchSize = (float)patchItr->size;
                    patchItr->saliencyValue += ((float)(saliencyMap(row, col, 0)) / patchSize);

                    // assign significance rgb color to each segmented patch
                    Eigen::array<Index, 3> offset = {row, col, 0};
                    Eigen::array<Index, 3> extent = {1, 1, 3};
                    patchItr->significanceColor += saliencyMapRGB.slice(offset, extent).cast<float>() / patchSize;
                }

            auto comparison = [](const Patch& patchA, const Patch& patchB) {
                return patchA.saliencyValue < patchB.saliencyValue;
            };

            // Normalized saliency value of each patch to 0.1 - 1
            float maxSaliencyValue = (*(std::max_element(patches.begin(), patches.end(), comparison))).saliencyValue;
            float minSaliencyValue = (*(std::min_element(patches.begin(), patches.end(), comparison))).saliencyValue;
            std::for_each(patches.begin(), patches.end(),
                [&maxSaliencyValue, &minSaliencyValue](Patch& p) {
                    p.saliencyValue = (p.saliencyValue - minSaliencyValue) / (maxSaliencyValue - minSaliencyValue);
                    p.saliencyValue = p.saliencyValue * 0.9f + 0.1f;
                });

            // Merge segementation and saliance value to create significance map
            for (int row = 0; row < H; row++)
                for (int col = 0; col < W; col++) {
                    int segId = segMapping(row, col, 0);

                    std::vector<Patch>::iterator patchItr = std::find_if(patches.begin(), patches.end(),
                        [&segId](const Patch& patch) { return patch.segmentId == segId; });
                    Eigen::array<Index, 3> offset = {row, col, 0};
                    Eigen::array<Index, 3> extent = {1, 1, 3};
                    significanceMap.slice(offset, extent) = patchItr->significanceColor;
                }
        }

    private:
        float alpha;
        std::size_t targetHeight, targetWidth;
        float weightDST;
        float weightDLT;
        float weightDOR;
        float quadSize; ///< grid size in pixels
        float quadWidth, quadHeight;
        int meshCols, meshRows;

        int nVertices{0}, nQuads{0}, nVleft{0}, nVright{0}, nVtop{0}, nVbottom{0}; ///< number of vertices
        // Each entry store an transformed coordinates
        std::vector<CachedCoordMapping> cache_mappings;
    };

    std::shared_ptr<Wrapping> createWrapping(std::size_t targetH, std::size_t targetW, float alpha, float quadSize, float weightDST, float weightDLT, float weightDOR)
    {
        std::shared_ptr<Wrapping> imageWarp = std::make_shared<Wrapping>(targetH, targetW, alpha, quadSize, weightDST, weightDLT, weightDOR);
        return imageWarp;
    }

} // namespace Image

#endif
