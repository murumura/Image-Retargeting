#ifndef WRAPPING_H
#define WRAPPING_H
#include <geometry/quad_mesh.h>
#include <image/image.h>
#include <image/utils.h>
#include <iostream>
#include <list>
#include <numerical/cg_solver.h>
#include <numerical/problem.h>
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

        void setPatchMesh(const std::vector<Eigen::Vector2f>& vertices_uv,
            const std::vector<std::pair<Geometry::LocationType, Geometry::LocationType>>& loc_types)
        {
            patchMesh = std::make_shared<Geometry::PatchMesh>(vertices_uv, loc_types);
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
        Eigen::Vector2i pixel_coord;
        int patch_index; // corresponding patch index of stored segment Id
    };

    using namespace Numerical;

    template <typename Scalar>
    class WrappingProblem : public Problem<Scalar> {
    public:
        using typename Problem<Scalar>::TVector;
        float alpha;
        int meshRows, meshCols;
        int newH, newW;
        int origH, origW;
        int nVertices;
        std::vector<Image::Patch> patches;

        explicit WrappingProblem(
            float alpha_,
            std::vector<Image::Patch> patches_,
            int meshRows_,
            int meshCols_,
            int nVertices_,
            int origH_, int origW_,
            int newH_, int newW_) : alpha{alpha_},
                                    patches{patches_},
                                    meshRows{meshRows_},
                                    meshCols{meshCols_},
                                    nVertices{nVertices_},
                                    origH{origH_},
                                    origW{origW_},
                                    newH{newH_},
                                    newW{newW_}
        {
        }

        // set up objective function
        Scalar value(const TVector& Vp)
        {
            const Eigen::Matrix2f L{
                {newH / origH, 0},
                {0, newW / origW}};

            const float high_ratio = L(0, 0);
            const float width_ratio = L(1, 1);

            Scalar D = 0;
            Scalar Dtop = 0;
            Scalar Dbottom = 0;
            Scalar Dleft = 0;
            Scalar Dright = 0;
            for (int i = 0; i < patches.size(); i++) {
                float s_i = patches[i].saliencyValue;
                const std::vector<std::shared_ptr<Geometry::MeshEdge>> edgesList = patches[i].patchMesh->edges;
                if (edgesList.empty())
                    continue;
                const std::shared_ptr<Geometry::MeshEdge> repr_edge = patches[i].reprEdge;
                Eigen::Vector2f c = repr_edge->v[0]->uv - repr_edge->v[1]->uv;

                Eigen::Matrix2f M{
                    {c(1), c(0)},
                    {-c(0), c(1)},
                };

                Eigen::Matrix2f M_inv = M.inverse();

                const Eigen::Vector2i repr_vertices = repr_edge->deserialize(meshCols);
                const int C1x = repr_vertices(0);
                const int C1y = repr_vertices(0) + nVertices;
                const int C2x = repr_vertices(1);
                const int C2y = repr_vertices(1) + nVertices;

                for (int j = 0; j < edgesList.size(); j++) {
                    const Eigen::Vector2f e = edgesList[j]->v[0]->uv - edgesList[j]->v[1]->uv;
                    const Eigen::Vector2f s_r = M_inv * e;
                    const float s = s_r(0);
                    const float r = s_r(1);

                    const Eigen::Vector2i vertices = edgesList[j]->deserialize(meshCols);
                    const int v1x = vertices(0);
                    const int v1y = vertices(0) + nVertices;
                    const int v2x = vertices(1);
                    const int v2y = vertices(1) + nVertices;

                    // clang-format off
                    // Set up patch transformation constraint
                    const Scalar DST =  \
                            alpha * s_i * 
                            (
                                ( (Vp[v1x] - Vp[v2x]) -  (s * (Vp[C1x] - Vp[C2x]) + r * (Vp[C1y] - Vp[C2y]) ) ) * 
                                ( (Vp[v1x] - Vp[v2x]) -  (s * (Vp[C1x] - Vp[C2x]) + r * (Vp[C1y] - Vp[C2y]) ) )
                                                                                     + 
                                ( (Vp[v1y] - Vp[v2y]) -  (-r * (Vp[C1x] - Vp[C2x]) + s * (Vp[C1y] - Vp[C2y]) ) ) * 
                                ( (Vp[v1y] - Vp[v2y]) -  (-r * (Vp[C1x] - Vp[C2x]) + s * (Vp[C1y] - Vp[C2y]) ) )
                            );

                    const Scalar DLT = \    
                            (1 - alpha) * (1 - s_i) *
                            (
                                ( (Vp[v1x] - Vp[v2x]) -  (high_ratio * s * (Vp[C1x] - Vp[C2x]) + high_ratio * r * (Vp[C1y] - Vp[C2y]) ) ) * 
                                ( (Vp[v1x] - Vp[v2x]) -  (high_ratio * s * (Vp[C1x] - Vp[C2x]) + high_ratio * r * (Vp[C1y] - Vp[C2y]) ) )
                                                                                     + 
                                ( (Vp[v1y] - Vp[v2y]) -  (width_ratio * -r * (Vp[C1x] - Vp[C2x]) + width_ratio * s * (Vp[C1y] - Vp[C2y]) ) ) * 
                                ( (Vp[v1y] - Vp[v2y]) -  (width_ratio * -r * (Vp[C1x] - Vp[C2x]) + width_ratio * s * (Vp[C1y] - Vp[C2y]) ) )
                            );
                    // clang-format on
                    D += (DST + DLT);
                }
            }

            // Set up grid orientation constraint
            for (int row = 0; row < meshRows - 1; row++) 
            {
                for (int col = 0; col < meshCols - 1; col++) 
                {
                    int vertices = row * meshCols + col;
                    const int vax = vertices;
                    const int vay = vertices + nVertices;
                    const int vbx = vertices + meshCols;
                    const int vby = vertices + meshCols + nVertices;
                    const int vcx = vertices + meshCols + 1;
                    const int vcy = vertices + meshCols + 1 + nVertices;
                    const int vdx = vertices + 1;
                    const int vdy = vertices + 1 + nVertices;
                    // clang-format off
                    const Scalar DOR = \
                        (Vp[vay] - Vp[vby]) * (Vp[vay] - Vp[vby]) + \
                        (Vp[vdy] - Vp[vcy]) * (Vp[vdy] - Vp[vcy]) + \
                        (Vp[vax] - Vp[vdx]) * (Vp[vax] - Vp[vdx]) + \
                        (Vp[vbx] - Vp[vcx]) * (Vp[vbx] - Vp[vcx]);
                    // clang-format on
                    D += DOR;
                }
            }

            return D;
        }
    };

    class Wrapping {
    public:
        explicit Wrapping(std::size_t targetHeight_, std::size_t targetWidth_, float alpha_, float quadSize_)
            : alpha{alpha_}, targetHeight{targetHeight_}, targetWidth{targetWidth_}, quadSize{quadSize_}
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
            cache_mappings.reserve(meshRows * meshCols);

            for (int row = 0; row < meshRows; row++)
                for (int col = 0; col < meshCols; col++) {
                    int index = row * meshCols + col;
                    cache_mappings[index].pixel_coord = coordTransform(row, col);
                    int segId = segMapping(cache_mappings[index].pixel_coord(0), cache_mappings[index].pixel_coord(1), 0);
                    cache_mappings[index].patch_index = findPatchIndex(segId);
                }

            std::vector<std::vector<Eigen::Vector2f>> vertices_uvs(patches.size());
            std::vector<std::vector<std::pair<Geometry::LocationType, Geometry::LocationType>>> vertices_locs(patches.size());

            std::function<std::pair<Geometry::LocationType, Geometry::LocationType>(int, int)> locationType = [&](int row, int col) {
                Geometry::LocationType typeY{Geometry::LocationType::Regular}, typeX{Geometry::LocationType::Regular};
                // precache boundary conditions for later constraint
                if (row == 0)
                    typeY = Geometry::LocationType::TopBoundary;
                else if (row == meshRows - 1)
                    typeY = Geometry::LocationType::BottomBoundary;
                if (col == 0)
                    typeX = Geometry::LocationType::LeftBoundary;
                else if (col == meshCols - 1)
                    typeX = Geometry::LocationType::RightBoundary;
                return std::make_pair(typeY, typeX);
            };

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
                        vertices_locs[pidx1].insert(vertices_locs[pidx1].end(), {
                                                                                    locationType(r1, c1),
                                                                                    locationType(r2, c2),
                                                                                });

                        if (pidx1 != pidx2) {
                            vertices_uvs[pidx2].insert(vertices_uvs[pidx2].end(), {Eigen::Vector2f{r1, c1}, Eigen::Vector2f{r2, c2}});
                            vertices_locs[pidx2].insert(vertices_locs[pidx2].end(), {
                                                                                        locationType(r1, c1),
                                                                                        locationType(r2, c2),
                                                                                    });
                        }
                    }
                }

            // Maintain edge list of each patch
            for (int i = 0; i < patches.size(); i++)
                patches[i].setPatchMesh(vertices_uvs[i], vertices_locs[i]);
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
            typedef WrappingProblem<double> WrappingProblem;
            typedef typename WrappingProblem::TVector TVector;
            typedef typename WrappingProblem::MatrixType MatrixType;
            // Initialize wrapping problem
            WrappingProblem f(alpha, patches, meshRows, meshCols, nVertices, origH, origW, targetHeight, targetWidth);
            std::cout << nVertices << std::endl;
            // Create deformed vercices to be solved
            TVector Vp = TVector::Zero(nVertices * 2);

            // first check the given derivative
            // there is output, if they are NOT similar to finite differences
            bool probably_correct = f.checkGradient(Vp);
            Criteria<double> crit = Criteria<double>::defaults(); // Create a Criteria class to set the solver's stop conditions
            crit.iterations = 10000;                              

            // choose a solver
            ConjugatedGradientDescentSolver<WrappingProblem> solver;
            solver.setStopCriteria(crit);
            std::cout << "Start to minimize ...";
            // and minimize the function
            solver.minimize(f, Vp);
            std::cout << "Done" << std::endl;

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

            // Normalized saliency value of each patch to 0 - 1
            float maxSaliencyValue = (*(std::max_element(patches.begin(), patches.end(), comparison))).saliencyValue;
            float minSaliencyValue = (*(std::min_element(patches.begin(), patches.end(), comparison))).saliencyValue;
            std::for_each(patches.begin(), patches.end(),
                [&maxSaliencyValue, &minSaliencyValue](Patch& p) {
                    p.saliencyValue = (p.saliencyValue - minSaliencyValue) / (maxSaliencyValue - minSaliencyValue);
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
        float quadSize; ///< grid size in pixels
        float quadWidth, quadHeight;
        int meshCols, meshRows;

        int nVertices; ///< number of vertices
        // Each entry store an transformed coordinates
        std::vector<CachedCoordMapping> cache_mappings;
    };

    std::shared_ptr<Wrapping> createWrapping(std::size_t targetH, std::size_t targetW, float alpha, float quadSize)
    {
        std::shared_ptr<Wrapping> imageWarp = std::make_shared<Wrapping>(targetH, targetW, alpha, quadSize);
        return imageWarp;
    }

} // namespace Image

#endif
