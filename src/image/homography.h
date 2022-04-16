#ifndef HOMOGRAPHY_H
#define HOMOGRAPHY_H
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Householder>
#include <image/image.h>
#include <numerical/types.h>
namespace Image {
    namespace {
        typedef Eigen::Matrix<float, 8, 1> Vector8f;
    }
    enum SystemSolverMethode {
        FULL_PIV_LU,
        PARTIAL_PIV_LU,
        FULL_PIV_QR,
        PARTIAL_QR,
        QR,
        COMPLETE_ORTHOGONAL_DECOMPOSITION,
        LLT,
        LDLT,
        JACOBI_SCD,
        BDCSVD //! warning: https://eigen.tuxfamily.org/dox/classEigen_1_1BDCSVD.html
    };

    template <typename Scalar>
    auto perspectiveMatrixToflatTensor(const Numerical::DenseMatrix<Scalar>& matrix)
    {
        assert(matrix.rows() == 3 && matrix.cols() == 3);
        Eigen::Tensor<Scalar, 1, Eigen::RowMajor> transform(8);
        transform.setValues(
            {matrix(0, 0), matrix(0, 1), matrix(0, 2),
                matrix(1, 0), matrix(1, 1), matrix(1, 2),
                matrix(2, 0), matrix(2, 1)});
        return transform;
    }

    /**
     * @brief   Calculates the homography matrix (3x3) to convert between two points of view:
     *  P_dst = H * P_src; useing 4 pairs of corresponding Points.
     * @param[in] src  The 4 Points representation source points.
     * @param[in] dst  The 4 Points representation destion points.
     * @param[in] method  Which methode should be used to calculate Ax=b. Have a look above which are avaiable.
     * 
     * @return Eigen::Matrix3d found homography.
     */
    template <SystemSolverMethode methode>
    static Eigen::Matrix3f findHomography(const std::array<Eigen::Vector2f, 4>& src, const std::array<Eigen::Vector2f, 4>& dst)
    {
        typedef Eigen::Matrix<float, 8, 8> HomograpyMatrix;
        HomograpyMatrix PH;
        Vector8f b;
        for (unsigned int i = 0, j = 0; i < 4; i++) {

            const float srcX = src[i](0);
            const float srcY = src[i](1);
            const float dstX = dst[i](0);
            const float dstY = dst[i](1);

            // clang-format off
            b(j) = dstX;
            PH.row(j++) << srcX, srcY,  1.,   0.,   0.,  0., -srcX*dstX, -srcY*dstX;
            b(j) = dstY;
            PH.row(j++) <<   0.,   0.,  0., srcX, srcY,  1., -srcX*dstY, -srcY*dstY;
            // clang-format on
        }

        // solv PH*x=b
        Vector8f x;
        Eigen::Matrix3f result;

        if constexpr (methode == SystemSolverMethode::FULL_PIV_LU) {
            Eigen::FullPivLU<HomograpyMatrix> lu(PH);
            x = lu.solve(b);
        }
        else if constexpr (methode == SystemSolverMethode::PARTIAL_PIV_LU) {
            Eigen::PartialPivLU<HomograpyMatrix> plu(PH);
            x = plu.solve(b);
        }
        else if constexpr (methode == SystemSolverMethode::FULL_PIV_QR) {
            Eigen::FullPivHouseholderQR<HomograpyMatrix> fqr(PH);
            x = fqr.solve(b);
        }
        else if constexpr (methode == SystemSolverMethode::PARTIAL_QR) {
            Eigen::ColPivHouseholderQR<HomograpyMatrix> cqr(PH);
            x = cqr.solve(b);
        }
        else if constexpr (methode == SystemSolverMethode::QR) {
            Eigen::HouseholderQR<HomograpyMatrix> qr(PH);
            x = qr.solve(b);
        }
        else if constexpr (methode == SystemSolverMethode::LLT) {
            x = PH.llt().solve(b);
            x = (PH.transpose() * PH).llt().solve(PH.transpose() * b);
        }
        else if constexpr (methode == SystemSolverMethode::LDLT) {
            x = (PH.transpose() * PH).ldlt().solve(PH.transpose() * b);
        }
        else if constexpr (methode == SystemSolverMethode::COMPLETE_ORTHOGONAL_DECOMPOSITION) {
            Eigen::CompleteOrthogonalDecomposition<HomograpyMatrix> cod(PH);
            x = cod.solve(b);
        }
        else {
            const std::string error = "findHomography: Given Template parameter is not supportet.";
            assert(error.c_str() && false);
        }

        // const float relative_error = (PH*x - b).norm() / b.norm(); // norm() is L2 norm

        Eigen::Vector3f x1, x2, x3;
        x1 << x.head<3>();
        x2 << x.segment<3>(3);
        x3 << x.tail<2>(), 1.;

        // result << x, 1.; //doesnt work for reasons
        result << x1, x2, x3;
        return result.transpose();
    }

} // namespace Image

#endif