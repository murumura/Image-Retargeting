#ifndef CG_SOLVER_H
#define CG_SOLVER_H
#include <numerical/types.h>
#include <unordered_map>
#include <vector>
namespace Numerical {

    enum class SolverStatus {
        NotStarted = -1,
        Continue = 0,
        RhsStable = 1,
        IterationLimit,
        XDeltaTolerance,
        FDeltaTolerance,
        GradNormTolerance,
        Condition,
        UserDefined
    };

    template <typename MatrixType, typename Rhs, typename Dest>
    void lscgSolve(const MatrixType& mat, const Rhs& rhs, Dest& x, Eigen::Index& iters,
        typename Dest::RealScalar& tol_error)
    {
        Eigen::LeastSquaresConjugateGradient<MatrixType> lscg;
        lscg.setMaxIterations(iters);
        lscg.setTolerance(tol_error);
        lscg.compute(mat);
        x = lscg.solveWithGuess(rhs, x);
        tol_error = lscg.tolerance();
        iters = lscg.iterations();
    }

    class CGSolver {
    public:
        CGSolver(int rows, int cols)
            : status{SolverStatus::NotStarted}, rows(rows), cols(cols)
        {
        }

        void addSysElement(int row, int col, double value)
        {
            tripletList.push_back(Eigen::Triplet<double>{row, col, value});
            status = SolverStatus::Continue;
        }

        void setStatus(SolverStatus status_)
        {
            status = status_;
        }

        template <typename Dest, typename Rhs>
        void solve(Dest& x, const Rhs& rhs, const int maxIters = 5000)
        {
            if (status != SolverStatus::RhsStable)
                throw std::runtime_error("Need to fill right handside");
            // build linear system
            Eigen::SparseMatrix<double> A;
            A.resize(rows, cols);
            A.setFromTriplets(tripletList.begin(), tripletList.end());
            double tol_error = Eigen::NumTraits<double>::epsilon();
            Eigen::Index iters = maxIters;
            lscgSolve(A, rhs, x, iters, tol_error);
            std::cout << "#iterations:     " << iters << std::endl;
            std::cout << "estimated error: " << tol_error << std::endl;
            setStatus(SolverStatus::NotStarted);
        }

    private:
        Eigen::VectorXd b;
        int rows, cols;
        std::vector<Eigen::Triplet<double>> tripletList;
        SolverStatus status;
    };
} // namespace Numerical

#endif
