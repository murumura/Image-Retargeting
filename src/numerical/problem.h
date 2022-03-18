#ifndef PROBLEM_H
#define PROBLEM_H
#include <array>
#include <numerical/types.h>
#include <vector>

namespace Numerical {

    template <typename Scalar_, int Dim_ = Eigen::Dynamic>
    class Problem {
    public:
        static const int Dim = Dim_;
        typedef Scalar_ Scalar;
        using TVector = Eigen::Matrix<Scalar, Dim, 1>;
        using TCriteria = Criteria<Scalar>;
        using TIndex = typename TVector::Index;
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    public:
        Problem() {}
        virtual ~Problem() = default;

        virtual bool callback(const Criteria<Scalar>& state, const TVector& x)
        {
            return true;
        }

        virtual bool detailed_callback(const Criteria<Scalar>& state, SimplexOp op, int index, const MatrixType& x, std::vector<Scalar> f)
        {
            return true;
        }

        /**
         * @brief returns objective value in x
         * @details [long description]
         *
         * @param x [description]
         * @return [description]
        */
        virtual Scalar value(const TVector& x) = 0;
        /**
         * @brief overload value for nice syntax
         * @details [long description]
         *
         * @param x [description]
         * @return [description]
        */
        Scalar operator()(const TVector& x)
        {
            return value(x);
        }
        /**
         * @brief returns gradient in x as reference parameter
         * @details should be overwritten by symbolic gradient
         *
         * @param grad [description]
        */
        virtual void gradient(const TVector& x, TVector& grad)
        {
            finiteGradient(x, grad);
        }

        virtual bool checkGradient(const TVector& x, int accuracy = 3)
        {
            // TODO: check if derived class exists:
            // int(typeid(&Rosenbrock<double>::gradient) == typeid(&Problem<double>::gradient)) == 1 --> overwritten
            const TIndex D = x.rows();
            TVector actual_grad(D);
            TVector expected_grad(D);
            gradient(x, actual_grad);
            finiteGradient(x, expected_grad, accuracy);
            for (TIndex d = 0; d < D; ++d) {
                Scalar scale = std::max(static_cast<Scalar>(std::max(fabs(actual_grad[d]), fabs(expected_grad[d]))), Scalar(1.));
                if (fabs(actual_grad[d] - expected_grad[d]) > 1e-2 * scale)
                    return false;
            }
            return true;
        }

        void finiteGradient(const TVector& x, TVector& grad, int accuracy = 0)
        {
            // accuracy can be 0, 1, 2, 3
            const Scalar eps = 2.2204e-6;
            static const std::array<std::vector<Scalar>, 4> coeff = {{{1, -1}, {1, -8, 8, -1}, {-1, 9, -45, 45, -9, 1}, {3, -32, 168, -672, 672, -168, 32, -3}}};
            static const std::array<std::vector<Scalar>, 4> coeff2 = {{{1, -1}, {-2, -1, 1, 2}, {-3, -2, -1, 1, 2, 3}, {-4, -3, -2, -1, 1, 2, 3, 4}}};
            static const std::array<Scalar, 4> dd = {2, 12, 60, 840};

            grad.resize(x.rows());
            TVector& xx = const_cast<TVector&>(x);

            const int innerSteps = 2 * (accuracy + 1);
            const Scalar ddVal = dd[accuracy] * eps;

            for (TIndex d = 0; d < x.rows(); d++) {
                grad[d] = 0;
                for (int s = 0; s < innerSteps; ++s) {
                    Scalar tmp = xx[d];
                    xx[d] += coeff2[accuracy][s] * eps;
                    grad[d] += coeff[accuracy][s] * value(xx);
                    xx[d] = tmp;
                }
                grad[d] /= ddVal;
            }
        }
    };
} // namespace Numerical

#endif /* PROBLEM_H */
