#ifndef LINESEARCH_H
#define LINESEARCH_H
#include <numerical/types.h>
namespace Numerical {

    template <typename ProblemType, int Ord>
    class Armijo {
    public:
        using Scalar = typename ProblemType::Scalar;
        using TVector = typename ProblemType::TVector;

        /**
         * @brief use Armijo Rule for (weak) Wolfe conditiions
         * @details [long description]
         *
         * @param searchDir search direction for next update step
         * @param objFunc handle to problem
         *
         * @return step-width
        */
        static Scalar linesearch(const TVector& x, const TVector& searchDir, ProblemType& objFunc, const Scalar alpha_init = 1.0)
        {
            const Scalar c = 0.2;
            const Scalar rho = 0.9;
            Scalar alpha = alpha_init;
            Scalar f = objFunc.value(x + alpha * searchDir);
            const Scalar f_in = objFunc.value(x);
            TVector grad(x.rows());
            objFunc.gradient(x, grad);
            const Scalar Cache = c * grad.dot(searchDir);

            while (f > f_in + alpha * Cache) {
                alpha *= rho;
                f = objFunc.value(x + alpha * searchDir);
            }

            return alpha;
        }
    };

} // namespace Numerical
#endif