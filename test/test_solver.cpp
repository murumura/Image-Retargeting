#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <numerical/cg_solver.h>
#include <numerical/problem.h>
using namespace Numerical;

// we define a new problem for optimizing the rosenbrock function
// we use a templated-class rather than "auto"-lambda function for a clean architecture
template <typename T>
class Rosenbrock : public Problem<T, 2> {
public:
    using typename Problem<T, 2>::TVector;

    // this is just the objective (NOT optional)
    T value(const TVector& x)
    {
        const T t1 = (1 - x[0]);
        const T t2 = (x[1] - x[0] * x[0]);
        return t1 * t1 + 100 * t2 * t2;
    }

    // if you calculated the derivative by hand
    // you can implement it here (OPTIONAL)
    // otherwise it will fall back to (bad) numerical finite differences
    void gradient(const TVector& x, TVector& grad)
    {
        grad[0] = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
        grad[1] = 200 * (x[1] - x[0] * x[0]);
    }
};

TEST(solver, Rosenbrock)
{
    typedef Rosenbrock<float> Rosenbrock;
    // initialize the Rosenbrock-problem
    Rosenbrock f;
    // choose a starting point
    Rosenbrock::TVector x(2);
    x << -1, 2;

    // first check the given derivative
    // there is output, if they are NOT similar to finite differences
    bool probably_correct = f.checkGradient(x);

    // choose a solver
    ConjugatedGradientDescentSolver<Rosenbrock> solver;

    // and minimize the function
    solver.minimize(f, x);
    // print argmin
    std::cout << "argmin      " << x.transpose() << std::endl;
    std::cout << "f in argmin " << f(x) << std::endl;
}
