#ifndef SOLVER_H
#define SOLVER_H
#include <numerical/problem.h>
#include <numerical/types.h>
namespace Numerical {
    template <typename ProblemType, int Ord>
    class Solver {
    public:
        using Scalar = typename ProblemType::Scalar;
        using TVector = typename ProblemType::TVector;
        using TCriteria = typename ProblemType::TCriteria;

    protected:
        const int order_ = Ord;
        TCriteria m_stop, m_current;
        Status m_status = Status::NotStarted;
        DebugLevel m_debug = DebugLevel::None;

    public:
        virtual ~Solver() = default;
        Solver()
        {
            m_stop = TCriteria::defaults();
            m_current.reset();
        }

        Solver(const TCriteria& s)
        {
            m_stop = s;
            m_current.reset();
        }

        void setStopCriteria(const TCriteria& s) { m_stop = s; }
        const TCriteria& criteria() { return m_current; }
        const Status& status() { return m_status; }
        void setDebug(const DebugLevel& d) { m_debug = d; }

        /**
     * @brief minimize an objective function given a gradient (and optinal a hessian)
     * @details this is just the abstract interface
     *
     * @param x0 starting point
     * @param funObjective objective function
     * @param funGradient gradient function
     * @param funcHession hessian function
     */
        virtual void minimize(ProblemType& objFunc, TVector& x0) = 0;
    };

} // namespace Numerical
#endif