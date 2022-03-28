#ifndef NUMERIC_TYPE_H
#define NUMERIC_TYPE_H
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Sparse>
#include <iostream>
#include <string>
namespace Numerical {
    // Convenience alias

    ///< Alias for dynamic vector, whose dimensions are known at compile time
    template <typename T>
    using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    ///< Alias for sparse matrix
    template <typename T>
    using SparseMatrix = Eigen::SparseMatrix<T>;

    ///< Alias for dense matrix, whose dimension are known at compile time
    template <typename T>
    using DenseMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

} // namespace Numerical

#endif
