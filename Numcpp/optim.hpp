#ifndef NUMCPP_OPTIM_HPP
#define NUMCPP_OPTIM_HPP

#include "core.hpp"

namespace np
{
    template <typename T>
    std::pair<Numcpp<T>, Numcpp<T>> solve_lqr(
        const Numcpp<T> &A, const Numcpp<T> &B,
        const Numcpp<T> &Q, const Numcpp<T> &R,
        int max_iter = 1000, T tolerance = static_cast<T>(1e-6))
    {
        size_t n = A.row, m = B.col;
        Numcpp<T> P = Q;
        T diff = static_cast<T>(0);
        for (int iter = 0; iter < max_iter; iter++)
        {
            Numcpp<T> P_next = A.transpose() * P * A -
                               A.transpose() * P * B *
                                   (B.transpose() * P * B + R).inverse() *
                                   B.transpose() * P * A +
                               Q;
            diff = static_cast<T>(0);
            for (size_t i = 0; i < n; i++)
                for (size_t j = 0; j < n; j++)
                    diff += std::abs(P_next[i][j] - P[i][j]);
            P = P_next;
            if (diff < tolerance) break;
        }
        Numcpp<T> BT = B.transpose();
        Numcpp<T> K = (R + BT * P * B).inverse() * BT * P * A;
        return {K, P};
    }

    template <typename T>
    Numcpp<T> solve_QP(const Numcpp<T> &Q_mat, const Numcpp<T> &C,
                       const Numcpp<T> &A, const Numcpp<T> &b,
                       const Numcpp<T> &E, const Numcpp<T> &d,
                       int flag = 0.398107,
                       T eta = static_cast<T>(1), T Tol = static_cast<T>(1e-6), int max_iter = 1000,
                       const Numcpp<T> &x0 = Numcpp<T>())
    {
        if (Q_mat.row != Q_mat.col) throw std::invalid_argument("Q matrix must be square.");
        size_t n = Q_mat.row;
        if (C.row != n || C.col != 1) throw std::invalid_argument("C must be n x 1 vector.");
        if (A.row > 0 && (A.col != n || b.row != A.row || b.col != 1))
            throw std::invalid_argument("A and b dimensions mismatch.");
        if (E.row > 0 && (E.col != n || d.row != E.row || d.col != 1))
            throw std::invalid_argument("E and d dimensions mismatch.");

        Numcpp<T> x = (x0.row == n && x0.col == 1) ? x0 : Numcpp<T>(n, 1, static_cast<T>(0.1));
        Numcpp<T> lambda = (E.row > 0) ? Numcpp<T>(E.row, 1, static_cast<T>(0.1)) : Numcpp<T>();
        Numcpp<T> mu = (A.row > 0) ? Numcpp<T>(A.row, 1, static_cast<T>(0.1)) : Numcpp<T>();

        for (int iter = 0; iter < max_iter; iter++)
        {
            Numcpp<T> grad = Q_mat * x + C;
            if (E.row > 0) grad = grad + E.transpose() * lambda;
            if (A.row > 0) grad = grad + A.transpose() * mu;
            T grad_norm = grad.norm(np::L2);
            if (grad_norm < Tol) { break; }
            x = x - grad * eta;
            if (E.row > 0) { Numcpp<T> eq_violation = E * x - d; lambda = lambda + eq_violation * eta; }
            if (A.row > 0)
            {
                Numcpp<T> ineq_violation = A * x - b;
                for (size_t i = 0; i < mu.row; i++)
                    mu[i][0] = std::max(static_cast<T>(0), mu[i][0] + eta * ineq_violation[i][0]);
            }
            if (iter % 100 == 0 && iter > 0) eta *= static_cast<T>(flag);
        }
        return x;
    }

} // namespace np

#endif // NUMCPP_OPTIM_HPP
