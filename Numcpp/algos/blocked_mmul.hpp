#ifndef NUMCPP_ALGOS_BLOCKED_MMUL_HPP
#define NUMCPP_ALGOS_BLOCKED_MMUL_HPP

#include "../core.hpp"

namespace np {

template <typename T>
Numcpp<T> blocked_multiply(const Numcpp<T> &A, const Numcpp<T> &B,
                           size_t block_size = 64, unsigned threads = 0)
{
    if (A.col != B.row)
        throw std::invalid_argument("blocked_multiply: A.col != B.row");

    size_t M = A.row, K = A.col, N = B.col;
    Numcpp<T> C(M, N, static_cast<T>(0));

    if (block_size == 0) block_size = 64;
    size_t bs = block_size;

    if (threads > 1)
    {
        auto &pool = units::pool();
        pool.ensure(threads);

        size_t n_tiles_i = (M + bs - 1) / bs;
        units::CompletionBarrier barrier(n_tiles_i);

        for (size_t ti = 0; ti < n_tiles_i; ti++)
        {
            size_t i0 = ti * bs;
            size_t i1 = std::min(i0 + bs, M);

            pool.enqueue([&A, &B, &C, i0, i1, K, N, bs, &barrier]()
            {
                for (size_t i = i0; i < i1; i++)
                {
                    for (size_t kk = 0; kk < K; kk += bs)
                    {
                        size_t k1 = std::min(kk + bs, K);
                        for (size_t j = 0; j < N; j += bs)
                        {
                            size_t j1 = std::min(j + bs, N);
                            for (size_t k = kk; k < k1; k++)
                            {
                                T aik = A[i][k];
                                for (size_t jj = j; jj < j1; jj++)
                                    C[i][jj] += aik * B[k][jj];
                            }
                        }
                    }
                }
                barrier.arrive();
            });
        }
        barrier.wait();
    }
    else
    {
        for (size_t kk = 0; kk < K; kk += bs)
        {
            size_t k1 = std::min(kk + bs, K);
            for (size_t i = 0; i < M; i++)
            {
                for (size_t k = kk; k < k1; k++)
                {
                    T aik = A[i][k];
                    for (size_t j = 0; j < N; j += bs)
                    {
                        size_t j1 = std::min(j + bs, N);
                        for (size_t jj = j; jj < j1; jj++)
                            C[i][jj] += aik * B[k][jj];
                    }
                }
            }
        }
    }

    return C;
}

} // namespace np

#endif // NUMCPP_ALGOS_BLOCKED_MMUL_HPP
