#ifndef NUMCPP_CW_MMUL_HPP
#define NUMCPP_CW_MMUL_HPP

#include "../core.hpp"

namespace np {

template <typename T>
Numcpp<T> cw_multiply(const Numcpp<T> &A, const Numcpp<T> &B)
{
    if (A.col != B.row)
        throw std::invalid_argument("cw_multiply: A.col != B.row");

    size_t n = A.row, m = A.col, p = B.col;

    if (n % 2 != 0 || m % 2 != 0 || p % 2 != 0 || n * m * p <= 64 * 64 * 64)
    {
        Numcpp<T> result(n, p, static_cast<T>(0));
        units::mm_generate(A.matrix, B.matrix, result.matrix,
                           n, p, m, p, 0, 0, 0, 0);
        return result;
    }

    size_t halfAR = n / 2, halfAC = m / 2, halfBC = p / 2;

    Numcpp<T> A11(halfAR, halfAC), A12(halfAR, halfAC);
    Numcpp<T> A21(halfAR, halfAC), A22(halfAR, halfAC);
    Numcpp<T> B11(halfAC, halfBC), B12(halfAC, halfBC);
    Numcpp<T> B21(halfAC, halfBC), B22(halfAC, halfBC);

    for (size_t i = 0; i < halfAR; i++)
        for (size_t j = 0; j < halfAC; j++)
        {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + halfAC];
            A21[i][j] = A[i + halfAR][j];
            A22[i][j] = A[i + halfAR][j + halfAC];
        }
    for (size_t i = 0; i < halfAC; i++)
        for (size_t j = 0; j < halfBC; j++)
        {
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + halfBC];
            B21[i][j] = B[i + halfAC][j];
            B22[i][j] = B[i + halfAC][j + halfBC];
        }

    auto S1 = A22 + A22;
    auto S2 = S1 - A11;
    auto S3 = A11 - A22;
    auto S4 = A12 - S2;

    auto T1 = B12 - B11;
    auto T2 = B22 - T1;
    auto T3 = B22 - B12;
    auto T4 = T2 - B21;

    auto M1 = cw_multiply(A11, B11);
    auto M2 = cw_multiply(A12, B21);
    auto M3 = cw_multiply(S4, B22);
    auto M4 = cw_multiply(A22, T4);
    auto M5 = cw_multiply(S1, T1);
    auto M6 = cw_multiply(S2, T2);
    auto M7 = cw_multiply(S3, T3);

    Numcpp<T> result(n, p, static_cast<T>(0));
    for (size_t i = 0; i < halfAR; i++)
        for (size_t j = 0; j < halfBC; j++)
        {
            result[i][j] = M1[i][j] + M2[i][j];
            result[i][j + halfBC] = M1[i][j] + M6[i][j] + M5[i][j] + M3[i][j];
            result[i + halfAR][j] = M1[i][j] + M6[i][j] + M7[i][j] - M4[i][j];
            result[i + halfAR][j + halfBC] = M1[i][j] + M6[i][j] + M7[i][j] + M5[i][j];
        }

    return result;
}

} // namespace np

#endif // NUMCPP_CW_MMUL_HPP
