#ifndef __NUMCPP__H__
#define __NUMCPP__H__
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <thread>
#include <functional>
#include <type_traits>
#include <complex>
#include <type_traits>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#define NP_PI 3.14159265358979

// 复数的特殊判断
template <typename T>
struct is_complex : std::false_type
{
};
template <typename T>
struct is_complex<std::complex<T>> : std::true_type
{
};
template <typename T>
constexpr bool is_complex_v = is_complex<T>::value;

#define MATtoPtr2D(T, value_name, change_name, row, col) \
    T *change_name[col];                                 \
    for (size_t i = 0; i < row; i++)                     \
    {                                                    \
        change_name[i] = mat[i];                         \
    }
#ifdef OPENCV_ALL_HPP
#define NUMCPP_OPENCV_SUPPORT
#endif // OPENCV_ALL_HPP

// cuda code
#if CUDA_CHECK
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cusolverDn.h>
#define DEVICE_CUDA 1
#define DEVICE_LOCAL 0
namespace cuda_op
{
    template <typename T>
    using func_At = T (*)(T);
    template <typename T>
    using func_Bt = T (*)(T, T);
    template <typename T>
    using func_Ct = T (*)(T, T, T);

    // x + y
    template <typename T>
    __device__ T r_add_opB(T x, T y)
    {
        return x + y;
    }
    template <typename T>
    __device__ func_Bt<T> add_opB = r_add_opB;

    // x - y
    template <typename T>
    __device__ T r_cut_opB(T x, T y)
    {
        return x - y;
    }
    template <typename T>
    __device__ func_Bt<T> cut_opB = r_cut_opB;

    // x * y
    template <typename T>
    __device__ T r_mul_opB(T x, T y)
    {
        return x * y;
    }
    template <typename T>
    __device__ func_Bt<T> mul_opB = r_mul_opB;

    // x / y (y != 0)
    template <typename T>
    __device__ T r_div_opB(T x, T y)
    {
        return x / y;
    }
    template <typename T>
    __device__ func_Bt<T> div_opB = r_div_opB;

    // x = y + z
    template <typename T>
    __device__ T r_add_opC(T x, T y, T z)
    {
        return y + z;
    }
    template <typename T>
    __device__ func_Ct<T> add_opC = r_add_opC;

    // x = y - z
    template <typename T>
    __device__ T r_cut_opC(T x, T y, T z)
    {
        return y - z;
    }
    template <typename T>
    __device__ func_Ct<T> cut_opC = r_cut_opC;

    // __global__ function
    template <typename T>
    __global__ void kernel_unary_op(T **mat, func_At<T> op)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        mat[row][col] = (*op)(mat[row][col]);
    }
    template <typename T>
    __global__ void kernel_nummul_op(T **mat, T value)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        mat[row][col] *= value;
    }
    template <typename T>
    __global__ void kernel_numdiv_op(T **mat, T value)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        mat[row][col] /= value;
    }
    template <typename T>
    __global__ void kernel_numadd_op(T **mat, T value)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        mat[row][col] += value;
    }
    template <typename T>
    __global__ void kernel_numcut_op(T **mat, T value)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        mat[row][col] -= value;
    }

    template <typename T>
    __global__ void kernel_binary_op(T **a, T **b, func_Bt<T> op)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        a[row][col] = (*op)(a[row][col], b[row][col]);
    }

    template <typename T>
    __global__ void kernel_ternary_op(T **a, T **b, T **c, func_Ct<T> op)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        a[row][col] = (*op)(a[row][col], b[row][col], c[row][col]);
    }

    template <typename T>
    __global__ void kernel_gemm(T **A, T **B, T **C,
                                size_t M, size_t N, size_t K,
                                T alpha, T beta)
    {
        size_t row = blockIdx.y * blockDim.y + threadIdx.y;
        size_t col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < M && col < K)
        {
            T sum = (T)0;
            for (size_t i = 0; i < N; ++i)
            {
                sum += A[row][i] * B[i][col];
            }
            C[row][col] = alpha * sum + beta * C[row][col];
        }
    }
    // 矩阵乘法 - 使用迭代器优化的CUDA实现
    template <typename T>
    void gemm(T **A, size_t A_row, size_t A_col, T **B, size_t B_col, T **C,
              T alpha = 1.0, T beta = 0.0)
    {
        dim3 block(A_row, B_col);
        kernel_gemm<T><<<block>>>(A, B, C, A_row, A_col, B_col,
                                  alpha, beta);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << __func__ << "()::" << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
    }
    // 矩阵转置CUDA核函数
    template <typename T>
    __global__ void kernel_transpose(T **mat, T **transposed, size_t rows, size_t cols)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < rows && col < cols)
        {
            transposed[col][row] = mat[row][col];
        }
    }

    // 计算A^T * A
    template <typename T>
    void compute_ATA(T **A, T **ATA, size_t m, size_t n)
    {
        // 分配转置矩阵内存
        T **A_T;
        cudaMalloc((void **)&A_T, n * sizeof(T *));
        for (size_t i = 0; i < n; i++)
        {
            cudaMalloc((void **)&A_T[i], m * sizeof(T));
        }

        // 执行转置
        dim3 block(16, 16);
        dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        kernel_transpose<T><<<grid, block>>>(A, A_T, m, n);
        cudaDeviceSynchronize();

        // 计算A^T * A
        cuda_op::gemm<T>(A_T, n, m, A, n, ATA, 1.0, 0.0);

        // 释放临时内存
        for (size_t i = 0; i < n; i++)
        {
            cudaFree(A_T[i]);
        }
        cudaFree(A_T);
    }

    // 幂法求最大特征值和特征向量
    template <typename T>
    T power_method(T **mat, T **vec, size_t size, int max_iter = 1000, T tol = 1e-6)
    {
        T *temp = new T[size];
        T lambda_old = 0.0, lambda_new = 0.0;

        // 初始化特征向量
        for (size_t i = 0; i < size; i++)
        {
            vec[i][0] = 1.0 / sqrt((T)size);
        }

        for (int iter = 0; iter < max_iter; iter++)
        {
            // 矩阵-向量乘法
            for (size_t i = 0; i < size; i++)
            {
                temp[i] = 0.0;
                for (size_t j = 0; j < size; j++)
                {
                    temp[i] += mat[i][j] * vec[j][0];
                }
            }

            // 计算新的特征值（Rayleigh商）
            lambda_new = 0.0;
            T vec_norm = 0.0;
            for (size_t i = 0; i < size; i++)
            {
                lambda_new += vec[i][0] * temp[i];
                vec_norm += temp[i] * temp[i];
            }
            vec_norm = sqrt(vec_norm);

            // 归一化特征向量
            for (size_t i = 0; i < size; i++)
            {
                vec[i][0] = temp[i] / vec_norm;
            }

            // 检查收敛
            if (fabs(lambda_new - lambda_old) < tol)
            {
                break;
            }
            lambda_old = lambda_new;
        }

        delete[] temp;
        return sqrt(lambda_new); // 奇异值是特征值的平方根
    }

    // SVD主函数
    template <typename T>
    void cuda_svd(T **A, T **U, T **S, T **V, size_t m, size_t n)
    {
        cudaError_t err;
        T **ATA = (T **)malloc(n * sizeof(T *));
        for (size_t i = 0; i < n; i++)
        {
            err = cudaMalloc((void **)&ATA[i], n * sizeof(T));
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA malloc error in cuda_svd: " << cudaGetErrorString(err) << std::endl;
                return;
            }
        }

        // 计算 A^T * A
        compute_ATA<T>(A, ATA, m, n);

        // 只计算最大奇异值（演示用）
        T **vec = (T **)malloc(n * sizeof(T *));
        for (size_t i = 0; i < n; i++)
        {
            err = cudaFree(ATA[i]);
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA free error: " << cudaGetErrorString(err) << std::endl;
            }
        }

        // 计算最大特征值
        T max_singular = power_method<T>(ATA, vec, n);

        // 设置S矩阵（只设置最大的）
        for (size_t i = 0; i < std::min(m, n); i++)
        {
            if (i == 0)
                S[i][i] = max_singular;
            else
                S[i][i] = (T)0;
        }

        // 清理内存
        for (size_t i = 0; i < n; i++)
        {
            cudaFree(ATA[i]);
            cudaFree(vec[i]);
        }
        free(ATA);
        free(vec);
    }

} // namespace cuda_iterator
#endif
namespace units
{
    template <typename T>
    void trans(std::complex<T> **x, size_t size_x)
    {
        size_t p = 0;
        size_t a, b;
        for (size_t i = 1; i < size_x; i *= 2)
        {
            p++; // 计算二进制位数
        }
        for (size_t i = 0; i < size_x; i++)
        {
            a = i;
            b = 0;
            for (size_t j = 0; j < p; j++)
            {
                b = (b << 1) + (a & 1); // b存储当前下标的回文值
                a = a >> 1;
            }
            if (b > i)
            { // 避免重复交换
                std::complex<T> temp = (*x)[i];
                (*x)[i] = (*x)[b];
                (*x)[b] = temp;
            }
        }
    }
    template <typename T>
    void fft(std::complex<T> **x, size_t size, std::complex<T> **X, int inv)
    {
        // 分配旋转因子数组
        std::complex<T> *Wn = new std::complex<T>[size];
        for (size_t i = 0; i < size; i++)
        {
            T angle = static_cast<T>(-2 * NP_PI * i / size);
            Wn[i] = std::complex<T>(cos(angle), inv * sin(angle));
        }

        // 复制输入到输出（支持原位计算）
        if (x != X)
        {
            for (size_t i = 0; i < size; i++)
            {
                (*X)[i] = (*x)[i];
            }
        }

        // 位反转置换
        std::complex<T> *p = *X;
        trans<T>(&p, size);

        // FFT计算
        for (size_t m = 2; m <= size; m *= 2)
        {
            for (size_t k = 0; k < size; k += m)
            {
                for (size_t j = 0; j < m / 2; j++)
                {
                    size_t index1 = k + j;
                    size_t index2 = index1 + m / 2;
                    size_t t = j * size / m; // 旋转因子索引
                    std::complex<T> temp1 = (*X)[index1];
                    std::complex<T> temp2 = (*X)[index2] * Wn[t];
                    (*X)[index1] = temp1 + temp2;
                    (*X)[index2] = temp1 - temp2;
                }
            }
        }

        delete[] Wn; // 释放旋转因子数组
    }
    /* 添加偏置offset用以支持分块迭代 */
    template <typename T>
    void mm_generate(T **this_matrix, T **other_matrix, T **result, const size_t A_row, const size_t B_row,
                     const size_t A_col, const size_t B_col, const size_t A_row_offset, const size_t A_col_offset, const size_t B_row_offset, const size_t B_col_offset)
    {
        for (size_t i = 0; i < A_row; i++)
        {
            for (size_t j = 0; j < B_col; j++)
            {
                for (size_t k = 0; k < A_col; k++)
                {
                    result[i][j] += (this_matrix[i + A_row_offset][k + A_col_offset] * other_matrix[k + B_row_offset][j + B_col_offset]);
                }
            }
        }
    };
    template <typename T>
    T **mat_create(size_t row, size_t col)
    {
        T **matrix = new T *[row];
        for (size_t i = 0; i < row; i++)
        {
            matrix[i] = new T[col];
            for (size_t j = 0; j < col; j++)
            {
                matrix[i][j] = (T)0;
            }
        }
        return matrix;
    };
    template <typename T>
    void mat_delete(T **mat, size_t row)
    {
        for (size_t i = 0; i < row; i++)
        {
            delete mat[i];
        }
        delete[] mat;
    };

    /*
     * this_matrix A_row * A_col
     * other_matrix A_col * B_col
     * result A_row * B_col
     * result = this_matrix * other_matrix
     * S1 = A21 + A22     T1 = B12 - B11
     * S2 = S1 - A11      T2 = B22 - T1
     * S3 = A11 - A21     T3 = B22 - B12
     * S4 = A12 - S2      T4 = T2 - B21
     * M1 = A11 * B11     U1 = M1 + M2
     * M2 = A12 * B21     U2 = M1 + M6
     * M3 = S4 * B22      U3 = U2 + M7
     * M4 = A22 * T4      U4 = U2 + M5
     * M5 = S1 * T1       U5 = U4 + M3
     * M6 = S2 * T2       U6 = U3 - U4
     * M7 = S3 * T3       U7 = U3 + M5
     * C11 = U1
     * C12 = U5
     * C21 = U6
     * C22 = U7
     */
    template <typename T>
    void mm_Coppersmith_Winograd(T **this_matrix, T **other_matrix, T **result, const size_t A_row, const size_t B_row,
                                 const size_t A_col, const size_t B_col, const size_t A_row_offset, const size_t A_col_offset, const size_t B_row_offset, const size_t B_col_offset)
    {
        if ((A_row <= 2) || (A_row % 2 != 0 || B_col % 2 != 0 || A_col % 2 != 0))
        {
            return mm_generate(this_matrix, other_matrix, result, A_row, B_col, A_col, B_col, A_row_offset, A_col_offset, B_row_offset, B_col_offset);
        }
        T **S1 = mat_create<T>((A_row / 2), (A_col / 2));
        T **S2 = mat_create<T>((A_row / 2), (A_col / 2));
        T **S3 = mat_create<T>((A_row / 2), (A_col / 2));
        T **S4 = mat_create<T>((A_row / 2), (A_col / 2));

        for (size_t i = 0; i < A_row / 2; i++)
        {
            for (size_t j = 0; j < A_col / 2; j++)
            {
                // S1     = A21 + A22
                S1[i][j] = this_matrix[(A_row / 2) + i + A_row_offset][(A_col / 2) + j + A_col_offset] + this_matrix[(A_row / 2) + i + A_row_offset][(A_col / 2) + j + A_col_offset];
                // S2     = S1 - A11
                S2[i][j] = S1[i][j] - this_matrix[i + A_row_offset][j + A_col_offset];
                // S3     = A11 - A21
                S3[i][j] = this_matrix[i + A_row_offset][j + A_col_offset] - this_matrix[(A_row / 2) + i + A_row_offset][(A_col / 2) + j + A_col_offset];
                // S4     = A12 - S2
                S4[i][j] = this_matrix[i + A_row_offset][(A_col / 2) + j + A_col_offset] - S2[i][j];
            }
        }
        T **T1 = mat_create<T>((B_row / 2), (B_col / 2));
        T **T2 = mat_create<T>((B_row / 2), (B_col / 2));
        T **T3 = mat_create<T>((B_row / 2), (B_col / 2));
        T **T4 = mat_create<T>((B_row / 2), (B_col / 2));
        for (size_t i = 0; i < B_row / 2; i++)
        {
            for (size_t j = 0; j < B_col / 2; j++)
            {
                // T1     = B12 - B11
                T1[i][j] = other_matrix[i + B_row_offset][(B_col / 2) + j + B_col_offset] - other_matrix[i + B_row_offset][j + B_col_offset];
                // T2     = B22 - T1
                T2[i][j] = other_matrix[(B_row / 2) + i + B_row_offset][(B_col / 2) + j + B_col_offset] - T1[i][j];
                // T3     = B22 - B12
                T3[i][j] = other_matrix[(B_row / 2) + i + B_row_offset][(B_col / 2) + j + B_col_offset] - other_matrix[i + B_row_offset][(B_col / 2) + j + B_col_offset];
                // T4     = T2 - B21
                T4[i][j] = T2[i][j] - other_matrix[(B_row / 2) + i + B_row_offset][j + B_col_offset];
            }
        }
        // M1 = A11 * B11
        T **M1 = mat_create<T>((A_row / 2), (B_col / 2));
        mm_Coppersmith_Winograd(this_matrix, other_matrix, M1, A_row / 2, B_row / 2, A_col / 2, B_col / 2, 0, 0, 0, 0);

        // M2 = A12 * B21
        T **M2 = mat_create<T>((A_row / 2), (B_col / 2));
        // T *A12 = &(this_matrix[0][(A_col / 2)]);
        // T *B21 = &(other_matrix[(A_col / 2)][0]);
        // T **A12 = this_matrix + (A_col / 2);
        // T **B21 = other_matrix + (A_col / 2);
        mm_Coppersmith_Winograd(this_matrix, other_matrix, M2, A_row / 2, B_row / 2, A_col / 2, B_col / 2, 0, A_col / 2, 0, 0);

        // M3 = S4 * B22
        T **M3 = mat_create<T>((A_row / 2), (B_col / 2));
        // T **B22 = other_matrix[(A_col / 2)][(B_col / 2)];
        mm_Coppersmith_Winograd(S4, other_matrix, M3, A_row / 2, B_row / 2, A_col / 2, B_col / 2, 0, 0, A_col / 2, B_col / 2);
        // M4 = A22 * T4
        T **M4 = mat_create<T>((A_row / 2), (B_col / 2));
        // T *A22 = &(this_matrix[(A_row / 2)][(A_col / 2)]);
        mm_Coppersmith_Winograd(this_matrix, T4, M4, A_row / 2, B_row / 2, A_col / 2, B_col / 2, A_row / 2, A_col / 2, 0, 0);
        // M5 = S1 * T1
        T **M5 = mat_create<T>((A_row / 2), (B_col / 2));
        mm_Coppersmith_Winograd(S1, T1, M5, A_row / 2, B_row / 2, A_col / 2, B_col / 2, 0, 0, 0, 0);
        // M6 = S2 * T2
        T **M6 = mat_create<T>((A_row / 2), (B_col / 2));
        mm_Coppersmith_Winograd(S2, T2, M6, A_row / 2, B_row / 2, A_col / 2, B_col / 2, 0, 0, 0, 0);
        // M7 = S3 * T3
        T **M7 = mat_create<T>((A_row / 2), (B_col / 2));
        mm_Coppersmith_Winograd(S3, T3, M7, A_row / 2, B_row / 2, A_col / 2, B_col / 2, 0, 0, 0, 0);

        for (size_t i = 0; i < A_row / 2; i++)
        {
            for (size_t j = 0; j < B_col / 2; j++)
            {
                result[i][j] = M1[i][j] + M2[i][j];
                result[i][(A_col / 2) + j] = M1[i][j] + M6[i][j] + M5[i][j] + M3[i][j];
                result[(A_row / 2) + i][j] = M1[i][j] + M6[i][j] + M7[i][j] - M4[i][j];
                result[(A_row / 2) + i][(A_col / 2) + j] = M1[i][j] + M6[i][j] + M7[i][j] + M5[i][j];
            }
        }
        mat_delete<T>(S1, A_row / 2);
        mat_delete<T>(S2, A_row / 2);
        mat_delete<T>(S3, A_row / 2);
        mat_delete<T>(S4, A_row / 2);
        mat_delete<T>(T1, A_col / 2);
        mat_delete<T>(T2, A_col / 2);
        mat_delete<T>(T3, A_col / 2);
        mat_delete<T>(T4, A_col / 2);
        mat_delete<T>(M1, A_row / 2);
        mat_delete<T>(M2, A_row / 2);
        mat_delete<T>(M3, A_row / 2);
        mat_delete<T>(M4, A_row / 2);
        mat_delete<T>(M5, A_row / 2);
        mat_delete<T>(M6, A_row / 2);
        mat_delete<T>(M7, A_row / 2);
    }

    template <typename T>
    void mm_auto(T **this_matrix, T **other_matrix, T **result, const size_t A_row, const size_t B_row,
                 const size_t A_col, const size_t B_col, const bool fast_flag)
    {
        if ((A_row * B_col * A_col <= 64 * 64 * 64) || (A_row % 2 != 0 || B_col % 2 != 0 || A_col % 2 != 0))
        {
            return mm_generate(this_matrix, other_matrix, result, A_row, B_row, A_col, B_col, 0, 0, 0, 0);
        }
        else if (fast_flag == true)
        {
            return mm_Coppersmith_Winograd(this_matrix, other_matrix, result, A_row, B_row, A_col, B_col, 0, 0, 0, 0);
        }
        else
        {
            throw std::invalid_argument("Matrix is too large or no a even matrix, use multicore optimization or do chunked iterative transport, turn on Coppersmith Winograd algorithm if needed");
        }
    }

    // CPU optimization
    template <typename T>
    void atomic_opalloc(T **a, size_t offset_i, size_t offset_j, size_t black_len_i, size_t black_len_j, size_t *sign, std::function<void(T **, size_t, size_t)> opalloc)
    {
        for (size_t i = 0; i < black_len_i; i++)
        {
            a[offset_i + i] = new T[black_len_j];
            for (size_t j = 0; j < black_len_j; j++)
            {
                opalloc(a, offset_i + i, offset_j + j);
            }
        }
        *sign += black_len_i * black_len_j;
    }
    template <typename T>
    void atomic_opcopy(T **a, T **b, size_t offset_i, size_t offset_j, size_t black_len_i, size_t black_len_j, size_t *sign, std::function<void(T **, T **, size_t, size_t)> opcopy)
    {
        for (size_t i = 0; i < black_len_i; i++)
        {
            a[offset_i + i] = new T[black_len_j];
            for (size_t j = 0; j < black_len_j; j++)
            {
                opcopy(a, b, offset_i + i, offset_j + j);
            }
        }
        *sign += black_len_i * black_len_j;
    }
    template <typename T>
    void atomic_op(T **a, size_t offset_i, size_t offset_j, size_t black_len_i, size_t black_len_j, size_t *sign, std::function<void(T **, size_t, size_t)> opA)
    {
        for (size_t i = 0; i < black_len_i; i++)
        {
            for (size_t j = 0; j < black_len_j; j++)
            {
                opA(a, offset_i + i, offset_j + j);
            }
        }
        *sign += black_len_i * black_len_j;
    }
    template <typename T>
    void atomic_op(T **a, T **b, size_t offset_i, size_t offset_j, size_t black_len_i, size_t black_len_j, size_t *sign, std::function<void(T **, T **, size_t, size_t)> opAB)
    {
        for (size_t i = 0; i < black_len_i; i++)
        {
            for (size_t j = 0; j < black_len_j; j++)
            {
                opAB(a, b, offset_i + i, offset_j + j);
            }
        }
        *sign += black_len_i * black_len_j;
    }
    template <typename T>
    void atomic_op(T **a, T **b, T **c, size_t offset_i, size_t offset_j, size_t black_len_i, size_t black_len_j, size_t *sign, std::function<void(T **, T **, T **, size_t, size_t)> opABC)
    {
        for (size_t i = 0; i < black_len_i; i++)
        {
            for (size_t j = 0; j < black_len_j; j++)
            {
                opABC(a, b, c, offset_i + i, offset_j + j);
            }
        }
        *sign += black_len_i * black_len_j;
    }
    template <typename T>
    int Alloc_thread_worker(T **a, size_t a_row, size_t a_col, size_t cpu_thread_max, std::function<void(T **, size_t, size_t)> opA)
    {
        size_t mat_n = sqrt(cpu_thread_max);
        std::thread *t_list = new std::thread[cpu_thread_max];
        size_t black_len_i = a_row / mat_n;
        size_t black_len_j = a_col / mat_n;
        size_t sign = 0;
        for (size_t i = 0; i < mat_n; i++)
        {
            for (size_t j = 0; j < mat_n; j++)
            {
                t_list[i * mat_n + j] = std::thread([=, &sign]()
                                                    { atomic_opalloc<T>(a, i * black_len_i, j * black_len_j, black_len_i, black_len_j, &sign, opA); });
                t_list[i * mat_n + j].detach();
            }
        }
        while (sign < a_row * a_col)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return 0;
    }
    template <typename T>
    int thread_worker(T **a, size_t a_row, size_t a_col, size_t cpu_thread_max, std::function<void(T **, size_t, size_t)> opA)
    {
        size_t mat_n = sqrt(cpu_thread_max);
        std::thread *t_list = new std::thread[cpu_thread_max];
        size_t black_len_i = a_row / mat_n;
        size_t black_len_j = a_col / mat_n;
        size_t sign = 0;
        for (size_t i = 0; i < mat_n; i++)
        {
            for (size_t j = 0; j < mat_n; j++)
            {
                t_list[i * mat_n + j] = std::thread([=, &sign]()
                                                    { atomic_op<T>(a, i * black_len_i, j * black_len_j, black_len_i, black_len_j, &sign, opA); });
                t_list[i * mat_n + j].detach();
            }
        }
        while (sign < a_row * a_col)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return 0;
    }
    template <typename T>
    int Copy_thread_worker(T **a, size_t a_row, size_t a_col, T **b, size_t cpu_thread_max, std::function<void(T **, T **, size_t, size_t)> opAB)
    {
        size_t mat_n = sqrt(cpu_thread_max);
        std::thread *t_list = new std::thread[cpu_thread_max];
        size_t black_len_i = a_row / mat_n;
        size_t black_len_j = a_col / mat_n;
        size_t sign = 0;
        for (size_t i = 0; i < mat_n; i++)
        {
            for (size_t j = 0; j < mat_n; j++)
            {
                t_list[i * mat_n + j] = std::thread([=, &sign]()
                                                    { atomic_opcopy<T>(a, b, i * black_len_i, j * black_len_j, black_len_i, black_len_j, &sign, opAB); });
                t_list[i * mat_n + j].detach();
            }
        }
        while (sign < a_row * a_col)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return 0;
    }
    template <typename T>
    int thread_worker(T **a, size_t a_row, size_t a_col, T **b, size_t cpu_thread_max, std::function<void(T **, T **, size_t, size_t)> opAB)
    {
        size_t mat_n = sqrt(cpu_thread_max);
        std::thread *t_list = new std::thread[cpu_thread_max];
        size_t black_len_i = a_row / mat_n;
        size_t black_len_j = a_col / mat_n;
        size_t sign = 0;
        for (size_t i = 0; i < mat_n; i++)
        {
            for (size_t j = 0; j < mat_n; j++)
            {
                t_list[i * mat_n + j] = std::thread([=, &sign]()
                                                    { atomic_op<T>(a, b, i * black_len_i, j * black_len_j, black_len_i, black_len_j, &sign, opAB); });
                t_list[i * mat_n + j].detach();
            }
        }
        while (sign < a_row * a_col)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return 0;
    }
    template <typename T>
    int thread_worker(T **a, size_t a_row, size_t a_col, T **b, T **c, size_t cpu_thread_max, std::function<void(T **, T **, T **, size_t, size_t)> opABC)
    {
        size_t mat_n = sqrt(cpu_thread_max);
        std::thread *t_list = new std::thread[cpu_thread_max];
        size_t black_len_i = a_row / mat_n;
        size_t black_len_j = a_col / mat_n;
        size_t sign = 0;
        for (size_t i = 0; i < mat_n; i++)
        {
            for (size_t j = 0; j < mat_n; j++)
            {
                t_list[i * mat_n + j] = std::thread([=, &sign]()
                                                    { atomic_op<T>(a, b, c, i * black_len_i, j * black_len_j, black_len_i, black_len_j, &sign, opABC); });
                t_list[i * mat_n + j].detach();
            }
        }
        while (sign < a_row * a_col)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return 0;
    }
    template <typename T>
    void qr_decomposition_gm(T **A, size_t n, T **Q, T **R)
    {
        // 初始化Q和R为0
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                Q[i][j] = 0;
                R[i][j] = 0;
            }
        }

        for (size_t j = 0; j < n; j++)
        {
            T *v = new T[n]; // 存储当前列向量
            for (size_t i = 0; i < n; i++)
            {
                v[i] = A[i][j];
            }

            for (size_t i = 0; i < j; i++)
            {
                // 计算R[i][j] = Q的第i列与A的第j列的点积
                T dot_product = 0;
                for (size_t k = 0; k < n; k++)
                {
                    dot_product += Q[k][i] * A[k][j];
                }
                R[i][j] = dot_product;

                // 从v中减去投影分量
                for (size_t k = 0; k < n; k++)
                {
                    v[k] -= R[i][j] * Q[k][i];
                }
            }

            // 计算R[j][j] = norm(v)
            T norm_v = 0;
            for (size_t i = 0; i < n; i++)
            {
                norm_v += v[i] * v[i];
            }
            norm_v = sqrt(norm_v);
            R[j][j] = norm_v;

            // Q的第j列 = v / norm_v
            for (size_t i = 0; i < n; i++)
            {
                Q[i][j] = v[i] / norm_v;
            }

            delete[] v;
        }
    }
}; // namespace units

#define mklamb(T, codes, ...) ([__VA_ARGS__](T x, T y)->T codes)

namespace np
{
    enum NormType
    {
        L1,
        L2,
        INF
    };

    static bool is_optimized = false;
    static size_t MAX_thread = 1;
    template <typename dataType>
    class Numcpp
    {
    private:
        bool optimization = is_optimized;
        size_t maxprocs = MAX_thread;
        bool is_destroy = false;
#if CUDA_CHECK
        bool mem_stat = false;
        bool mem_synced = false;
#endif
    public:
        dataType **matrix = nullptr;
        size_t row = 0, col = 0;
#if CUDA_CHECK
        bool MUL_GPU = true;
        bool auto_sync = false;
        dataType **device_data = nullptr;
#endif
        Numcpp() = default;
        Numcpp(const size_t _row, const size_t _col);
        Numcpp(const size_t _row, const size_t _col, dataType value);
        Numcpp(const Numcpp<dataType> &other);
        Numcpp(dataType *mat, const size_t _row, const size_t _col);
        Numcpp(dataType **mat, const size_t _row, const size_t _col);
        Numcpp(char *filename);

        void ensure() const
        {
            if (matrix == nullptr && is_destroy == true)
            {
                std::runtime_error("matrix is a nullptr && is_destroy = true");
            }
        }

// operators
#if CUDA_CHECK
        void to(const int device)
        {
            if (device == DEVICE_CUDA && (mem_stat == false || mem_synced == false))
            {
                if (device_data == nullptr)
                {
                    cudaMalloc((void **)&device_data, row * sizeof(dataType *));
                    dataType **temp;
                    cudaHostAlloc((void ***)&temp, row * sizeof(dataType *), cudaHostAllocDefault);
                    for (int i = 0; i < row; i++)
                    {
                        cudaMalloc((void **)&(temp[i]), col * sizeof(dataType));
                        cudaMemcpy(temp[i], matrix[i], col * sizeof(dataType), cudaMemcpyHostToDevice);
                    }
                    cudaMemcpy(device_data, temp, row * sizeof(dataType *), cudaMemcpyHostToDevice);
                    cudaFreeHost(temp);
                }
                else
                {
                    dataType **temp;
                    cudaHostAlloc((void ***)&temp, row * sizeof(dataType *), cudaHostAllocDefault);
                    for (int i = 0; i < row; i++)
                    {
                        cudaMalloc((void **)&(temp[i]), col * sizeof(dataType));
                        cudaMemcpy(temp[i], matrix[i], col * sizeof(dataType), cudaMemcpyHostToDevice);
                    }
                    cudaMemcpy(device_data, temp, row * sizeof(dataType *), cudaMemcpyHostToDevice);
                    cudaFreeHost(temp);
                }
                cudaDeviceSynchronize();
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    std::cerr << __func__ << "()::" << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                }
                mem_stat = true;
                mem_synced = true;
            }
            else if (device == DEVICE_LOCAL && mem_stat == true)
            {
                dataType **temp;
                cudaHostAlloc((void ***)&temp, row * sizeof(dataType *), cudaHostAllocDefault);
                cudaMemcpy(temp, device_data, row * sizeof(dataType *), cudaMemcpyDeviceToHost);
                for (size_t i = 0; i < row; i++)
                {
                    cudaMemcpy(matrix[i], temp[i], col * sizeof(dataType), cudaMemcpyDeviceToHost);
                }
                cudaFreeHost(temp);
                cudaDeviceSynchronize();
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    std::cerr << __func__ << "()::" << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                }
                mem_synced = false;
            }
        }
        void cuda_free()
        {
            if (device_data != nullptr)
            {
                cudaFree(device_data);
                cudaDeviceSynchronize();
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    std::cerr << __func__ << "()::" << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                }
                device_data = nullptr;
                mem_stat = false;
                mem_synced = false;
            }
        }
#endif

        void operator=(const Numcpp<dataType> &other)
        {
            ensure();
            if (other.row != this->row || other.col != this->col)
            {
                if (matrix == nullptr)
                {
#if CUDA_CHECK
                    cuda_free();
                    for (size_t i = 0; i < this->row; i++)
                    {
                        delete matrix[i];
                    }
                    delete[] matrix;
#else
                    for (size_t i = 0; i < this->row; i++)
                    {
                        delete matrix[i];
                    }
                    delete[] matrix;
#endif
                }
                row = other.row;
                col = other.col;
                matrix = new dataType *[row];
                if (this->optimization == false)
                {
                    for (size_t i = 0; i < row; i++)
                    {
                        matrix[i] = new dataType[col];
                        for (size_t j = 0; j < col; j++)
                        {
                            matrix[i][j] = other.matrix[i][j];
                        }
                    }
                }
                else
                {
                    units::Copy_thread_worker<dataType>(matrix, this->row, this->col, other.matrix, this->maxprocs, [](dataType **a, dataType **b, size_t i, size_t j)
                                                        { a[i][j] = b[i][j]; });
                }
            }
            else
            {
                if (this->optimization == false)
                {
                    for (size_t i = 0; i < this->row; i++)
                    {
                        for (size_t j = 0; j < this->col; j++)
                        {
                            this->matrix[i][j] = other.matrix[i][j];
                        }
                    }
                }
                else
                {
                    units::thread_worker<dataType>(this->matrix, this->row, this->col, other.matrix, this->maxprocs, [](dataType **a, dataType **b, size_t i, size_t j)
                                                   { a[i][j] = b[i][j]; });
                }
            }
        }
        void operator+=(const Numcpp<dataType> &other)
        {
            ensure();
            if (other.row != this->row || other.col != this->col)
            {
                throw std::invalid_argument("Invalid Matrix");
            }
            else
            {
                if (this->optimization == false)
                {
                    for (size_t i = 0; i < this->row; i++)
                    {
                        for (size_t j = 0; j < this->col; j++)
                        {
                            this->matrix[i][j] = other.matrix[i][j];
                        }
                    }
                }
                else
                {
#if CUDA_CHECK
                    if (this->mem_stat == true && other.mem_stat == true)
                    {
                        dim3 block(this->row, this->col);
                        cuda_op::func_Bt<dataType> d_p;
                        cudaMemcpyFromSymbol(&d_p, cuda_op::add_opB<dataType>, sizeof(cuda_op::func_Bt<dataType>));
                        cuda_op::kernel_binary_op<dataType><<<block>>>(this->device_data, other.device_data, d_p);

                        cudaError_t err = cudaGetLastError();
                        if (err != cudaSuccess)
                        {
                            std::cerr << __func__ << "()::__global__ function error "
                                      << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                            throw std::runtime_error("CUDA runtime error");
                        }
                    }
                    else
                    {
                        throw std::invalid_argument("Invalid Matrix Device: Both parties involved in the operation should be on the same device.");
                    }
#else
                    units::thread_worker<dataType>(this->matrix, this->row, this->col, other.matrix, this->maxprocs, [](dataType **a, dataType **b, size_t i, size_t j)
                                                   { a[i][j] += b[i][j]; });
#endif
                }
            }
        }
        Numcpp<dataType> operator+(const Numcpp<dataType> &other) const
        {
            ensure();
            if (other.row != this->row || other.col != this->col)
            {
                throw std::invalid_argument("Invalid Matrix");
            }
            else
            {
                Numcpp<dataType> result(this->row, this->col);
                if (this->optimization == false)
                {
                    for (size_t i = 0; i < this->row; i++)
                    {
                        for (size_t j = 0; j < this->col; j++)
                        {
                            result.matrix[i][j] = this->matrix[i][j] + other.matrix[i][j];
                        }
                    }
                }
                else
                {
#if CUDA_CHECK
                    if (this->mem_stat == true && other.mem_stat == true)
                    {
                        result.to(DEVICE_CUDA);

                        dim3 block(this->row, this->col);
                        cuda_op::func_Ct<dataType> d_p;
                        cudaMemcpyFromSymbol(&d_p, cuda_op::add_opC<dataType>, sizeof(cuda_op::func_Ct<dataType>));
                        cuda_op::kernel_ternary_op<dataType><<<block>>>(result.device_data, this->device_data, other.device_data, d_p);

                        if (result.auto_sync == true)
                        {
                            result.to(DEVICE_LOCAL);
                        }

                        cudaError_t err = cudaGetLastError();
                        if (err != cudaSuccess)
                        {
                            std::cerr << __func__ << "()::__global__ function error "
                                      << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                            throw std::runtime_error("CUDA runtime error");
                        }
                    }
                    else
                    {
                        throw std::invalid_argument("Invalid Matrix Device: Both parties involved in the operation should be on the same device.");
                    }
#else
                    dataType **temp = other.matrix;
                    units::thread_worker<dataType>(this->matrix, this->row, this->col, temp, result.matrix, this->maxprocs, [](dataType **a, dataType **b, dataType **c, size_t i, size_t j)
                                                   { c[i][j] = a[i][j] + b[i][j]; });
#endif
                }
                return result;
            }
        }
        void operator-=(const Numcpp<dataType> &other)
        {
            ensure();
            if (other.row != this->row || other.col != this->col)
            {
                throw std::invalid_argument("Invalid Matrix");
            }
            else
            {
                if (this->optimization == false)
                {
                    for (size_t i = 0; i < this->row; i++)
                    {
                        for (size_t j = 0; j < this->col; j++)
                        {
                            this->matrix[i][j] -= other.matrix[i][j];
                        }
                    }
                }
                else
                {
#if CUDA_CHECK
                    if (this->mem_stat == true && other.mem_stat == true)
                    {
                        dim3 block(this->row, this->col);
                        cuda_op::func_Bt<dataType> d_p;
                        cudaMemcpyFromSymbol(&d_p, cuda_op::cut_opB<dataType>, sizeof(cuda_op::func_Bt<dataType>));
                        cuda_op::kernel_binary_op<dataType><<<block>>>(this->device_data, other.device_data, d_p);

                        cudaError_t err = cudaGetLastError();
                        if (err != cudaSuccess)
                        {
                            std::cerr << __func__ << "()::__global__ function error "
                                      << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                            throw std::runtime_error("CUDA runtime error");
                        }
                    }
                    else
                    {
                        throw std::invalid_argument("Invalid Matrix Device: Both parties involved in the operation should be on the same device.");
                    }
#else
                    units::thread_worker<dataType>(this->matrix, this->row, this->col, other.matrix, this->maxprocs, [](dataType **a, dataType **b, size_t i, size_t j)
                                                   { a[i][j] -= b[i][j]; });
#endif
                }
            }
        }
        Numcpp<dataType> operator-(const Numcpp<dataType> &other) const
        {
            ensure();
            if (other.row != this->row || other.col != this->col)
            {
                throw std::invalid_argument("Invalid Matrix");
            }
            else
            {
                Numcpp<dataType> result(this->row, this->col);
                if (this->optimization == false)
                {
                    for (size_t i = 0; i < this->row; i++)
                    {
                        for (size_t j = 0; j < this->col; j++)
                        {
                            result.matrix[i][j] = this->matrix[i][j] - other.matrix[i][j];
                        }
                    }
                }
                else
                {
#if CUDA_CHECK
                    if (this->mem_stat == true && other.mem_stat == true)
                    {
                        result.to(DEVICE_CUDA);

                        dim3 block(this->row, this->col);
                        cuda_op::func_Ct<dataType> d_p;
                        cudaMemcpyFromSymbol(&d_p, cuda_op::cut_opC<dataType>, sizeof(cuda_op::func_Ct<dataType>));
                        cuda_op::kernel_ternary_op<dataType><<<block>>>(result.device_data, this->device_data, other.device_data, d_p);

                        if (result.auto_sync == true)
                        {
                            result.to(DEVICE_LOCAL);
                        }
                        cudaError_t err = cudaGetLastError();
                        if (err != cudaSuccess)
                        {
                            std::cerr << __func__ << "()::__global__ function error "
                                      << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                            throw std::runtime_error("CUDA runtime error");
                        }
                    }
                    else
                    {
                        throw std::invalid_argument("Invalid Matrix Device: Both parties involved in the operation should be on the same device.");
                    }
#else

                    dataType **temp = other.matrix;
                    units::thread_worker<dataType>(this->matrix, this->row, this->col, temp, result.matrix, this->maxprocs, [](dataType **a, dataType **b, dataType **c, size_t i, size_t j)
                                                   { c[i][j] = a[i][j] + b[i][j]; });
#endif
                }
                return result;
            }
        }
        Numcpp<dataType> operator+(dataType n) const
        {
            ensure();
            Numcpp<dataType> result(this->row, this->col, n);
            if (this->optimization == false)
            {
                for (size_t i = 0; i < this->row; i++)
                {
                    for (size_t j = 0; j < this->col; j++)
                    {
                        result.matrix[i][j] += this->matrix[i][j];
                    }
                }
            }
            else
            {
#if CUDA_CHECK
                if (this->mem_stat == true)
                {
                    result.to(DEVICE_CUDA);

                    dim3 block(this->row, this->col);
                    cuda_op::func_Bt<dataType> d_p;
                    cudaMemcpyFromSymbol(&d_p, cuda_op::add_opB<dataType>, sizeof(cuda_op::func_Bt<dataType>));
                    cuda_op::kernel_binary_op<dataType><<<block>>>(result.device_data, this->device_data, d_p);

                    if (result.auto_sync == true)
                    {
                        result.to(DEVICE_LOCAL);
                    }
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        std::cerr << __func__ << "()::__global__ function error "
                                  << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                        throw std::runtime_error("CUDA runtime error");
                    }
                }
                else
                {
                    units::thread_worker<dataType>(result.matrix, this->row, this->col, this->matrix, this->maxprocs, [n](dataType **a, dataType **b, size_t i, size_t j)
                                                   { a[i][j] += b[i][j]; });
                }
#else
                units::thread_worker<dataType>(result.matrix, this->row, this->col, this->matrix, this->maxprocs, [n](dataType **a, dataType **b, size_t i, size_t j)
                                               { a[i][j] += b[i][j]; });
#endif
            }
            return result;
        }
        void operator+=(dataType n)
        {
            ensure();
            if (this->optimization == false)
            {
                for (size_t i = 0; i < this->row; i++)
                {
                    for (size_t j = 0; j < this->col; j++)
                    {
                        this->matrix[i][j] += n;
                    }
                }
            }
            else
            {
#if CUDA_CHECK
                if (this->mem_stat == true)
                {
                    dim3 block(this->row, this->col);
                    cuda_op::kernel_numadd_op<dataType><<<block>>>(this->device_data, n);

                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        std::cerr << __func__ << "()::__global__ function error "
                                  << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                        throw std::runtime_error("CUDA runtime error");
                    }
                }
                else
                {
                    units::thread_worker<dataType>(this->matrix, this->row, this->col, this->maxprocs, [n](dataType **a, size_t i, size_t j)
                                                   { a[i][j] += n; });
                }
#else
                units::thread_worker<dataType>(this->matrix, this->row, this->col, this->maxprocs, [n](dataType **a, size_t i, size_t j)
                                               { a[i][j] += n; });
#endif
            }
        }
        Numcpp<dataType> operator-(dataType n) const
        {
            ensure();
            Numcpp<dataType> result(this->matrix, this->row, this->col);
            if (this->optimization == false)
            {
                for (size_t i = 0; i < this->row; i++)
                {
                    for (size_t j = 0; j < this->col; j++)
                    {
                        result.matrix[i][j] -= n;
                    }
                }
            }
            else
            {
#if CUDA_CHECK
                if (this->mem_stat == true)
                {
                    result.to(DEVICE_CUDA);

                    dim3 block(this->row, this->col);
                    cuda_op::func_Bt<dataType> d_p;
                    cudaMemcpyFromSymbol(&d_p, cuda_op::mul_opB<dataType>, sizeof(cuda_op::func_Bt<dataType>));
                    cuda_op::kernel_numcut_op<dataType><<<block>>>(result.device_data, n);

                    if (result.auto_sync == true)
                    {
                        result.to(DEVICE_LOCAL);
                    }
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        std::cerr << __func__ << "()::__global__ function error "
                                  << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                        throw std::runtime_error("CUDA runtime error");
                    }
                }
                else
                {
                    units::thread_worker<dataType>(result.matrix, this->row, this->col, this->maxprocs, [n](dataType **a, size_t i, size_t j)
                                                   { a[i][j] -= n; });
                }
#else
                units::thread_worker<dataType>(result.matrix, this->row, this->col, this->maxprocs, [n](dataType **a, size_t i, size_t j)
                                               { a[i][j] -= n; });
#endif
            }
            return result;
        }
        void operator-=(dataType n)
        {
            ensure();
            if (this->optimization == false)
            {
                for (size_t i = 0; i < this->row; i++)
                {
                    for (size_t j = 0; j < this->col; j++)
                    {
                        this->matrix[i][j] -= n;
                    }
                }
            }
            else
            {
#if CUDA_CHECK
                if (this->mem_stat == true)
                {
                    dim3 block(this->row, this->col);
                    cuda_op::kernel_numcut_op<dataType><<<block>>>(this->device_data, n);

                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        std::cerr << __func__ << "()::__global__ function error "
                                  << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                        throw std::runtime_error("CUDA runtime error");
                    }
                }
                else
                {
                    units::thread_worker<dataType>(this->matrix, this->row, this->col, this->maxprocs, [n](dataType **a, size_t i, size_t j)
                                                   { a[i][j] -= n; });
                }
#else
                units::thread_worker<dataType>(this->matrix, this->row, this->col, this->maxprocs, [n](dataType **a, size_t i, size_t j)
                                               { a[i][j] -= n; });
#endif
            }
        }

        Numcpp<dataType> operator*(dataType n) const
        {
            ensure();
            Numcpp<dataType> result(this->row, this->col, n);
            if (this->optimization == false)
            {
                for (size_t i = 0; i < this->row; i++)
                {
                    for (size_t j = 0; j < this->col; j++)
                    {
                        result.matrix[i][j] *= this->matrix[i][j];
                    }
                }
            }
            else
            {
#if CUDA_CHECK
                if (this->mem_stat == true)
                {
                    result.to(DEVICE_CUDA);

                    dim3 block(this->row, this->col);
                    cuda_op::func_Bt<dataType> d_p;
                    cudaMemcpyFromSymbol(&d_p, cuda_op::mul_opB<dataType>, sizeof(cuda_op::func_Bt<dataType>));
                    cuda_op::kernel_binary_op<dataType><<<block>>>(result.device_data, this->device_data, d_p);

                    if (result.auto_sync == true)
                    {
                        result.to(DEVICE_LOCAL);
                    }
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        std::cerr << __func__ << "()::__global__ function error "
                                  << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                        throw std::runtime_error("CUDA runtime error");
                    }
                }
                else
                {
                    units::thread_worker<dataType>(result.matrix, this->row, this->col, this->matrix, this->maxprocs, [n](dataType **a, dataType **b, size_t i, size_t j)
                                                   { a[i][j] *= b[i][j]; });
                }
#else
                units::thread_worker<dataType>(result.matrix, this->row, this->col, this->matrix, this->maxprocs, [n](dataType **a, dataType **b, size_t i, size_t j)
                                               { a[i][j] *= b[i][j]; });
#endif
            }
            return result;
        }
        void operator*=(dataType n)
        {
            ensure();
            if (this->optimization == false)
            {
                for (size_t i = 0; i < this->row; i++)
                {
                    for (size_t j = 0; j < this->col; j++)
                    {
                        this->matrix[i][j] *= n;
                    }
                }
            }
            else
            {
#if CUDA_CHECK
                if (this->mem_stat == true)
                {
                    dim3 block(this->row, this->col);
                    cuda_op::kernel_nummul_op<dataType><<<block>>>(this->device_data, n);

                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        std::cerr << __func__ << "()::__global__ function error "
                                  << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                        throw std::runtime_error("CUDA runtime error");
                    }
                }
                else
                {
                    units::thread_worker<dataType>(this->matrix, this->row, this->col, this->maxprocs, [n](dataType **a, size_t i, size_t j)
                                                   { a[i][j] *= n; });
                }
#else
                units::thread_worker<dataType>(this->matrix, this->row, this->col, this->maxprocs, [n](dataType **a, size_t i, size_t j)
                                               { a[i][j] *= n; });
#endif
            }
        }
        Numcpp<dataType> operator/(dataType n) const
        {
            ensure();
            assert(n != 0);
            Numcpp<dataType> result(this->row, this->col);
            if (this->optimization == false)
            {
                for (size_t i = 0; i < this->row; i++)
                {
                    for (size_t j = 0; j < this->col; j++)
                    {
                        result.matrix[i][j] = this->matrix[i][j] / n;
                    }
                }
            }
            else
            {
#if CUDA_CHECK
                if (this->mem_stat == true)
                {
                    result.to(DEVICE_CUDA);

                    dim3 block(this->row, this->col);
                    cuda_op::func_Bt<dataType> d_p;
                    cudaMemcpyFromSymbol(&d_p, cuda_op::div_opB<dataType>, sizeof(cuda_op::func_Bt<dataType>));
                    cuda_op::kernel_binary_op<dataType><<<block>>>(result.device_data, this->device_data, d_p);

                    if (result.auto_sync == true)
                    {
                        result.to(DEVICE_LOCAL);
                    }
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        std::cerr << __func__ << "()::__global__ function error "
                                  << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                        throw std::runtime_error("CUDA runtime error");
                    }
                }
                else
                {
                    units::thread_worker<dataType>(result.matrix, this->row, this->col, this->matrix, this->maxprocs, [n](dataType **a, dataType **b, size_t i, size_t j)
                                                   { a[i][j] = b[i][j] / n; });
                }
#else
                units::thread_worker<dataType>(result.matrix, this->row, this->col, this->matrix, this->maxprocs, [n](dataType **a, dataType **b, size_t i, size_t j)
                                               { a[i][j] = b[i][j] / n; });
#endif
            }
            return result;
        }
        void operator/=(dataType n)
        {
            ensure();
            assert(n != 0);
            if (this->optimization == false)
            {
                for (size_t i = 0; i < this->row; i++)
                {
                    for (size_t j = 0; j < this->col; j++)
                    {
                        this->matrix[i][j] /= n;
                    }
                }
            }
            else
            {
#if CUDA_CHECK
                if (this->mem_stat == true)
                {
                    dim3 block(this->row, this->col);
                    cuda_op::kernel_numdiv_op<dataType><<<block>>>(this->device_data, n);

                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        std::cerr << __func__ << "()::__global__ function error "
                                  << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                        throw std::runtime_error("CUDA runtime error");
                    }
                }
                else
                {
                    units::thread_worker<dataType>(this->matrix, this->row, this->col, this->maxprocs, [n](dataType **a, size_t i, size_t j)
                                                   { a[i][j] /= n; });
                }
#else
                units::thread_worker<dataType>(this->matrix, this->row, this->col, this->maxprocs, [n](dataType **a, size_t i, size_t j)
                                               { a[i][j] /= n; });
#endif
            }
        }
        /*Matrix function complex*/
        /*the col of first matrix is the same as the row of second matrix*/
        Numcpp<dataType> operator*(const Numcpp<dataType> &other) const
        {
            ensure();
            if (this->col != other.row)
            {
                throw std::invalid_argument("Invalid Matrix");
            }
            else
            {
                Numcpp<dataType> result(this->row, other.col, 0);
                if (this->optimization == true)
                {
                    units::mm_auto(this->matrix, other.matrix, result.matrix, this->row, other.row, this->col, other.col, true);
                }
                else
                {
#if CUDA_CHECK
                    if (this->MUL_GPU == true)
                    {
                        if (this->mem_stat == true && other.mem_stat == true)
                        {
                            result.to(DEVICE_CUDA);
                            cuda_op::gemm<dataType>(this->device_data, this->row, this->col, other.device_data, other.col, result.device_data);
                            if (result.auto_sync == true)
                            {
                                result.to(DEVICE_LOCAL);
                            }
                        }
                        else
                        {
                            throw std::invalid_argument("Invalid Matrix Device: Both parties involved in the operation should be on the same device.");
                        }
                    }
                    else
                    {
                        units::mm_generate(this->matrix, other.matrix, result.matrix, this->row, other.row, this->col, other.col, 0, 0, 0, 0);
                    }
#else
                    units::mm_generate(this->matrix, other.matrix, result.matrix, this->row, other.row, this->col, other.col, 0, 0, 0, 0);
#endif
                }
                return result;
            }
        }
        /*index for each*/
        dataType *operator[](const size_t index) const
        {
            ensure();
            return index < this->row ? this->matrix[index] : NULL;
        }
        Numcpp<dataType> srow(const size_t index) const
        {
            ensure();
            return index < this->row ? Numcpp<dataType>(this->matrix[index], 1, this->col) : Numcpp<dataType>();
        }
        Numcpp<dataType> scol(const size_t index) const
        {
            ensure();
            if (index < this->col)
            {
                Numcpp<dataType> result(this->row, 1);
                for (size_t i = 0; i < this->row; i++)
                {
                    result[i][0] = this->matrix[i][index];
                }
                return result;
            }
            else
            {
                return Numcpp();
            }
        }
        /*transposed this matrix*/
        void transposed();
        /*the transposition of this matrix*/
        Numcpp transpose() const;
        void Hadamard_self(const Numcpp<dataType> &);
        Numcpp Hadamard(const Numcpp<dataType> &) const;

        void optimized(bool flag)
        {
            this->optimization = flag;
        }
        /*
         * set the maxprocs for this matrix, used in multi-threaded operations
         * Usually the maxprocs is under the number of CPU cores
         */
        void maxprocs_set(size_t thread_num)
        {
            if (sqrt(thread_num) * sqrt(thread_num) > std::thread::hardware_concurrency() || thread_num < 1)
            {
                throw std::invalid_argument("Invalid maxprocs");
            }
            else
            {
                this->maxprocs = thread_num;
            }
        }
        // delete
        ~Numcpp();
// FFT only the cuda disable can used
#if !CUDA_CHECK
        // 正向/反向FFT（返回新矩阵）
        Numcpp<dataType> fft(int inv) const
        {
            ensure();
            if (is_complex_v<dataType>)
            {
                Numcpp<dataType> result(this->row, this->col);
                using ValueType = typename dataType::value_type; // 提取底层数值类型

                for (size_t i = 0; i < this->row; i++)
                {
                    units::fft<ValueType>(
                        &(this->matrix[i]),
                        this->col,
                        &(result.matrix[i]),
                        inv);
                }

                // 逆变换归一化
                if (inv < 0)
                {
                    result *= dataType(1.0 / this->col, 0);
                }
                return result;
            }
            else
            {
                // 不是复数转为复数
                Numcpp<std::complex<dataType>> temp(row, col);
                if (this->optimization == false)
                {
                    for (size_t i = 0; i < row; i++)
                    {
                        for (size_t j = 0; j < col; j++)
                        {
                            temp.matrix[i][j] = std::complex<dataType>(matrix[i][j]);
                        }
                    }
                }
                else
                {
                    units::Copy_thread_worker<dataType>(temp.matrix, this->row, this->col, matrix, this->maxprocs, [](dataType **a, dataType **b, size_t i, size_t j)
                                                        { a[i][j] = std::complex<dataType>(b[i][j]); });
                }
                for (size_t i = 0; i < this->row; i++)
                {
                    // 原地计算（输入输出指向相同内存）
                    units::fft<dataType>(
                        &(temp.matrix[i]),
                        this->col,
                        &(temp.matrix[i]),
                        inv);
                }

                // 逆变换归一化
                if (inv < 0)
                {
                    temp *= std::complex<dataType>(1.0 / this->col, 0);
                }
            }
        }

        // 自体FFT
        void ffted(int inv)
        {
            ensure();
            if (is_complex_v<dataType>)
            {
                using ValueType = typename dataType::value_type;
                for (size_t i = 0; i < this->row; i++)
                {
                    // 原地计算（输入输出指向相同内存）
                    units::fft<ValueType>(
                        &(this->matrix[i]),
                        this->col,
                        &(this->matrix[i]),
                        inv);
                }

                // 逆变换归一化
                if (inv < 0)
                {
                    *this *= dataType(1.0 / this->col, 0);
                }
            }
            else
            {
                std::invalid_argument("FFT in self must require the complex type");
            }
        }
#endif
        dataType sum() const
        {
            ensure();
            dataType sum_value = 0;
            if (this->optimization == false)
            {
                for (size_t i = 0; i < this->row; i++)
                {
                    for (size_t j = 0; j < this->col; j++)
                    {
                        sum_value += this->matrix[i][j];
                    }
                }
            }
            else
            {
                dataType *p = &sum_value;
                units::thread_worker<dataType>(this->matrix, this->row, this->col, this->maxprocs, [p](dataType **a, size_t i, size_t j)
                                               { (*p) += a[i][j]; });
            }
            return sum_value;
        }
        void save(const char *path)
        {
            ensure();
            FILE *fp = fopen(path, "ab");
            if (fp == NULL)
            {
                throw std::invalid_argument("Invalid path");
            }
#if CUDA_CHECK
            to(DEVICE_LOCAL);
#endif
            fwrite(&row, sizeof(size_t), 1, fp);
            fwrite(&col, sizeof(size_t), 1, fp);
            for (size_t i = 0; i < row; i++)
            {
                fwrite(matrix[i], sizeof(dataType), col, fp);
            }
            fclose(fp);
        }
        friend std::ostream &operator<<(std::ostream &stream, const Numcpp<dataType> &m)
        {
            m.ensure();
            stream << '(' << m.row << ',' << m.col << ')' << "[\n";
            for (size_t i = 0; i < m.row; ++i)
            {
                stream << "    [" << i << "][";
                for (size_t j = 0; j < m.col; ++j)
                {
                    stream << (dataType)(m.matrix[i][j]) << (j == m.col - 1 ? "]\n" : " , ");
                }
            }
            stream << "]\n";
            return stream;
        }

        dataType determinant() const;
        Numcpp<dataType> inverse() const;
        Numcpp<dataType> pseudoinverse() const; // 伪逆计算

        class CommaInitializer
        {
        public:
            CommaInitializer(Numcpp *mat, size_t current_index) : mat_(mat), current_index_(current_index) {}

            CommaInitializer &operator,(dataType value)
            {
                size_t row = current_index_ / mat_->col;
                size_t col = current_index_ % mat_->col;
                if (row < mat_->row && col < mat_->col)
                {
                    mat_->matrix[row][col] = value;
                    current_index_++;
                }
                else
                {
                    throw std::out_of_range("Too many elements for matrix");
                }
                return *this;
            }
            operator Numcpp() const
            {
                return *(this->mat_);
            }

        private:
            Numcpp *mat_;
            size_t current_index_;
        };
        CommaInitializer operator<<(dataType value)
        {
            ensure();
            if (row * col == 0)
                throw std::out_of_range("Matrix is empty");
            matrix[0][0] = value;
            return CommaInitializer(this, 1);
        }
        /*
        优化的SVD方法,A = (-U) * S * (-V^T)
        U,V左右两个正交矩阵中的值均为相反值
        */
        void svd(Numcpp<dataType> &U, Numcpp<dataType> &S, Numcpp<dataType> &V) const;
        std::vector<Numcpp<dataType>> svd() const;

        void zero_approximation()
        {
            ensure();
            if (this->optimization == false)
            {
                for (size_t i = 0; i < this->row; i++)
                {
                    for (size_t j = 0; j < this->col; j++)
                    {
                        if (this->matrix[i][j] < 1e-6)
                        {
                            this->matrix[i][j] *= 0;
                        }
                    }
                }
            }
            else
            {
                units::thread_worker<dataType>(this->matrix, this->row, this->col, this->maxprocs, [](dataType **a, size_t i, size_t j)
                                               { a[i][j] *= 0; });
            }
        }
        // 检查矩阵是否对称
        bool is_symmetric(double tolerance = 1e-6) const
        {
            ensure();
            if (row != col)
                return false;
            for (size_t i = 0; i < row; i++)
            {
                for (size_t j = 0; j < i; j++)
                {
                    if (fabs(matrix[i][j] - matrix[j][i]) > tolerance)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        // 设置矩阵为单位矩阵
        void set_identity()
        {
            ensure();
            for (size_t i = 0; i < row; i++)
            {
                for (size_t j = 0; j < col; j++)
                {
                    matrix[i][j] = (i == j) ? (dataType)1 : (dataType)0;
                }
            }
        }
        // 计算特征值和特征向量
        std::vector<Numcpp<dataType>> eig(int max_iter = 1000, double tolerance = 1e-6) const
        {
            ensure();
            if (!is_symmetric(tolerance))
            {
                throw std::invalid_argument("eig only supported for symmetric matrices");
            }

            size_t n = row;
            Numcpp<dataType> A(*this); // 工作矩阵
            Numcpp<dataType> Q_total(n, n);
            Q_total.set_identity(); // 初始化为单位矩阵

            for (int iter = 0; iter < max_iter; iter++)
            {
                Numcpp<dataType> Q(n, n);
                Numcpp<dataType> R(n, n);

                // 对A进行QR分解
                units::qr_decomposition_gm<dataType>(A.matrix, n, Q.matrix, R.matrix);

                // 更新A = R * Q
                A = R * Q;

                // 更新Q_total = Q_total * Q
                Q_total = Q_total * Q;

                // 检查A是否接近对角矩阵
                bool is_diag = true;
                for (size_t i = 0; i < n; i++)
                {
                    for (size_t j = 0; j < i; j++)
                    {
                        if (fabs(A.matrix[i][j]) > tolerance)
                        {
                            is_diag = false;
                            break;
                        }
                    }
                    if (!is_diag)
                        break;
                }
                if (is_diag)
                    break;
            }

            // 从A中提取特征值（对角线元素）
            Numcpp<dataType> eigenvalues(1, n);
            for (size_t i = 0; i < n; i++)
            {
                eigenvalues.matrix[0][i] = A.matrix[i][i];
            }
            std::sort((eigenvalues[0]), (eigenvalues[0]) + eigenvalues.col, std::greater<double>());
            A.zero_approximation();
            for (size_t i = 0; i < n; i++)
            {
                A.matrix[i][i] = eigenvalues.matrix[0][i];
            }
            return {eigenvalues, Q_total, A};
        }
#ifdef NUMCPP_OPENCV_SUPPORT
        /**
         * @brief 从 OpenCV Mat 构造 Numcpp 对象
         * @param mat OpenCV 矩阵
         * @param copy_data 是否复制数据（默认为 true）
         */
        Numcpp(const cv::Mat &mat, bool copy_data = true)
        {
            ensure();
            if (mat.empty())
            {
                throw std::invalid_argument("OpenCV matrix is empty");
            }

            row = mat.rows;
            col = mat.cols;
            matrix = new dataType *[row];

            // 数据类型映射
            if (mat.type() == CV_8U)
            {
                initializeFromCV<uchar>(mat, copy_data);
            }
            else if (mat.type() == CV_8S)
            {
                initializeFromCV<schar>(mat, copy_data);
            }
            else if (mat.type() == CV_16U)
            {
                initializeFromCV<ushort>(mat, copy_data);
            }
            else if (mat.type() == CV_16S)
            {
                initializeFromCV<short>(mat, copy_data);
            }
            else if (mat.type() == CV_32S)
            {
                initializeFromCV<int>(mat, copy_data);
            }
            else if (mat.type() == CV_32F)
            {
                initializeFromCV<float>(mat, copy_data);
            }
            else if (mat.type() == CV_64F)
            {
                initializeFromCV<double>(mat, copy_data);
            }
            else
            {
                throw std::invalid_argument("Unsupported OpenCV matrix type");
            }
        }

        /**
         * @brief 转换为 OpenCV Mat
         * @param mat_type OpenCV 矩阵类型（如 CV_32F, CV_64F 等），-1 表示自动推断
         * @return OpenCV 矩阵
         */
        cv::Mat toMat(int mat_type = -1) const
        {
            ensure();
            if (row == 0 || col == 0)
            {
                return cv::Mat();
            }

            // 自动推断类型
            if (mat_type == -1)
            {
                if (std::is_same<dataType, float>::value)
                {
                    mat_type = CV_32F;
                }
                else if (std::is_same<dataType, double>::value)
                {
                    mat_type = CV_64F;
                }
                else if (std::is_same<dataType, uchar>::value)
                {
                    mat_type = CV_8U;
                }
                else if (std::is_same<dataType, schar>::value)
                {
                    mat_type = CV_8S;
                }
                else if (std::is_same<dataType, ushort>::value)
                {
                    mat_type = CV_16U;
                }
                else if (std::is_same<dataType, short>::value)
                {
                    mat_type = CV_16S;
                }
                else if (std::is_same<dataType, int>::value)
                {
                    mat_type = CV_32S;
                }
                else
                {
                    mat_type = CV_64F; // 默认转换为 double
                }
            }

            cv::Mat result(row, col, mat_type);

            // 根据目标类型进行数据拷贝
            if (mat_type == CV_32F)
            {
                copyToCV<float>(result);
            }
            else if (mat_type == CV_64F)
            {
                copyToCV<double>(result);
            }
            else if (mat_type == CV_8U)
            {
                copyToCV<uchar>(result);
            }
            else if (mat_type == CV_8S)
            {
                copyToCV<schar>(result);
            }
            else if (mat_type == CV_16U)
            {
                copyToCV<ushort>(result);
            }
            else if (mat_type == CV_16S)
            {
                copyToCV<short>(result);
            }
            else if (mat_type == CV_32S)
            {
                copyToCV<int>(result);
            }
            else
            {
                throw std::invalid_argument("Unsupported OpenCV matrix type");
            }

            return result;
        }

        /**
         * @brief 从 OpenCV Mat 赋值
         * @param mat OpenCV 矩阵
         * @return 当前对象的引用
         */
        Numcpp<dataType> &fromMat(const cv::Mat &mat)
        {
            ensure();
            if (mat.empty())
            {
                throw std::invalid_argument("OpenCV matrix is empty");
            }

            // 清理现有数据
            if (matrix != nullptr)
            {
                for (size_t i = 0; i < row; i++)
                {
                    delete[] matrix[i];
                }
                delete[] matrix;
#if CUDA_CHECK
                cuda_free();
#endif
            }

            // 使用构造函数逻辑
            *this = Numcpp<dataType>(mat);
            return *this;
        }

    private:
        /**
         * @brief 从 OpenCV Mat 初始化数据（模板辅助函数）
         */
        void initializeFromCV(const cv::Mat &mat, bool copy_data)
        {
            ensure();
            if (copy_data)
            {
                // 深拷贝：分配新内存并复制数据
                for (size_t i = 0; i < row; i++)
                {
                    matrix[i] = new dataType[col];
                    const dataType *row_ptr = mat.ptr<dataType>(i);
                    for (size_t j = 0; j < col; j++)
                    {
                        matrix[i][j] = static_cast<dataType>(row_ptr[j]);
                    }
                }
            }
            else
            {
                // 浅拷贝：直接使用 OpenCV 数据（危险，不推荐）
                // 注意：这要求 OpenCV Mat 的生命周期长于 Numcpp 对象
                for (size_t i = 0; i < row; i++)
                {
                    matrix[i] = reinterpret_cast<dataType *>(const_cast<uchar *>(mat.ptr(i)));
                }
            }
        }
        /**
         * @brief 拷贝数据到 OpenCV Mat（模板辅助函数）
         */
        void copyToCV(cv::Mat &result) const
        {
            for (size_t i = 0; i < row; i++)
            {
                dataType *row_ptr = result.ptr<dataType>(i);
                for (size_t j = 0; j < col; j++)
                {
                    row_ptr[j] = static_cast<dataType>(matrix[i][j]);
                }
            }
        }
#endif // NUMCPP_OPENCV_SUPPORT
        bool is_vector() const
        {
            // 向量定义为行数为1或列数为1的矩阵
            return (row == 1 || col == 1);
        }
        size_t size() const
        {
            // 向量元素总数
            return row * col;
        }

        /**
         * 计算矩阵/向量的范数
         * @param type 范数类型（默认L2）
         * @return 范数计算结果
         */
        dataType norm(NormType type = L2) const
        {
            ensure();
            dataType result = 0;

            switch (type)
            {
            case L1:
                // L1范数：元素绝对值之和（向量）或列绝对值之和的最大值（矩阵）
                if (is_vector()) // 假设已实现判断是否为向量的方法
                {
                    result += sum();
                }
                else
                {
                    dataType max_col_sum = 0;
                    for (size_t j = 0; j < col; ++j)
                    {
                        dataType col_sum = 0;
                        for (size_t i = 0; i < row; ++i)
                        {
                            col_sum += std::fabs(matrix[i][j]);
                        }
                        max_col_sum = std::max(max_col_sum, col_sum);
                    }
                    result = max_col_sum;
                }
                break;

            case L2:
                // L2范数：元素平方和的平方根（向量）或F-范数（矩阵）
                if (is_vector())
                {
                    dataType sum_sq = ((*this) * (this->transpose())).sum();
                    result = std::sqrt(sum_sq);
                }
                else
                {
                    dataType sum_sq = 0;
                    for (size_t i = 0; i < row; ++i)
                    {
                        for (size_t j = 0; j < col; ++j)
                        {
                            sum_sq += matrix[i][j] * matrix[i][j];
                        }
                    }
                    result = std::sqrt(sum_sq);
                }
                break;

            case INF:
                // 无穷范数：元素绝对值的最大值（向量）或行绝对值之和的最大值（矩阵）
                if (is_vector())
                {
                    if (row == 1)
                    {
                        for (size_t i = 0; i < col; i++)
                        {
                            if (matrix[0][i] > result)
                            {
                                result = matrix[0][i];
                            }
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < row; i++)
                        {
                            if (matrix[i][0] > result)
                            {
                                result = matrix[i][0];
                            }
                        }
                    }
                }
                else
                {
                    auto temp = ((*this) * Numcpp<dataType>(col, 1))<mklamb(dataType, {
                        return std::abs(x);
                    })>
                        NULL;
                    for (size_t i = 0; i < row; i++)
                    {
                        if (temp[i][0] > result)
                        {
                            result = temp[i][0];
                        }
                    }
                }
                break;

            default:
                throw std::invalid_argument("Unsupported norm type");
            }

            return result;
        }
        dataType dot(const Numcpp<dataType> &other) const
        {
            ensure();
            if (!(this->is_vector() && other.is_vector()))
            {
                throw std::invalid_argument("Dot require two vectors");
            }
            if (this->row * this->col != other.row * other.col)
            {
                throw std::invalid_argument("Two vectors must in a same dim");
            }
            if (row == other.row)
            {
                dataType result = 0;
                if (row == 1)
                {
                    for (size_t i = 0; i < col; i++)
                    {
                        result += other.matrix[0][i] * matrix[0][i];
                    }
                    return result;
                }
                else
                {
                    for (size_t i = 0; i < row; i++)
                    {
                        result += other.matrix[i][0] * matrix[i][0];
                    }
                    return result;
                }
            }
            else
            {
                auto temp = (*this) * other;
                return temp[0][0];
            }
        }
    };
    // matrix special operate
    template <typename T>
    class smul_object
    {
    public:
        size_t row, col;
        T **matrix;
        T (*function_object)(T A, T B);
        smul_object(const Numcpp<T> &A, T (*function_object)(T A, T B))
        {
            this->row = A.row;
            this->col = A.col;
            this->matrix = (A.matrix);
            this->function_object = function_object;
        };
    };
    template <typename T>
#if CUDA_CHECK
    smul_object<T> operator<(const Numcpp<T> &A, T (*function_object)(T A, T B))
#elif __cplusplus < 202000L
    smul_object<T> operator<(const Numcpp<T> &A, T (*function_object)(T A, T B))
#else
    smul_object<T> operator<(const Numcpp<T> &A, auto function_object)
#endif // CUDA_CHECK
    {
        smul_object<T> oper(A, function_object);
        return oper;
    }
    template <typename T>
    Numcpp<T> operator>(const smul_object<T> &oper, const Numcpp<T> &B)
    {
        // A.col = B.row
        if (oper.col != B.row)
        {
            throw std::invalid_argument("Invalid Matrix");
        }
        else
        {
            Numcpp<T> result(oper.row * B.col, B.row);
            for (size_t i = 0; i < B.col; i++)
            {
                for (size_t j = 0; j < oper.row; j++)
                {
                    for (size_t k = 0; k < oper.col; k++)
                    {
                        result.matrix[j + i * oper.row][k] = oper.function_object((oper.matrix)[j][k], B.matrix[k][i]);
                    }
                }
            }
            return result;
        }
    }
    template <typename T>
    Numcpp<T> operator>(const smul_object<T> &oper, void *data)
    {
        Numcpp<T> result(oper.row, oper.col);
        for (size_t i = 0; i < oper.row; i++)
        {
            for (size_t j = 0; j < oper.col; j++)
            {
                result[i][j] = oper.function_object((oper.matrix)[i][j], (T)0);
            }
        }
        return result;
    }
    // defined in class functions
    template <typename T>
    Numcpp<T>::Numcpp(const size_t _row, const size_t _col)
    {
        if (_row == 0 || _col == 0)
        {
            throw "Invalid creation";
        }
        else
        {
            row = _row;
            col = _col;
            matrix = new T *[_row];
            if (this->optimization == false)
            {
                for (size_t i = 0; i < _row; i++)
                {
                    matrix[i] = new T[_col];
                    for (size_t j = 0; j < _col; j++)
                    {
                        matrix[i][j] = (T)1;
                    }
                }
            }
            else
            {
                units::Alloc_thread_worker<T>(matrix, _row, _col, this->maxprocs, [](T **a, size_t i, size_t j)
                                              { a[i][j] = (T)1; });
            }
        }
    }
    template <typename T>
    inline Numcpp<T>::Numcpp(const size_t _row, const size_t _col, T value)
    {
        if (_row == 0 || _col == 0)
        {
            throw "Invalid creation";
        }
        else
        {
            row = _row;
            col = _col;
            matrix = new T *[_row];
            if (this->optimization == false)
            {
                for (size_t i = 0; i < _row; i++)
                {
                    matrix[i] = new T[_col];
                    for (size_t j = 0; j < _col; j++)
                    {
                        matrix[i][j] = value;
                    }
                }
            }
            else
            {
                units::Alloc_thread_worker<T>(matrix, _row, _col, this->maxprocs, [value](T **a, size_t i, size_t j)
                                              { a[i][j] = value; });
            }
        }
    }
    template <typename T>
    inline Numcpp<T>::Numcpp(T *mat, const size_t _row, const size_t _col)
    {
        if (_row == 0 || _col == 0)
        {
            throw "Invalid creation";
        }
        else
        {
            row = _row;
            col = _col;
            matrix = new T *[_row];
            if (this->optimization == false)
            {
                for (size_t i = 0; i < _row; i++)
                {
                    matrix[i] = new T[_col];
                    for (size_t j = 0; j < _col; j++)
                    {
                        matrix[i][j] = mat[i * _col + j];
                    }
                }
            }
            else
            {
                units::Copy_thread_worker<T>(matrix, _row, _col, &mat, this->maxprocs, [&](T **a, T **b, size_t i, size_t j)
                                             { a[i][j] = (*b)[i * _col + j]; });
            }
        }
    }
    template <typename T>
    inline Numcpp<T>::Numcpp(T **mat, const size_t _row, const size_t _col)
    {
        if (_row == 0 || _col == 0)
        {
            throw "Invalid creation";
        }
        else
        {
            row = _row;
            col = _col;
            matrix = new T *[_row];
            if (this->optimization == false)
            {
                for (size_t i = 0; i < _row; i++)
                {
                    matrix[i] = new T[_col];
                    for (size_t j = 0; j < _col; j++)
                    {
                        matrix[i][j] = mat[i][j];
                    }
                }
            }
            else
            {
                units::Copy_thread_worker<T>(matrix, _row, _col, mat, this->maxprocs, [](T **a, T **b, size_t i, size_t j)
                                             { a[i][j] = b[i][j]; });
            }
        }
    }
    template <typename T>
    Numcpp<T>::Numcpp(const Numcpp<T> &other)
    {
        if (other.row == 0 || other.col == 0)
        {
            throw std::invalid_argument("Invalid Matrix");
        }
        else
        {
            row = other.row;
            col = other.col;
            matrix = new T *[row];
            if (this->optimization == false)
            {
                for (size_t i = 0; i < row; i++)
                {
                    matrix[i] = new T[col];
                    for (size_t j = 0; j < col; j++)
                    {
                        matrix[i][j] = other.matrix[i][j];
                    }
                }
            }
            else
            {
                units::Copy_thread_worker<T>(matrix, this->row, this->col, other.matrix, this->maxprocs, [](T **a, T **b, size_t i, size_t j)
                                             { a[i][j] = b[i][j]; });
            }
        }
    }
    template <typename T>
    Numcpp<T>::Numcpp(char *path)
    {
        FILE *fp = fopen(path, "rb");
        if (fp == NULL)
        {
            throw std::invalid_argument("Invalid path");
        }
        fread(&row, sizeof(size_t), 1, fp);
        fread(&col, sizeof(size_t), 1, fp);
        matrix = new T *[row];
        for (size_t i = 0; i < row; i++)
        {
            matrix[i] = new T[col];
            fread(matrix[i], sizeof(T), col, fp);
        }
        fclose(fp);
    }

    template <typename T>
    Numcpp<T>::~Numcpp()
    {
        if (matrix != nullptr && is_destroy != true)
        {
#if CUDA_CHECK
            cuda_free();
#endif
            for (size_t i = 0; i < this->row; i++)
            {
                delete matrix[i];
                matrix[i] = nullptr;
            }
            delete[] matrix;
            matrix = nullptr;
            is_destroy = false;
        }
    }
    template <typename T>
    Numcpp<T> Numcpp<T>::transpose() const
    {
        this->ensure();
        Numcpp<T> result(this->col, this->row);
        if (this->optimization == false)
        {
            for (size_t i = 0; i < this->row; i++)
            {
                for (size_t j = 0; j < this->col; j++)
                {
                    result.matrix[j][i] = this->matrix[i][j];
                }
            }
        }
        else
        {
            units::thread_worker<T>(this->matrix, this->row, this->col, result.matrix, this->maxprocs, [](T **a, T **b, size_t i, size_t j)
                                    { b[j][i] = a[i][j]; });
        }
        return result;
    }
    template <typename T>
    void Numcpp<T>::transposed()
    {
        this->ensure();
        size_t x = this->col;
        size_t y = this->row;
        T **temp = new T *[x];

        if (this->optimization == false)
        {
            for (size_t i = 0; i < x; i++)
            {
                temp[i] = new T[y];
                for (size_t j = 0; j < y; j++)
                {
                    temp[i][j] = this->matrix[j][i];
                }
            }
        }
        else
        {
            for (size_t i = 0; i < x; i++)
            {
                temp[i] = new T[y];
            }
            units::thread_worker<T>(this->matrix, this->row, this->col, temp, this->maxprocs, [](T **a, T **b, size_t i, size_t j)
                                    { b[j][i] = a[i][j]; });
        }

        for (size_t i = 0; i < this->row; i++)
        {
            delete matrix[i];
        }
        delete[] matrix;

        this->matrix = temp;
        this->col = y;
        this->row = x;
    }
    template <typename T>
    void Numcpp<T>::Hadamard_self(const Numcpp<T> &other)
    {
        this->ensure();
        if (other.row != this->row || other.col != this->col)
        {
            throw std::invalid_argument("Invalid Matrix");
        }
        else
        {
            if (this->optimization == false)
            {
                for (size_t i = 0; i < this->row; i++)
                {
                    for (size_t j = 0; j < this->col; j++)
                    {
                        this->matrix[i][j] *= other.matrix[i][j];
                    }
                }
            }
            else
            {
#if CUDA_CHECK
                if (this->mem_stat == true && other.mem_stat == true)
                {
                    dim3 block(this->row, this->col);
                    cuda_op::func_Bt<T> d_p = new cuda_op::func_Bt<T>;
                    cudaMemcpyFromSymbol(&d_p, cuda_op::mul_opB<T>, sizeof(cuda_op::func_Bt<T>));
                    cuda_op::kernel_binary_op<T><<<block>>>(this->device_data, other.device_data, d_p);

                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        std::cerr << __func__ << "()::__global__ function error "
                                  << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                        throw std::runtime_error("CUDA runtime error");
                    }
                }
                else
                {
                    throw std::invalid_argument("Invalid Matrix Device: Both parties involved in the operation should be on the same device.");
                }
#else
                units::thread_worker<T>(this->matrix, this->row, this->col, other.matrix, this->maxprocs, [](T **a, T **b, size_t i, size_t j)
                                        { a[i][j] *= b[i][j]; });
#endif
            }
        }
    }
    template <typename T>
    Numcpp<T> Numcpp<T>::Hadamard(const Numcpp<T> &other) const
    {
        this->ensure();
        if (other.row != this->row || other.col != this->col)
        {
            throw std::invalid_argument("Invalid Matrix");
        }
        else
        {
            Numcpp<T> result(other);
            result.Hadamard_self(*this);
            return result;
        }
    }
    template <typename T>
    static Numcpp<T> load(const char *path)
    {
        FILE *fp = fopen(path, "rb");
        if (fp == NULL)
        {
            throw std::invalid_argument("Invalid path");
        }
        size_t row, col;
        fread(&row, sizeof(size_t), 1, fp);
        fread(&col, sizeof(size_t), 1, fp);
        Numcpp<T> result(row, col);
        for (size_t i = 0; i < row; i++)
        {
            fread(result.matrix[i], sizeof(T), col, fp);
        }
        fclose(fp);
        return result;
    }
    template <typename T>
    T det_cal(T **det, size_t n)
    {

        T detVal = 0; // 行列式的值

        if (n == 1)
        {
            return det[0][0]; // 递归终止条件
        }
        T **tempdet = new T *[n - 1]; // 用来存储余相应的余子式
        for (size_t i = 0; i < n - 1; i++)
        {
            tempdet[i] = new T[n - 1];
        }
        for (size_t i = 0; i < n; i++) // 第一重循环，行列式按第一行展开
        {
            for (size_t j = 0; j < n - 1; j++)
                for (size_t k = 0; k < n - 1; k++)
                {
                    if (k < i)
                    {
                        tempdet[j][k] = det[j + 1][k];
                    }
                    else
                    {
                        tempdet[j][k] = det[j + 1][k + 1];
                    }
                }
            detVal += det[0][i] * pow(-1.0, i) * det_cal(tempdet, n - 1);
        }
        for (size_t i = 0; i < n - 1; i++)
        {
            delete tempdet[i];
        }
        delete[] tempdet;
        return detVal;
    }

    template <typename T>
    T Numcpp<T>::determinant() const
    {
        this->ensure();
        if (row != col)
        {
            throw std::invalid_argument("Matrix must be square to compute determinant.");
        }
        return det_cal(matrix, row);
    }

    template <typename T>
    Numcpp<T> Numcpp<T>::inverse() const
    {
        this->ensure();
        if (row != col)
        {
            throw std::invalid_argument("Standard inverse is only defined for square matrices. Use pseudoinverse() for non-square matrices.");
        }

        T det = determinant();
        if (fabs(det) < 1e-10) // 使用适当容差检查行列式是否为0
        {
            throw std::invalid_argument("Matrix is singular (determinant is zero), cannot compute inverse.");
        }

        Numcpp<T> result(row, col);

        if (row == 1)
        {
            // 1x1矩阵的特殊情况
            result.matrix[0][0] = ((T)1) / matrix[0][0];
        }
        else
        {
            // 计算伴随矩阵
            Numcpp<T> adjugate(row, col);

            for (size_t i = 0; i < row; i++)
            {
                for (size_t j = 0; j < col; j++)
                {
                    // 计算余子式
                    Numcpp<T> minor(row - 1, col - 1);

                    // 构建余子矩阵
                    size_t minor_i = 0;
                    for (size_t ii = 0; ii < row; ii++)
                    {
                        if (ii == i)
                            continue;

                        size_t minor_j = 0;
                        for (size_t jj = 0; jj < col; jj++)
                        {
                            if (jj == j)
                                continue;

                            minor.matrix[minor_i][minor_j] = matrix[ii][jj];
                            minor_j++;
                        }
                        minor_i++;
                    }

                    // 计算代数余子式并放入伴随矩阵（注意转置）
                    T sign = ((i + j) % 2 == 0) ? 1 : -1;
                    adjugate.matrix[j][i] = sign * minor.determinant();
                }
            }

            // 逆矩阵 = 伴随矩阵 / 行列式
            result = adjugate * (1.0 / det);
        }

        return result;
    }
    template <typename T>
    Numcpp<T> Numcpp<T>::pseudoinverse() const
    {
        this->ensure();
        if (row == 0 || col == 0)
        {
            throw std::invalid_argument("Matrix is empty");
        }

        // 计算 A^T * A
        Numcpp<T> ATA = this->transpose() * (*this);

        // 计算 ATA 的特征值和特征向量
        auto eig_result = ATA.eig();
        Numcpp<T> V = eig_result[1]; // 特征向量矩阵

        // 计算奇异值（特征值的平方根）
        auto S_values = (eig_result[0])<mklamb(T, {
            return sqrt(x);
        })>
            NULL;
        // 构建 Σ 矩阵（对角线为奇异值）
        size_t min_dim = std::min(row, col);
        Numcpp<T> Sigma(row, col, 0);
        for (size_t i = 0; i < min_dim; i++)
        {
            Sigma[i][i] = S_values[0][i];
        }

        // 计算 Σ 的伪逆（对角线元素取倒数，处理零奇异值）
        Numcpp<T> Sigma_plus(col, row, 0);
        T tolerance = 1e-6; // 设置一个小的阈值来处理零奇异值

        for (size_t i = 0; i < min_dim; i++)
        {
            if (fabs(Sigma[i][i]) > tolerance)
            {
                Sigma_plus[i][i] = 1.0 / Sigma[i][i];
            }
            // 否则保持为零（已经是零）
        }

        // 计算伪逆：A⁺ = V * Σ⁺ * U^T
        // 但 U = A * V * Σ⁺，所以 A⁺ = V * Σ⁺ * (A * V * Σ⁺)^T
        Numcpp<T> this_copy(*this);
        Numcpp<T> U = this_copy * V * Sigma_plus;
        Numcpp<T> A_plus = V * Sigma_plus * U.transpose();

        return A_plus;
    }
    template <typename T>
    void Numcpp<T>::svd(Numcpp<T> &U, Numcpp<T> &S, Numcpp<T> &V) const
    {
        this->ensure();
        Numcpp<T> AT(*this);
        AT.transposed();
        Numcpp<T> ATA = (*this) * AT;
        auto result = ATA.eig();

        // 特征向量矩阵，即U
        U = result[1];
        Numcpp<T> Sv = (result[0])<mklamb(T, { return sqrt(x); })> NULL;

        // 奇异值矩阵
        S = Numcpp<T>(row, col, 0.0);
        size_t mindim = std::min(row, col);
        for (size_t i = 0; i < mindim; i++)
        {
            S[i][i] = Sv[0][i];
        }
        Numcpp<T> S_inv(col, row, 0.0);
        for (size_t i = 0; i < mindim; i++)
        {
            if (S[i][i] != 0)
            {
                S_inv[i][i] = 1.0 / S[i][i];
            }
        }
        V = AT * U * S_inv.transpose();
    };
    template <typename T>
    std::vector<Numcpp<T>> Numcpp<T>::svd() const
    {
        this->ensure();
        Numcpp<T> AT(*this);
        AT.transposed();
        Numcpp<T> ATA = (*this) * AT;
        auto result = ATA.eig();

        // 特征向量矩阵，即U
        Numcpp<T> U = result[1];
        Numcpp<T> Sv = (result[0])<mklamb(T, { return sqrt(x); })> NULL;

        // 奇异值矩阵
        Numcpp<T> S = Numcpp<T>(row, col, 0.0);
        size_t mindim = std::min(row, col);
        for (size_t i = 0; i < mindim; i++)
        {
            S[i][i] = Sv[0][i];
        }
        Numcpp<T> S_inv(col, row, 0.0);
        for (size_t i = 0; i < mindim; i++)
        {
            if (S[i][i] != 0)
            {
                S_inv[i][i] = 1.0 / S[i][i];
            }
        }
        Numcpp<T> V = AT * U * S_inv.transpose();
        return {U, S, V};
    };
#if CUDA_CHECK
    template <typename T>
    void cuda_svd(Numcpp<T> &A, Numcpp<T> &U, Numcpp<T> &S, Numcpp<T> &Vt)
    {
        // 仅支持实数类型
        static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, "cuda_svd only supports float and double");

        U = Numcpp<T>(A.row, A.row, 0);
        S = Numcpp<T>(A.row, A.col, 0);
        Vt = Numcpp<T>(A.col, A.col, 0);

        U.to(DEVICE_CUDA);
        S.to(DEVICE_CUDA);
        Vt.to(DEVICE_CUDA);
        A.to(DEVICE_CUDA);
        cuda_op::cuda_svd<T>(A.device_data, U.device_data, S.device_data, Vt.device_data, A.row, A.col);
        U.to(DEVICE_LOCAL);
        S.to(DEVICE_LOCAL);
        Vt.to(DEVICE_LOCAL);
    }
#endif
#define MATtoNumcpp(mat_name, Numcpp, row, col) \
    for (size_t i = 0; i < row; i++)            \
    {                                           \
        for (size_t j = 0; j < col; j++)        \
        {                                       \
            Numcpp[i][j] = mat_name[i][j];      \
        }                                       \
    }
} // namespace np

// tools of matrix
namespace np
{
    /**
     * 将矩阵二值化（用于处理非二值矩阵）
     * @param mat 输入矩阵
     * @param threshold 阈值，大于等于此值的视为1，否则为0
     * @return 二值化后的矩阵
     */
    template <typename T>
    Numcpp<T> binarizeMatrix(const Numcpp<T> &mat, T threshold)
    {
        Numcpp<T> result(mat.row, mat.col);

        for (size_t i = 0; i < mat.row; i++)
        {
            for (size_t j = 0; j < mat.col; j++)
            {
                result[i][j] = (mat[i][j] >= threshold) ? 1 : 0;
            }
        }

        return result;
    }
    /**
     * 生成高斯随机矩阵的配置参数
     */
    struct GaussianConfig
    {
        double mean = 0.0;     // 均值
        double stddev = 1.0;   // 标准差
        unsigned int seed = 0; // 随机种子 (0表示使用随机设备)
    };

    /**
     * 方法1: 使用Box-Muller变换生成高斯随机数
     * 这是经典的高斯随机数生成方法
     */
    template <typename T>
    class BoxMullerGenerator
    {
    private:
        std::mt19937 generator;
        std::uniform_real_distribution<T> uniform;
        T z0, z1;
        bool hasSpare = false;

    public:
        BoxMullerGenerator(unsigned int seed = 0)
        {
            if (seed == 0)
            {
                std::random_device rd;
                seed = rd();
            }
            generator.seed(seed);
            uniform = std::uniform_real_distribution<T>(0.0, 1.0);
        }

        T generate(T mean = 0.0, T stddev = 1.0)
        {
            if (hasSpare)
            {
                hasSpare = false;
                return z1 * stddev + mean;
            }

            T u, v, s;
            do
            {
                u = uniform(generator) * 2.0 - 1.0;
                v = uniform(generator) * 2.0 - 1.0;
                s = u * u + v * v;
            } while (s >= 1.0 || s == 0.0);

            T mul = std::sqrt(-2.0 * std::log(s) / s);
            z0 = u * mul;
            z1 = v * mul;
            hasSpare = true;

            return z0 * stddev + mean;
        }
    };

    /**
     * 方法2: 使用C++11的std::normal_distribution
     * 更现代且高效的方法
     */
    template <typename T>
    class StandardGaussianGenerator
    {
    private:
        std::mt19937 generator;
        std::normal_distribution<T> normal;

    public:
        StandardGaussianGenerator(unsigned int seed = 0) : normal(0.0, 1.0)
        {
            if (seed == 0)
            {
                std::random_device rd;
                seed = rd();
            }
            generator.seed(seed);
        }

        T generate(T mean = 0.0, T stddev = 1.0)
        {
            return normal(generator) * stddev + mean;
        }
    };

    /**
     * 生成高斯随机矩阵 - 主函数
     * @param rows 行数
     * @param cols 列数
     * @param config 高斯分布配置
     * @param useBoxMuller 是否使用Box-Muller方法 (默认使用std::normal_distribution)
     * @return 高斯随机矩阵
     */
    template <typename T>
    Numcpp<T> randn(size_t rows, size_t cols,
                    const GaussianConfig &config = GaussianConfig(),
                    bool useBoxMuller = false)
    {

        if (rows == 0 || cols == 0)
        {
            throw std::invalid_argument("Matrix dimensions must be positive");
        }

        Numcpp<T> result(rows, cols);

        if (useBoxMuller)
        {
            // 使用Box-Muller方法
            BoxMullerGenerator<T> generator(config.seed);
            for (size_t i = 0; i < rows; i++)
            {
                for (size_t j = 0; j < cols; j++)
                {
                    result[i][j] = generator.generate(config.mean, config.stddev);
                }
            }
        }
        else
        {
            // 使用标准库方法
            StandardGaussianGenerator<T> generator(config.seed);
            for (size_t i = 0; i < rows; i++)
            {
                for (size_t j = 0; j < cols; j++)
                {
                    result[i][j] = generator.generate(config.mean, config.stddev);
                }
            }
        }

        return result;
    }

    /**
     * 多线程版本的高斯随机矩阵生成
     * 利用Numcpp的多线程优化功能
     */
    template <typename T>
    Numcpp<T> randn_parallel(size_t rows, size_t cols,
                             const GaussianConfig &config = GaussianConfig(),
                             size_t thread_count = 4)
    {

        if (rows == 0 || cols == 0)
        {
            throw std::invalid_argument("Matrix dimensions must be positive");
        }

        Numcpp<T> result(rows, cols);
        result.optimized(true);
        result.maxprocs_set(thread_count);

        // 为每个线程创建独立的随机数生成器
        std::vector<StandardGaussianGenerator<T>> generators;
        for (size_t i = 0; i < thread_count; i++)
        {
            generators.emplace_back(config.seed + i);
        }

        // 使用多线程填充矩阵
        units::thread_worker<T>(
            result.matrix, rows, cols,
            thread_count,
            [&](T **mat, size_t i, size_t j)
            {
                size_t thread_id = (i * cols + j) % thread_count;
                mat[i][j] = generators[thread_id].generate(config.mean, config.stddev);
            });

        return result;
    }

    /**
     * 生成协方差矩阵相关的高斯随机矩阵
     * 用于生成具有特定协方差结构的多变量高斯数据
     */
    template <typename T>
    Numcpp<T> multivariate_randn(size_t n_samples, const Numcpp<T> &covariance,
                                 const Numcpp<T> &mean = Numcpp<T>())
    {

        if (covariance.row != covariance.col)
        {
            throw std::invalid_argument("Covariance matrix must be square");
        }

        size_t n_features = covariance.row;

        // 如果没有提供均值向量，使用零向量
        Numcpp<T> mean_vector;
        if (mean.row == 0)
        {
            mean_vector = Numcpp<T>(1, n_features, 0.0);
        }
        else if (mean.row == 1 && mean.col == n_features)
        {
            mean_vector = mean;
        }
        else
        {
            throw std::invalid_argument("Mean must be a 1 x n_features vector");
        }

        // 对协方差矩阵进行Cholesky分解: covariance = L * L^T
        Numcpp<T> L = cholesky_decomposition(covariance);

        // 生成标准高斯随机矩阵
        Numcpp<T> Z = randn<T>(n_samples, n_features);

        // 转换: X = mean + Z * L^T
        Numcpp<T> X = Z * L.transpose();

        // 添加均值
        for (size_t i = 0; i < n_samples; i++)
        {
            for (size_t j = 0; j < n_features; j++)
            {
                X[i][j] += mean_vector[0][j];
            }
        }

        return X;
    }

    /**
     * Cholesky分解实现
     * 用于多变量高斯随机数生成
     */
    template <typename T>
    Numcpp<T> cholesky_decomposition(const Numcpp<T> &A)
    {
        if (A.row != A.col)
        {
            throw std::invalid_argument("Matrix must be square for Cholesky decomposition");
        }

        size_t n = A.row;
        Numcpp<T> L(n, n, 0.0);

        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j <= i; j++)
            {
                T sum = 0.0;

                if (j == i)
                {
                    // 对角线元素
                    for (size_t k = 0; k < j; k++)
                    {
                        sum += L[j][k] * L[j][k];
                    }
                    L[j][j] = std::sqrt(A[j][j] - sum);
                }
                else
                {
                    // 非对角线元素
                    for (size_t k = 0; k < j; k++)
                    {
                        sum += L[i][k] * L[j][k];
                    }
                    L[i][j] = (A[i][j] - sum) / L[j][j];
                }
            }
        }

        return L;
    }

    /**
     * 验证生成的高斯矩阵的统计特性
     */
    template <typename T>
    void validate_gaussian(const Numcpp<T> &matrix,
                           T expected_mean = 0.0,
                           T expected_stddev = 1.0,
                           T tolerance = 0.1)
    {

        T sum = 0.0;
        T sum_sq = 0.0;
        size_t total_elements = matrix.row * matrix.col;

        for (size_t i = 0; i < matrix.row; i++)
        {
            for (size_t j = 0; j < matrix.col; j++)
            {
                sum += matrix[i][j];
                sum_sq += matrix[i][j] * matrix[i][j];
            }
        }

        T mean = sum / total_elements;
        T variance = (sum_sq / total_elements) - (mean * mean);
        T stddev = std::sqrt(variance);

        std::cout << "高斯矩阵统计验证:\n";
        std::cout << "样本数量: " << total_elements << "\n";
        std::cout << "计算均值: " << mean << " (期望: " << expected_mean << ")\n";
        std::cout << "计算标准差: " << stddev << " (期望: " << expected_stddev << ")\n";

        if (std::abs(mean - expected_mean) < tolerance &&
            std::abs(stddev - expected_stddev) < tolerance)
        {
            std::cout << "✓ 统计特性在容差范围内\n";
        }
        else
        {
            std::cout << "✗ 统计特性超出容差范围\n";
        }
        std::cout << std::endl;
    }

    /**
     * 生成混合高斯分布矩阵
     * 用于创建具有多个高斯分量混合的数据
     */
    template <typename T>
    Numcpp<T> gaussian_mixture(size_t rows, size_t cols,
                               const std::vector<GaussianConfig> &components,
                               const std::vector<T> &weights = {})
    {

        if (components.empty())
        {
            throw std::invalid_argument("At least one Gaussian component required");
        }

        // 如果没有提供权重，使用均匀权重
        std::vector<T> actual_weights = weights;
        if (actual_weights.empty())
        {
            actual_weights = std::vector<T>(components.size(), 1.0 / components.size());
        }

        if (components.size() != actual_weights.size())
        {
            throw std::invalid_argument("Number of components and weights must match");
        }

        Numcpp<T> result(rows, cols);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> component_selector(actual_weights.begin(), actual_weights.end());

        // 为每个分量创建生成器
        std::vector<StandardGaussianGenerator<T>> generators;
        for (const auto &config : components)
        {
            generators.emplace_back(config.seed);
        }

        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < cols; j++)
            {
                int component = component_selector(gen);
                const auto &config = components[component];
                result[i][j] = generators[component].generate(config.mean, config.stddev);
            }
        }

        return result;
    }

    // LQR solver
    template <typename T>
    std::pair<np::Numcpp<T>, np::Numcpp<T>> solve_lqr(
        const np::Numcpp<T> &A, const np::Numcpp<T> &B,
        const np::Numcpp<T> &Q, const np::Numcpp<T> &R,
        int max_iter = 1000, T tolerance = 1e-6)
    {
        size_t n = A.row;
        size_t m = B.col;

        // 求解Riccati方程得到P
        np::Numcpp<T> P = Q; // 初始猜测

        T diff = 0.0;
        for (int iter = 0; diff < tolerance; iter++)
        {
            np::Numcpp<T> P_next = A.transpose() * P * A -
                                   A.transpose() * P * B *
                                       (B.transpose() * P * B + R).inverse() *
                                       B.transpose() * P * A +
                                   Q;
            for (size_t i = 0; i < n; i++)
            {
                for (size_t j = 0; j < n; j++)
                {
                    diff += std::abs(P_next[i][j] - P[i][j]);
                }
            }
            P = P_next;
        }

        // 计算反馈增益K
        np::Numcpp<T> BT = B.transpose();
        np::Numcpp<T> K = (R + BT * P * B).inverse() * BT * P * A;

        return {K, P};
    }

} // namespace np
#endif //!__NUMCPP__H__