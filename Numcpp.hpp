#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <thread>
#include <functional>
#include <type_traits>
#include <complex>
#include <fstream>
#define NP_PI 3.14159265358979

#define CUDA_CHECK __has_include(<cuda.h>)

// cuda code
#if CUDA_CHECK
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
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

}; // namespace units
namespace np
{
    static bool is_optimized = false;
    static size_t MAX_thread = 1;
    template <typename dataType>
    class Numcpp
    {
    private:
        bool optimization = is_optimized;
        size_t maxprocs = MAX_thread;
#if CUDA_CHECK
        bool mem_stat = false;
        bool mem_synced = false;
#endif
    public:
        dataType **matrix;
        size_t row, col;
#if CUDA_CHECK
        bool MUL_GPU = true;
        bool auto_sync = false;
        dataType **device_data = nullptr;
#endif
        Numcpp(const size_t _row, const size_t _col);
        Numcpp(const size_t _row, const size_t _col, dataType value);
        Numcpp(const Numcpp<dataType> &other);
        Numcpp(dataType **mat, const size_t _row, const size_t _col);
        Numcpp(char *filename);
// operators
#if CUDA_CHECK
        void to(const int device)
        {
            if (device == DEVICE_CUDA && mem_stat == false)
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
                    units::thread_worker<dataType>(this->matrix, this->row, this->col, other.matrix, this->maxprocs, [](dataType **a, dataType **b, size_t i, size_t j)
                                                   { a[i][j] = b[i][j]; });
                }
            }
        }
        void operator+=(const Numcpp<dataType> &other)
        {
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
        Numcpp<dataType> operator+(const Numcpp<dataType> &other)
        {
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
        Numcpp<dataType> operator-(const Numcpp<dataType> &other)
        {
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
        Numcpp<dataType> operator+(dataType n)
        {
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
        Numcpp<dataType> operator-(dataType n)
        {
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

        Numcpp<dataType> operator*(dataType n)
        {
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
        Numcpp<dataType> operator/(dataType n)
        {
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
        Numcpp<dataType> operator*(const Numcpp<dataType> &other)
        {
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
        dataType *operator[](const size_t index)
        {
            return index < this->row ? this->matrix[index] : NULL;
        }
        /*transposed this matrix*/
        void transposed();
        /*the transposition of this matrix*/
        Numcpp transpose();
        void Hadamard_self(const Numcpp<dataType> &);
        Numcpp Hadamard(const Numcpp<dataType> &);

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
        Numcpp<dataType> fft(int inv)
        {
            static_assert(
                std::is_same_v<dataType, std::complex<float>> ||
                    std::is_same_v<dataType, std::complex<double>>,
                "FFT requires complex types");

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

        // 原地FFT
        void ffted(int inv)
        {
            static_assert(
                std::is_same_v<dataType, std::complex<float>> ||
                    std::is_same_v<dataType, std::complex<double>>,
                "FFT requires complex types");

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
#endif
        void save(char *path)
        {
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
        template <typename T>
        friend std::ostream &operator<<(std::ostream &stream, const Numcpp<T> &m)
        {
            stream << '(' << m.row << ',' << m.col << ')' << "[\n";
            for (size_t i = 0; i < m.row; ++i)
            {
                stream << "    [" << i << "][";
                for (size_t j = 0; j < m.col; ++j)
                {
                    stream << (T)(m.matrix[i][j]) << (j == m.col - 1 ? "]\n" : " , ");
                }
            }
            stream << "]\n";
            return stream;
        }
        dataType determinant() const;
        Numcpp<dataType> inverse() const;
        Numcpp<dataType> pseudoinverse() const; // 伪逆计算
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
    smul_object<T> operator<(const Numcpp<T> &A, auto function_object)
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
    template <typename T>
    Numcpp<T> Numcpp<T>::transpose()
    {
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
    Numcpp<T> Numcpp<T>::Hadamard(const Numcpp<T> &other)
    {
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
    T Numcpp<T>::determinant() const
    {
        if (row != col)
        {
            throw std::invalid_argument("Matrix must be square to compute determinant.");
        }

        // 复制矩阵数据以避免修改原矩阵
        T **temp = units::mat_create<T>(row, col);

        if (this->optimization == false)
        {
            for (size_t i = 0; i < row; i++)
            {
                for (size_t j = 0; j < col; j++)
                {
                    temp[i][j] = matrix[i][j];
                }
            }
        }
        else
        {
            units::Copy_thread_worker<T>(temp, row, col, matrix, this->maxprocs,
                                         [](T **a, T **b, size_t i, size_t j)
                                         { a[i][j] = b[i][j]; });
        }

        T det = static_cast<T>(1);
        int sign = 1;

        // LU分解 with partial pivoting
        for (size_t k = 0; k < row; k++)
        {
            // 寻找主元
            size_t max_row = k;
            T max_val = std::abs(temp[k][k]);
            for (size_t i = k + 1; i < row; i++)
            {
                T val = std::abs(temp[i][k]);
                if (std::abs(val) > std::abs(max_val))
                {
                    max_val = val;
                    max_row = i;
                }
            }

            // 如果主元为0，则行列式为0
            if (max_val == static_cast<T>(0))
            {
                units::mat_delete(temp, row);
                return static_cast<T>(0);
            }

            // 交换行
            if (max_row != k)
            {
                std::swap(temp[k], temp[max_row]);
                sign *= -1; // 行交换改变符号
            }

            det *= temp[k][k];

            // 消元 - 使用多线程优化
            if (this->optimization && (row - k - 1) > 100)
            {
                units::thread_worker<T>(temp, row - k - 1, col - k, this->maxprocs,
                                        [k, temp](T **a, size_t i, size_t j)
                                        {
                                            size_t row_idx = k + 1 + i;
                                            size_t col_idx = k + j;
                                            T factor = a[row_idx][k] / a[k][k];
                                            a[row_idx][col_idx] -= factor * a[k][col_idx];
                                        });
            }
            else
            {
                for (size_t i = k + 1; i < row; i++)
                {
                    T factor = temp[i][k] / temp[k][k];
                    for (size_t j = k + 1; j < col; j++)
                    {
                        temp[i][j] -= factor * temp[k][j];
                    }
                }
            }
        }

        units::mat_delete(temp, row);
        return det * static_cast<T>(sign);
    }

    template <typename T>
    Numcpp<T> Numcpp<T>::inverse() const
    {
        if (row != col)
        {
            throw std::invalid_argument("Standard inverse is only defined for square matrices. Use pseudoinverse() for non-square matrices.");
        }

        // 创建增广矩阵 [A | I]
        T **aug = units::mat_create<T>(row, 2 * col);

        // 初始化增广矩阵
        if (this->optimization)
        {
            // 使用多线程初始化增广矩阵
            units::thread_worker<T>(aug, row, 2 * col, this->maxprocs,
                                    [this](T **a, size_t i, size_t j)
                                    {
                                        if (j < col)
                                        {
                                            a[i][j] = matrix[i][j];
                                        }
                                        else
                                        {
                                            a[i][j] = (j == i + col) ? static_cast<T>(1) : static_cast<T>(0);
                                        }
                                    });
        }
        else
        {
            for (size_t i = 0; i < row; i++)
            {
                for (size_t j = 0; j < col; j++)
                {
                    aug[i][j] = matrix[i][j];
                }
                for (size_t j = col; j < 2 * col; j++)
                {
                    aug[i][j] = (j == i + col) ? static_cast<T>(1) : static_cast<T>(0);
                }
            }
        }

        // 全选主元高斯-约旦消元
        std::vector<size_t> col_swap(col); // 记录列交换
        for (size_t i = 0; i < col; i++)
            col_swap[i] = i;

        for (size_t k = 0; k < row; k++)
        {
            // 寻找主元
            size_t max_row = k, max_col = k;
            T max_val = std::abs(aug[k][k]);
            for (size_t i = k; i < row; i++)
            {
                for (size_t j = k; j < col; j++)
                {
                    T val = std::abs(aug[i][j]);
                    if (std::abs(val) > std::abs(max_val))
                    {
                        max_val = val;
                        max_row = i;
                        max_col = j;
                    }
                }
            }

            if (max_val == static_cast<T>(0))
            {
                units::mat_delete(aug, row);
                throw std::runtime_error("Matrix is singular and cannot be inverted.");
            }

            // 交换行
            if (max_row != k)
            {
                std::swap(aug[k], aug[max_row]);
            }

            // 交换列
            if (max_col != k)
            {
                for (size_t i = 0; i < row; i++)
                {
                    std::swap(aug[i][k], aug[i][max_col]);
                }
                std::swap(col_swap[k], col_swap[max_col]);
            }

            // 归一化主元行
            T pivot = aug[k][k];

            if (this->optimization)
            {
                // 使用多线程归一化
                units::thread_worker<T>(aug, 1, 2 * col - k, this->maxprocs,
                                        [k, pivot](T **a, size_t i, size_t j)
                                        {
                                            a[k][k + j] /= pivot;
                                        });
            }
            else
            {
                for (size_t j = k; j < 2 * col; j++)
                {
                    aug[k][j] /= pivot;
                }
            }

            // 消元 - 使用多线程优化
            if (this->optimization && row > 100)
            {
                units::thread_worker<T>(aug, row, 2 * col, this->maxprocs,
                                        [k](T **a, size_t i, size_t j)
                                        {
                                            if (i != k)
                                            {
                                                T factor = a[i][k];
                                                a[i][j] -= factor * a[k][j];
                                            }
                                        });
            }
            else
            {
                for (size_t i = 0; i < row; i++)
                {
                    if (i == k)
                        continue;
                    T factor = aug[i][k];
                    for (size_t j = k; j < 2 * col; j++)
                    {
                        aug[i][j] -= factor * aug[k][j];
                    }
                }
            }
        }

        // 提取逆矩阵
        Numcpp<T> result(row, col);

        // 根据列交换调整逆矩阵
        if (this->optimization)
        {
            units::thread_worker<T>(aug, row, col, result.matrix, this->maxprocs,
                                    [&](T **a, T **b, size_t i, size_t j)
                                    {
                                        b[i][col_swap[j]] = a[i][j + col];
                                    });
        }
        else
        {
            for (size_t i = 0; i < row; i++)
            {
                for (size_t j = 0; j < col; j++)
                {
                    result.matrix[i][col_swap[j]] = aug[i][j + col];
                }
            }
        }

        units::mat_delete(aug, row);
        return result;
    }
    template <typename T>
    Numcpp<T> Numcpp<T>::pseudoinverse() const
    {
        // 计算伪逆: A⁺ = (AᵀA)⁻¹Aᵀ (对于满列秩矩阵)
        // 或者使用SVD分解，但这里使用更简单的方法

        // 计算转置
        Numcpp<T> A_T = this->transpose();

        // 计算 AᵀA
        Numcpp<T> ATA = A_T * (*this);

        try
        {
            // 尝试计算逆
            Numcpp<T> ATA_inv = ATA.inverse();

            // 计算伪逆: (AᵀA)⁻¹Aᵀ
            return ATA_inv * A_T;
        }
        catch (const std::runtime_error &e)
        {
            // 如果矩阵是奇异的，使用正则化方法
            // 添加一个小的正则化参数
            T lambda = static_cast<T>(1e-10);
            Numcpp<T> regularized = ATA;

            // 添加λI到对角线
            if (this->optimization)
            {
                units::thread_worker<T>(regularized.matrix, regularized.row, regularized.col, this->maxprocs,
                                        [lambda](T **a, size_t i, size_t j)
                                        {
                                            if (i == j)
                                                a[i][j] += lambda;
                                        });
            }
            else
            {
                for (size_t i = 0; i < regularized.row; i++)
                {
                    regularized.matrix[i][i] += lambda;
                }
            }

            // 计算逆
            Numcpp<T> regularized_inv = regularized.inverse();

            // 计算伪逆: (AᵀA + λI)⁻¹Aᵀ
            return regularized_inv * A_T;
        }
    }
} // namespace np
