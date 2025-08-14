#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <thread>
#include <functional>
#include <type_traits>
#include <complex>
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
    __device__ T add_opB(T x, T y)
    {
        return x + y;
    }

    // x - y
    template <typename T>
    __device__ T cut_opB(T x, T y)
    {
        return x - y;
    }
    // x * y
    template <typename T>
    __device__ T mul_opB(T x, T y)
    {
        return x * y;
    }
    // x / y (y != 0)
    template <typename T>
    __device__ T div_opB(T x, T y)
    {
        return x / y;
    }

    // x = y + z
    template <typename T>
    __device__ T add_opC(T x, T y, T z)
    {
        return y + z;
    }
    // x = y - z
    template <typename T>
    __device__ T cut_opC(T x, T y, T z)
    {
        return y - z;
    }

    template <typename T>
    __global__ static void kernel_cuda_opA(T **mat, size_t rows, size_t cols, size_t n, func_At<T> op)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            size_t r = idx / cols;
            size_t c = idx % cols;
            T *row = mat[r];
            row[c] = op(row[c]);
        }
    }
    template <typename T>
    __global__ static void kernel_cuda_memset(T **mat, T value, size_t rows, size_t cols, size_t n, func_Bt<T> op)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            size_t r = idx / cols;
            size_t c = idx % cols;
            T *row = mat[r];
            row[c] = op(row[c], value);
        }
    }
    template <typename T>
    __global__ static void kernel_cuda_opAB(T **a, T **b, size_t rows, size_t cols, size_t n, func_Bt<T> op)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            size_t r = idx / cols;
            size_t c = idx % cols;
            T *a_row = a[r];
            T *b_row = b[r];
            a_row[c] = op(a_row[c], b_row[c]);
        }
    }

    template <typename T>
    __global__ static void kernel_cuda_opABC(T **a, T **b, T **c, size_t rows, size_t cols, size_t n, func_Ct<T> op)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            size_t r = idx / cols;
            size_t col = idx % cols;
            T *a_row = a[r];
            T *b_row = b[r];
            T *c_row = c[r];
            a_row[col] = op(a_row[col], b_row[col], c_row[col]);
        }
    }
    template <typename T>
    void cuda_iterator(T **a, size_t rows_, size_t cols_, func_At<T> op)
    {
        constexpr size_t block_size = 256;
        size_t size_ = rows_ * cols_;
        const size_t grid_size = (size_ + block_size - 1) / block_size;

        kernel_cuda_opA<T><<<grid_size, block_size>>>(a, rows_, cols_, size_, op);
        cudaDeviceSynchronize();
    }

    template <typename T>
    void cuda_memset(T **a, T value, size_t rows_, size_t cols_, func_Bt<T> op)
    {
        constexpr size_t block_size = 256;
        size_t size_ = rows_ * cols_;
        const size_t grid_size = (size_ + block_size - 1) / block_size;

        kernel_cuda_memset<T><<<grid_size, block_size>>>(a, value, rows_, cols_, size_, op);
        cudaDeviceSynchronize();
    }

    template <typename T>
    void cuda_iterator(T **a, T **b, size_t rows_, size_t cols_, func_Bt<T> op)
    {

        constexpr size_t block_size = 256;
        size_t size_ = rows_ * cols_;
        const size_t grid_size = (size_ + block_size - 1) / block_size;

        kernel_cuda_opAB<T><<<grid_size, block_size>>>(a, b, rows_, cols_, size_, op);
        cudaDeviceSynchronize();
    }

    template <typename T>
    void cuda_iterator(T **a, T **b, T **c, size_t rows_, size_t cols_, func_Ct<T> op)
    {
        constexpr size_t block_size = 256;
        size_t size_ = rows_ * cols_;
        const size_t grid_size = (size_ + block_size - 1) / block_size;

        kernel_cuda_opABC<T><<<grid_size, block_size>>>(a, b, c, rows_, cols_, size_, op);
        cudaDeviceSynchronize();
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
            T sum = 0;
            for (size_t i = 0; i < N; ++i)
            {
                T *A_row = A[row];
                T *B_row = B[i];
                sum += A_row[i] * B_row[col];
            }
            T *C_row = C[row];
            C_row[col] = alpha * sum + beta * C_row[col];
        }
    }
    // 矩阵乘法 - 使用迭代器优化的CUDA实现
    template <typename T>
    void gemm(T **A, size_t A_row, size_t A_col, T **B, size_t B_col, T **C,
              T alpha = 1.0, T beta = 0.0)
    {
        constexpr size_t block_size = 16;
        dim3 block(block_size, block_size);
        dim3 grid((B_col + block_size - 1) / block_size,
                  (A_row + block_size - 1) / block_size);

        kernel_gemm<T><<<grid, block>>>(A, B, C, A_row, A_col, B_col,
                                        alpha, beta);
        cudaDeviceSynchronize();
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

    public:
        dataType **matrix;
        size_t row, col;
#if CUDA_CHECK
        bool mem_stat = false;
        bool MUL_GPU = true;
        dataType **device_data;
#endif
        Numcpp(const size_t _row, const size_t _col);
        Numcpp(const size_t _row, const size_t _col, dataType value);
        Numcpp(const Numcpp<dataType> &other);
        Numcpp(const dataType **mat, const size_t _row, const size_t _col);
// operators
#if CUDA_CHECK
        void to(const int device)
        {
            if (device == DEVICE_CUDA)
            {
                if (device_data == nullptr)
                {
                    size_t pitch;
                    cudaMallocPitch(device_data, &pitch, col * sizeof(dataType), row);
                    cudaMemcpy2D(device_data, pitch, matrix, col * sizeof(dataType), col * sizeof(dataType), row, cudaMemcpyHostToDevice);
                }
                else
                {
                    for (size_t i = 0; i < row; i++)
                    {
                        cudaMemcpy2D(device_data, col * sizeof(dataType), matrix, col * sizeof(dataType), col * sizeof(dataType), row, cudaMemcpyHostToDevice);
                    }
                }
                mem_stat = true;
            }
            else if (device == DEVICE_LOCAL)
            {
                for (size_t i = 0; i < row; i++)
                {
                    cudaMemcpy2D(matrix, col * sizeof(dataType), device_data, col * sizeof(dataType), col * sizeof(dataType), row, cudaMemcpyHostToDevice);
                }
            }
            else
            {
                std::invalid_argument("Invalid Device");
            }
        };
        void cuda_free()
        {
            cudaFree(device_data);
            mem_stat = false;
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
                        cuda_op::cuda_iterator<dataType>(this->device_data, row, col, cuda_op::add_opB);
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
                        cuda_op::cuda_iterator<dataType>(result.device_data, this->device_data, other.device_data, row, col, cuda_op::add_opC);
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
                        cuda_op::cuda_iterator<dataType>(this->device_data, other.device_data, row, col, cuda_op::cut_opB);
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
                        cuda_op::cuda_iterator<dataType>(result.device_data, this->device_data, other.device_data, row, col, cuda_op::cut_opC);
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
                    cuda_op::cuda_iterator<dataType>(result.device_data, this->device_data, row, col, cuda_op::mul_opB);
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
                    cuda_op::cuda_memset<dataType>(this->device_data, n, row, col, cuda_op::mul_opB);
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
            Numcpp<dataType> result(this->row, this->col);
            if (this->optimization == false)
            {
                for (size_t i = 0; i < this->row; i++)
                {
                    for (size_t j = 0; j < this->col; j++)
                    {
                        result.matrix[i][j] = this->matrix[i][j] * n;
                    }
                }
            }
            else
            {
#if CUDA_CHECK
                if (this->mem_stat == true)
                {
                    result.to(DEVICE_CUDA);
                    cuda_op::cuda_iterator<dataType>(result.device_data, this->device_data, row, col, cuda_op::div_opB);
                }
                else
                {
                    units::thread_worker<dataType>(result.matrix, this->row, this->col, this->maxprocs, [n](dataType **a, size_t i, size_t j)
                                                   { a[i][j] /= n; });
                }
#else
                units::thread_worker<dataType>(result.matrix, this->row, this->col, this->maxprocs, [n](dataType **a, size_t i, size_t j)
                                               { a[i][j] /= n; });
#endif
            }
            return result;
        }
        void operator/=(dataType n)
        {
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
                    cuda_op::cuda_memset<dataType>(this->device_data, n, row, col, cuda_op::div_opB);
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
                        }
                        else
                        {
                            throw std::invalid_argument("Invalid Matrix Device: Both parties involved in the operation should be on the same device.");
                        }
                    }
                    else
                    {
                        units::mm_generate(this->matrix, other.matrix, result.matrix, this->row, other.row, this->col, other.row, 0, 0, 0, 0);
                    }
#else
                    units::mm_generate(this->matrix, other.matrix, result.matrix, this->row, other.row, this->col, other.row, 0, 0, 0, 0);
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
    };
    // matrix special operate
    template <typename T>
    class oper_object
    {
    public:
        size_t row, col;
        T **matrix;
        T (*function_object)(T A, T B);
        oper_object(const Numcpp<T> &A, T (*function_object)(T A, T B))
        {
            this->row = A.row;
            this->col = A.col;
            this->matrix = (A.matrix);
            this->function_object = function_object;
        };
    };
    template <typename T>
    oper_object<T> operator<(const Numcpp<T> &A, T (*function_object)(T A, T B))
    {
        oper_object<T> oper(A, function_object);
        return oper;
    }
    template <typename T>
    Numcpp<T> operator>(const oper_object<T> &oper, const Numcpp<T> &B)
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
    Numcpp<T> operator>(const oper_object<T> &oper, void *data)
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
    inline Numcpp<T>::Numcpp(const T **mat, const size_t _row, const size_t _col)
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
    Numcpp<T>::~Numcpp()
    {
#if CUDA_CHECK
        if (mem_stat == true)
        {
            for (size_t i = 0; i < this->row; i++)
            {
                cudaFree(matrix[i]);
            }
            cudaFree(matrix);
        }
        else
        {
            for (size_t i = 0; i < this->row; i++)
            {
                delete matrix[i];
            }
            delete[] matrix;
        }
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
                units::thread_worker<T>(this->matrix, this->row, this->col, other.matrix, this->maxprocs, [](T **a, T **b, size_t i, size_t j)
                                        { a[i][j] *= b[i][j]; });
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

} // namespace np
