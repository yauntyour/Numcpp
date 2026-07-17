#ifndef NUMCPP_CORE_HPP
#define NUMCPP_CORE_HPP

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <thread>
#include <functional>
#include <type_traits>
#include <complex>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdint>
#include <limits>

#define NP_PI 3.14159265358979

template <typename T>
struct is_complex : std::false_type {};
template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};
template <typename T>
constexpr bool is_complex_v = is_complex<T>::value;

#define MATtoPtr2D(T, value_name, change_name, row, col) \
    T *change_name[row];                                  \
    for (size_t i = 0; i < row; i++) { change_name[i] = value_name[i]; }

#define mklamb(T, codes, ...) ([__VA_ARGS__](T x, T y)->T codes)

namespace units
{
    template <typename T>
    void trans(std::complex<T> **x, size_t size_x)
    {
        size_t p = 0;
        size_t a, b;
        for (size_t i = 1; i < size_x; i *= 2) { p++; }
        for (size_t i = 0; i < size_x; i++)
        {
            a = i;
            b = 0;
            for (size_t j = 0; j < p; j++) { b = (b << 1) + (a & 1); a >>= 1; }
            if (b > i)
            {
                std::complex<T> temp = (*x)[i];
                (*x)[i] = (*x)[b];
                (*x)[b] = temp;
            }
        }
    }

    template <typename T>
    void fft(std::complex<T> **x, size_t size, std::complex<T> **X, int inv)
    {
        std::complex<T> *Wn = new std::complex<T>[size];
        for (size_t i = 0; i < size; i++)
        {
            T angle = static_cast<T>(-2 * NP_PI * i / size);
            Wn[i] = std::complex<T>(cos(angle), inv * sin(angle));
        }
        if (x != X)
            for (size_t i = 0; i < size; i++) { (*X)[i] = (*x)[i]; }
        std::complex<T> *p = *X;
        trans<T>(&p, size);
        for (size_t m = 2; m <= size; m *= 2)
            for (size_t k = 0; k < size; k += m)
                for (size_t j = 0; j < m / 2; j++)
                {
                    size_t index1 = k + j, index2 = index1 + m / 2, t = j * size / m;
                    std::complex<T> temp1 = (*X)[index1], temp2 = (*X)[index2] * Wn[t];
                    (*X)[index1] = temp1 + temp2;
                    (*X)[index2] = temp1 - temp2;
                }
        delete[] Wn;
    }

    template <typename T>
    void mm_generate(T **this_matrix, T **other_matrix, T **result,
                     const size_t A_row, const size_t B_row, const size_t A_col, const size_t B_col,
                     const size_t A_row_offset, const size_t A_col_offset,
                     const size_t B_row_offset, const size_t B_col_offset)
    {
        for (size_t i = 0; i < A_row; i++)
            for (size_t j = 0; j < B_col; j++)
                for (size_t k = 0; k < A_col; k++)
                    result[i][j] += (this_matrix[i + A_row_offset][k + A_col_offset] *
                                     other_matrix[k + B_row_offset][j + B_col_offset]);
    }

    template <typename T>
    T **mat_create(size_t row, size_t col)
    {
        T **matrix = new T *[row];
        for (size_t i = 0; i < row; i++)
        {
            matrix[i] = new T[col];
            for (size_t j = 0; j < col; j++) { matrix[i][j] = (T)0; }
        }
        return matrix;
    }

    template <typename T>
    void mat_delete(T **mat, size_t row)
    {
        for (size_t i = 0; i < row; i++) { delete[] mat[i]; }
        delete[] mat;
    }

    template <typename T>
    void mm_Coppersmith_Winograd(T **this_matrix, T **other_matrix, T **result,
                                 const size_t A_row, const size_t B_row, const size_t A_col, const size_t B_col,
                                 const size_t A_row_offset, const size_t A_col_offset,
                                 const size_t B_row_offset, const size_t B_col_offset)
    {
        if ((A_row <= 2) || (A_row % 2 != 0 || B_col % 2 != 0 || A_col % 2 != 0))
            return mm_generate(this_matrix, other_matrix, result,
                               A_row, B_col, A_col, B_col,
                               A_row_offset, A_col_offset, B_row_offset, B_col_offset);

        size_t halfAR = A_row / 2, halfAC = A_col / 2, halfBR = B_row / 2, halfBC = B_col / 2;

        T **S1 = mat_create<T>(halfAR, halfAC);
        T **S2 = mat_create<T>(halfAR, halfAC);
        T **S3 = mat_create<T>(halfAR, halfAC);
        T **S4 = mat_create<T>(halfAR, halfAC);
        for (size_t i = 0; i < halfAR; i++)
            for (size_t j = 0; j < halfAC; j++)
            {
                S1[i][j] = this_matrix[halfAR + i + A_row_offset][halfAC + j + A_col_offset] +
                           this_matrix[halfAR + i + A_row_offset][halfAC + j + A_col_offset];
                S2[i][j] = S1[i][j] - this_matrix[i + A_row_offset][j + A_col_offset];
                S3[i][j] = this_matrix[i + A_row_offset][j + A_col_offset] -
                           this_matrix[halfAR + i + A_row_offset][halfAC + j + A_col_offset];
                S4[i][j] = this_matrix[i + A_row_offset][halfAC + j + A_col_offset] - S2[i][j];
            }

        T **T1 = mat_create<T>(halfBR, halfBC);
        T **T2 = mat_create<T>(halfBR, halfBC);
        T **T3 = mat_create<T>(halfBR, halfBC);
        T **T4 = mat_create<T>(halfBR, halfBC);
        for (size_t i = 0; i < halfBR; i++)
            for (size_t j = 0; j < halfBC; j++)
            {
                T1[i][j] = other_matrix[i + B_row_offset][halfBC + j + B_col_offset] -
                           other_matrix[i + B_row_offset][j + B_col_offset];
                T2[i][j] = other_matrix[halfBR + i + B_row_offset][halfBC + j + B_col_offset] - T1[i][j];
                T3[i][j] = other_matrix[halfBR + i + B_row_offset][halfBC + j + B_col_offset] -
                           other_matrix[i + B_row_offset][halfBC + j + B_col_offset];
                T4[i][j] = T2[i][j] - other_matrix[halfBR + i + B_row_offset][j + B_col_offset];
            }

        T **M1 = mat_create<T>(halfAR, halfBC);
        mm_Coppersmith_Winograd(this_matrix, other_matrix, M1, halfAR, halfBR, halfAC, halfBC, 0, 0, 0, 0);
        T **M2 = mat_create<T>(halfAR, halfBC);
        mm_Coppersmith_Winograd(this_matrix, other_matrix, M2, halfAR, halfBR, halfAC, halfBC, 0, halfAC, 0, 0);
        T **M3 = mat_create<T>(halfAR, halfBC);
        mm_Coppersmith_Winograd(S4, other_matrix, M3, halfAR, halfBR, halfAC, halfBC, 0, 0, halfAC, halfBC);
        T **M4 = mat_create<T>(halfAR, halfBC);
        mm_Coppersmith_Winograd(this_matrix, T4, M4, halfAR, halfBR, halfAC, halfBC, halfAR, halfAC, 0, 0);
        T **M5 = mat_create<T>(halfAR, halfBC);
        mm_Coppersmith_Winograd(S1, T1, M5, halfAR, halfBR, halfAC, halfBC, 0, 0, 0, 0);
        T **M6 = mat_create<T>(halfAR, halfBC);
        mm_Coppersmith_Winograd(S2, T2, M6, halfAR, halfBR, halfAC, halfBC, 0, 0, 0, 0);
        T **M7 = mat_create<T>(halfAR, halfBC);
        mm_Coppersmith_Winograd(S3, T3, M7, halfAR, halfBR, halfAC, halfBC, 0, 0, 0, 0);

        for (size_t i = 0; i < halfAR; i++)
            for (size_t j = 0; j < halfBC; j++)
            {
                result[i][j] = M1[i][j] + M2[i][j];
                result[i][halfBC + j] = M1[i][j] + M6[i][j] + M5[i][j] + M3[i][j];
                result[halfAR + i][j] = M1[i][j] + M6[i][j] + M7[i][j] - M4[i][j];
                result[halfAR + i][halfBC + j] = M1[i][j] + M6[i][j] + M7[i][j] + M5[i][j];
            }

        mat_delete<T>(S1, halfAR); mat_delete<T>(S2, halfAR);
        mat_delete<T>(S3, halfAR); mat_delete<T>(S4, halfAR);
        mat_delete<T>(T1, halfBR); mat_delete<T>(T2, halfBR);
        mat_delete<T>(T3, halfBR); mat_delete<T>(T4, halfBR);
        mat_delete<T>(M1, halfAR); mat_delete<T>(M2, halfAR);
        mat_delete<T>(M3, halfAR); mat_delete<T>(M4, halfAR);
        mat_delete<T>(M5, halfAR); mat_delete<T>(M6, halfAR);
        mat_delete<T>(M7, halfAR);
    }

    template <typename T>
    void mm_auto(T **this_matrix, T **other_matrix, T **result,
                 const size_t A_row, const size_t B_row, const size_t A_col, const size_t B_col, const bool fast_flag)
    {
        if ((A_row * B_col * A_col <= 64 * 64 * 64) || (A_row % 2 != 0 || B_col % 2 != 0 || A_col % 2 != 0))
            return mm_generate(this_matrix, other_matrix, result, A_row, B_row, A_col, B_col, 0, 0, 0, 0);
        else if (fast_flag)
            return mm_Coppersmith_Winograd(this_matrix, other_matrix, result, A_row, B_row, A_col, B_col, 0, 0, 0, 0);
        else
            throw std::invalid_argument("Matrix too large, use multicore or enable Coppersmith-Winograd.");
    }

    template <typename T>
    void atomic_opalloc(T **a, size_t offset_i, size_t offset_j, size_t black_len_i, size_t black_len_j,
                        size_t *sign, std::function<void(T **, size_t, size_t)> opalloc)
    {
        for (size_t i = 0; i < black_len_i; i++)
        {
            a[offset_i + i] = new T[black_len_j];
            for (size_t j = 0; j < black_len_j; j++) { opalloc(a, offset_i + i, offset_j + j); }
        }
        *sign += black_len_i * black_len_j;
    }

    template <typename T>
    void atomic_opcopy(T **a, T **b, size_t offset_i, size_t offset_j, size_t black_len_i, size_t black_len_j,
                       size_t *sign, std::function<void(T **, T **, size_t, size_t)> opcopy)
    {
        for (size_t i = 0; i < black_len_i; i++)
        {
            a[offset_i + i] = new T[black_len_j];
            for (size_t j = 0; j < black_len_j; j++) { opcopy(a, b, offset_i + i, offset_j + j); }
        }
        *sign += black_len_i * black_len_j;
    }

    template <typename T>
    void atomic_op(T **a, size_t offset_i, size_t offset_j, size_t black_len_i, size_t black_len_j,
                   size_t *sign, std::function<void(T **, size_t, size_t)> opA)
    {
        for (size_t i = 0; i < black_len_i; i++)
            for (size_t j = 0; j < black_len_j; j++) { opA(a, offset_i + i, offset_j + j); }
        *sign += black_len_i * black_len_j;
    }

    template <typename T>
    void atomic_op(T **a, T **b, size_t offset_i, size_t offset_j, size_t black_len_i, size_t black_len_j,
                   size_t *sign, std::function<void(T **, T **, size_t, size_t)> opAB)
    {
        for (size_t i = 0; i < black_len_i; i++)
            for (size_t j = 0; j < black_len_j; j++) { opAB(a, b, offset_i + i, offset_j + j); }
        *sign += black_len_i * black_len_j;
    }

    template <typename T>
    void atomic_op(T **a, T **b, T **c, size_t offset_i, size_t offset_j, size_t black_len_i, size_t black_len_j,
                   size_t *sign, std::function<void(T **, T **, T **, size_t, size_t)> opABC)
    {
        for (size_t i = 0; i < black_len_i; i++)
            for (size_t j = 0; j < black_len_j; j++) { opABC(a, b, c, offset_i + i, offset_j + j); }
        *sign += black_len_i * black_len_j;
    }

    template <typename T>
    int Alloc_thread_worker(T **a, size_t a_row, size_t a_col, size_t cpu_thread_max,
                            std::function<void(T **, size_t, size_t)> opA)
    {
        size_t mat_n = static_cast<size_t>(std::sqrt(static_cast<double>(cpu_thread_max)));
        std::thread *t_list = new std::thread[cpu_thread_max];
        size_t black_len_i = a_row / mat_n, black_len_j = a_col / mat_n, sign = 0;
        for (size_t i = 0; i < mat_n; i++)
            for (size_t j = 0; j < mat_n; j++)
            {
                t_list[i * mat_n + j] = std::thread([=, &sign]() {
                    atomic_opalloc<T>(a, i * black_len_i, j * black_len_j, black_len_i, black_len_j, &sign, opA);
                });
                t_list[i * mat_n + j].detach();
            }
        while (sign < a_row * a_col) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
        delete[] t_list;
        return 0;
    }

    template <typename T>
    int thread_worker(T **a, size_t a_row, size_t a_col, size_t cpu_thread_max,
                      std::function<void(T **, size_t, size_t)> opA)
    {
        size_t mat_n = static_cast<size_t>(std::sqrt(static_cast<double>(cpu_thread_max)));
        std::thread *t_list = new std::thread[cpu_thread_max];
        size_t black_len_i = a_row / mat_n, black_len_j = a_col / mat_n, sign = 0;
        for (size_t i = 0; i < mat_n; i++)
            for (size_t j = 0; j < mat_n; j++)
            {
                t_list[i * mat_n + j] = std::thread([=, &sign]() {
                    atomic_op<T>(a, i * black_len_i, j * black_len_j, black_len_i, black_len_j, &sign, opA);
                });
                t_list[i * mat_n + j].detach();
            }
        while (sign < a_row * a_col) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
        delete[] t_list;
        return 0;
    }

    template <typename T>
    int Copy_thread_worker(T **a, size_t a_row, size_t a_col, T **b, size_t cpu_thread_max,
                           std::function<void(T **, T **, size_t, size_t)> opAB)
    {
        size_t mat_n = static_cast<size_t>(std::sqrt(static_cast<double>(cpu_thread_max)));
        std::thread *t_list = new std::thread[cpu_thread_max];
        size_t black_len_i = a_row / mat_n, black_len_j = a_col / mat_n, sign = 0;
        for (size_t i = 0; i < mat_n; i++)
            for (size_t j = 0; j < mat_n; j++)
            {
                t_list[i * mat_n + j] = std::thread([=, &sign]() {
                    atomic_opcopy<T>(a, b, i * black_len_i, j * black_len_j, black_len_i, black_len_j, &sign, opAB);
                });
                t_list[i * mat_n + j].detach();
            }
        while (sign < a_row * a_col) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
        delete[] t_list;
        return 0;
    }

    template <typename T>
    int thread_worker(T **a, size_t a_row, size_t a_col, T **b, size_t cpu_thread_max,
                      std::function<void(T **, T **, size_t, size_t)> opAB)
    {
        size_t mat_n = static_cast<size_t>(std::sqrt(static_cast<double>(cpu_thread_max)));
        std::thread *t_list = new std::thread[cpu_thread_max];
        size_t black_len_i = a_row / mat_n, black_len_j = a_col / mat_n, sign = 0;
        for (size_t i = 0; i < mat_n; i++)
            for (size_t j = 0; j < mat_n; j++)
            {
                t_list[i * mat_n + j] = std::thread([=, &sign]() {
                    atomic_op<T>(a, b, i * black_len_i, j * black_len_j, black_len_i, black_len_j, &sign, opAB);
                });
                t_list[i * mat_n + j].detach();
            }
        while (sign < a_row * a_col) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
        delete[] t_list;
        return 0;
    }

    template <typename T>
    int thread_worker(T **a, size_t a_row, size_t a_col, T **b, T **c, size_t cpu_thread_max,
                      std::function<void(T **, T **, T **, size_t, size_t)> opABC)
    {
        size_t mat_n = static_cast<size_t>(std::sqrt(static_cast<double>(cpu_thread_max)));
        std::thread *t_list = new std::thread[cpu_thread_max];
        size_t black_len_i = a_row / mat_n, black_len_j = a_col / mat_n, sign = 0;
        for (size_t i = 0; i < mat_n; i++)
            for (size_t j = 0; j < mat_n; j++)
            {
                t_list[i * mat_n + j] = std::thread([=, &sign]() {
                    atomic_op<T>(a, b, c, i * black_len_i, j * black_len_j, black_len_i, black_len_j, &sign, opABC);
                });
                t_list[i * mat_n + j].detach();
            }
        while (sign < a_row * a_col) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
        delete[] t_list;
        return 0;
    }

    template <typename T>
    void qr_decomposition_gm(T **A, size_t n, T **Q, T **R)
    {
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++) { Q[i][j] = 0; R[i][j] = 0; }
        for (size_t j = 0; j < n; j++)
        {
            T *v = new T[n];
            for (size_t i = 0; i < n; i++) { v[i] = A[i][j]; }
            for (size_t i = 0; i < j; i++)
            {
                T dot_product = 0;
                for (size_t k = 0; k < n; k++) { dot_product += Q[k][i] * A[k][j]; }
                R[i][j] = dot_product;
                for (size_t k = 0; k < n; k++) { v[k] -= R[i][j] * Q[k][i]; }
            }
            T norm_v = 0;
            for (size_t i = 0; i < n; i++) { norm_v += v[i] * v[i]; }
            norm_v = std::sqrt(norm_v);
            R[j][j] = norm_v;
            for (size_t i = 0; i < n; i++) { Q[i][j] = v[i] / norm_v; }
            delete[] v;
        }
    }
} // namespace units

namespace np
{
    enum NormType { L1, L2, INF };

    static bool is_optimized = false;
    static size_t MAX_thread = 1;

    template <typename T>
    class Numcpp;

    template <typename T>
    struct is_numcpp : std::false_type {};
    template <typename T>
    struct is_numcpp<Numcpp<T>> : std::true_type {};
    template <typename T>
    constexpr bool is_numcpp_v = is_numcpp<T>::value;

    template <typename T>
    struct fft_result { using type = std::complex<T>; };
    template <typename T>
    struct fft_result<std::complex<T>> { using type = std::complex<T>; };
    template <typename T>
    using fft_result_t = typename fft_result<T>::type;

    template <typename T>
    std::ostream &operator<<(std::ostream &, const Numcpp<T> &);

    template <typename dataType>
    class Numcpp
    {
    private:
        bool optimization = is_optimized;
        size_t maxprocs = MAX_thread;
        bool is_destroy = true;

    public:
        using value_type = dataType;
        dataType **matrix = nullptr;
        size_t row = 0, col = 0;

        Numcpp() = default;
        Numcpp(const size_t _row, const size_t _col);
        Numcpp(const size_t _row, const size_t _col, dataType value);
        Numcpp(const Numcpp<dataType> &other);
        Numcpp(dataType *mat, const size_t _row, const size_t _col);
        Numcpp(dataType **mat, const size_t _row, const size_t _col);
        Numcpp(const char *filename);

        void ensure() const
        {
            if (matrix == nullptr && is_destroy == true)
            {
                if (row == 0 && col == 0)
                    throw std::runtime_error("The matrix maybe had not been init.");
                else
                    throw std::runtime_error("The matrix maybe had been destoried.");
            }
        }

        void operator=(const Numcpp<dataType> &other);
        void operator+=(const Numcpp<dataType> &other);
        Numcpp<dataType> operator+(const Numcpp<dataType> &other) const;
        void operator-=(const Numcpp<dataType> &other);
        Numcpp<dataType> operator-(const Numcpp<dataType> &other) const;
        Numcpp<dataType> operator+(dataType n) const;
        void operator+=(dataType n);
        Numcpp<dataType> operator-(dataType n) const;
        void operator-=(dataType n);
        Numcpp<dataType> operator*(dataType n) const;
        void operator*=(dataType n);
        Numcpp<dataType> operator/(dataType n) const;
        void operator/=(dataType n);
        Numcpp<dataType> operator*(const Numcpp<dataType> &other) const;

        dataType *operator[](const size_t index) const
        {
            ensure();
            return index < this->row ? this->matrix[index] : nullptr;
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
                for (size_t i = 0; i < this->row; i++) { result[i][0] = this->matrix[i][index]; }
                return result;
            }
            return Numcpp<dataType>();
        }

        void transposed();
        Numcpp<dataType> transpose() const;
        void Hadamard_self(const Numcpp<dataType> &);
        Numcpp<dataType> Hadamard(const Numcpp<dataType> &) const;

        void optimized(bool flag) { this->optimization = flag; }
        void maxprocs_set(size_t thread_num)
        {
            if (std::sqrt(static_cast<double>(thread_num)) * std::sqrt(static_cast<double>(thread_num)) > std::thread::hardware_concurrency() || thread_num < 1)
                throw std::invalid_argument("Invalid maxprocs.");
            this->maxprocs = thread_num;
        }
        ~Numcpp();

        // FFT: returns complex result for all types
        Numcpp<fft_result_t<dataType>> fft(int inv) const;
        void ffted(int inv);

        dataType sum() const;
        void save(const char *path);
        template <typename U>
        friend std::ostream &operator<<(std::ostream &stream, const Numcpp<U> &m);

        dataType determinant() const;
        Numcpp<dataType> inverse() const;
        Numcpp<dataType> pseudoinverse() const;

        class CommaInitializer
        {
        public:
            CommaInitializer(Numcpp *mat, size_t current_index) : mat_(mat), current_index_(current_index) {}
            CommaInitializer &operator,(dataType value)
            {
                size_t r = current_index_ / mat_->col;
                size_t c = current_index_ % mat_->col;
                if (r < mat_->row && c < mat_->col) { mat_->matrix[r][c] = value; current_index_++; }
                else throw std::out_of_range("Too many elements for matrix.");
                return *this;
            }
            operator Numcpp() const { return *(this->mat_); }
        private:
            Numcpp *mat_;
            size_t current_index_;
        };

        CommaInitializer operator<<(dataType value)
        {
            ensure();
            if (row * col == 0) throw std::out_of_range("Matrix is empty.");
            matrix[0][0] = value;
            return CommaInitializer(this, 1);
        }

        void svd(Numcpp<dataType> &U, Numcpp<dataType> &S, Numcpp<dataType> &V) const;
        std::vector<Numcpp<dataType>> svd() const;

        void zero_approximation(double tolerance = 1e-6)
        {
            ensure();
            if (this->optimization == false)
            {
                for (size_t i = 0; i < this->row; i++)
                    for (size_t j = 0; j < this->col; j++)
                        if (std::abs(this->matrix[i][j]) < tolerance)
                            this->matrix[i][j] = 0;
            }
            else
            {
                units::thread_worker<dataType>(this->matrix, this->row, this->col, this->maxprocs,
                    [=](dataType **a, size_t i, size_t j) {
                        if (std::abs(a[i][j]) < tolerance) a[i][j] = 0;
                    });
            }
        }

        bool is_symmetric(double tolerance = 1e-6) const
        {
            ensure();
            if (row != col) return false;
            for (size_t i = 0; i < row; i++)
                for (size_t j = 0; j < i; j++)
                    if (std::abs(matrix[i][j] - matrix[j][i]) > tolerance)
                        return false;
            return true;
        }

        void set_identity()
        {
            ensure();
            if (row != col) throw std::runtime_error("row != col.");
            for (size_t i = 0; i < row; i++)
                for (size_t j = 0; j < col; j++)
                    matrix[i][j] = (i == j) ? static_cast<dataType>(1) : static_cast<dataType>(0);
        }

        Numcpp<dataType> identity()
        {
            ensure();
            Numcpp<dataType> result(*this);
            result.set_identity();
            return result;
        }

        std::vector<Numcpp<dataType>> eig(int max_iter = 1000, double tolerance = 1e-6) const;

        bool is_vector() const { return (row == 1 || col == 1); }
        size_t size() const { return row * col; }

        dataType norm(NormType type = L2) const;
        dataType dot(const Numcpp<dataType> &other) const;

        template <typename NewType>
        Numcpp<NewType> as() const
        {
            ensure();
            Numcpp<NewType> result(row, col);
            for (size_t i = 0; i < row; i++)
                for (size_t j = 0; j < col; j++)
                    result[i][j] = static_cast<NewType>(matrix[i][j]);
            return result;
        }
    };

    // smul_object for special function-multiply syntax
    template <typename T>
    class smul_object
    {
    public:
        size_t row, col;
        T **matrix;
        T (*function_object)(T A, T B);
        smul_object(const Numcpp<T> &A, T (*func)(T, T))
            : row(A.row), col(A.col), matrix(A.matrix), function_object(func) {}
    };

#if __cplusplus >= 202000L
    template <typename T>
    smul_object<T> operator<(const Numcpp<T> &A, auto function_object)
    {
        return smul_object<T>(A, function_object);
    }
#else
    template <typename T>
    smul_object<T> operator<(const Numcpp<T> &A, T (*function_object)(T, T))
    {
        return smul_object<T>(A, function_object);
    }
#endif

    template <typename T>
    Numcpp<T> operator>(const smul_object<T> &oper, const Numcpp<T> &B)
    {
        if (oper.col != B.row) throw std::invalid_argument("Invalid Matrix.");
        Numcpp<T> result(oper.row * B.col, B.row);
        for (size_t i = 0; i < B.col; i++)
            for (size_t j = 0; j < oper.row; j++)
                for (size_t k = 0; k < oper.col; k++)
                    result.matrix[j + i * oper.row][k] = oper.function_object(oper.matrix[j][k], B.matrix[k][i]);
        return result;
    }

    template <typename T>
    Numcpp<T> operator>(const smul_object<T> &oper, void *)
    {
        Numcpp<T> result(oper.row, oper.col);
        for (size_t i = 0; i < oper.row; i++)
            for (size_t j = 0; j < oper.col; j++)
                result[i][j] = oper.function_object(oper.matrix[i][j], (T)0);
        return result;
    }

    // ==================== Class method definitions ====================

    template <typename T>
    Numcpp<T>::Numcpp(const size_t _row, const size_t _col)
    {
        if (_row == 0 || _col == 0) throw "Invalid creation.";
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
                    if constexpr (is_numcpp_v<T>)
                        matrix[i][j] = T();
                    else
                        matrix[i][j] = static_cast<T>(0);
                }
            }
        }
        else
        {
            if constexpr (is_numcpp_v<T>)
                units::Alloc_thread_worker<T>(matrix, _row, _col, this->maxprocs,
                    [](T **a, size_t i, size_t j) { a[i][j] = T(); });
            else
                units::Alloc_thread_worker<T>(matrix, _row, _col, this->maxprocs,
                    [](T **a, size_t i, size_t j) { a[i][j] = static_cast<T>(0); });
        }
        is_destroy = false;
    }

    template <typename T>
    inline Numcpp<T>::Numcpp(const size_t _row, const size_t _col, T value)
    {
        if (_row == 0 || _col == 0) throw "Invalid creation.";
        row = _row;
        col = _col;
        matrix = new T *[_row];
        if (this->optimization == false)
        {
            for (size_t i = 0; i < _row; i++)
            {
                matrix[i] = new T[_col];
                for (size_t j = 0; j < _col; j++) { matrix[i][j] = value; }
            }
        }
        else
        {
            units::Alloc_thread_worker<T>(matrix, _row, _col, this->maxprocs,
                [value](T **a, size_t i, size_t j) { a[i][j] = value; });
        }
        is_destroy = false;
    }

    template <typename T>
    inline Numcpp<T>::Numcpp(T *mat, const size_t _row, const size_t _col)
    {
        if (_row == 0 || _col == 0) throw "Invalid creation.";
        row = _row;
        col = _col;
        matrix = new T *[_row];
        if (this->optimization == false)
        {
            for (size_t i = 0; i < _row; i++)
            {
                matrix[i] = new T[_col];
                for (size_t j = 0; j < _col; j++) { matrix[i][j] = mat[i * _col + j]; }
            }
        }
        else
        {
            units::Copy_thread_worker<T>(matrix, _row, _col, &mat, this->maxprocs,
                [&](T **a, T **b, size_t i, size_t j) { a[i][j] = (*b)[i * _col + j]; });
        }
        is_destroy = false;
    }

    template <typename T>
    inline Numcpp<T>::Numcpp(T **mat, const size_t _row, const size_t _col)
    {
        if (_row == 0 || _col == 0) throw "Invalid creation.";
        row = _row;
        col = _col;
        matrix = new T *[_row];
        if (this->optimization == false)
        {
            for (size_t i = 0; i < _row; i++)
            {
                matrix[i] = new T[_col];
                for (size_t j = 0; j < _col; j++) { matrix[i][j] = mat[i][j]; }
            }
        }
        else
        {
            units::Copy_thread_worker<T>(matrix, _row, _col, mat, this->maxprocs,
                [](T **a, T **b, size_t i, size_t j) { a[i][j] = b[i][j]; });
        }
        is_destroy = false;
    }

    template <typename T>
    Numcpp<T>::Numcpp(const Numcpp<T> &other)
    {
        if (other.row == 0 || other.col == 0) throw std::invalid_argument("Invalid Matrix.");
        row = other.row;
        col = other.col;
        matrix = new T *[row];
        if (this->optimization == false)
        {
            for (size_t i = 0; i < row; i++)
            {
                matrix[i] = new T[col];
                for (size_t j = 0; j < col; j++) { matrix[i][j] = other.matrix[i][j]; }
            }
        }
        else
        {
            units::Copy_thread_worker<T>(matrix, this->row, this->col, other.matrix, this->maxprocs,
                [](T **a, T **b, size_t i, size_t j) { a[i][j] = b[i][j]; });
        }
        is_destroy = false;
    }

    template <typename T>
    Numcpp<T>::Numcpp(const char *path)
    {
        FILE *fp = fopen(path, "rb");
        if (fp == nullptr) throw std::invalid_argument("Invalid path.");
        fread(&row, sizeof(size_t), 1, fp);
        fread(&col, sizeof(size_t), 1, fp);
        matrix = new T *[row];
        for (size_t i = 0; i < row; i++)
        {
            matrix[i] = new T[col];
            fread(matrix[i], sizeof(T), col, fp);
        }
        fclose(fp);
        is_destroy = false;
    }

    template <typename T>
    Numcpp<T>::~Numcpp()
    {
        if (matrix != nullptr && is_destroy != true)
        {
            for (size_t i = 0; i < this->row; i++)
            {
                if constexpr (is_numcpp_v<T>)
                {
                    for (size_t j = 0; j < this->col; j++) { matrix[i][j].~T(); }
                }
                else
                {
                    delete[] matrix[i];
                }
                matrix[i] = nullptr;
            }
            delete[] matrix;
            matrix = nullptr;
            is_destroy = true;
        }
    }

    template <typename T>
    void Numcpp<T>::operator=(const Numcpp<T> &other)
    {
        if (other.row != this->row || other.col != this->col)
        {
            if (matrix != nullptr)
            {
                for (size_t i = 0; i < this->row; i++)
                {
                    if constexpr (is_numcpp_v<T>)
                    {
                        for (size_t j = 0; j < this->col; j++) { matrix[i][j].~T(); }
                    }
                    else { delete[] matrix[i]; }
                    matrix[i] = nullptr;
                }
                delete[] matrix;
            }
            row = other.row;
            col = other.col;
            matrix = new T *[row];
            if (this->optimization == false)
            {
                for (size_t i = 0; i < row; i++)
                {
                    matrix[i] = new T[col];
                    for (size_t j = 0; j < col; j++) { matrix[i][j] = other.matrix[i][j]; }
                }
            }
            else
            {
                units::Copy_thread_worker<T>(matrix, this->row, this->col, other.matrix, this->maxprocs,
                    [](T **a, T **b, size_t i, size_t j) { a[i][j] = b[i][j]; });
            }
        }
        else
        {
            if (this->optimization == false)
            {
                for (size_t i = 0; i < this->row; i++)
                    for (size_t j = 0; j < this->col; j++)
                        this->matrix[i][j] = other.matrix[i][j];
            }
            else
            {
                units::thread_worker<T>(this->matrix, this->row, this->col, other.matrix, this->maxprocs,
                    [](T **a, T **b, size_t i, size_t j) { a[i][j] = b[i][j]; });
            }
        }
    }

    template <typename T>
    void Numcpp<T>::operator+=(const Numcpp<T> &other)
    {
        ensure();
        if (other.row != this->row || other.col != this->col) throw std::invalid_argument("Invalid Matrix.");
        if (this->optimization == false)
        {
            for (size_t i = 0; i < this->row; i++)
                for (size_t j = 0; j < this->col; j++)
                    this->matrix[i][j] += other.matrix[i][j];
        }
        else
        {
            units::thread_worker<T>(this->matrix, this->row, this->col, other.matrix, this->maxprocs,
                [](T **a, T **b, size_t i, size_t j) { a[i][j] += b[i][j]; });
        }
    }

    template <typename T>
    Numcpp<T> Numcpp<T>::operator+(const Numcpp<T> &other) const
    {
        ensure();
        if (other.row != this->row || other.col != this->col) throw std::invalid_argument("Invalid Matrix.");
        Numcpp<T> result;
        if constexpr (is_numcpp_v<T>)
            result = Numcpp<T>(this->row, this->col, this->matrix[0][0]);
        else
            result = Numcpp<T>(this->row, this->col);
        if (this->optimization == false)
        {
            for (size_t i = 0; i < this->row; i++)
                for (size_t j = 0; j < this->col; j++)
                    result.matrix[i][j] = this->matrix[i][j] + other.matrix[i][j];
        }
        else
        {
            T **temp = other.matrix;
            units::thread_worker<T>(this->matrix, this->row, this->col, temp, result.matrix, this->maxprocs,
                [](T **a, T **b, T **c, size_t i, size_t j) { c[i][j] = a[i][j] + b[i][j]; });
        }
        return result;
    }

    template <typename T>
    void Numcpp<T>::operator-=(const Numcpp<T> &other)
    {
        ensure();
        if (other.row != this->row || other.col != this->col) throw std::invalid_argument("Invalid Matrix.");
        if (this->optimization == false)
        {
            for (size_t i = 0; i < this->row; i++)
                for (size_t j = 0; j < this->col; j++)
                    this->matrix[i][j] -= other.matrix[i][j];
        }
        else
        {
            units::thread_worker<T>(this->matrix, this->row, this->col, other.matrix, this->maxprocs,
                [](T **a, T **b, size_t i, size_t j) { a[i][j] -= b[i][j]; });
        }
    }

    template <typename T>
    Numcpp<T> Numcpp<T>::operator-(const Numcpp<T> &other) const
    {
        ensure();
        if (other.row != this->row || other.col != this->col) throw std::invalid_argument("Invalid Matrix.");
        Numcpp<T> result;
        if constexpr (is_numcpp_v<T>)
            result = Numcpp<T>(this->row, this->col, this->matrix[0][0]);
        else
            result = Numcpp<T>(this->row, this->col);
        if (this->optimization == false)
        {
            for (size_t i = 0; i < this->row; i++)
                for (size_t j = 0; j < this->col; j++)
                    result.matrix[i][j] = this->matrix[i][j] - other.matrix[i][j];
        }
        else
        {
            T **temp = other.matrix;
            units::thread_worker<T>(this->matrix, this->row, this->col, temp, result.matrix, this->maxprocs,
                [](T **a, T **b, T **c, size_t i, size_t j) { c[i][j] = a[i][j] - b[i][j]; });
        }
        return result;
    }

    template <typename T>
    Numcpp<T> Numcpp<T>::operator+(T n) const
    {
        ensure();
        Numcpp<T> result(this->row, this->col, n);
        if (this->optimization == false)
        {
            for (size_t i = 0; i < this->row; i++)
                for (size_t j = 0; j < this->col; j++)
                    result.matrix[i][j] += this->matrix[i][j];
        }
        else
        {
            units::thread_worker<T>(result.matrix, this->row, this->col, this->matrix, this->maxprocs,
                [n](T **a, T **b, size_t i, size_t j) { a[i][j] += b[i][j]; });
        }
        return result;
    }

    template <typename T>
    void Numcpp<T>::operator+=(T n)
    {
        ensure();
        if (this->optimization == false)
        {
            for (size_t i = 0; i < this->row; i++)
                for (size_t j = 0; j < this->col; j++)
                    this->matrix[i][j] += n;
        }
        else
        {
            units::thread_worker<T>(this->matrix, this->row, this->col, this->maxprocs,
                [n](T **a, size_t i, size_t j) { a[i][j] += n; });
        }
    }

    template <typename T>
    Numcpp<T> Numcpp<T>::operator-(T n) const
    {
        ensure();
        Numcpp<T> result(this->matrix, this->row, this->col);
        if (this->optimization == false)
        {
            for (size_t i = 0; i < this->row; i++)
                for (size_t j = 0; j < this->col; j++)
                    result.matrix[i][j] -= n;
        }
        else
        {
            units::thread_worker<T>(result.matrix, this->row, this->col, this->maxprocs,
                [n](T **a, size_t i, size_t j) { a[i][j] -= n; });
        }
        return result;
    }

    template <typename T>
    void Numcpp<T>::operator-=(T n)
    {
        ensure();
        if (this->optimization == false)
        {
            for (size_t i = 0; i < this->row; i++)
                for (size_t j = 0; j < this->col; j++)
                    this->matrix[i][j] -= n;
        }
        else
        {
            units::thread_worker<T>(this->matrix, this->row, this->col, this->maxprocs,
                [n](T **a, size_t i, size_t j) { a[i][j] -= n; });
        }
    }

    template <typename T>
    Numcpp<T> Numcpp<T>::operator*(T n) const
    {
        ensure();
        Numcpp<T> result(this->row, this->col, n);
        if (this->optimization == false)
        {
            for (size_t i = 0; i < this->row; i++)
                for (size_t j = 0; j < this->col; j++)
                    result.matrix[i][j] *= this->matrix[i][j];
        }
        else
        {
            units::thread_worker<T>(result.matrix, this->row, this->col, this->matrix, this->maxprocs,
                [n](T **a, T **b, size_t i, size_t j) { a[i][j] *= b[i][j]; });
        }
        return result;
    }

    template <typename T>
    void Numcpp<T>::operator*=(T n)
    {
        ensure();
        if (this->optimization == false)
        {
            for (size_t i = 0; i < this->row; i++)
                for (size_t j = 0; j < this->col; j++)
                    this->matrix[i][j] *= n;
        }
        else
        {
            units::thread_worker<T>(this->matrix, this->row, this->col, this->maxprocs,
                [n](T **a, size_t i, size_t j) { a[i][j] *= n; });
        }
    }

    template <typename T>
    Numcpp<T> Numcpp<T>::operator/(T n) const
    {
        ensure();
        assert(n != 0);
        Numcpp<T> result(this->row, this->col);
        if (this->optimization == false)
        {
            for (size_t i = 0; i < this->row; i++)
                for (size_t j = 0; j < this->col; j++)
                    result.matrix[i][j] = this->matrix[i][j] / n;
        }
        else
        {
            units::thread_worker<T>(result.matrix, this->row, this->col, this->matrix, this->maxprocs,
                [n](T **a, T **b, size_t i, size_t j) { a[i][j] = b[i][j] / n; });
        }
        return result;
    }

    template <typename T>
    void Numcpp<T>::operator/=(T n)
    {
        ensure();
        assert(n != 0);
        if (this->optimization == false)
        {
            for (size_t i = 0; i < this->row; i++)
                for (size_t j = 0; j < this->col; j++)
                    this->matrix[i][j] /= n;
        }
        else
        {
            units::thread_worker<T>(this->matrix, this->row, this->col, this->maxprocs,
                [n](T **a, size_t i, size_t j) { a[i][j] /= n; });
        }
    }

    template <typename T>
    Numcpp<T> Numcpp<T>::operator*(const Numcpp<T> &other) const
    {
        ensure();
        if (this->col != other.row) throw std::invalid_argument("Invalid Matrix.");
        Numcpp<T> result;
        if constexpr (is_numcpp_v<T>)
        {
            size_t n = matrix[0][0].row;
            size_t m = other.matrix[0][0].col;
            using VT = typename T::value_type;
            result = Numcpp<T>(this->row, other.col, Numcpp<VT>(n, m, 0));
        }
        else
        {
            result = Numcpp<T>(this->row, other.col, static_cast<T>(0));
        }
        if (this->optimization == true)
        {
            units::mm_auto(this->matrix, other.matrix, result.matrix,
                           this->row, other.row, this->col, other.col, true);
        }
        else
        {
            units::mm_generate(this->matrix, other.matrix, result.matrix,
                               this->row, other.row, this->col, other.col, 0, 0, 0, 0);
        }
        return result;
    }

    template <typename T>
    void Numcpp<T>::transposed()
    {
        ensure();
        size_t x = this->col, y = this->row;
        T **temp = new T *[x];
        if (this->optimization == false)
        {
            for (size_t i = 0; i < x; i++)
            {
                temp[i] = new T[y];
                for (size_t j = 0; j < y; j++) { temp[i][j] = this->matrix[j][i]; }
            }
        }
        else
        {
            for (size_t i = 0; i < x; i++) { temp[i] = new T[y]; }
            units::thread_worker<T>(this->matrix, this->row, this->col, temp, this->maxprocs,
                [](T **a, T **b, size_t i, size_t j) { b[j][i] = a[i][j]; });
        }
        for (size_t i = 0; i < this->row; i++)
        {
            if constexpr (is_numcpp_v<T>)
            {
                for (size_t j = 0; j < this->col; j++) { matrix[i][j].~T(); }
            }
            else { delete[] matrix[i]; }
            matrix[i] = nullptr;
        }
        delete[] matrix;
        this->matrix = temp;
        this->col = y;
        this->row = x;
    }

    template <typename T>
    Numcpp<T> Numcpp<T>::transpose() const
    {
        ensure();
        Numcpp<T> result(this->col, this->row);
        if (this->optimization == false)
        {
            for (size_t i = 0; i < this->row; i++)
                for (size_t j = 0; j < this->col; j++)
                    result.matrix[j][i] = this->matrix[i][j];
        }
        else
        {
            units::thread_worker<T>(this->matrix, this->row, this->col, result.matrix, this->maxprocs,
                [](T **a, T **b, size_t i, size_t j) { b[j][i] = a[i][j]; });
        }
        return result;
    }

    template <typename T>
    void Numcpp<T>::Hadamard_self(const Numcpp<T> &other)
    {
        ensure();
        if (other.row != this->row || other.col != this->col) throw std::invalid_argument("Invalid Matrix.");
        if (this->optimization == false)
        {
            for (size_t i = 0; i < this->row; i++)
                for (size_t j = 0; j < this->col; j++)
                    this->matrix[i][j] *= other.matrix[i][j];
        }
        else
        {
            units::thread_worker<T>(this->matrix, this->row, this->col, other.matrix, this->maxprocs,
                [](T **a, T **b, size_t i, size_t j) { a[i][j] *= b[i][j]; });
        }
    }

    template <typename T>
    Numcpp<T> Numcpp<T>::Hadamard(const Numcpp<T> &other) const
    {
        ensure();
        if (other.row != this->row || other.col != this->col) throw std::invalid_argument("Invalid Matrix.");
        Numcpp<T> result(other);
        result.Hadamard_self(*this);
        return result;
    }

    template <typename T>
    T Numcpp<T>::sum() const
    {
        ensure();
        T sum_value;
        if constexpr (is_numcpp_v<T>)
            sum_value = T(matrix[0][0].row, matrix[0][0].col, static_cast<typename T::value_type>(0));
        else
            sum_value = static_cast<T>(0);
        if (this->optimization == false)
        {
            for (size_t i = 0; i < this->row; i++)
                for (size_t j = 0; j < this->col; j++)
                    sum_value += this->matrix[i][j];
        }
        else
        {
            T *p = &sum_value;
            units::thread_worker<T>(this->matrix, this->row, this->col, this->maxprocs,
                [p](T **a, size_t i, size_t j) { (*p) += a[i][j]; });
        }
        return sum_value;
    }

    template <typename T>
    void Numcpp<T>::save(const char *path)
    {
        ensure();
        FILE *fp = fopen(path, "wb");
        if (fp == nullptr) throw std::invalid_argument("Invalid path.");
        fwrite(&row, sizeof(size_t), 1, fp);
        fwrite(&col, sizeof(size_t), 1, fp);
        for (size_t i = 0; i < row; i++) { fwrite(matrix[i], sizeof(T), col, fp); }
        fclose(fp);
    }

    template <typename T>
    std::ostream &operator<<(std::ostream &stream, const Numcpp<T> &m)
    {
        m.ensure();
        if constexpr (is_numcpp_v<T>)
        {
            stream << '(' << m.row << ',' << m.col << "){\n";
            for (size_t i = 0; i < m.row; ++i)
                for (size_t j = 0; j < m.col; ++j)
                    stream << "    [" << i << "][" << j << ']' << m.matrix[i][j] << (j == m.col - 1 ? "}\n" : " \n");
            stream << "}\n";
        }
        else
        {
            stream << '(' << m.row << ',' << m.col << ")[\n";
            for (size_t i = 0; i < m.row; ++i)
            {
                stream << "    [" << i << "][";
                for (size_t j = 0; j < m.col; ++j)
                    stream << m.matrix[i][j] << (j == m.col - 1 ? "]\n" : " , ");
            }
            stream << ']';
        }
        return stream;
    }

    // ==================== FFT ====================
    template <typename T>
    Numcpp<fft_result_t<T>> Numcpp<T>::fft(int inv) const
    {
        ensure();
        using Cplx = fft_result_t<T>;
        using VT = typename Cplx::value_type;

        Numcpp<Cplx> result(this->row, this->col);
        if constexpr (is_complex_v<T>)
        {
            for (size_t i = 0; i < this->row; i++)
                units::fft<VT>(&(this->matrix[i]), this->col, &(result.matrix[i]), inv);
        }
        else
        {
            Numcpp<Cplx> temp(row, col);
            for (size_t i = 0; i < row; i++)
                for (size_t j = 0; j < col; j++)
                    temp.matrix[i][j] = Cplx(matrix[i][j]);
            for (size_t i = 0; i < this->row; i++)
                units::fft<VT>(&(temp.matrix[i]), this->col, &(temp.matrix[i]), inv);
            if (inv < 0) temp *= Cplx(static_cast<VT>(1.0 / this->col), 0);
            return temp;
        }
        if (inv < 0) result *= Cplx(static_cast<VT>(1.0 / this->col), 0);
        return result;
    }

    template <typename T>
    void Numcpp<T>::ffted(int inv)
    {
        ensure();
        if constexpr (is_complex_v<T>)
        {
            using VT = typename T::value_type;
            for (size_t i = 0; i < this->row; i++)
                units::fft<VT>(&(this->matrix[i]), this->col, &(this->matrix[i]), inv);
            if (inv < 0) *this *= T(1.0 / this->col, 0);
        }
        else
        {
            throw std::invalid_argument("In-place FFT requires complex type. Use fft() instead.");
        }
    }

    // ==================== Determinant ====================
    template <typename T>
    T det_cal(T **det, size_t n)
    {
        T detVal;
        if constexpr (is_numcpp_v<T>)
            detVal = T(det[0][0].row, det[0][0].col, static_cast<typename T::value_type>(0));
        else
            detVal = static_cast<T>(0);
        if (n == 1) return det[0][0];
        T **tempdet = new T *[n - 1];
        for (size_t i = 0; i < n - 1; i++) tempdet[i] = new T[n - 1];
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < n - 1; j++)
                for (size_t k = 0; k < n - 1; k++)
                    tempdet[j][k] = (k < i) ? det[j + 1][k] : det[j + 1][k + 1];
            detVal += det[0][i] * std::pow(-1.0, static_cast<int>(i)) * det_cal(tempdet, n - 1);
        }
        for (size_t i = 0; i < n - 1; i++)
        {
            if constexpr (is_numcpp_v<T>)
            {
                for (size_t j = 0; j < n - 1; j++) tempdet[i][j].~T();
            }
            else
                delete[] tempdet[i];
        }
        delete[] tempdet;
        return detVal;
    }

    template <typename T>
    T Numcpp<T>::determinant() const
    {
        ensure();
        if (row != col) throw std::invalid_argument("Matrix must be square to compute determinant.");
        return det_cal(matrix, row);
    }

    // ==================== Inverse ====================
    template <typename T>
    Numcpp<T> Numcpp<T>::inverse() const
    {
        ensure();
        if (row != col)
            throw std::invalid_argument("Standard inverse is only defined for square matrices. Use pseudoinverse().");
        T det = determinant();
        if (std::abs(det) < 1e-10)
            throw std::invalid_argument("Matrix is singular, cannot compute inverse.");
        Numcpp<T> result(row, col);
        if (row == 1)
        {
            result.matrix[0][0] = T(1) / matrix[0][0];
        }
        else
        {
            Numcpp<T> adjugate(row, col);
            for (size_t i = 0; i < row; i++)
            {
                for (size_t j = 0; j < col; j++)
                {
                    Numcpp<T> minor(row - 1, col - 1);
                    size_t minor_i = 0;
                    for (size_t ii = 0; ii < row; ii++)
                    {
                        if (ii == i) continue;
                        size_t minor_j = 0;
                        for (size_t jj = 0; jj < col; jj++)
                        {
                            if (jj == j) continue;
                            minor.matrix[minor_i][minor_j] = matrix[ii][jj];
                            minor_j++;
                        }
                        minor_i++;
                    }
                    T sign = ((i + j) % 2 == 0) ? T(1) : T(-1);
                    adjugate.matrix[j][i] = sign * minor.determinant();
                }
            }
            result = adjugate * (T(1) / det);
        }
        return result;
    }

    // ==================== Pseudoinverse ====================
    template <typename T>
    Numcpp<T> Numcpp<T>::pseudoinverse() const
    {
        ensure();
        if (row == 0 || col == 0) throw std::invalid_argument("Matrix is empty.");
        Numcpp<T> ATA = this->transpose() * (*this);
        auto eig_result = ATA.eig();
        Numcpp<T> V = eig_result[1];
        auto S_values = (eig_result[0])<mklamb(T, { return std::sqrt(x); })> nullptr;
        size_t min_dim = std::min(row, col);
        Numcpp<T> Sigma(row, col, T(0));
        for (size_t i = 0; i < min_dim; i++) Sigma[i][i] = S_values[0][i];
        Numcpp<T> Sigma_plus(col, row, T(0));
        T tolerance = static_cast<T>(1e-6);
        for (size_t i = 0; i < min_dim; i++)
            if (std::abs(Sigma[i][i]) > tolerance) Sigma_plus[i][i] = T(1) / Sigma[i][i];
        Numcpp<T> this_copy(*this);
        Numcpp<T> U = this_copy * V * Sigma_plus;
        Numcpp<T> A_plus = V * Sigma_plus * U.transpose();
        return A_plus;
    }

    // ==================== SVD ====================
    template <typename T>
    void Numcpp<T>::svd(Numcpp<T> &U, Numcpp<T> &S, Numcpp<T> &V) const
    {
        ensure();
        Numcpp<T> AT(*this);
        AT.transposed();
        Numcpp<T> ATA = (*this) * AT;
        auto result = ATA.eig();
        U = result[1];
        Numcpp<T> Sv = (result[0])<mklamb(T, { return std::sqrt(x); })> nullptr;
        S = Numcpp<T>(row, col, T(0));
        size_t mindim = std::min(row, col);
        for (size_t i = 0; i < mindim; i++) S[i][i] = Sv[0][i];
        Numcpp<T> S_inv(col, row, T(0));
        for (size_t i = 0; i < mindim; i++)
            if (S[i][i] != T(0)) S_inv[i][i] = T(1) / S[i][i];
        V = AT * U * S_inv.transpose();
    }

    template <typename T>
    std::vector<Numcpp<T>> Numcpp<T>::svd() const
    {
        ensure();
        Numcpp<T> AT(*this);
        AT.transposed();
        Numcpp<T> ATA = (*this) * AT;
        auto result = ATA.eig();
        Numcpp<T> U = result[1];
        Numcpp<T> Sv = (result[0])<mklamb(T, { return std::sqrt(x); })> nullptr;
        Numcpp<T> S = Numcpp<T>(row, col, T(0));
        size_t mindim = std::min(row, col);
        for (size_t i = 0; i < mindim; i++) S[i][i] = Sv[0][i];
        Numcpp<T> S_inv(col, row, T(0));
        for (size_t i = 0; i < mindim; i++)
            if (S[i][i] != T(0)) S_inv[i][i] = T(1) / S[i][i];
        Numcpp<T> V = AT * U * S_inv.transpose();
        return {U, S, V};
    }

    // ==================== Eig ====================
    template <typename T>
    std::vector<Numcpp<T>> Numcpp<T>::eig(int max_iter, double tolerance) const
    {
        ensure();
        if (!is_symmetric(tolerance))
            throw std::invalid_argument("eig only supported for symmetric matrices.");
        size_t n = row;
        Numcpp<T> A(*this);
        Numcpp<T> Q_total(n, n);
        Q_total.set_identity();
        for (int iter = 0; iter < max_iter; iter++)
        {
            Numcpp<T> Q(n, n);
            Numcpp<T> R(n, n);
            units::qr_decomposition_gm<T>(A.matrix, n, Q.matrix, R.matrix);
            A = R * Q;
            Q_total = Q_total * Q;
            bool is_diag = true;
            for (size_t i = 0; i < n; i++)
            {
                for (size_t j = 0; j < i; j++)
                    if (std::abs(A.matrix[i][j]) > tolerance) { is_diag = false; break; }
                if (!is_diag) break;
            }
            if (is_diag) break;
        }
        Numcpp<T> eigenvalues(1, n);
        for (size_t i = 0; i < n; i++) eigenvalues.matrix[0][i] = A.matrix[i][i];
        std::sort(eigenvalues[0], eigenvalues[0] + eigenvalues.col, std::greater<double>());
        A.zero_approximation();
        for (size_t i = 0; i < n; i++) A.matrix[i][i] = eigenvalues.matrix[0][i];
        return {eigenvalues, Q_total, A};
    }

    // ==================== Norm ====================
    template <typename T>
    T Numcpp<T>::norm(NormType type) const
    {
        ensure();
        T result = T(0);
        switch (type)
        {
        case L1:
            if (is_vector()) { result = sum(); }
            else
            {
                T max_col_sum = T(0);
                for (size_t j = 0; j < col; ++j)
                {
                    T col_sum = T(0);
                    for (size_t i = 0; i < row; ++i) col_sum += std::abs(matrix[i][j]);
                    max_col_sum = std::max(max_col_sum, col_sum);
                }
                result = max_col_sum;
            }
            break;
        case L2:
        {
            T sum_sq = T(0);
            for (size_t i = 0; i < row; ++i)
                for (size_t j = 0; j < col; ++j)
                    sum_sq += matrix[i][j] * matrix[i][j];
            result = std::sqrt(sum_sq);
            break;
        }
        case INF:
            if (is_vector())
            {
                result = matrix[0][0];
                if (row == 1)
                    for (size_t i = 0; i < col; i++) result = std::max(result, matrix[0][i]);
                else
                    for (size_t i = 0; i < row; i++) result = std::max(result, matrix[i][0]);
            }
            else
            {
                auto temp = ((*this) * Numcpp<T>(col, 1))<mklamb(T, { return std::abs(x); })> nullptr;
                for (size_t i = 0; i < row; i++) result = std::max(result, temp[i][0]);
            }
            break;
        default:
            throw std::invalid_argument("Unsupported norm type.");
        }
        return result;
    }

    // ==================== Dot ====================
    template <typename T>
    T Numcpp<T>::dot(const Numcpp<T> &other) const
    {
        ensure();
        if (!(this->is_vector() && other.is_vector()))
            throw std::invalid_argument("Dot requires two vectors.");
        if (this->row * this->col != other.row * other.col)
            throw std::invalid_argument("Two vectors must be in the same dimension.");
        if (row == other.row)
        {
            T result = T(0);
            if (row == 1)
                for (size_t i = 0; i < col; i++) result += other.matrix[0][i] * matrix[0][i];
            else
                for (size_t i = 0; i < row; i++) result += other.matrix[i][0] * matrix[i][0];
            return result;
        }
        else
        {
            auto temp = (*this) * other;
            return temp[0][0];
        }
    }

    // ==================== Free functions ====================

    template <typename T>
    static Numcpp<T> load(const char *path)
    {
        FILE *fp = fopen(path, "rb");
        if (fp == nullptr) throw std::invalid_argument("Invalid path.");
        size_t r, c;
        fread(&r, sizeof(size_t), 1, fp);
        fread(&c, sizeof(size_t), 1, fp);
        Numcpp<T> result(r, c);
        for (size_t i = 0; i < r; i++) fread(result.matrix[i], sizeof(T), c, fp);
        fclose(fp);
        return result;
    }

    template <typename T>
    Numcpp<T> binarizeMatrix(const Numcpp<T> &mat, T threshold)
    {
        Numcpp<T> result(mat.row, mat.col);
        for (size_t i = 0; i < mat.row; i++)
            for (size_t j = 0; j < mat.col; j++)
                result[i][j] = (mat[i][j] >= threshold) ? T(1) : T(0);
        return result;
    }

#define MATtoNumcpp(mat_name, Numcpp, row, col) \
    for (size_t i = 0; i < row; i++)            \
        for (size_t j = 0; j < col; j++)        \
            Numcpp[i][j] = mat_name[i][j];

} // namespace np

#endif // NUMCPP_CORE_HPP
