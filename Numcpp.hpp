#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <complex>
#include <math.h>
#include <thread>
#include <vector>

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
            if (b > i) // 避免重复交换
            {
                std::complex<T> temp;
                temp = (*x)[i];
                (*x)[i] = (*x)[b];
                (*x)[b] = temp;
            }
        }
    }
    template <typename T>
    void fft(std::complex<T> **x, size_t size, std::complex<T> **X, size_t inv)
    {
        std::complex<T> *Wn = new std::complex<T>[size]; // 这里可以自己新建长度为size的数组
        for (size_t i = 0; i < size; i++)
        {
            (*X)[i] = (*x)[i];
            long double real = cos(-2 * M_PI * i / size);
            long double img = inv * sin(-2 * M_PI * i / size);
            Wn[i] = std::complex<T>(real, img); // 初始化Wn
        }
        std::complex<T> *p = (*X);
        trans(&p, size); // 位反转置换
        size_t t;
        for (size_t m = 2; m <= size; m *= 2) // 小序列点数
        {
            for (size_t k = 0; k < size; k += m) // 小序列起始下标
            {
                for (size_t j = 0; j < m / 2; j++) // 小序列的DFT计算
                {
                    size_t index1 = k + j;
                    size_t index2 = index1 + m / 2;
                    t = j * size / m; // t是在完整序列中的下标，找到对应的旋转因子
                    std::complex<T> temp1, temp2;
                    temp2 = (*X)[index2] * Wn[t];
                    temp1 = (*X)[index1];
                    (*X)[index1] = temp1 + temp2; // Wn的性质
                    (*X)[index2] = temp1 - temp2;
                }
            }
        }
    }
    /* 添加偏置offset用以支持分块迭代 */
    template <typename T>
    static void mm_generate(T **this_matrix, T **other_matrix, T **result, const size_t A_row, const size_t B_row,
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
    static T **mat_create(size_t row, size_t col)
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
    static void mat_delete(T **mat, size_t row)
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
    static void mm_Coppersmith_Winograd(T **this_matrix, T **other_matrix, T **result, const size_t A_row, const size_t B_row,
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
    static void mm_auto(T **this_matrix, T **other_matrix, T **result, const size_t A_row, const size_t B_row,
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
}; // namespace units

template <typename dataType>
class Numcpp
{
private:
    bool optimization = false;
    size_t maxprocs = 1;

public:
    dataType **matrix;
    size_t row, col;
    Numcpp(const size_t _row, const size_t _col);
    Numcpp(const size_t _row, const size_t _col, dataType value);
    Numcpp(const Numcpp<dataType> &other);
    Numcpp(const dataType **mat, const size_t _row, const size_t _col);
    /*
    operators
    */
    void operator=(const Numcpp<dataType> &other)
    {
        if (other.row != this->row || other.col != this->col)
        {
            throw std::invalid_argument("Invalid Matrix");
        }
        else
        {
            for (size_t i = 0; i < this->row; i++)
            {
                for (size_t j = 0; j < this->col; j++)
                {
                    this->matrix[i][j] = other.matrix[i][j];
                }
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
            for (size_t i = 0; i < this->row; i++)
            {
                for (size_t j = 0; j < this->col; j++)
                {
                    this->matrix[i][j] += other.matrix[i][j];
                }
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
            for (size_t i = 0; i < this->row; i++)
            {
                for (size_t j = 0; j < this->col; j++)
                {
                    result.matrix[i][j] = this->matrix[i][j] + other.matrix[i][j];
                }
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
            for (size_t i = 0; i < this->row; i++)
            {
                for (size_t j = 0; j < this->col; j++)
                {
                    this->matrix[i][j] -= other.matrix[i][j];
                }
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
            for (size_t i = 0; i < this->row; i++)
            {
                for (size_t j = 0; j < this->col; j++)
                {
                    result.matrix[i][j] = this->matrix[i][j] - other.matrix[i][j];
                }
            }
            return result;
        }
    }
    Numcpp<dataType> operator*(dataType n)
    {
        Numcpp<dataType> result(this->row, this->col);
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                result.matrix[i][j] = this->matrix[i][j] * n;
            }
        }
        return result;
    }
    void operator*=(dataType n)
    {
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                this->matrix[i][j] *= n;
            }
        }
    }
    Numcpp<dataType> operator/(dataType n)
    {
        Numcpp<dataType> result(this->row, this->col);
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                result.matrix[i][j] = this->matrix[i][j] / n;
            }
        }
        return result;
    }
    void operator/=(dataType n)
    {
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                this->matrix[i][j] /= n;
            }
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
            if (this->optimization = true)
            {
                units::mm_auto(this->matrix, other.matrix, result.matrix, this->row, other.row, this->col, other.col, true);
            }
            else
            {
                units::mm_generate(this->matrix, other.matrix, result.matrix, this->row, other.row, this->col, other.row, 0, 0, 0, 0);
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
    void optimized(bool b);
    //
    ~Numcpp();
    void ffted(size_t inv)
    {
        dataType **result = new dataType *[this->row];
        for (size_t i = 0; i < this->row; i++)
        {
            result[i] = new dataType[this->col];
        }

        for (size_t i = 0; i < this->row; i++)
        {
            units::fft(this->matrix + i, this->col, result + i, inv);
        }
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                if (inv < 0)
                {
                    this->matrix[i][j] = result[i][j];
                    this->matrix[i][j] /= this->col;
                }
                else
                {
                    this->matrix[i][j] = result[i][j];
                }
            }
        }
    }
    Numcpp fft(size_t inv)
    {
        Numcpp<dataType> result(this->row, this->col);
        for (size_t i = 0; i < this->row; i++)
        {
            units::fft(this->matrix + i, this->col, result.matrix + i, inv);
        }
        if (inv < 0)
        {
            result *= (1 / this->col);
        }
        return result;
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
};
// matrix special operate
template <typename T>
class oper_object
{
public:
    size_t row, col;
    T **matrix;
    T(*function_object)
    (T A, T B);
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
        for (size_t i = 0; i < _row; i++)
        {
            matrix[i] = new T[_col];
            for (size_t j = 0; j < _col; j++)
            {
                matrix[i][j] = (T)1;
            }
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
        for (size_t i = 0; i < _row; i++)
        {
            matrix[i] = new T[_col];
            for (size_t j = 0; j < _col; j++)
            {
                matrix[i][j] = value;
            }
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
        for (size_t i = 0; i < _row; i++)
        {
            matrix[i] = new T[_col];
            for (size_t j = 0; j < _col; j++)
            {
                matrix[i][j] = mat[i][j];
            }
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
        for (size_t i = 0; i < row; i++)
        {
            matrix[i] = new T[col];
            for (size_t j = 0; j < col; j++)
            {
                matrix[i][j] = other.matrix[i][j];
            }
        }
    }
}
template <typename T>
Numcpp<T>::~Numcpp()
{
    for (size_t i = 0; i < this->row; i++)
    {
        delete matrix[i];
    }
    delete[] matrix;
}
template <typename T>
Numcpp<T> Numcpp<T>::transpose()
{
    Numcpp<T> result(this->col, this->row);
    for (size_t i = 0; i < this->col; i++)
    {
        for (size_t j = 0; j < this->row; j++)
        {
            result[i][j] = matrix[j][i];
        }
    }
    return result;
}
template <typename T>
void Numcpp<T>::transposed()
{
    size_t x = this->col;
    size_t y = this->row;
    T **temp = new T *[x];

    for (size_t i = 0; i < x; i++)
    {
        temp[i] = new T[y];
        for (size_t j = 0; j < y; j++)
        {
            temp[i][j] = this->matrix[j][i];
        }
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
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                this->matrix[i][j] *= other.matrix[i][j];
            }
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
        Numcpp<T> result(this->row, this->col);
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                result.matrix[i][j] = this->matrix[i][j] * other.matrix[i][j];
            }
        }
        return result;
    }
}
template <typename T>
void Numcpp<T>::optimized(bool b)
{
    this->optimization = b;
    return;
}
