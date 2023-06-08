#include <iostream>
#include <stdlib.h>
#include <complex>
#include <math.h>
namespace units
{
    template <typename T>
    void trans(std::complex<T> **x, size_t size_x)
    {
        int p = 0;
        int a, b;
        for (int i = 1; i < size_x; i *= 2)
        {
            p++; // 计算二进制位数
        }
        for (int i = 0; i < size_x; i++)
        {
            a = i;
            b = 0;
            for (int j = 0; j < p; j++)
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
    void fft(std::complex<T> **x, size_t size, std::complex<T> **X, int inv)
    {
        std::complex<T> *Wn = new std::complex<T>[size]; // 这里可以自己新建长度为size的数组
        for (int i = 0; i < size; i++)
        {
            (*X)[i] = (*x)[i];
            long double real = cos(-2 * M_PI * i / size);
            long double img = inv * sin(-2 * M_PI * i / size);
            Wn[i] = std::complex<T>(real, img); // 初始化Wn
        }
        std::complex<T> *p = (*X);
        trans(&p, size); // 位反转置换
        int t;
        for (int m = 2; m <= size; m *= 2) // 小序列点数
        {
            for (int k = 0; k < size; k += m) // 小序列起始下标
            {
                for (int j = 0; j < m / 2; j++) // 小序列的DFT计算
                {
                    int index1 = k + j;
                    int index2 = index1 + m / 2;
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
}; // namespace units
template <typename dataType>
class Numcpp
{
public:
    dataType **matrix;
    size_t row, col;
    Numcpp(const size_t _row, const size_t _col);
    Numcpp(const size_t _row, const size_t _col, dataType value);
    Numcpp(const Numcpp<dataType> &other);
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
            return 0;
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
            for (size_t i = 0; i < this->row; i++)
            {
                for (size_t j = 0; j < other.col; j++)
                {
                    for (size_t k = 0; k < this->col; k++)
                    {
                        result[i][j] += (this->matrix[i][k] * other.matrix[k][j]);
                    }
                }
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
    ~Numcpp();
    void ffted(int inv)
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
    Numcpp fft(int inv)
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
        for (int i = 0; i < m.row; ++i)
        {
            stream << "    [" << i << "][";
            for (int j = 0; j < m.col; ++j)
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
                    result.matrix[j + i * B.row][k] = oper.function_object((oper.matrix)[j][k], B.matrix[k][i]);
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