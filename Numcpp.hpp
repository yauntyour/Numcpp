#include <iostream>
#include <stdlib.h>
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

    template <typename T>
    friend std::ostream &operator<<(std::ostream &stream, const Numcpp<T> &m)
    {
        stream << '(' << m.row << ',' << m.col << ')' << "[\n";
        for (int i = 0; i < m.row; ++i)
        {
            stream << "    [";
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
                    result.matrix[i * (B.col - 1) + j][k] = oper.function_object((oper.matrix)[j][k], B.matrix[k][i]);
                }
            }
        }
        return result;
    }
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
