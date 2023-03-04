#include <iostream>
typedef unsigned long long size_t;
template <typename dataType>
class Numcpp
{
private:
    size_t row, col;
    dataType **matrix;

public:
    Numcpp(const size_t _row, const size_t _col);
    Numcpp(const Numcpp<dataType> &other);
    /*
    operators
    */
    void operator=(const Numcpp<dataType> &other)
    {
        if (other.row != this->row || other.col != this->col)
        {
            throw "Invalid Matrix";
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
            throw "Invalid Matrix";
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
            throw "Invalid Matrix";
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
            throw "Invalid Matrix";
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
            throw "Invalid Matrix";
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
            throw "Invalid Matrix";
        }
        else
        {
            Numcpp<dataType> result(this->row, other.col);
            for (size_t i = 0; i < this->row; i++)
            {
                for (size_t k = 0; k < other.col; k++)
                {
                    for (size_t j = 0; j < this->col; j++)
                    {
                        result.matrix[i][j] = this->matrix[i][j] * other.matrix[j][k];
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
        for (int i = 0; i < m.row; ++i)
        {
            for (int j = 0; j < m.col; ++j)
                stream << m.matrix[i][j] << (j == m.col - 1 ? '\n' : ' ');
        }
        return stream;
    }
};

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
Numcpp<T>::Numcpp(const Numcpp<T> &other)
{
    if (other.row == 0 || other.col == 0)
    {
        throw "Invalid Matrix";
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
        throw "Invalid Matrix";
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
        throw "Invalid Matrix";
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
