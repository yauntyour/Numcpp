#include <iostream>
#include <math.h>

#include "Numcpp.hpp"

// 理论上模板要被同一个类型实例化
#define nc_t long long

void func(Numcpp<nc_t> &nc)
{
    for (size_t i = 0; i < nc.row; i++)
    {
        for (size_t j = 0; j < nc.col; j++)
        {
            double U1 = rand() * 1.0f / RAND_MAX;               // 0~1均匀分布
            double U2 = rand() * 1.0f / RAND_MAX;               // 0~1均匀分布
            double Z = sqrt(-2 * log(U1)) * cos(2 * M_PI * U2); // 标准正态分布
            // 期望为1，方差为3^2的正态分布
            nc[i][j] *= 1 + 3 * Z;
        }
    }
}

int main(int argc, char const *argv[])
{
    /*使用Numcpp<type> np(row,col)创建一个row * col的矩阵*/
    Numcpp<nc_t> n(4, 4), m(4, 4);
    /*矩阵中所有元素的默认值为1，也可以手动设置*/
    Numcpp<nc_t> c(6, 7, 3.0);
    n *= 2.0;
    m *= 3.0;
    try
    {
        // 矩阵乘法：
        Numcpp<nc_t> result = n * m;
        std::cout << result << "\n";
        // 矩阵运算：
        result = n + m;
        std::cout << result << "\n";
        // 哈达马乘积：
        result = n.Hadamard(m);
        std::cout << result << "\n";

        // 生成正态分布的矩阵
        func(c);
        // 矩阵转置：
        std::cout << c << "\n";
        c.transposed();
        std::cout << c << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
    return 0;
}
