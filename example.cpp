#include <iostream>
#include "Numcpp.hpp"

int main(int argc, char const *argv[])
{
    /*使用Numcpp<type> np(row,col)创建一个row * col的矩阵*/
    Numcpp<int> nc(4, 4), m(4, 4);
    /*矩阵中所有元素的默认值为1*/
    nc *= 2;
    m *= 3;
    try
    {
        // 矩阵乘法：
        Numcpp<int> result = nc * m;
        std::cout << result << std::endl;
        // 矩阵运算：
        result = nc + m;
        std::cout << result << std::endl;
        // 哈达马乘积：
        result = nc.Hadamard(m);
        std::cout << result << std::endl;
        // 矩阵转置：
        result = nc.transpose();
        std::cout << result << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
    return 0;
}