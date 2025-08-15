#include <iostream>
#include "Numcpp.hpp"

using namespace np;
#define nc_t double
int main()
{
    np::is_optimized = true;
    Numcpp<nc_t> n(16, 16);
    Numcpp<nc_t> m(16, 16);
    std::cout << n;
    std::cout << m;
    // 上传到GPU
    n.to(DEVICE_CUDA);
    m.to(DEVICE_CUDA);

    // 在GPU上操作
    n *= 2.0; // 广播操作
    m *= 3.0;

    // 同步回CPU查看结果
    n.to(DEVICE_LOCAL);
    m.to(DEVICE_LOCAL);
    std::cout << n;
    std::cout << m;

    // GPU加速的矩阵乘法（无优化算法）
    Numcpp<nc_t> result = n * m;
    std::cout << result;

    // 同位加法
    result = n + m;
    std::cout << result;
    // 本位加法
    n += m;
    std::cout << n;
    return 0;
}