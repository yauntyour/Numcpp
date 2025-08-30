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
    n += 4;
    m -= 6;
    n /= 8;
    m /= 12;

    // 同步回CPU查看结果
    n.to(DEVICE_LOCAL);
    m.to(DEVICE_LOCAL);
    std::cout << n;
    std::cout << m;

    // GPU加速的矩阵乘法（无优化算法）
    Numcpp<nc_t> result = n * m;
    result.to(DEVICE_LOCAL);
    std::cout << result;

    // 同位加法
    result = n + m;
    result.to(DEVICE_LOCAL);
    std::cout << result;

    // 本位减法
    n -= m;
    n.to(DEVICE_LOCAL);
    std::cout << n;

    // 同位广播 & 开启自动同步
    n.auto_sync = true;
    std::cout << (n - 10) / 8.0 * 5 + 3 - 2 * 4 / 2 + 1 << std::endl;
    return 0;
}