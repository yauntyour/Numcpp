#include <iostream>
#include "Numcpp.hpp"

using namespace np;
#define nc_t float
int main()
{
    np::is_optimized = true;
    Numcpp<nc_t> n(16, 16);
    Numcpp<nc_t> m(16, 16);
    std::cout << "n:" << n;
    std::cout << "m:" << m;
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
    std::cout << "n:" << n;
    std::cout << "m:" << m;

    // GPU加速的矩阵乘法（无优化算法）
    Numcpp<nc_t> result = n * m;
    result.to(DEVICE_LOCAL);
    std::cout << "n * m:" << result;
    // 注意：GPU运算产生的结果无法拷贝（Lock），因为不同的Numcpp对象无法持有同一个GPU内存

    // 本位减法
    n += m;
    n.to(DEVICE_LOCAL);
    std::cout << "n -= m" << n;

    std::cout << "L1 norm:" << n.norm(L1) << std::endl;
    std::cout << "L2 norm:" << n.norm() << std::endl;
    std::cout << "inf norm:" << n.norm(INF) << std::endl;

    return 0;
}