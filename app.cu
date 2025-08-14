#include <iostream>
#include "Numcpp.hpp"

using namespace np;
#define nc_t double
#define M_PI 3.1415926535
int main()
{
    Numcpp<nc_t> n(16, 16);
    n.to(DEVICE_CUDA);

    // 在GPU上操作
    n *= 2.0; // 广播操作

    // 同步回CPU查看结果
    n.to(DEVICE_LOCAL);
    std::cout << n;

    return 0;
}