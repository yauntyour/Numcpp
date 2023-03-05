# **基于原生C++的通用矩阵——Numcpp**

原生的C++矩阵类封装，基本使用：

```c++
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
```

# **矩阵的运算**

### 下面是支持的操纵符的类型：

```c++
Numcpp<typename> nc(4, 4),matrix(4,4);
```

### 赋值运算符：

```c++
//矩阵简单运算的赋值运算
nc += matrix;
nc -= matrix;
nc = matrix;
//矩阵数乘广播的赋值运算
nc *= number//number为一个数值
```

### 矩阵的基本运算

```c++
Numcpp<typename> result = nc + matrix;
Numcpp<typename> result = nc - matrix;

Numcpp<typename> result = nc * matrix;//数乘
Numcpp<typename> result = nc * matrix;//矩阵乘法
```

### 解矩阵运算（下标访问）

```c++
typename data = nc[x][y];
```

### 矩阵的哈达马乘积（Hadamard product）

```c++
//哈达马乘积的基本运算
Numcpp<typename> result = nc.Hadamard(matrix);
//哈达马乘积的赋值运算
nc.Hadamard_self(matrix);
```

### 矩阵的置转运算

```c++
//获取矩阵的转置
Numcpp<typename> result = nc.transpose();
//转置矩阵
nc.transposed();
```

# 相关信息

作者：[yauntyour]([yauntyour (yauntyour) · GitHub](https://github.com/yauntyour/))

授权协议：MIT开源协议

参考：[数学基础 - 矩阵的基本运算（Matrix Operations）_沙沙的兔子的博客-CSDN博客_矩阵运算](https://blog.csdn.net/darkrabbit/article/details/80025935)
