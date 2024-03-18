# **基于原生C++的通用矩阵——Numcpp**

原生的C++矩阵类封装，基本使用：

```c++
#include <iostream>
#include <math.h>
#include <complex>
#include "Numcpp.hpp"
// 复数
typedef std::complex<double> complex_double;

// 理论上模板要被同一个类型实例化
#define nc_t complex_double

void generate(Numcpp<nc_t> &nc)
{
    srand(time(NULL));
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

nc_t func(nc_t n, nc_t m)
{
    nc_t result = n * m;
    return result;
}
nc_t sigmoid(nc_t n, nc_t m)
{
    return nc_t(1, 0) / (nc_t(1, 0) + exp(-n));
}

int main(int argc, char const *argv[])
{
    /*使用Numcpp<type> np(row,col)创建一个row * col的矩阵*/
    Numcpp<nc_t> n(4, 4), m(4, 4);

    /*矩阵中所有元素的默认值为1，也可以手动设置*/
    Numcpp<nc_t> c(6, 7, 3.0);
    Numcpp<nc_t> e(6, 8);

    /*广播操作*/
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
        generate(c);
        generate(e);
        std::cout << c << "\n";
        std::cout << e << "\n";
        // 矩阵转置：
        c.transposed();
        std::cout << c << "\n";

        // 矩阵的特殊乘法：
        Numcpp<nc_t> Out = c<func> e; // 会创建一个新的矩阵
        std::cout << Out << "\n";

        // 函数数乘特殊乘法：
        Numcpp<nc_t> act = result<sigmoid> NULL;
        std::cout << act << "\n";

        // 矩阵fft
        std::cout << result << "\n";
        result.ffted(1);
        std::cout << result << "\n";
        // ifft
        result.ffted(-1);
        std::cout << result << "\n";
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

### 矩阵的特殊乘法

```c++
Numcpp<nc_t> Out = c<func> e; // 会创建一个新的矩阵
```

### 函数数乘特殊乘法：

```c++
Numcpp<nc_t> act = result<sigmoid> NULL;
```

### FFT

```c++
// 矩阵fft
result.ffted(1);
// ifft
result.ffted(-1);
```

# 利用矩阵进行神经网络计算示例

利用偏异化随机生成产生符合特定要求的训练数据，然后进行神经网络模型的训练。

使用了两层隐含层。

```c++
#include <iostream>
#include <math.h>
#include <complex>
#include "Numcpp.hpp"
// 复数
typedef std::complex<double> complex_double;

// 理论上模板要被同一个类型实例化
#define nc_t complex_double

void generate(Numcpp<nc_t> &nc)
{
    srand(time(NULL));
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

nc_t func(nc_t n, nc_t m)
{
    nc_t result = n * m;
    return result;
}
nc_t sigmoid(nc_t n, nc_t m)
{
    return nc_t(1, 0) / (nc_t(1, 0) + exp(-n));
}

int main(int argc, char const *argv[])
{
    /*使用Numcpp<type> np(row,col)创建一个row * col的矩阵*/
    Numcpp<nc_t> n(4, 4), m(4, 4);

    /*矩阵中所有元素的默认值为1，也可以手动设置*/
    Numcpp<nc_t> c(6, 7, 3.0);
    Numcpp<nc_t> e(6, 8);

    /*广播操作*/
    n *= 2.0;
    m *= 3.0;

    try
    {

        // 矩阵乘法：
        Numcpp<nc_t> result = n * m;
        std::cout << result << "\n";

        // 使用矩阵乘法优化算法，在M*K*N >64*64*64时生效,对特殊乘法无效
        n.optimized(true);

        // 矩阵运算：
        result = n + m;
        std::cout << result << "\n";
        // 哈达马乘积：
        result = n.Hadamard(m);
        std::cout << result << "\n";

        // 生成正态分布的矩阵
        generate(c);
        generate(e);
        std::cout << c << "\n";
        std::cout << e << "\n";
        // 矩阵转置：
        c.transposed();
        std::cout << c << "\n";

        // 矩阵的特殊乘法：
        Numcpp<nc_t> Out = c<func> e; // 会创建一个新的矩阵
        std::cout << Out << "\n";

        // 函数数乘特殊乘法：
        Numcpp<nc_t> act = result<sigmoid> NULL;
        std::cout << act << "\n";

        // 矩阵fft
        std::cout << result << "\n";
        result.ffted(1);
        std::cout << result << "\n";
        // ifft
        result.ffted(-1);
        std::cout << result << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
    return 0;
}
```



# 相关信息

作者：[yauntyour]([yauntyour (yauntyour) · GitHub](https://github.com/yauntyour/))

授权协议：MIT开源协议

参考：[数学基础 - 矩阵的基本运算（Matrix Operations）_沙沙的兔子的博客-CSDN博客_矩阵运算](https://blog.csdn.net/darkrabbit/article/details/80025935)
