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
#include "Numcpp/Numcpp.hpp"
#include <math.h>
#include <vector>

#define nc_t double

#define OWN(x, a, b) (a / b) * sqrt(a *x) * sqrt(a *x)
nc_t OWN_A = 1, OWN_K = 1;

void gIN(Numcpp<nc_t> &nc)
{
    for (size_t j = 0; j < nc.row; j++)
    {
        nc.matrix[j][0] = 20 + 50 * rand() * 1.0f / RAND_MAX;  // age
        nc.matrix[j][1] = 50 + 10 * rand() * 1.0f / RAND_MAX;  // weight
        nc.matrix[j][2] = 170 + 30 * rand() * 1.0f / RAND_MAX; // high

        double fam = rand() * 1.0f / RAND_MAX;
        if (fam > 0.5)
        {
            // boy
            nc.matrix[j][3] = 1;
            nc.matrix[j][1] *= 1.05;
            nc.matrix[j][2] *= 1.05;
        }
        else
        {
            // girl
            nc.matrix[j][3] = 0;
            nc.matrix[j][1] *= 0.95;
            nc.matrix[j][2] *= 0.95;
        }
    }
}

void average(Numcpp<nc_t> &nc)
{
    auto asum = 0, wsum = 0, hsum = 0;
    for (size_t i = 0; i < nc.row; i++)
    {
        asum += nc.matrix[i][0];
        wsum += nc.matrix[i][1];
        hsum += nc.matrix[i][2];
    }
    asum /= nc.row;
    wsum /= nc.row;
    hsum /= nc.row;
    for (size_t i = 0; i < nc.row; i++)
    {
        nc.matrix[i][0] -= asum;
        nc.matrix[i][1] -= wsum;
        nc.matrix[i][2] -= hsum;
    }
}

void gWei(Numcpp<nc_t> &nc, nc_t age, nc_t weight, nc_t high)
{
    for (size_t i = 0; i < nc.row; i++)
    {
        nc.matrix[i][0] = age;
    }
    for (size_t i = 0; i < nc.row; i++)
    {
        nc.matrix[i][1] = weight;
    }
    for (size_t i = 0; i < nc.row; i++)
    {
        nc.matrix[i][2] = high;
    }
    for (size_t i = 0; i < nc.row; i++)
    {
        nc.matrix[i][3] = 0;
    }
}

void gCost(Numcpp<nc_t> &Inputs, Numcpp<nc_t> &Cost)
{
    for (size_t i = 0; i < Inputs.row; i++)
    {
        Cost.matrix[i][0] = Inputs.matrix[i][3];
    }
}

nc_t sigmoid(nc_t x, nc_t y)
{
    return 1 / (1 + exp(-x));
}
nc_t d_sigmoid(nc_t x, nc_t y)
{
    return sigmoid(x, y) * (1 - sigmoid(x, y));
}
nc_t Squdiff(nc_t x, nc_t y)
{
    return x * x;
}
nc_t eta = 0.01;

int main(int argc, char const *argv[])
{
    // generate random data with random values
    Numcpp<nc_t> Inputs(64, 4);
    gIN(Inputs);
    // std::cout << "RAW: " << Inputs << "\n";
    Numcpp<nc_t> Cost(Inputs.row, 1);
    gCost(Inputs, Cost);
    // std::cout << "COST: " << Cost << "\n";

    // set weight
    Numcpp<nc_t> Wei(Inputs.col, Inputs.col);
    gWei(Wei, 1, 1, 1);
    Numcpp<nc_t> Aver = Inputs * Wei;
    average(Aver);
    Aver.col -= 1;
    // std::cout << "Averaged:" << Aver << "\n";

    // Weight & BaisOut_Wei
    Numcpp<nc_t> Hider_Wei(Aver.col, 2);                 // 3*2
    Numcpp<nc_t> Hider_Bais(Aver.row, Hider_Wei.col, 0); // 16*2
    Numcpp<nc_t> Out_Wei(Hider_Wei.col, 1);              // 2*1
    Numcpp<nc_t> Out_Bais(Aver.row, Out_Wei.col, 0);     // 16*1
    //10w次
    for (size_t i = 0; i < 100000; i++)
    // while (1)
    {
        // std::cout << "############################################################################\n";
        //   Hide layer computation
        Numcpp<nc_t> z1 = Aver * Hider_Wei + Hider_Bais;
        Numcpp<nc_t> Hider = z1<sigmoid> NULL;
        // std::cout << "Hider: " << Hider << "\n";
        //   Out layer computation
        Numcpp<nc_t> z2 = Hider * Out_Wei + Out_Bais; // 16 * 2
        Numcpp<nc_t> Out = z2<sigmoid> NULL;
        // std::cout << "Out" << Out << "\n";
        //    loss computation
        Numcpp<nc_t> L = Out - Cost;
        Numcpp<nc_t> s_L = Numcpp<nc_t>(1, Inputs.row) * (L<Squdiff> NULL) / Inputs.row;
        std::cout << "Loss: " << s_L[0][0] << "\n";

        // updata wei & bais computation
        // Out
        /*
        std::cout << "Out_Wei: " << Out_Wei << "\n";
        std::cout << "Out_Bais: " << Out_Bais << "\n";
        */
        Numcpp<nc_t> dow = Hider.transpose() * (L.Hadamard(z2<d_sigmoid> NULL)) * 2;
        Out_Wei = Out_Wei - dow * eta;
        Numcpp<nc_t> dob = (L.Hadamard(z2<d_sigmoid> NULL)) * 2;
        Out_Bais = Out_Bais - dob * eta;
        /*
        std::cout << "dow: " << dow << "\n";
        std::cout << "dob: " << dob << "\n";
        std::cout << "updata Out_Wei: " << Out_Wei << "\n";
        std::cout << "updata Out_Bais: " << Out_Bais << "\n";
        */
        //  Hider
        /*
        std::cout << "Hider_Wei: " << Hider_Wei << "\n";
        std::cout << "Hider_Bais: " << Hider_Bais << "\n";
        */
        Numcpp<nc_t> dhw = (Aver.transpose() * (z1<d_sigmoid> NULL).Hadamard(L.Hadamard(z2<d_sigmoid> NULL) * Out_Wei.transpose())) * 2;
        Hider_Wei = Hider_Wei - dhw * eta;
        Numcpp<nc_t> dhb = (z1<d_sigmoid> NULL).Hadamard(L.Hadamard(z2<d_sigmoid> NULL) * Out_Wei.transpose()) * 2;
        Hider_Bais = Hider_Bais - dhb * eta;
        /*
        std::cout << "dhw: " << dhw << "\n";
        std::cout << "dhb: " << dhb << "\n";
        std::cout << "updata Hider_Wei: " << Hider_Wei << "\n";
        std::cout << "updata Hider_Bais: " << Hider_Bais << "\n";
        std::cout << "############################################################################\n";
        */
        //_sleep(500);
    }
    std::cout << "Train done.\n";
    // testing
    /*
    //  Age weight high
    Numcpp<nc_t> T(Aver.row, 3, 0);
    while (1)
    {
        printf("scan: Age && Weight && High\n");
        std::cin >> T.matrix[0][0];
        std::cin >> T.matrix[0][1];
        std::cin >> T.matrix[0][2];
        std::cout << "scan: " << (T.matrix[0][0]) << "&&" << (T.matrix[0][1]) << "&&" << (T.matrix[0][2]) << "\n";

        Numcpp<nc_t> T_Hider = (T * Hider_Wei + Hider_Bais)<sigmoid> NULL;
        Numcpp<nc_t> T_Out = (T_Hider * Out_Wei + Out_Bais)<sigmoid> NULL;
        T_Out.row = 1;
        std::cout << "Out" << T_Out << "\n";
    }
    */
    Numcpp<nc_t> T(Aver.row, 3);
    gIN(T);
    Numcpp<nc_t> T_Hider = (T * Hider_Wei + Hider_Bais)<sigmoid> NULL;
    Numcpp<nc_t> T_Out = (T_Hider * Out_Wei + Out_Bais)<sigmoid> NULL;
    std::cout << "Out" << T_Out << "\n";
    return 0;
}

```



# 相关信息

作者：[yauntyour]([yauntyour (yauntyour) · GitHub](https://github.com/yauntyour/))

授权协议：MIT开源协议

参考：[数学基础 - 矩阵的基本运算（Matrix Operations）_沙沙的兔子的博客-CSDN博客_矩阵运算](https://blog.csdn.net/darkrabbit/article/details/80025935)
