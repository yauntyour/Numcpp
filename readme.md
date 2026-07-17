# Numcpp —— 基于原生 C++ 的通用矩阵库 (Header-Only)

原生 C++ 矩阵类封装，header-only 结构，零依赖（核心），无需安装。

## 快速开始

```cpp
#include "Numcpp.hpp"

using namespace np;

<<<<<<< HEAD
=======
// 复数，推荐使用C++的复数类型，支持FFT变换
typedef std::complex<double> complex_double;

#define nc_t complex_double

nc_t sinxy(nc_t x, nc_t y)
{
    return nc_t(sin(x.real()), sin(x.imag()));
}

void generate(Numcpp<nc_t> &nc)
{
    srand(time(NULL));
    for (size_t i = 0; i < nc.row; i++)
    {
        for (size_t j = 0; j < nc.col; j++)
        {
            double U1 = rand() * 1.0f / RAND_MAX;                // 0~1均匀分布
            double U2 = rand() * 1.0f / RAND_MAX;                // 0~1均匀分布
            double Z = sqrt(-2 * log(U1)) * cos(2 * NP_PI * U2); // 标准正态分布
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
    // 开启多核优化
    np::is_optimized = true;
    /*使用Numcpp<type> np(row,col)创建一个row * col的矩阵*/
    Numcpp<nc_t> n(16, 16), m(16, 16);

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
        std::cout << "n * m:" << result << "\n";

        // 使用矩阵乘法优化算法，在M*K*N >64*64*64时生效,对特殊乘法无效;

        // 矩阵运算：
        result = n + m;
        std::cout << "n + m:" << result << "\n";
        // 哈达马乘积：
        result = n.Hadamard(m);
        std::cout << "n (h*) m:" << result << "\n";

        // 生成正态分布的矩阵
        generate(c);
        generate(e);
        std::cout << c << "\n";
        std::cout << e << "\n";
        // 矩阵转置：
        c.transposed();
        std::cout << "c transposed:" << c << "\n";

        // 矩阵的特殊乘法：
        Numcpp<nc_t> Out = c<func> e; // 会创建一个新的矩阵
        std::cout << "Out:" << Out << "\n";

        // 函数数乘特殊乘法：
        Numcpp<nc_t> act = result<sigmoid> NULL;
        std::cout << "act:" << act << "\n";

        // 矩阵fft
        std::cout << "RAW:" << result << "\n";
        result.ffted(1);
        std::cout << "FFT:" << result << "\n";
        // ifft
        result.ffted(-1);
        std::cout << "iFFT" << result << "\n";
        // 保存矩阵
        Out.save("mat");
        // 读取矩阵
        Numcpp<nc_t> temp = load<nc_t>("mat");
        std::cout << "temp load in Out:" << temp << "\n";

        // 流式创建一个方阵
        Numcpp<int> mat(3, 3);
        mat << 4, 1, 1,
            1, 3, 2,
            1, 2, 5;

        // 方阵的逆、行列式
        std::cout << "mat:" << mat << "\n";
        std::cout << "mat's sum:" << mat.sum() << "\n";
        std::cout << "mat Determinant value:" << mat.determinant() << "\n";
        std::cout << "mat Inverse mat:" << mat.inverse() << "\n";

        // 直接赋值式流
        Numcpp<double> nmat = (Numcpp<double>(4, 3) << 4, 1, 1, 1,
                               3, 2, 1, 2,
                               5, 5, 1, 1);
        // 矩阵阵的逆
        std::cout << "nmat:" << nmat << "\n";
        std::cout << "nmat pseudoinverse mat:" << nmat.pseudoinverse() << "\n";
        std::cout << "nmat[0:]:" << nmat.srow(0) << std::endl;
        std::cout << "nmat[:2]:" << nmat.scol(2) << std::endl;

        Numcpp<double> U, S, V;
        nmat.svd(U, S, V);

        std::cout << "SVD_U:" << U << "\n";
        std::cout << "SVD_S:" << S << "\n";
        std::cout << "SVD_V:" << V << "\n";
        std::cout << "rebuild nmat:" << U * S * V.transpose() << "\n";

        // lambda支持
        std::cout << "<lambda>:" << (temp<[](nc_t x, nc_t y) -> nc_t
                                          { return nc_t(sin(x.real()), sin(x.imag())); }>
                                         NULL)
                  << std::endl;
        std::cout << "<func>:" << (n<sinxy> NULL)
                  << std::endl;

        // 生成高斯矩阵基本用法
        auto gmat = np::randn<double>(100, 100); // 100x100标准高斯矩阵

        // 自定义参数
        np::GaussianConfig config;
        config.mean = 5.0;
        config.stddev = 2.0;
        config.seed = 12345;
        auto custom_mat = np::randn<double>(50, 50, config);

        // 多线程生成大矩阵
        auto big_mat = np::randn_parallel<double>(1000, 1000, config, 8);

        // 多变量高斯
        np::Numcpp<double> cov(2, 2);
        cov << 1.0, 0.8, 0.8, 1.0;
        auto multi_mat = np::multivariate_randn<double>(1000, cov);

        // 向量的点积
        Numcpp<int> v1(1, 9), v2(9, 1, 8);
        std::cout << "Dot: " << v1.dot(v1) << std::endl;

        // 范数计算
        auto normat = np::randn<double>(16, 16);
        std::cout << "normat: " << normat << std::endl;
        std::cout << "normat's L1: " << normat.norm(np::L1) << std::endl;
        std::cout << "normat's L2: " << normat.norm(np::L2) << std::endl;
        std::cout << "normat's INF: " << normat.norm(np::INF) << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
    return 0;
}

```

# **矩阵的运算**

### 下面是支持的操纵符的类型

```c++
Numcpp<typename> nc(4, 4),matrix(4,4);
```

### 赋值运算符

```c++
//矩阵简单运算的赋值运算
nc += matrix;
nc -= matrix;
nc = matrix;
//矩阵数乘广播的赋值运算
nc *= number//number为一个数值
nc += number
nc -= number
nc /= number
nc + number -> Numcpp
nc - number -> Numcpp
nc * number -> Numcpp
nc / number -> Numcpp
```

### 矩阵的基本运算

```c++
Numcpp<typename> result = nc + matrix;
Numcpp<typename> result = nc - matrix;

Numcpp<typename> result = nc * number;//数乘
Numcpp<typename> result = nc * matrix;//矩阵乘法
```

### 解矩阵运算

```c++
typename data = nc[x][y];//下标访问
nc.srow(index) -> Numcpp //行提取
nc.scol(index) -> Numcpp //列提取
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

## 方阵的逆以及矩形矩阵的伪逆

```c++
// 流式创建一个方阵
Numcpp<nc_t> mat(3, 3);
mat << 1, 2, 3, 4, 5, 6, 7, 8, 9;

// 方阵的逆
mat.inverse() -> Numcpp

// 矩阵的逆
Numcpp<nc_t> nmat(4, 3);
nmat << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
nmat.pseudoinverse() -> Numcpp
```

## 方阵的行列式计算

```c++
mat.determinant() -> Number
```

### 矩阵的特殊乘法

```c++
Numcpp<nc_t> Out = c<func> e; // 会创建一个新的矩阵
```

### 函数数乘特殊乘法

```c++
Numcpp<nc_t> act = result<sigmoid> NULL;
```

### FFT（只对复数矩阵有效）

```c++
// 矩阵fft
result.ffted(1);
// ifft
result.ffted(-1);
```

### **启用矩阵乘法优化算法和CPU多核优化加速计算**

```c++
//默认同时开启矩阵乘法优化算法和CPU多核优化加速计算
nc.optimized(true);
//设置最大并发数
nc.maxprocs_set(thread_num);
```

值得一提的是：

```c++
if (sqrt(thread_num) * sqrt(thread_num) > std::thread::hardware_concurrency() || thread_num < 1)
{
 throw std::invalid_argument("Invalid maxprocs");
}
```

如果最大并发数大于CPU的最大线程数优化将毫无意义，我们禁止这种行为。同时根据约定，当`thread_num`的值为一个可开方数，同时能够和矩阵的形状成几何倍数关系，此时优化效果最佳。同时，如果`nc.row/sqrt(thread_num)`或`nc.col/sqrt(thread_num)`不为整数，则可能存在分块缺陷。看起来像这样：

```
(16,8)[
    [0][(2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (1,0) , (1,0)]
    [1][(2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (1,0) , (1,0)]
    [2][(2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (1,0) , (1,0)]
    [3][(2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (1,0) , (1,0)]
    [4][(2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (1,0) , (1,0)]
    [5][(2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (1,0) , (1,0)]
    [6][(2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (1,0) , (1,0)]
    [7][(2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (1,0) , (1,0)]
    [8][(2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (1,0) , (1,0)]
    [9][(2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (1,0) , (1,0)]
    [10][(2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (1,0) , (1,0)]
    [11][(2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (1,0) , (1,0)]
    [12][(2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (1,0) , (1,0)]
    [13][(2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (1,0) , (1,0)]
    [14][(2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (2,0) , (1,0) , (1,0)]
    [15][(1,0) , (1,0) , (1,0) , (1,0) , (1,0) , (1,0) , (1,0) , (1,0)]
]
```

正常被处理的输出值应该为2，这是由于分块算法无法切割残块导致的，所以应该确保矩阵可以被分割为N个正方形大小的矩阵，如此可以保证效率与数据安全。

关于矩阵乘法的优化算法，则是采用了Coppersmith和Winograd在1990年由Strassen算法改进而来的Coppersmith-Winograd算法：论文地址：[107.pdf](https://eprint.iacr.org/2013/107.pdf)

```c++
/*
 The mathematical principles mentioned in the paper:
     * this_matrix A_row * A_col
     * other_matrix A_col * B_col
     * result A_row * B_col
     * result = this_matrix * other_matrix
     * S1 = A21 + A22     T1 = B12 - B11
     * S2 = S1 - A11      T2 = B22 - T1
     * S3 = A11 - A21     T3 = B22 - B12
     * S4 = A12 - S2      T4 = T2 - B21
     * M1 = A11 * B11     U1 = M1 + M2
     * M2 = A12 * B21     U2 = M1 + M6
     * M3 = S4 * B22      U3 = U2 + M7
     * M4 = A22 * T4      U4 = U2 + M5
     * M5 = S1 * T1       U5 = U4 + M3
     * M6 = S2 * T2       U6 = U3 - U4
     * M7 = S3 * T3       U7 = U3 + M5
     * C11 = U1
     * C12 = U5
     * C21 = U6
     * C22 = U7
*/
```

### 矩阵有关的实用工具

```c++
// 创建矩阵
np::Numcpp<int> mat(4, 5);
mat << 1, 1, 0, 1, 0,
       1, 1, 1, 1, 1,
       1, 1, 1, 1, 1,
       1, 0, 1, 1, 0;

// 查找最大矩形
np::Rectangle rect = np::findMaximalRectangle(mat);
std::cout << "最大矩形: " << rect.area << " 面积" << std::endl;

// 基本用法
auto mat = np::randn<double>(100, 100);  // 100x100标准高斯矩阵

// 自定义参数
np::GaussianConfig config;
config.mean = 5.0;
config.stddev = 2.0;
config.seed = 12345;
auto custom_mat = np::randn<double>(50, 50, config);

// 多线程生成大矩阵
auto big_mat = np::randn_parallel<double>(1000, 1000, config, 8);

// 多变量高斯
np::Numcpp<double> cov(2, 2);
cov << 1.0, 0.8, 0.8, 1.0;
auto multi_mat = np::multivariate_randn<double>(1000, cov);

```



# 对Nvidia GPU的CUDA加速支持

提供所有基础操作的CUDA加速，同时基于性能考量，内存自动同步默认为关，在**完成所有运算后**应**手动同步数据**回主机。

```c++
#include <iostream>
#include "Numcpp.hpp"

using namespace np;
#define nc_t double
>>>>>>> ba984c9bca88375fd70002cc78fce1a7188e962d
int main()
{
    Numcpp<double> n(16, 16, 2.0);
    Numcpp<double> m(16, 16, 3.0);

<<<<<<< HEAD
    auto result = n * m;
    std::cout << result << std::endl;
=======
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

    // 本位减法
    n -= m;
    n.to(DEVICE_LOCAL);
    std::cout << n;

    // 同位广播 & 开启自动同步(默认关闭状态，避免运算期间反复拷贝内存)
    n.auto_sync = true;
    std::cout << (n - 10) / 8.0 * 5 + 3 - 2 * 4 / 2 + 1 << std::endl;
>>>>>>> ba984c9bca88375fd70002cc78fce1a7188e962d
    return 0;
}
```

编译：`g++ -std=c++20 example.cpp -o example`

---

## 模块化引入

支持按需引入子模块，降低编译依赖：

```cpp
// 仅需核心矩阵功能
#include "Numcpp/core.hpp"

// 需要高斯随机数生成
#include "Numcpp/random.hpp"

// 需要 LQR / QP 优化求解器
#include "Numcpp/optim.hpp"

// 需要 OpenCV 互操作（需已安装 OpenCV）
#include "Numcpp/opencv.hpp"

// 引入全部模块
#include "Numcpp.hpp"
```

---

## 创建矩阵

```cpp
Numcpp<double> a(3, 4);           // 3x4, 元素默认 0
Numcpp<double> b(3, 4, 5.0);      // 3x4, 元素默认 5.0
Numcpp<double> c(a);              // 拷贝构造

double  data[] = {1,2,3,4,5,6};
Numcpp<double> d(data, 2, 3);     // 从一维数组

// 流式初始化
Numcpp<double> e(3, 3);
e << 4, 1, 1,
     1, 3, 2,
     1, 2, 5;
```

---

## 算术运算

```cpp
Numcpp<double> a(4, 4), b(4, 4);

// 矩阵运算
auto s = a + b;
auto d = a - b;
auto m = a * b;          // 标准矩阵乘法

// 数乘运算
auto x = a * 3.0;
auto y = a / 2.0;
auto z = a + 1.0;

// 原位运算
a += b;  a -= b;
a *= 2;  a /= 2;  a += 1;  a -= 1;
```

---

## 索引与提取

```cpp
auto v  = a[2][3];        // 下标访问
auto r  = a.srow(0);      // 提取第 0 行，返回 1×n 矩阵
auto c  = a.scol(1);      // 提取第 1 列，返回 m×1 矩阵
bool iv = a.is_vector();   // 是否向量
size_t sz = a.size();      // 元素总数
auto s  = a.sum();         // 元素求和
```

---

## 哈达马乘积 (Hadamard)

```cpp
auto h = a.Hadamard(b);        // 返回新矩阵
a.Hadamard_self(b);            // 原位
```

---

## 转置

```cpp
auto t = a.transpose();       // 返回新矩阵
a.transposed();               // 原位转置
```

---

## 特殊乘法

```cpp
// 二元函数乘法: C[i][j] = sum_k func(A[i][k], B[k][j])
double mul(double x, double y) { return x * y; }
auto out = a <mul> b;

// 一元函数映射: 对每个元素调用 func
double sigmoid(double x, double) { return 1.0 / (1.0 + exp(-x)); }
auto act = a <sigmoid> NULL;

// C++20 lambda 映射
auto r = a <[](double x, double y) { return x * 2; }> nullptr;
```

---

## 行列式 / 逆矩阵 / 伪逆

```cpp
double det = mat.determinant();         // 行列式
auto inv   = mat.inverse();             // 逆矩阵（方阵）
auto pinv  = mat.pseudoinverse();       // 伪逆 (Moore-Penrose)
```

---

## SVD（奇异值分解）

```cpp
Numcpp<double> U, S, V;
nmat.svd(U, S, V);                        // 引用方式
auto rebuilt = U * S * V.transpose();     // 可重建成原矩阵

auto svd_rst = nmat.svd();                // 返回 {U, S, V}
```

---

## 特征值与特征向量

```cpp
auto r = mat.eig();              // {特征值, 特征向量矩阵, 对角矩阵}
// 仅支持对称矩阵, QR 迭代法
```

---

## 范数与点积

```cpp
auto l1 = mat.norm(np::L1);        // L1 范数（列和最大）
auto l2 = mat.norm(np::L2);        // L2 / Frobenius 范数
auto inf= mat.norm(np::INF);       // 无穷范数（行和最大）
auto d  = v1.dot(v2);              // 向量点积
```

---

## FFT / IFFT

```cpp
// 复数矩阵
Numcpp<complex<double>> nc(16, 16);
auto f = nc.fft(1);             // 正向 FFT, 返回复数矩阵
auto g = f.fft(-1);             // 逆向 (含 1/N 归一化)
nc.ffted(1);                    // 原位 FFT（仅复数类型）

// 实数矩阵 — 自动转为复数返回
Numcpp<double> rn(16, 16);
auto cplx = rn.fft(1);         // 返回 Numcpp<complex<double>>
// 不支持实数矩阵原位 FFT — 请用 fft() 替代
```

---

## 类型转换 `.as<>()`

所有数值类型可互转：

```cpp
Numcpp<int>     imat(3, 3, 42);
auto fmat = imat.as<float>();                       // int → float
auto dmat = fmat.as<double>();                      // float → double
auto i8   = dmat.as<int8_t>();                      // double → int8_t
auto cmat = imat.as<std::complex<double>>();         // int → complex
```

---

## 实用工具

```cpp
// 单位矩阵
a.set_identity();
auto I = a.identity();

// 对称性检查
bool sym = a.is_symmetric();

// 零逼近（小于 tolerance 的元素置零）
a.zero_approximation(1e-6);

// 二值化
auto binary = binarizeMatrix(a, 0.5);

// Cholesky 分解 L*L^T = A
auto L = cholesky_decomposition(A);
```

---

## 高斯随机矩阵

```cpp
auto gmat = randn<double>(100, 100);

GaussianConfig cfg{ .mean = 5.0, .stddev = 2.0, .seed = 12345 };
auto cmat = randn<double>(50, 50, cfg);

// 多线程生成
auto big = randn_parallel<double>(1000, 1000, cfg, 8);

// 多变量高斯
Numcpp<double> cov(2, 2);
cov << 1.0, 0.8, 0.8, 1.0;
auto multi = multivariate_randn<double>(1000, cov);

// 混合高斯
auto mix = gaussian_mixture<double>(100, 100,
    {{.mean=0, .stddev=1}, {.mean=5, .stddev=0.5}});
```

---

## LQR / QP 优化求解器

```cpp
// LQR（离散 Riccati 方程迭代求解）
auto [K, P] = solve_lqr(A, B, Q, R);

// QP（原-对偶法，支持等式/不等式约束）
auto x_opt = solve_QP(Q_mat, C, A, b, E, d);
```

---

## 嵌套矩阵

支持 `Numcpp<Numcpp<T>>`（矩阵元素为矩阵），所有运算符自动适配：

```cpp
Numcpp<Numcpp<double>> nm(1, 1, Numcpp<double>(3, 2, 1.0));
auto mm = nm * nm;         // 内层矩阵乘法 + 外层矩阵乘法
auto ss = nm.sum();        // 内层矩阵逐元素求和
```

---

## 多核优化

```cpp
nc.optimized(true);                      // 开启多线程加速
nc.maxprocs_set(4);                      // 设置线程数（需为完全平方数）
```

线程数必须 ≤ `hardware_concurrency()`。矩阵乘法在 M×K×N > 64³ 且维度为偶数时自动启用 Coppersmith-Winograd 算法。

---

## OpenCV 互操作

需安装 OpenCV，单独引入 `Numcpp/opencv.hpp`：

```cpp
#include "Numcpp/opencv.hpp"

cv::Mat img = cv::imread("image.png", cv::IMREAD_GRAYSCALE);
auto nm = np::fromCvMat<uint8_t>(img);    // cv::Mat → Numcpp
cv::Mat result = np::toCvMat(nm);         // Numcpp → cv::Mat
```

---

## 保存与加载

```cpp
mat.save("matrix.bin");
auto loaded = np::load<double>("matrix.bin");
```

---

## 文件结构

```
Numcpp/
├── Numcpp.hpp          # 兼容头（#include "Numcpp/Numcpp.hpp"）
└── Numcpp/
    ├── Numcpp.hpp      # 总引入（core + random + optim）
    ├── core.hpp        # 核心矩阵类 + 线性代数 + FFT
    ├── random.hpp      # 高斯随机数生成 + Cholesky
    ├── optim.hpp       # LQR + QP 求解器
    └── opencv.hpp      # OpenCV 互操作（fromCvMat / toCvMat）
```

---

## 授权

MIT License

作者：[yauntyour](https://github.com/yauntyour/)
