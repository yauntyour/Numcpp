# Numcpp —— 基于原生 C++ 的通用矩阵库 (Header-Only)

原生 C++ 矩阵类封装，header-only 结构，零依赖（核心），无需安装。

## 快速开始

```cpp
#include "Numcpp.hpp"

using namespace np;

int main()
{
    Numcpp<double> n(16, 16, 2.0);
    Numcpp<double> m(16, 16, 3.0);

    auto result = n * m;
    std::cout << result << std::endl;
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
