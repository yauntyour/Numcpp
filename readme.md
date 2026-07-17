# Numcpp

纯头文件、模板化的 C++ 线性代数与矩阵计算库。

## 特性

- **纯头文件** — `#include "Numcpp/Numcpp.hpp"` 即用
- **模板化** — 支持 `float`、`double`、`int`、`std::complex` 及自定义类型
- **多线程** — 通过 `optimized(true)` + `maxprocs_set(N)` 启用并行执行，内部使用全局线程池（延迟初始化、RAII 管理），工作线程跨操作复用
- **丰富 API** — 矩阵运算、矩阵分解、FFT、随机采样、优化求解、OpenCV 互操作、算法扩展库

## 快速开始

```cpp
#include <iostream>
#include "Numcpp/Numcpp.hpp"
using namespace np;

int main() {
    // 两种构造方式
    Numcpp<double> a(2, 3, 1.0);               // 2x3，全 1
    auto b = (Numcpp<double>(2, 2) << 1, 2,     // 2x2 流式初始化
                                          3, 4);

    // 算术运算
    auto c = a + b;         // 矩阵加法
    auto d = a * 2.0;       // 标量乘法
    auto e = a.transpose(); // 转置
    auto f = a.Hadamard(a); // Hadamard 逐元素乘积

    // 矩阵乘法
    auto g = (Numcpp<double>(2, 3) << 1, 2, 3, 4, 5, 6)
           * (Numcpp<double>(3, 2) << 7, 8, 9, 10, 11, 12);

    // 矩阵分解
    double det = b.determinant();
    auto inv = b.inverse();

    // 奇异值分解
    Numcpp<double> U, S, V;
    a.svd(U, S, V);

    // 元素访问
    double x = a[0][1];

    // 打印
    std::cout << a << std::endl;

    return 0;
}
```

使用 C++17 或更高版本编译：

```bash
g++ -std=c++17 -O2 main.cpp -o main
```

## API 参考

### 构造与赋值

| 表达式 | 说明 |
|---|---|
| `Numcpp<T> M(rows, cols)` | 零初始化矩阵 |
| `Numcpp<T> M(rows, cols, val)` | 用 `val` 填充的矩阵 |
| `Numcpp<T> M(T**, rows, cols)` | 从原始二维数组构造（深拷贝） |
| `Numcpp<T> M(T*, rows, cols)` | 从一维数组构造（深拷贝） |
| `Numcpp<T> M("file.mat")` | 从二进制文件加载 |
| `M << v1, v2, v3, ...` | 流式初始化（行优先） |
| `M = other` | 深拷贝赋值 |

### 元素访问

| 表达式 | 说明 |
|---|---|
| `M[i][j]` | 访问第 i 行、第 j 列元素 |
| `M.srow(i)` | 提取单行，返回 1×N 矩阵 |
| `M.scol(j)` | 提取单列，返回 M×1 矩阵 |

### 矩阵运算

| 表达式 | 说明 |
|---|---|
| `A + B`、`A += B` | 逐元素加法 |
| `A - B`、`A -= B` | 逐元素减法 |
| `A * B` | 矩阵乘法 |
| `A * s`、`s * A`、`A *= s` | 标量乘法 |
| `A / s`、`A /= s` | 标量除法 |
| `A + s`、`A += s` | 每个元素加标量 |
| `A - s`、`A -= s` | 每个元素减标量 |
| `A.Hadamard(B)` | 逐元素（Hadamard）乘积 |
| `A.Hadamard_self(B)` | 原地 Hadamard 乘积 |

### 矩阵属性与变换

| 表达式 | 说明 |
|---|---|
| `M.transpose()` | 返回转置副本 |
| `M.transposed()` | 原地转置 |
| `M.sum()` | 所有元素求和 |
| `M.norm(type)` | 范数：`L1`、`L2` 或 `INF` |
| `M.dot(v)` | 两个向量的点积 |
| `M.is_vector()` | 判断是否为向量 |
| `M.size()` | 元素总数 `row * col` |
| `M.is_symmetric()` | 判断是否对称 |
| `M.set_identity()` | 设为单位矩阵（需为方阵） |
| `M.zero_approximation(tol)` | 将小于容差的元素置零 |

### 线性代数

| 表达式 | 说明 |
|---|---|
| `M.determinant()` | 行列式（方阵） |
| `M.inverse()` | 矩阵逆（方阵，非奇异） |
| `M.pseudoinverse()` | Moore-Penrose 伪逆 |
| `M.eig(max_iter, tol)` | 特征分解（对称矩阵）— 返回 `{特征值, 特征向量, 对角矩阵}` |
| `M.svd(U, S, V)` | 奇异值分解 — `U * S * V^T = M` |

### FFT 快速傅里叶变换

```cpp
// 复 FFT
Numcpp<std::complex<double>> c(1, 8);
auto freq = c.fft(1);   // 正变换
auto time = c.fft(-1);  // 逆变换

// 实数到复数 FFT
Numcpp<double> r(1, 8);
auto freq_cpx = r.fft(1);   // 返回 Numcpp<std::complex<double>>
```

### 类型转换

```cpp
Numcpp<int> a(2, 2, 42);
auto f = a.as<float>();                 // → Numcpp<float>
auto d = a.as<double>();                // → Numcpp<double>
auto c = a.as<std::complex<double>>();  // → 复数类型
```

### Lambda 函数映射

```cpp
// 逐元素 lambda 映射
auto r = a < [](double x, double y) { return x * 2; } > nullptr;

// 矩阵乘法风格的二元函数映射
auto r2 = a < [](double x, double y) { return x + y; } > b;
```

### 文件读写

```cpp
M.save("matrix.mat");
auto M2 = load<double>("matrix.mat");
```

### 随机与分布

```cpp
// 标准正态分布
auto g = randn<double>(100, 100);

// 自定义参数正态分布
GaussianConfig cfg{ .mean = 3.0, .stddev = 0.5, .seed = 42 };
auto gc = randn<double>(50, 50, cfg);

// Box-Muller 生成器
auto bm = randn<double>(100, 100, cfg, true);

// 并行随机生成
auto gp = randn_parallel<double>(100, 100, cfg, 8);

// 多元正态分布
Numcpp<double> cov = (Numcpp<double>(3, 3) << 4, 1, 1, 1, 3, 2, 1, 2, 5);
Numcpp<double> mean(1, 3, 0.0);
auto samples = multivariate_randn(1000, cov, mean);

// 高斯混合模型
std::vector<GaussianConfig> components = {
    { .mean =  -2.0, .stddev = 0.5, .seed = 1 },
    { .mean =   2.0, .stddev = 0.5, .seed = 2 }
};
auto mixture = gaussian_mixture<double>(500, 500, components);
```

### Cholesky 分解

```cpp
Numcpp<double> A = (Numcpp<double>(3, 3) << 4, 1, 1, 1, 3, 2, 1, 2, 5);
auto L = cholesky_decomposition(A);  // L * L^T = A
```

### 优化求解

**LQR（线性二次型调节器）**

```cpp
auto [K, P] = solve_lqr(A, B, Q, R, max_iter, tolerance);
// 返回最优增益 K 和代价矩阵 P
```

**QP（二次规划）**

```cpp
// min  0.5 * x^T * Q * x + c^T * x
// s.t. A * x <= b,  E * x == d
auto x = solve_QP(Q, c, A, b, E, d);
```

### OpenCV 互操作

`#include "Numcpp/opencv.hpp"`

```cpp
cv::Mat cvMat = ...;
auto numMat = fromCvMat<double>(cvMat);
auto back = toCvMat(numMat);
```

### 多线程

内部使用全局线程池（延迟初始化、RAII 管理），工作线程跨操作复用，避免每次调用创建线程的开销。

```cpp
auto M = Numcpp<double>(1000, 1000);
M.optimized(true);          // 启用并行执行
M.maxprocs_set(8);          // 设置线程数
// 后续所有操作均使用线程池
```

### 工具函数

```cpp
auto bin = binarizeMatrix(A, 0.5);  // 阈值二值化
```

### 算法扩展库

`#include "Numcpp/algos/algos.hpp"` （也可单独引入各头文件）

```cpp
#include "Numcpp/algos/algos.hpp"

// Coppersmith-Winograd — O(n^2.807) 递归算法，N >= 128 时串行最优
auto C = cw_multiply(A, B);

// 分块乘法 — 缓存友好的瓦片化乘法，可选并行
auto D = blocked_multiply(A, B, 64, 0);   // 串行，64×64 分块
auto E = blocked_multiply(A, B, 64, 8);   // 并行，8 线程
```

## 文件结构

```
Numcpp/
├── Numcpp.hpp           主引入头文件
├── core.hpp             核心矩阵类、算术运算、线性代数、FFT
├── random.hpp           高斯随机、Cholesky 分解、多元分布、混合模型
├── optim.hpp            LQR 与 QP 求解器
├── opencv.hpp           OpenCV cv::Mat 互转
└── algos/
    ├── algos.hpp         算法库统一引入
    ├── cw_mmul.hpp       Coppersmith-Winograd 乘法
    └── blocked_mmul.hpp  缓存分块乘法
```

## 性能基准

测试环境：20 核 CPU，使用 `example.cpp` 运行，取多次平均。

### 逐元素操作（1500×1500，225 万元素）

| 操作 | 1 核 | 4 核 | 加速比 | 20 核 | 加速比 |
|---|---|---|---|---|---|
| construct | 7.7ms | 7.4ms | 1.05x | 7.0ms | 1.10x |
| operator+= | 16.4ms | 17.4ms | 0.94x | 15.4ms | 1.07x |
| operator*= | 7.4ms | 10.1ms | 0.73x | 9.7ms | 0.76x |
| Hadamard | 18.1ms | 19.1ms | 0.95x | 16.4ms | 1.10x |
| transpose | 20.7ms | 20.4ms | 1.02x | 17.4ms | 1.19x |
| copy ctor | 14.1ms | 13.4ms | 1.05x | 12.7ms | 1.11x |
| sum | 7.0ms | 7.4ms | 0.95x | 6.4ms | 1.11x |

逐元素操作受内存带宽限制 — 对 O(n) 级运算，多线程在此规模下仅提供边际收益。

### 矩阵乘法（400×400）

| 操作 | 1 核 | 4 核 | 加速比 | 20 核 | 加速比 |
|---|---|---|---|---|---|
| A × B | 47.0ms | 13.4ms | 3.52x | 7.4ms | 6.41x |

### 算法对比（方阵 N×N，2 的幂，20 线程，分块大小 64）

| N | 朴素 | CW | 分块串行 | 分块并行 | CW/朴素 | 分块/朴素 | 并分/朴素 |
|---|---|---|---|---|---|---|---|
| 64 | 119μs | 118μs | 157μs | 342μs | 1.00x | 0.76x | 0.35x |
| 128 | 1,230μs | 1,162μs | 1,476μs | 1,277μs | 1.05x | 0.83x | 0.96x |
| 256 | 11,236μs | 9,311μs | 10,669μs | 7,355μs | 1.20x | 1.05x | 1.53x |
| 512 | 95,866μs | 68,453μs | 81,442μs | 29,601μs | 1.40x | 1.18x | 3.24x |
| 1024 | 1,184ms | — | 1,278ms | 242ms | — | 0.93x | 4.88x |

- **CW**（`cw_multiply`）：串行最优算法，相对朴素 1.05–1.40x，优势随 N 增长而扩大
- **分块串行**（`blocked_multiply`，threads=0）：缓存友好的瓦片化，N≥256 后开始见效
- **分块并行**（`blocked_multiply`，threads=N）：结合缓存分块与线程池，N=512 时 3.24x，N=1024 时 4.88x

### OpenBLAS 对比

与 [OpenBLAS](https://www.openblas.net/) 0.3.30 进行矩阵乘法（`cblas_dgemm`）性能对比。测试代码：`benchmark_openblas.cpp`，编译参数 `-std=c++20 -O3`。

测试环境：20 核 CPU，分块大小 64，OpenBLAS 线程数 = 20。

| N | 朴素 (ms) | 分块串行 (ms) | 分块并行 (ms) | OpenBLAS (ms) | BLAS/并行 |
|---|---|---|---|---|---|
| 128 | 1.24 | 0.95 | 2.30 | 0.24 | 9.62x |
| 256 | 11.14 | 8.59 | 5.53 | 0.36 | 15.23x |
| 512 | 95.91 | 62.45 | 26.64 | 1.23 | 21.66x |
| 1024 | 797.78 | 512.00 | 172.70 | 7.40 | 23.34x |
| 2048 | — | 4,405.41 | 1,588.95 | 54.77 | 29.01x |

**GFLOPS 对比（理论浮点运算 = 2 × N³）：**

| N | 朴素 GFLOPS | 分块串行 GFLOPS | 分块并行 GFLOPS | OpenBLAS GFLOPS |
|---|---|---|---|---|
| 128 | 3.37 | 4.40 | 1.82 | 17.53 |
| 256 | 3.01 | 3.91 | 6.07 | 92.36 |
| 512 | 2.80 | 4.30 | 10.08 | 218.20 |
| 1024 | 2.69 | 4.19 | 12.44 | 290.10 |
| 2048 | — | 3.90 | 10.81 | 313.66 |

**OpenBLAS 相对 Numcpp 最优实现的加速比：**

| N | Numcpp 最优 (ms) | OpenBLAS (ms) | BLAS 加速比 |
|---|---|---|---|
| 128 | 0.95 | 0.24 | 4.0x |
| 256 | 5.53 | 0.36 | 15.2x |
| 512 | 26.64 | 1.23 | 21.7x |
| 1024 | 172.70 | 7.40 | 23.3x |
| 2048 | 1,588.95 | 54.77 | 29.0x |

结论：OpenBLAS 在矩阵乘法上大幅领先 Numcpp 纯 C++ 实现，大矩阵（2048×2048）差距达 **29 倍**。差距主要来自：
- **SIMD 指令级并行**：OpenBLAS 使用手写汇编/内联 SIMD（AVX2/AVX-512），Numcpp 仅依赖编译器自动向量化
- **缓存优化**：OpenBLAS 的分块打包策略经高度调优，远超通用 64×64 分块
- **多线程效率**：OpenBLAS 线程调度针对 NUMA 拓扑优化

## 依赖要求

- C++17 或更高（C++20 可启用 `auto` lambda 参数）
- OpenCV（可选，仅 `opencv.hpp` 需要）
- `algos/` 目录为独立可选模块，按需引入

## 开源协议

本项目使用MIT开源协议