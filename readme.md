# Numcpp

A header-only, template-based C++ linear algebra and matrix computation library.

## Features

- **Header-only** — `#include "Numcpp/Numcpp.hpp"` and you're ready
- **Template-based** — supports `float`, `double`, `int`, `std::complex`, and custom types
- **Multi-threading** — optional parallel execution via `optimized(true)` + `maxprocs_set(N)`
- **API** — covers matrix arithmetic, decompositions, FFT, random sampling, optimization, and OpenCV interop

## Quick Start

```cpp
#include <iostream>
#include "Numcpp/Numcpp.hpp"
using namespace np;

int main() {
    // Two ways to initialize
    Numcpp<double> a(2, 3, 1.0);               // 2x3, all ones
    auto b = (Numcpp<double>(2, 2) << 1, 2,     // 2x2
                                          3, 4);

    // Arithmetic
    auto c = a + b;         // matrix addition
    auto d = a * 2.0;       // scalar multiplication
    auto e = a.transpose(); // transpose
    auto f = a.Hadamard(a); // element-wise product

    // Matrix multiply
    auto g = (Numcpp<double>(2, 3) << 1, 2, 3, 4, 5, 6)
           * (Numcpp<double>(3, 2) << 7, 8, 9, 10, 11, 12);

    // Decompositions
    double det = b.determinant();
    auto inv = b.inverse();

    // SVD
    Numcpp<double> U, S, V;
    a.svd(U, S, V);

    // Access elements
    double x = a[0][1];

    // Print
    std::cout << a << std::endl;

    return 0;
}
```

Compile with C++17 or later:

```bash
g++ -std=c++17 -O2 main.cpp -o main
```

## API Reference

### Construction & Assignment

| Expression | Description |
|---|---|
| `Numcpp<T> M(rows, cols)` | Zero-initialized matrix |
| `Numcpp<T> M(rows, cols, val)` | Matrix filled with `val` |
| `Numcpp<T> M(T**, rows, cols)` | From raw 2D array (copies data) |
| `Numcpp<T> M(T*, rows, cols)` | From flat array (copies data) |
| `Numcpp<T> M("file.mat")` | Load from binary file |
| `M << v1, v2, v3, ...` | Stream initializer (row-major) |
| `M = other` | Deep copy |

### Element Access

| Expression | Description |
|---|---|
| `M[i][j]` | Access element at row `i`, column `j` |
| `M.srow(i)` | Extract single row as 1xN matrix |
| `M.scol(j)` | Extract single column as Mx1 matrix |

### Matrix Arithmetic

| Expression | Description |
|---|---|
| `A + B`, `A += B` | Element-wise addition |
| `A - B`, `A -= B` | Element-wise subtraction |
| `A * B` | Matrix multiplication |
| `A * s`, `s * A`, `A *= s` | Scalar multiplication |
| `A / s`, `A /= s` | Scalar division |
| `A + s`, `A += s` | Add scalar to each element |
| `A - s`, `A -= s` | Subtract scalar from each element |
| `A.Hadamard(B)` | Element-wise (Hadamard) product |
| `A.Hadamard_self(B)` | In-place Hadamard product |

### Matrix Properties & Transforms

| Expression | Description |
|---|---|
| `M.transpose()` | Return transposed copy |
| `M.transposed()` | Transpose in-place |
| `M.sum()` | Sum of all elements |
| `M.norm(type)` | Norm: `L1`, `L2`, or `INF` |
| `M.dot(v)` | Dot product of two vectors |
| `M.is_vector()` | Check if matrix is a vector |
| `M.size()` | `row * col` |
| `M.is_symmetric()` | Check symmetry |
| `M.set_identity()` | Set to identity matrix (must be square) |
| `M.zero_approximation(tol)` | Set values below tolerance to zero |

### Linear Algebra

| Expression | Description |
|---|---|
| `M.determinant()` | Determinant (square matrices) |
| `M.inverse()` | Matrix inverse (square, non-singular) |
| `M.pseudoinverse()` | Moore-Penrose pseudoinverse |
| `M.eig(max_iter, tol)` | Eigen-decomposition (symmetric) — returns `{eigenvalues, eigenvectors, diagonal}` |
| `M.svd(U, S, V)` | SVD via eigen-decomposition — returns `U * S * V^T = M` |

### FFT

```cpp
// Complex FFT
Numcpp<std::complex<double>> c(1, 8);
auto freq = c.fft(1);   // forward FFT
auto time = c.fft(-1);  // inverse FFT

// Real-to-complex FFT
Numcpp<double> r(1, 8);
auto freq_cpx = r.fft(1);   // returns Numcpp<std::complex<double>>
```

### Type Conversion

```cpp
Numcpp<int> a(2, 2, 42);
auto f = a.as<float>();    // Numcpp<float>
auto d = a.as<double>();   // Numcpp<double>
auto c = a.as<std::complex<double>>();  // converts to complex type
```

### Special Function Multiply (Lambda Map)

```cpp
// Element-wise lambda
auto r = a < [](double x, double y) { return x * 2; } > nullptr;

// Matrix-multiply style with bilateral function
auto r2 = a < [](double x, double y) { return x + y; } > b;
```

### File I/O

```cpp
M.save("matrix.mat");
auto M2 = load<double>("matrix.mat");
```

### Random & Distributions

```cpp
// Standard normal
auto g = randn<double>(100, 100);

// Configured normal
GaussianConfig cfg{ .mean = 3.0, .stddev = 0.5, .seed = 42 };
auto gc = randn<double>(50, 50, cfg);

// Box-Muller generator
auto bm = randn<double>(100, 100, cfg, true);

// Parallel random
auto gp = randn_parallel<double>(100, 100, cfg, 8);

// Multivariate normal
Numcpp<double> cov = (Numcpp<double>(3, 3) << 4, 1, 1, 1, 3, 2, 1, 2, 5);
Numcpp<double> mean(1, 3, 0.0);
auto samples = multivariate_randn(1000, cov, mean);

// Gaussian mixture
std::vector<GaussianConfig> components = {
    { .mean =  -2.0, .stddev = 0.5, .seed = 1 },
    { .mean =   2.0, .stddev = 0.5, .seed = 2 }
};
auto mixture = gaussian_mixture<double>(500, 500, components);
```

### Cholesky Decomposition

```cpp
Numcpp<double> A = (Numcpp<double>(3, 3) << 4, 1, 1, 1, 3, 2, 1, 2, 5);
auto L = cholesky_decomposition(A);  // L * L^T = A
```

### Optimization

**LQR (Linear Quadratic Regulator)**

```cpp
auto [K, P] = solve_lqr(A, B, Q, R, max_iter, tolerance);
// Returns optimal gain K and cost-to-go P
```

**QP (Quadratic Programming)**

```cpp
// minimize   0.5 * x^T * Q * x + c^T * x
// subject to A * x <= b,  E * x == d
auto x = solve_QP(Q, c, A, b, E, d);
```

### OpenCV Interop

`#include "Numcpp/opencv.hpp"`

```cpp
cv::Mat cvMat = ...;
auto numMat = fromCvMat<double>(cvMat);
auto back = toCvMat(numMat);
```

### Multi-threading

A global thread pool (lazy-init, RAII-managed) is used internally. Workers persist across operations, avoiding per-call thread creation overhead.

```cpp
auto M = Numcpp<double>(1000, 1000);
M.optimized(true);          // enable parallel execution
M.maxprocs_set(8);          // set thread count
// All subsequent operations use the thread pool
```

### Utility

```cpp
auto bin = binarizeMatrix(A, 0.5);  // threshold binarization
```

### Algorithm Library

`#include "Numcpp/algos/algos.hpp"` (or include individual headers)

```cpp
#include "Numcpp/algos/algos.hpp"

// Coppersmith-Winograd — O(n^2.807) recursive, best serial perf at N >= 128
auto C = cw_multiply(A, B);

// Cache-blocked — tiled multiplication, optional thread parallelism
auto D = blocked_multiply(A, B, 64, 0);   // serial, 64×64 tiles
auto E = blocked_multiply(A, B, 64, 8);   // parallel, 8 threads
```

## File Structure

```
Numcpp/
├── Numcpp.hpp          Main include header
├── core.hpp            Core matrix class, arithmetic, linear algebra, FFT
├── random.hpp          Gaussian random, Cholesky, multivariate, mixtures
├── optim.hpp           LQR and QP solvers
├── opencv.hpp          OpenCV cv::Mat conversion
└── algos/
    ├── algos.hpp       Single include for all algorithms
    ├── cw_mmul.hpp     Coppersmith-Winograd multiplication
    └── blocked_mmul.hpp  Cache-blocked multiplication
```

## Benchmark

Tests run on a 20-core CPU with `example.cpp`. All times averaged over multiple iterations.

### Element-wise operations (1500×1500, 2.25M elements)

| Operation | 1-core | 4-core | Speedup | 20-core | Speedup |
|---|---|---|---|---|---|
| construct | 7.7ms | 7.4ms | 1.05x | 7.0ms | 1.10x |
| operator+= | 16.4ms | 17.4ms | 0.94x | 15.4ms | 1.07x |
| operator*= | 7.4ms | 10.1ms | 0.73x | 9.7ms | 0.76x |
| Hadamard | 18.1ms | 19.1ms | 0.95x | 16.4ms | 1.10x |
| transpose | 20.7ms | 20.4ms | 1.02x | 17.4ms | 1.19x |
| copy ctor | 14.1ms | 13.4ms | 1.05x | 12.7ms | 1.11x |
| sum | 7.0ms | 7.4ms | 0.95x | 6.4ms | 1.11x |

Element-wise operations are memory-bandwidth bound — threading provides marginal gains for O(n) operations on matrices of this size.

### Matrix multiplication (400×400)

| Operation | 1-core | 4-core | Speedup | 20-core | Speedup |
|---|---|---|---|---|---|
| A × B | 47.0ms | 13.4ms | 3.52x | 7.4ms | 6.41x |

### Algorithm Comparison (square N×N power-of-2, 20 threads, block=64)

| N | Naive | CW | Blk-serial | Blk-par | CW/Nai | Blk/Nai | BPar/Nai |
|---|---|---|---|---|---|---|---|
| 64 | 119μs | 118μs | 157μs | 342μs | 1.00x | 0.76x | 0.35x |
| 128 | 1,230μs | 1,162μs | 1,476μs | 1,277μs | 1.05x | 0.83x | 0.96x |
| 256 | 11,236μs | 9,311μs | 10,669μs | 7,355μs | 1.20x | 1.05x | 1.53x |
| 512 | 95,866μs | 68,453μs | 81,442μs | 29,601μs | 1.40x | 1.18x | 3.24x |
| 1024 | 1,184ms | — | 1,278ms | 242ms | — | 0.93x | 4.88x |

- **CW** (`cw_multiply`): Best serial algorithm, 1.05–1.40x over naive. Strengthens with N.
- **Blocked-serial** (`blocked_multiply` with threads=0): Cache-friendly tiling, modest gains from N≥256.
- **Blocked-parallel** (`blocked_multiply` with threads=N): Combines cache blocking + thread pool. 3.24x at N=512, 4.88x at N=1024.

## Requirements

- C++17 or later (C++20 for `auto` lambda parameters)
- OpenCV (optional, for `opencv.hpp`)

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
