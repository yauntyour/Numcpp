#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <cassert>
#include <chrono>
#include <string>
#include "Numcpp/Numcpp.hpp"
#include "Numcpp/algos/algos.hpp"

using namespace np;

typedef std::complex<double> cd;
cd sigmoid(cd x, cd) { return cd(1, 0) / (cd(1, 0) + exp(-x)); }

int main()
{
    int err = 0;
    auto check = [&](bool ok, const char *name) {
        std::cout << (ok ? "  PASS " : "  FAIL ") << name << std::endl;
        if (!ok) err++;
    };

    std::cout << "=== Numcpp Tests ===\n" << std::endl;

    // ---- 1. construct / assign ----
    {
        Numcpp<double> a(2, 3, 5.0);
        check(a.row == 2 && a.col == 3, "construct(row,col,val)");
        check(a[0][0] == 5.0 && a[1][2] == 5.0, "all elements = init value");

        Numcpp<double> b(2, 3);
        check(b.row == 2 && b.col == 3, "construct(row,col) default");
        b = a;
        check(b[0][0] == 5.0, "operator=");
    }

    // ---- 2. scalar arithmetic ----
    {
        Numcpp<double> a(2, 2, 10.0);
        auto m = a * 3.0;
        check(m[0][0] == 30.0, "scalar multiply");
        auto d = a / 2.0;
        check(std::abs(d[0][0] - 5.0) < 1e-10, "scalar divide");
        auto p = a + 1.0;
        check(p[0][0] == 11.0, "scalar add");
        auto s = a - 2.0;
        check(s[0][0] == 8.0, "scalar subtract");
    }

    // ---- 3. matrix arithmetic ----
    {
        Numcpp<double> a = (Numcpp<double>(2, 2) << 1, 2, 3, 4);
        Numcpp<double> b = (Numcpp<double>(2, 2) << 5, 6, 7, 8);
        auto sum = a + b;
        check(sum[0][0] == 6 && sum[1][1] == 12, "matrix add");
        auto diff = a - b;
        check(diff[0][0] == -4 && diff[1][1] == -4, "matrix subtract");
    }

    // ---- 4. matrix multiply ----
    {
        Numcpp<double> a = (Numcpp<double>(2, 3) << 1, 2, 3, 4, 5, 6);
        Numcpp<double> b = (Numcpp<double>(3, 2) << 7, 8, 9, 10, 11, 12);
        auto c = a * b;
        check(c.row == 2 && c.col == 2, "mm row/col");
        check(c[0][0] == 58 && c[0][1] == 64, "mm values");  // 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
    }

    // ---- 5. Hadamard / transpose ----
    {
        Numcpp<double> a = (Numcpp<double>(2, 2) << 1, 2, 3, 4);
        auto h = a.Hadamard(a);
        check(h[0][0] == 1 && h[1][1] == 16, "Hadamard");
        auto t = a.transpose();
        check(t[0][0] == 1 && t[0][1] == 3 && t[1][0] == 2, "transpose");
        a.transposed();
        check(a[0][0] == 1 && a[0][1] == 3, "transposed in-place");
    }

    // ---- 6. row / col / index / sum ----
    {
        Numcpp<double> a = (Numcpp<double>(2, 2) << 1, 2, 3, 4);
        check(a.srow(0)[0][0] == 1 && a.srow(0)[0][1] == 2, "srow");
        check(a.scol(1)[0][0] == 2 && a.scol(1)[1][0] == 4, "scol");
        check(a.sum() == 10.0, "sum");
    }

    // ---- 7. determinant / inverse ----
    {
        Numcpp<double> a = (Numcpp<double>(3, 3) << 4, 1, 1, 1, 3, 2, 1, 2, 5);
        double det = a.determinant();
        check(std::abs(det - 40.0) < 1e-10, "determinant");
        auto inv = a.inverse();
        auto id = a * inv;
        check(std::abs(id[0][0] - 1.0) < 1e-10 && std::abs(id[0][1]) < 1e-10, "inverse * A = I");
    }

    // ---- 8. pseudoinverse ----
    {
        Numcpp<double> a = (Numcpp<double>(4, 3) << 4, 1, 1, 1, 3, 2, 1, 2, 5, 5, 1, 1);
        auto pinv = a.pseudoinverse();
        check(pinv.row == 3 && pinv.col == 4, "pseudoinverse dims");
    }

    // ---- 9. SVD ----
    {
        Numcpp<double> a = (Numcpp<double>(4, 3) << 4, 1, 1, 1, 3, 2, 1, 2, 5, 5, 1, 1);
        Numcpp<double> U, S, V;
        a.svd(U, S, V);
        auto rebuilt = U * S * V.transpose();
        double maxdiff = 0;
        for (size_t i = 0; i < a.row; i++)
            for (size_t j = 0; j < a.col; j++)
                maxdiff = std::max(maxdiff, std::abs(a[i][j] - rebuilt[i][j]));
        check(maxdiff < 1e-10, "SVD rebuild");
    }

    // ---- 10. eigenvalues (symmetric) ----
    {
        Numcpp<double> a = (Numcpp<double>(3, 3) << 4, 1, 1, 1, 3, 2, 1, 2, 5);
        auto r = a.eig();
        check(r[0].col == 3, "eigenvalues count");
    }

    // ---- 11. norm / dot ----
    {
        Numcpp<double> v(3, 1, 3.0);
        check(std::abs(v.norm(L2) - std::sqrt(27.0)) < 1e-10, "L2 norm");
        Numcpp<double> u(3, 1, 2.0);
        check(v.dot(u) == 18.0, "dot product");
    }

    // ---- 12. FFT complex ----
    {
        Numcpp<cd> a(1, 4);
        a << cd(1, 0), cd(1, 0), cd(1, 0), cd(1, 0);
        auto f = a.fft(1);
        check(std::abs(f[0][0] - cd(4, 0)) < 1e-10, "FFT DC bin");
        auto g = f.fft(-1);
        check(std::abs(g[0][0] - cd(1, 0)) < 1e-10, "IFFT round-trip");
    }

    // ---- 13. FFT real -> complex ----
    {
        Numcpp<double> a(1, 4);
        a << 1, 1, 1, 1;
        auto f = a.fft(1);
        check(std::abs(f[0][0].real() - 4.0) < 1e-10, "real FFT DC bin");
    }

    // ---- 14. type conversion (.as<>) ----
    {
        Numcpp<int> a(2, 2, 42);
        auto f = a.as<float>();
        check(std::abs(f[0][0] - 42.0f) < 1e-6, "int->float");
        auto d = f.as<double>();
        check(std::abs(d[0][0] - 42.0) < 1e-10, "float->double");
        auto i8 = d.as<int8_t>();
        check(i8[0][0] == 42, "double->int8");
        auto cpx = a.as<cd>();
        check(std::abs(cpx[0][0].real() - 42.0) < 1e-10, "int->complex");
    }

    // ---- 15. stream init and save/load ----
    {
        Numcpp<double> a(2, 2);
        a << 1, 2, 3, 4;
        check(a[0][0] == 1 && a[1][1] == 4, "stream << init");
        a.save("_test.mat");
        auto b = load<double>("_test.mat");
        check(b[0][0] == 1 && b[1][1] == 4, "save/load");
        remove("_test.mat");
    }

    // ---- 16. special multiply (<func>) ----
    {
        Numcpp<double> a = (Numcpp<double>(2, 2) << 1, 2, 3, 4);
        auto r = a<[](double x, double y) { return x * 2; }> nullptr;
        check(r[0][0] == 2 && r[1][1] == 8, "lambda map");
    }

    // ---- 17. Gaussian random ----
    {
        auto g = randn<double>(10, 10);
        check(g.row == 10 && g.col == 10, "randn");
        GaussianConfig cfg{.mean = 3.0, .stddev = 0.5, .seed = 42};
        auto gc = randn<double>(6, 6, cfg);
        check(gc.row == 6 && gc.col == 6, "randn with config");
    }

    // ---- 18. Cholesky decomposition ----
    {
        Numcpp<double> a = (Numcpp<double>(2, 2) << 4, 1, 1, 3);
        auto L = cholesky_decomposition(a);
        check(std::abs(L[0][0] - 2.0) < 1e-10, "Cholesky L[0][0]");
        auto re = L * L.transpose();
        check(std::abs(re[0][0] - a[0][0]) < 1e-10, "Cholesky L*L^T = A");
    }

    // ---- 19. is_symmetric / identity / zero_approx ----
    {
        Numcpp<double> a = (Numcpp<double>(3, 3) << 4, 1, 1, 1, 3, 2, 1, 2, 5);
        check(a.is_symmetric(), "is_symmetric");
        a.set_identity();
        check(a[0][0] == 1 && a[0][1] == 0, "set_identity");
    }

    // ---- 20. binarize ----
    {
        Numcpp<double> a = (Numcpp<double>(2, 2) << 0.3, 0.7, 0.5, 0.9);
        auto b = binarizeMatrix(a, 0.5);
        check(b[0][0] == 0 && b[1][0] == 1, "binarize");
    }

    // ---- 21. LDL decomposition (inverse) ----
    //  (already tested above)

    // ==================== Performance Benchmark ====================
    {
        using Clock = std::chrono::high_resolution_clock;
        using Ms = std::chrono::milliseconds;
        const unsigned int cores = std::thread::hardware_concurrency();
        const unsigned int threads4 = std::min(4u, cores);
        const int N_elem = 1500;
        const int N_mmul = 400;
        const int N_iters = 3;

        std::cout << "\n=== Performance Benchmark (CPU cores: " << cores
                  << ", matrix sizes: " << N_elem << "x" << N_elem << " / " << N_mmul << "x" << N_mmul << ") ===\n"
                  << std::endl;

        struct Entry {
            std::string name;
            double single;
            double t4;
            double t4_sp;
            double tmax;
            double tmax_sp;
        };
        std::vector<Entry> entries;

        auto timeit = [&](std::function<void()> fn) {
            auto t0 = Clock::now();
            for (int r = 0; r < N_iters; r++) fn();
            return (double)std::chrono::duration_cast<Ms>(Clock::now() - t0).count() / N_iters;
        };

        auto add_bench = [&](const std::string &name,
                             std::function<void()> single,
                             std::function<void(unsigned)> multi) {
            double s = timeit(single);
            double t4 = timeit([&] { multi(threads4); });
            double tm = timeit([&] { multi(cores); });
            entries.push_back({name, s, t4, s / t4, tm, s / tm});
        };

        // 1. construct
        add_bench("construct(row,col,val)",
            [&] { Numcpp<double> a(N_elem, N_elem, 1.0); },
            [&](unsigned th) {
                Numcpp<double> a(N_elem, N_elem, 1.0);
                a.optimized(true); a.maxprocs_set(th);
            });

        // 2. operator+=
        add_bench("operator+=",
            [&] {
                Numcpp<double> a(N_elem, N_elem, 2.0), b(N_elem, N_elem, 3.0);
                a += b;
            },
            [&](unsigned th) {
                Numcpp<double> a(N_elem, N_elem, 2.0), b(N_elem, N_elem, 3.0);
                a.optimized(true); a.maxprocs_set(th);
                b.optimized(true); b.maxprocs_set(th);
                a += b;
            });

        // 3. operator*=
        add_bench("operator*=",
            [&] { Numcpp<double> a(N_elem, N_elem, 2.0); a *= 3.0; },
            [&](unsigned th) {
                Numcpp<double> a(N_elem, N_elem, 2.0);
                a.optimized(true); a.maxprocs_set(th); a *= 3.0;
            });

        // 4. Hadamard
        add_bench("Hadamard",
            [&] {
                Numcpp<double> a(N_elem, N_elem, 2.0), b(N_elem, N_elem, 3.0);
                a.Hadamard_self(b);
            },
            [&](unsigned th) {
                Numcpp<double> a(N_elem, N_elem, 2.0), b(N_elem, N_elem, 3.0);
                a.optimized(true); a.maxprocs_set(th);
                a.Hadamard_self(b);
            });

        // 5. transpose
        add_bench("transpose",
            [&] {
                Numcpp<double> a(N_elem, N_elem, 1.0);
                for (size_t i = 0; i < a.row; i++)
                    for (size_t j = 0; j < a.col; j++)
                        a[i][j] = (double)(i * a.col + j);
                auto t = a.transpose();
            },
            [&](unsigned th) {
                Numcpp<double> a(N_elem, N_elem, 1.0);
                a.optimized(true); a.maxprocs_set(th);
                for (size_t i = 0; i < a.row; i++)
                    for (size_t j = 0; j < a.col; j++)
                        a[i][j] = (double)(i * a.col + j);
                auto t = a.transpose();
            });

        // 6. matrix multiply
        add_bench("A * B (mmul)",
            [&] {
                Numcpp<double> a(N_mmul, N_mmul, 1.0), b(N_mmul, N_mmul, 2.0);
                auto c = a * b;
            },
            [&](unsigned th) {
                Numcpp<double> a(N_mmul, N_mmul, 1.0), b(N_mmul, N_mmul, 2.0);
                a.optimized(true); a.maxprocs_set(th);
                b.optimized(true); b.maxprocs_set(th);
                auto c = a * b;
            });

        // 7. copy construct
        add_bench("copy construct",
            [&] {
                Numcpp<double> a(N_elem, N_elem, 1.0);
                Numcpp<double> b(a);
            },
            [&](unsigned th) {
                Numcpp<double> a(N_elem, N_elem, 1.0);
                a.optimized(true); a.maxprocs_set(th);
                Numcpp<double> b(a);
            });

        // 8. sum
        add_bench("sum",
            [&] {
                Numcpp<double> a(N_elem, N_elem, 1.0);
                volatile double s = a.sum(); (void)s;
            },
            [&](unsigned th) {
                Numcpp<double> a(N_elem, N_elem, 1.0);
                a.optimized(true); a.maxprocs_set(th);
                volatile double s = a.sum(); (void)s;
            });

        // ---- Print table ----
        std::cout << std::left
                  << std::setw(24) << "Operation"
                  << std::setw(10) << "1-core"
                  << std::setw(10) << "4-core"
                  << std::setw(7)  << "x"
                  << std::setw(10) << std::to_string(cores) + "-core"
                  << std::setw(7)  << "x" << std::endl;
        std::cout << std::string(68, '-') << std::endl;

        for (auto &e : entries)
        {
            std::cout << std::left << std::fixed << std::setprecision(1)
                      << std::setw(24) << e.name
                      << std::setw(10) << e.single + 0.05
                      << std::setw(10) << e.t4 + 0.05
                      << std::setw(6)  << std::setprecision(2) << e.t4_sp << "x"
                      << std::setw(10) << std::setprecision(1) << e.tmax + 0.05
                      << std::setw(6)  << std::setprecision(2) << e.tmax_sp << "x"
                      << std::endl;
        }
    }

    // ==================== Algorithm Comparison: Naive vs CW vs Blocked vs Parallel ====================
    {
        using Clock = std::chrono::high_resolution_clock;
        using Us = std::chrono::microseconds;
        const unsigned int th = std::thread::hardware_concurrency();

        std::cout << "\n=== Algorithm Comparison (square NxN power-of-2, " << th << " threads) ===\n"
                  << "    times in us, block size = 64\n" << std::endl;

        std::cout << std::left
                  << std::setw(8)  << "N"
                  << std::setw(12) << "Naive"
                  << std::setw(12) << "CW"
                  << std::setw(12) << "Blk-serial"
                  << std::setw(12) << "Blk-par"
                  << std::setw(10) << "CW/Nai"
                  << std::setw(10) << "Blk/Nai"
                  << std::setw(10) << "BPar/Nai"
                  << std::endl;
        std::cout << std::string(86, '-') << std::endl;

        for (int N : {64, 128, 256, 512, 1024})
        {
            Numcpp<double> A(N, N), B(N, N);
            for (size_t i = 0; i < A.row; i++)
                for (size_t j = 0; j < A.col; j++)
                {
                    A[i][j] = (double)((i * 37 + j * 53) % 100) / 10.0;
                    B[i][j] = (double)((i * 71 + j * 13) % 100) / 10.0;
                }

            auto timeit = [&](auto &&fn) {
                fn(); // warm up
                auto t0 = Clock::now();
                fn();
                return (double)std::chrono::duration_cast<Us>(Clock::now() - t0).count();
            };

            double t_naive = 0, t_cw = 0, t_blk = 0, t_blk_par = 0;

            if (N <= 512) {
                t_naive = timeit([&] {
                    Numcpp<double> R(N, N, 0.0);
                    units::mm_generate(A.matrix, B.matrix, R.matrix, N, N, N, N, 0, 0, 0, 0);
                });

                t_cw = timeit([&] {
                    auto C = cw_multiply(A, B);
                });
            } else {
                t_naive = timeit([&] {
                    auto C = A * B;
                });
                t_cw = 0; // CW too slow for 1024, skip
            }

            t_blk = timeit([&] {
                auto C = blocked_multiply(A, B, 64, 0);
            });

            t_blk_par = timeit([&] {
                auto C = blocked_multiply(A, B, 64, th);
            });

            std::cout << std::left << std::fixed
                      << std::setw(8)  << N
                      << std::setprecision(0)
                      << std::setw(12) << t_naive
                      << std::setw(12) << (t_cw > 0 ? std::to_string((long long)t_cw) : "  --")
                      << std::setw(12) << t_blk
                      << std::setw(12) << t_blk_par
                      << std::setprecision(2)
                      << std::setw(10) << (t_cw > 0 ? std::to_string(t_naive / t_cw).substr(0,4) + "x" : "  --")
                      << std::setw(10) << (t_naive / t_blk) << "x"
                      << std::setw(10) << (t_naive / t_blk_par) << "x"
                      << std::endl;
        }
    }

    std::cout << "\n=== " << (err ? "FAILED" : "ALL PASSED") << " (" << err << " failures) ===" << std::endl;
    return err;
}
