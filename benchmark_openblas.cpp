#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <cassert>

#include "Numcpp/Numcpp.hpp"
#include "Numcpp/algos/algos.hpp"

#include <openblas/cblas.h>

using namespace np;

int main()
{
    using Clock = std::chrono::high_resolution_clock;
    using Ms = std::chrono::milliseconds;
    using Us = std::chrono::microseconds;

    const unsigned int cores = std::thread::hardware_concurrency();

    std::cout << "OpenBLAS Config: " << OPENBLAS_VERSION << std::endl;
    std::cout << "CPU cores: " << cores << std::endl;
    openblas_set_num_threads(cores);

    auto to_millis = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };

    struct Result {
        int N;
        double t_naive;
        double t_cpp_blocked;
        double t_cpp_parallel;
        double t_openblas;
        double gflops_naive;
        double gflops_cpp_blk;
        double gflops_cpp_par;
        double gflops_blas;
    };

    std::vector<Result> results;
    int warmup = 3;
    int iters = 5;

    for (int N : {128, 256, 512, 1024, 2048})
    {
        Numcpp<double> A(N, N), B(N, N);
        for (size_t i = 0; i < A.row; i++)
            for (size_t j = 0; j < A.col; j++)
            {
                A[i][j] = (double)((i * 37 + j * 53) % 100) / 10.0;
                B[i][j] = (double)((i * 71 + j * 13) % 100) / 10.0;
            }

        // reference result (blocked serial for correctness)
        auto C_ref = blocked_multiply(A, B, 64, 0);

        double t1 = 0, t2 = 0, t3 = 0, t4 = 0;

        // 1. naive triple-loop
        if (N <= 1024)
        {
            for (int w = 0; w < warmup; w++)
            {
                Numcpp<double> R(N, N, 0.0);
                units::mm_generate(A.matrix, B.matrix, R.matrix, N, N, N, N, 0, 0, 0, 0);
            }
            for (int r = 0; r < iters; r++)
            {
                Numcpp<double> R(N, N, 0.0);
                auto t0 = Clock::now();
                units::mm_generate(A.matrix, B.matrix, R.matrix, N, N, N, N, 0, 0, 0, 0);
                t1 += to_millis(Clock::now() - t0);
            }
            t1 /= iters;
        }

        // 2. C++ blocked serial (threads=0)
        for (int w = 0; w < warmup; w++)
            blocked_multiply(A, B, 64, 0);
        for (int r = 0; r < iters; r++)
        {
            auto t0 = Clock::now();
            auto C = blocked_multiply(A, B, 64, 0);
            t2 += to_millis(Clock::now() - t0);
        }
        t2 /= iters;

        // 3. C++ blocked parallel (threads=cores)
        for (int w = 0; w < warmup; w++)
            blocked_multiply(A, B, 64, cores);
        for (int r = 0; r < iters; r++)
        {
            auto t0 = Clock::now();
            auto C = blocked_multiply(A, B, 64, cores);
            t3 += to_millis(Clock::now() - t0);
        }
        t3 /= iters;

        // 4. OpenBLAS cblas_dgemm
        std::vector<double> A_flat(N * N), B_flat(N * N), C_flat(N * N);
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
            {
                A_flat[j * N + i] = A[i][j]; // col-major
                B_flat[j * N + i] = B[i][j];
            }

        for (int w = 0; w < warmup; w++)
        {
            std::fill(C_flat.begin(), C_flat.end(), 0.0);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        N, N, N, 1.0, A_flat.data(), N, B_flat.data(), N,
                        0.0, C_flat.data(), N);
        }
        for (int r = 0; r < iters; r++)
        {
            std::fill(C_flat.begin(), C_flat.end(), 0.0);
            auto t0 = Clock::now();
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        N, N, N, 1.0, A_flat.data(), N, B_flat.data(), N,
                        0.0, C_flat.data(), N);
            t4 += to_millis(Clock::now() - t0);
        }
        t4 /= iters;

        // verify OpenBLAS result
        double maxdiff = 0;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                maxdiff = std::max(maxdiff, std::abs(C_flat[j * N + i] - C_ref[i][j]));
        bool ok = maxdiff < 1e-8;

        double flops = 2.0 * N * N * N;
        double gflop = 1e9;

        results.push_back({
            N,
            t1,
            t2, t3, t4,
            t1 > 0 ? (flops / (t1 * 1e-3)) / gflop : 0,
            t2 > 0 ? (flops / (t2 * 1e-3)) / gflop : 0,
            t3 > 0 ? (flops / (t3 * 1e-3)) / gflop : 0,
            t4 > 0 ? (flops / (t4 * 1e-3)) / gflop : 0
        });

        std::string status = ok ? "OK" : "FAIL";
        if (N > 1024)
            std::cout << "N=" << N << " prepared. (max diff=" << maxdiff << " " << status << ")" << std::endl;
        else
            std::cout << "N=" << N << " done. (max diff=" << maxdiff << " " << status << ")" << std::endl;
    }

    // ---- Print results ----
    std::cout << std::endl;
    std::cout << std::left
              << std::setw(8)  << "N"
              << std::setw(14) << "Naive(ms)"
              << std::setw(14) << "Blk-Ser(ms)"
              << std::setw(14) << "Blk-Par(ms)"
              << std::setw(14) << "OpenBLAS(ms)"
              << std::setw(10) << "BLAS/Par"
              << std::endl;
    std::cout << std::string(74, '=') << std::endl;

    for (auto &r : results)
    {
        std::cout << std::left << std::fixed
                  << std::setw(8)  << r.N
                  << std::setprecision(3)
                  << std::setw(14) << (r.t_naive > 0 ? std::to_string(r.t_naive) : "  --")
                  << std::setw(14) << r.t_cpp_blocked
                  << std::setw(14) << r.t_cpp_parallel
                  << std::setw(14) << r.t_openblas
                  << std::setprecision(2)
                  << std::setw(10) << (r.t_openblas > 0 ? std::to_string(r.t_cpp_parallel / r.t_openblas).substr(0, 4) + "x" : "  --")
                  << std::endl;
    }

    // ---- GFLOPS table ----
    std::cout << std::endl;
    std::cout << std::left
              << std::setw(8)  << "N"
              << std::setw(14) << "Naive GF/s"
              << std::setw(14) << "Blk-Ser GF/s"
              << std::setw(14) << "Blk-Par GF/s"
              << std::setw(14) << "OpenBLAS GF/s"
              << std::endl;
    std::cout << std::string(64, '=') << std::endl;

    for (auto &r : results)
    {
        std::cout << std::left << std::fixed
                  << std::setw(8)  << r.N
                  << std::setprecision(3)
                  << std::setw(14) << (r.gflops_naive > 0 ? std::to_string(r.gflops_naive) : "  --")
                  << std::setw(14) << r.gflops_cpp_blk
                  << std::setw(14) << r.gflops_cpp_par
                  << std::setw(14) << r.gflops_blas
                  << std::endl;
    }

    // ---- Speedup summary ----
    std::cout << std::endl;
    std::cout << "=== Summary: OpenBLAS speedup vs Numcpp best ===" << std::endl;
    std::cout << std::left
              << std::setw(8)  << "N"
              << std::setw(12) << "Cpp-Best"
              << std::setw(12) << "OpenBLAS"
              << std::setw(10) << "BLAS-x"
              << std::endl;
    std::cout << std::string(42, '-') << std::endl;

    for (auto &r : results)
    {
        double best_cpp = std::min(r.t_cpp_blocked, r.t_cpp_parallel);
        std::cout << std::left << std::fixed
                  << std::setw(8)  << r.N
                  << std::setprecision(3)
                  << std::setw(12) << best_cpp
                  << std::setw(12) << r.t_openblas
                  << std::setprecision(1)
                  << std::setw(10) << (best_cpp / r.t_openblas) << "x"
                  << std::endl;
    }

    return 0;
}
