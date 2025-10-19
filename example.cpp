#include <iostream>
#include <math.h>
#include <complex>
#include "Numcpp.hpp"

using namespace np;

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
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
    return 0;
}
