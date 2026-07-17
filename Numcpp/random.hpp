#ifndef NUMCPP_RANDOM_HPP
#define NUMCPP_RANDOM_HPP

#include "core.hpp"

namespace np
{
    struct GaussianConfig
    {
        double mean = 0.0;
        double stddev = 1.0;
        unsigned int seed = 0;
    };

    template <typename T>
    class BoxMullerGenerator
    {
    private:
        std::mt19937 generator;
        std::uniform_real_distribution<T> uniform;
        T z0, z1;
        bool hasSpare = false;

    public:
        BoxMullerGenerator(unsigned int seed = 0)
        {
            if (seed == 0) { std::random_device rd; seed = rd(); }
            generator.seed(seed);
            uniform = std::uniform_real_distribution<T>(0.0, 1.0);
        }

        T generate(T mean = 0.0, T stddev = 1.0)
        {
            if (hasSpare) { hasSpare = false; return z1 * stddev + mean; }
            T u, v, s;
            do { u = uniform(generator) * 2.0f - 1.0f; v = uniform(generator) * 2.0f - 1.0f; s = u * u + v * v; }
            while (s >= 1.0 || s == 0.0);
            T mul = std::sqrt(-2.0 * std::log(s) / s);
            z0 = u * mul; z1 = v * mul; hasSpare = true;
            return z0 * stddev + mean;
        }
    };

    template <typename T>
    class StandardGaussianGenerator
    {
    private:
        std::mt19937 generator;
        std::normal_distribution<T> normal;

    public:
        StandardGaussianGenerator(unsigned int seed = 0) : normal(0.0, 1.0)
        {
            if (seed == 0) { std::random_device rd; seed = rd(); }
            generator.seed(seed);
        }

        T generate(T mean = 0.0, T stddev = 1.0) { return normal(generator) * stddev + mean; }
    };

    template <typename T>
    Numcpp<T> randn(size_t rows, size_t cols,
                    const GaussianConfig &config = GaussianConfig(),
                    bool useBoxMuller = false)
    {
        if (rows == 0 || cols == 0)
            throw std::invalid_argument("Matrix dimensions must be positive.");
        Numcpp<T> result(rows, cols);
        if (useBoxMuller)
        {
            BoxMullerGenerator<T> generator(config.seed);
            for (size_t i = 0; i < rows; i++)
                for (size_t j = 0; j < cols; j++)
                    result[i][j] = generator.generate(static_cast<T>(config.mean), static_cast<T>(config.stddev));
        }
        else
        {
            StandardGaussianGenerator<T> generator(config.seed);
            for (size_t i = 0; i < rows; i++)
                for (size_t j = 0; j < cols; j++)
                    result[i][j] = generator.generate(static_cast<T>(config.mean), static_cast<T>(config.stddev));
        }
        return result;
    }

    template <typename T>
    Numcpp<T> randn_parallel(size_t rows, size_t cols,
                             const GaussianConfig &config = GaussianConfig(),
                             size_t thread_count = 4)
    {
        if (rows == 0 || cols == 0)
            throw std::invalid_argument("Matrix dimensions must be positive.");
        Numcpp<T> result(rows, cols);
        result.optimized(true);
        result.maxprocs_set(thread_count);
        std::vector<StandardGaussianGenerator<T>> generators;
        for (size_t i = 0; i < thread_count; i++)
            generators.emplace_back(config.seed + static_cast<unsigned int>(i));
        units::thread_worker<T>(result.matrix, rows, cols, thread_count,
            [&](T **mat, size_t i, size_t j) {
                size_t tid = (i * cols + j) % thread_count;
                mat[i][j] = generators[tid].generate(static_cast<T>(config.mean), static_cast<T>(config.stddev));
            });
        return result;
    }

    template <typename T>
    Numcpp<T> cholesky_decomposition(const Numcpp<T> &A)
    {
        if (A.row != A.col) throw std::invalid_argument("Matrix must be square for Cholesky decomposition.");
        size_t n = A.row;
        Numcpp<T> L(n, n, static_cast<T>(0));
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j <= i; j++)
            {
                T sum = static_cast<T>(0);
                if (j == i)
                {
                    for (size_t k = 0; k < j; k++) sum += L[j][k] * L[j][k];
                    L[j][j] = std::sqrt(A[j][j] - sum);
                }
                else
                {
                    for (size_t k = 0; k < j; k++) sum += L[i][k] * L[j][k];
                    L[i][j] = (A[i][j] - sum) / L[j][j];
                }
            }
        }
        return L;
    }

    template <typename T>
    Numcpp<T> multivariate_randn(size_t n_samples, const Numcpp<T> &covariance,
                                 const Numcpp<T> &mean = Numcpp<T>())
    {
        if (covariance.row != covariance.col)
            throw std::invalid_argument("Covariance matrix must be square.");
        size_t n_features = covariance.row;
        Numcpp<T> mean_vector;
        if (mean.row == 0)
            mean_vector = Numcpp<T>(1, n_features, static_cast<T>(0));
        else if (mean.row == 1 && mean.col == n_features)
            mean_vector = mean;
        else
            throw std::invalid_argument("Mean must be a 1 x n_features vector.");
        Numcpp<T> L = cholesky_decomposition(covariance);
        Numcpp<T> Z = randn<T>(n_samples, n_features);
        Numcpp<T> X = Z * L.transpose();
        for (size_t i = 0; i < n_samples; i++)
            for (size_t j = 0; j < n_features; j++)
                X[i][j] += mean_vector[0][j];
        return X;
    }

    template <typename T>
    void validate_gaussian(const Numcpp<T> &matrix,
                           T expected_mean = static_cast<T>(0),
                           T expected_stddev = static_cast<T>(1),
                           T tolerance = static_cast<T>(0.1))
    {
        T sum = static_cast<T>(0), sum_sq = static_cast<T>(0);
        size_t total_elements = matrix.row * matrix.col;
        for (size_t i = 0; i < matrix.row; i++)
            for (size_t j = 0; j < matrix.col; j++)
            {
                sum += matrix[i][j];
                sum_sq += matrix[i][j] * matrix[i][j];
            }
        T m = sum / static_cast<T>(total_elements);
        T variance = (sum_sq / static_cast<T>(total_elements)) - (m * m);
        T stddev = std::sqrt(variance);
        std::cout << "Gaussian validation:\n"
                  << "Samples: " << total_elements << "\n"
                  << "Computed mean: " << m << " (expected: " << expected_mean << ")\n"
                  << "Computed stddev: " << stddev << " (expected: " << expected_stddev << ")\n";
        if (std::abs(m - expected_mean) < tolerance && std::abs(stddev - expected_stddev) < tolerance)
            std::cout << "OK: within tolerance\n";
        else
            std::cout << "FAIL: outside tolerance\n";
        std::cout << std::endl;
    }

    template <typename T>
    Numcpp<T> gaussian_mixture(size_t rows, size_t cols,
                               const std::vector<GaussianConfig> &components,
                               const std::vector<T> &weights = {})
    {
        if (components.empty()) throw std::invalid_argument("At least one Gaussian component required.");
        std::vector<T> actual_weights = weights;
        if (actual_weights.empty())
            actual_weights = std::vector<T>(components.size(), static_cast<T>(1.0 / components.size()));
        if (components.size() != actual_weights.size())
            throw std::invalid_argument("Number of components and weights must match.");
        Numcpp<T> result(rows, cols);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> component_selector(actual_weights.begin(), actual_weights.end());
        std::vector<StandardGaussianGenerator<T>> generators;
        for (const auto &config : components) generators.emplace_back(config.seed);
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
            {
                int component = component_selector(gen);
                const auto &config = components[component];
                result[i][j] = generators[component].generate(
                    static_cast<T>(config.mean), static_cast<T>(config.stddev));
            }
        return result;
    }

} // namespace np

#endif // NUMCPP_RANDOM_HPP
