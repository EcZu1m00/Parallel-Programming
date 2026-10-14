#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

using namespace std;
namespace fs = std::filesystem;

using Matrix = vector<int64_t>;

struct ExperimentResult
{
    int size;
    int threads;
    double timeMs;
    double speedup;
    uint64_t operations;
};

Matrix generateMatrix(int n, uint32_t seed)
{
    mt19937 generator(seed);
    uniform_int_distribution<int> distribution(-9, 9);

    Matrix matrix(static_cast<size_t>(n) * n);

    for (auto& value : matrix)
    {
        value = distribution(generator);
    }

    return matrix;
}

void saveMatrix(const string& filename, const Matrix& matrix, int n)
{
    ofstream file(filename);

    if (!file)
    {
        throw runtime_error("Cannot create file: " + filename);
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (j > 0)
            {
                file << ' ';
            }

            file << matrix[static_cast<size_t>(i) * n + j];
        }

        file << '\n';
    }
}

Matrix loadMatrix(const string& filename, int n)
{
    ifstream file(filename);

    if (!file)
    {
        throw runtime_error("Cannot open file: " + filename);
    }

    Matrix matrix(static_cast<size_t>(n) * n);

    for (auto& value : matrix)
    {
        if (!(file >> value))
        {
            throw runtime_error("Invalid matrix data in: " + filename);
        }
    }

    return matrix;
}

void prepareInputFiles(int n)
{
    const string fileA = "data/matrixA_" + to_string(n) + ".txt";
    const string fileB = "data/matrixB_" + to_string(n) + ".txt";

    if (!fs::exists(fileA) || !fs::exists(fileB))
    {
        cout << "Generating input matrices for N = " << n << "...\n";

        Matrix A = generateMatrix(n, static_cast<uint32_t>(n));
        Matrix B = generateMatrix(n, static_cast<uint32_t>(n + 10000));

        saveMatrix(fileA, A, n);
        saveMatrix(fileB, B, n);
    }
}

Matrix multiplyMatrices(
    const Matrix& A,
    const Matrix& B,
    int n,
    int threadCount)
{
    Matrix C(static_cast<size_t>(n) * n, 0);

#pragma omp parallel for num_threads(threadCount) schedule(static)
    for (int i = 0; i < n; ++i)
    {
        const size_t row = static_cast<size_t>(i) * n;

        for (int k = 0; k < n; ++k)
        {
            const int64_t a = A[row + k];
            const size_t bRow = static_cast<size_t>(k) * n;

            for (int j = 0; j < n; ++j)
            {
                C[row + j] += a * B[bRow + j];
            }
        }
    }

    return C;
}

int main()
{
    try
    {
        fs::create_directories("data");
        fs::create_directories("results");

        const vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000 };
        const vector<int> threadCounts = { 1, 2, 4, 8 };

        omp_set_dynamic(0);

        ofstream csv("experiments.csv");

        if (!csv)
        {
            throw runtime_error("Cannot create experiments.csv");
        }

        csv << "size,threads,time_ms,operations,speedup\n";
        csv << fixed << setprecision(3);

        vector<ExperimentResult> results;

        cout << "OpenMP matrix multiplication experiments\n";
        cout << "Maximum available OpenMP threads: "
            << omp_get_max_threads() << "\n\n";

        for (int n : sizes)
        {
            cout << "========================================\n";
            cout << "Matrix size: " << n << " x " << n << '\n';

            prepareInputFiles(n);

            const Matrix A = loadMatrix(
                "data/matrixA_" + to_string(n) + ".txt", n);

            const Matrix B = loadMatrix(
                "data/matrixB_" + to_string(n) + ".txt", n);

            double baselineMs = 0.0;

            for (int threadCount : threadCounts)
            {
                cout << "Threads: " << threadCount << " ... "
                    << flush;

                const auto start = chrono::steady_clock::now();

                Matrix C = multiplyMatrices(A, B, n, threadCount);

                const auto finish = chrono::steady_clock::now();

                const double timeMs =
                    chrono::duration<double, milli>(finish - start).count();

                if (threadCount == 1)
                {
                    baselineMs = timeMs;
                }

                const double speedup = baselineMs / timeMs;
                const uint64_t operations =
                    2ULL * n * n * n;

                const string resultFile =
                    "results/result_" + to_string(n) +
                    "_threads_" + to_string(threadCount) + ".txt";

                saveMatrix(resultFile, C, n);

                csv << n << ','
                    << threadCount << ','
                    << timeMs << ','
                    << operations << ','
                    << speedup << '\n';

                results.push_back({
                    n, threadCount, timeMs, speedup, operations
                    });

                cout << fixed << setprecision(3)
                    << timeMs << " ms"
                    << " | Speedup: " << speedup << "x\n";
            }
        }

        csv.close();

        cout << "\nAll experiments completed successfully.\n";
        cout << "Results saved to experiments.csv\n";
        cout << "Matrix results saved to the results folder.\n";
    }
    catch (const exception& e)
    {
        cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }

    return 0;
}