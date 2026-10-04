#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdint>
#include <random>
#include <filesystem>
#include <cstdlib>

using namespace std;
namespace fs = std::filesystem;

using Matrix = vector<vector<int>>;
using ResultMatrix = vector<vector<long long>>;



Matrix generateMatrix(size_t n, unsigned int seed)
{
    Matrix matrix(n, vector<int>(n));

    mt19937 generator(seed);
    uniform_int_distribution<int> distribution(1, 9);

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            matrix[i][j] = distribution(generator);
        }
    }

    return matrix;
}


void writeMatrix(
    const string& filename,
    const Matrix& matrix)
{
    ofstream file(filename);

    if (!file)
    {
        throw runtime_error(
            "Cannot create file: " + filename
        );
    }

    const size_t n = matrix.size();

    file << n << '\n';

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            file << matrix[i][j];

            if (j + 1 < n)
            {
                file << ' ';
            }
        }

        file << '\n';
    }
}



void writeResultMatrix(
    const string& filename,
    const ResultMatrix& matrix)
{
    ofstream file(filename);

    if (!file)
    {
        throw runtime_error(
            "Cannot create file: " + filename
        );
    }

    const size_t n = matrix.size();

    file << n << '\n';

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            file << matrix[i][j];

            if (j + 1 < n)
            {
                file << ' ';
            }
        }

        file << '\n';
    }
}



ResultMatrix multiplyMatrices(
    const Matrix& A,
    const Matrix& B,
    size_t n)
{
    ResultMatrix C(
        n,
        vector<long long>(n, 0)
    );

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t k = 0; k < n; ++k)
        {
            for (size_t j = 0; j < n; ++j)
            {
                C[i][j] +=
                    static_cast<long long>(A[i][k]) *
                    B[k][j];
            }
        }
    }

    return C;
}



void writeExperiment(
    ofstream& file,
    size_t n,
    unsigned long long multiplications,
    unsigned long long additions,
    unsigned long long operations,
    double timeMs)
{
    file << n << ','
        << multiplications << ','
        << additions << ','
        << operations << ','
        << fixed << setprecision(6)
        << timeMs
        << '\n';
}


int main()
{
    try
    {
        const vector<size_t> sizes =
        {
            200,
            400,
            800,
            1200,
            1600,
            2000
        };



        fs::create_directories("data");
        fs::create_directories("results");



        ofstream csv("experiments.csv");

        if (!csv)
        {
            throw runtime_error(
                "Cannot create experiments.csv"
            );
        }

        csv << "size,"
            << "multiplications,"
            << "additions,"
            << "operations,"
            << "time_ms\n";



        cout << "============================================\n";
        cout << "     MATRIX MULTIPLICATION EXPERIMENT\n";
        cout << "============================================\n\n";

        cout << "Algorithm: sequential matrix multiplication\n";
        cout << "Parallelization: disabled\n";
        cout << "Complexity: O(N^3)\n\n";



        for (size_t experiment = 0;
            experiment < sizes.size();
            ++experiment)
        {
            const size_t n = sizes[experiment];

            cout << "--------------------------------------------\n";
            cout << "Experiment "
                << experiment + 1
                << " / "
                << sizes.size()
                << '\n';

            cout << "Matrix size: "
                << n << " x " << n << '\n';



            cout << "Generating input matrices...\n";

            Matrix A = generateMatrix(
                n,
                static_cast<unsigned int>(1000 + n)
            );

            Matrix B = generateMatrix(
                n,
                static_cast<unsigned int>(2000 + n)
            );



            const string matrixAFile =
                "data/matrixA_" +
                to_string(n) +
                ".txt";

            const string matrixBFile =
                "data/matrixB_" +
                to_string(n) +
                ".txt";


            writeMatrix(matrixAFile, A);
            writeMatrix(matrixBFile, B);

            cout << "Input matrices saved.\n";


            const unsigned long long cube =
                static_cast<unsigned long long>(n) *
                static_cast<unsigned long long>(n) *
                static_cast<unsigned long long>(n);

            const unsigned long long multiplications = cube;

            const unsigned long long additions = cube;

            const unsigned long long operations =
                multiplications + additions;


            cout << "Multiplications: "
                << multiplications
                << '\n';

            cout << "Additions: "
                << additions
                << '\n';

            cout << "Total arithmetic operations: "
                << operations
                << '\n';



            cout << "Starting multiplication...\n";

            const auto start =
                chrono::high_resolution_clock::now();



            ResultMatrix C =
                multiplyMatrices(A, B, n);



            const auto finish =
                chrono::high_resolution_clock::now();


            const chrono::duration<double, milli> elapsed =
                finish - start;


            const double timeMs =
                elapsed.count();


            cout << fixed << setprecision(6);

            cout << "Execution time: "
                << timeMs
                << " ms\n";



            const string resultFile =
                "results/result_" +
                to_string(n) +
                ".txt";


            writeResultMatrix(resultFile, C);

            cout << "Result saved to: "
                << resultFile
                << '\n';



            writeExperiment(
                csv,
                n,
                multiplications,
                additions,
                operations,
                timeMs
            );


            cout << "Experiment completed.\n\n";
        }


        csv.close();



        cout << "============================================\n";
        cout << "All experiments completed.\n";
        cout << "============================================\n\n";

        cout << "Experimental data:\n";
        cout << "experiments.csv\n\n";

        cout << "Input matrices:\n";
        cout << "data/\n\n";

        cout << "Result matrices:\n";
        cout << "results/\n\n";


        cout << "Starting automatic verification...\n\n";

        const int verificationResult =
            system("python verify.py");


        if (verificationResult == 0)
        {
            cout << "\nVerification completed successfully.\n";
        }
        else
        {
            cout << "\nVerification script was not found "
                "or verification failed.\n";

            cout << "You can run verify.py manually.\n";
        }
    }
    catch (const exception& e)
    {
        cerr << "\nERROR: "
            << e.what()
            << '\n';

        return 1;
    }


    return 0;
}