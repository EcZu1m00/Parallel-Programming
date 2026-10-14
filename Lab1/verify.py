import numpy as np
import os
import sys


SIZES = [200, 400, 800, 1200, 1600, 2000]


def read_matrix(filename):
    """
    Read matrix from a text file.

    File format:
        n
        a11 a12 a13 ...
        a21 a22 a23 ...
        ...
    """

    with open(filename, "r") as file:
        n = int(file.readline())

        matrix = np.loadtxt(
            file,
            dtype=np.int64
        )

    matrix = np.atleast_2d(matrix)

    if matrix.shape != (n, n):
        raise ValueError(
            f"Invalid matrix size in {filename}: "
            f"expected {n}x{n}, got {matrix.shape}"
        )

    return matrix


def verify_experiment(n):
    """
    Verify one experiment.
    """

    matrix_a_file = f"data/matrixA_{n}.txt"
    matrix_b_file = f"data/matrixB_{n}.txt"
    result_file = f"results/result_{n}.txt"

    print("--------------------------------------------")
    print(f"Checking matrix size: {n} x {n}")

    for filename in [
        matrix_a_file,
        matrix_b_file,
        result_file
    ]:
        if not os.path.exists(filename):
            print(f"ERROR: file not found: {filename}")
            return False

    try:
        print("Reading matrix A...")
        A = read_matrix(matrix_a_file)

        print("Reading matrix B...")
        B = read_matrix(matrix_b_file)

        print("Reading C++ result...")
        C_cpp = read_matrix(result_file)

        print("Calculating reference result with NumPy...")

        C_numpy = np.matmul(A, B)

        if np.array_equal(C_cpp, C_numpy):
            print("VERIFICATION: PASSED")
            print("C++ result matches NumPy result.")
            return True

        print("VERIFICATION: FAILED")
        print("C++ result does NOT match NumPy result.")

        difference = np.abs(C_cpp - C_numpy)

        print(
            "Maximum difference:",
            difference.max()
        )

        incorrect = np.argwhere(C_cpp != C_numpy)

        if len(incorrect) > 0:
            i, j = incorrect[0]

            print(
                f"First mismatch at [{i}][{j}]"
            )

            print(
                "C++ value:",
                C_cpp[i, j]
            )

            print(
                "NumPy value:",
                C_numpy[i, j]
            )

        return False

    except Exception as error:
        print(f"ERROR: {error}")
        return False


def main():

    print("============================================")
    print("       NUMPY AUTOMATIC VERIFICATION")
    print("============================================")
    print()

    passed = 0
    failed = 0

    for n in SIZES:

        result = verify_experiment(n)

        if result:
            passed += 1
        else:
            failed += 1

        print()

    print("============================================")
    print("             VERIFICATION SUMMARY")
    print("============================================")

    print(f"Total experiments: {len(SIZES)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    print()

    if failed == 0:
        print("VERIFICATION: PASSED")
        print("All C++ results match NumPy results.")
        return 0

    else:
        print("VERIFICATION: FAILED")
        print("Some results do not match NumPy.")
        return 1


if __name__ == "__main__":
    sys.exit(main())