from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

SIZES = [200, 400, 800, 1200, 1600, 2000]
THREAD_COUNTS = [1, 2, 4, 8]


def read_matrix(path, size):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    matrix = np.loadtxt(path, dtype=np.int64)

    if matrix.shape != (size, size):
        raise ValueError(
            f"Invalid matrix shape in {path}: {matrix.shape}; "
            f"expected ({size}, {size})"
        )

    return matrix


def main():
    total = 0
    passed = 0
    failed = 0

    print("=" * 55)
    print("OPENMP MATRIX MULTIPLICATION VERIFICATION")
    print("=" * 55)

    for size in SIZES:
        try:
            matrix_a = read_matrix(
                DATA_DIR / f"matrixA_{size}.txt", size
            )
            matrix_b = read_matrix(
                DATA_DIR / f"matrixB_{size}.txt", size
            )

            print(f"\nMatrix size: {size} x {size}")
            print("Calculating reference result with NumPy...")

            reference = matrix_a @ matrix_b

            for threads in THREAD_COUNTS:
                total += 1

                result_path = (
                    RESULTS_DIR
                    / f"result_{size}_threads_{threads}.txt"
                )

                try:
                    actual = read_matrix(result_path, size)

                    if np.array_equal(actual, reference):
                        passed += 1
                        print(
                            f"  Threads {threads}: PASSED"
                        )
                    else:
                        failed += 1
                        difference = np.max(
                            np.abs(actual - reference)
                        )
                        print(
                            f"  Threads {threads}: FAILED "
                            f"(max difference: {difference})"
                        )

                except (FileNotFoundError, ValueError) as error:
                    failed += 1
                    print(f"  Threads {threads}: FAILED ({error})")

        except (FileNotFoundError, ValueError) as error:
            missing_count = len(THREAD_COUNTS)
            total += missing_count
            failed += missing_count
            print(f"\nMatrix size {size}: FAILED ({error})")

    print("\n" + "=" * 55)
    print("VERIFICATION SUMMARY")
    print("=" * 55)
    print(f"Total experiments: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\nVERIFICATION: PASSED")
        print("All OpenMP results match NumPy results.")
    else:
        print("\nVERIFICATION: FAILED")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())