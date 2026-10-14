import csv
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "experiments.csv"


def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {CSV_FILE}. Run the C++ program first."
        )

    rows = []

    with CSV_FILE.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        required = {
            "size", "threads", "time_ms", "operations", "speedup"
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "experiments.csv has an unexpected format."
            )

        for row in reader:
            rows.append({
                "size": int(row["size"]),
                "threads": int(row["threads"]),
                "time_ms": float(row["time_ms"]),
                "speedup": float(row["speedup"]),
            })

    sizes = sorted({row["size"] for row in rows})
    thread_counts = sorted({row["threads"] for row in rows})

    # Execution time
    plt.figure(figsize=(10, 6))

    for size in sizes:
        selected = sorted(
            (row for row in rows if row["size"] == size),
            key=lambda row: row["threads"],
        )

        plt.plot(
            [row["threads"] for row in selected],
            [row["time_ms"] for row in selected],
            marker="o",
            label=f"N = {size}",
        )

    plt.title("OpenMP Matrix Multiplication: Execution Time")
    plt.xlabel("Number of threads")
    plt.ylabel("Execution time (ms)")
    plt.xticks(thread_counts)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "performance.png", dpi=300)
    plt.close()

    # Speedup
    plt.figure(figsize=(10, 6))

    for size in sizes:
        selected = sorted(
            (row for row in rows if row["size"] == size),
            key=lambda row: row["threads"],
        )

        plt.plot(
            [row["threads"] for row in selected],
            [row["speedup"] for row in selected],
            marker="o",
            label=f"N = {size}",
        )

    plt.plot(
        thread_counts,
        thread_counts,
        linestyle="--",
        label="Ideal speedup",
    )

    plt.title("OpenMP Matrix Multiplication: Speedup")
    plt.xlabel("Number of threads")
    plt.ylabel("Speedup")
    plt.xticks(thread_counts)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "speedup.png", dpi=300)
    plt.close()

    print("Graphs saved:")
    print(BASE_DIR / "performance.png")
    print(BASE_DIR / "speedup.png")


if __name__ == "__main__":
    main()