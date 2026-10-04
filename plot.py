import csv
import matplotlib.pyplot as plt


def read_experiments(filename):
    sizes = []
    times = []
    operations = []

    with open(filename, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            sizes.append(int(row["size"]))
            times.append(float(row["time_ms"]))
            operations.append(int(row["operations"]))

    return sizes, times, operations


def create_time_graph(sizes, times):
    plt.figure(figsize=(10, 6))

    plt.plot(
        sizes,
        times,
        marker="o",
        linewidth=2
    )

    plt.title("Matrix Multiplication Performance")
    plt.xlabel("Matrix size N")
    plt.ylabel("Execution time (ms)")

    plt.grid(True)

    plt.xticks(sizes)

    plt.tight_layout()

    plt.savefig(
        "performance.png",
        dpi=300
    )

    plt.show()


def main():
    filename = "experiments.csv"

    sizes, times, operations = read_experiments(filename)

    print("============================================")
    print("       PERFORMANCE GRAPH")
    print("============================================")
    print()

    print("Experimental results:")

    for i in range(len(sizes)):
        print(
            f"N = {sizes[i]:4d} | "
            f"Time = {times[i]:.6f} ms | "
            f"Operations = {operations[i]}"
        )

    print()

    create_time_graph(sizes, times)

    print("Graph saved to:")
    print("performance.png")


if __name__ == "__main__":
    main()