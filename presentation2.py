import random
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import copy

class SortAlgorithm:
    def __init__(self, data):
        self.data = data
        self.comparisons = 0
        self.swaps = 0

class BubbleSort(SortAlgorithm):
    def sort_steps(self):
        n = len(self.data)
        for i in range(n):
            swapped = False
            for j in range(n - i - 1):
                self.comparisons += 1
                if self.data[j] > self.data[j + 1]:
                    self.data[j], self.data[j + 1] = self.data[j + 1], self.data[j]
                    self.swaps += 1
                    swapped = True
                yield self.data[:], (j, j + 1)
            if not swapped:
                break
        yield self.data[:], (), list(range(n))

class InsertionSort(SortAlgorithm):
    def sort_steps(self):
        for i in range(1, len(self.data)):
            key = self.data[i]
            j = i - 1
            while j >= 0 and key < self.data[j]:
                self.comparisons += 1
                self.data[j + 1] = self.data[j]
                self.swaps += 1
                yield self.data[:], (j, j + 1)
                j -= 1
            self.data[j + 1] = key
            yield self.data[:], (j + 1,)
        yield self.data[:], (), list(range(len(self.data)))

class QuickSort(SortAlgorithm):
    def sort_steps(self):
        def quick_sort(arr, low, high):
            if low < high:
                pivot = arr[high]
                i = low
                for j in range(low, high):
                    self.comparisons += 1
                    if arr[j] < pivot:
                        arr[i], arr[j] = arr[j], arr[i]
                        self.swaps += 1
                        yield arr[:], (i, j), None, high
                        i += 1
                arr[i], arr[high] = arr[high], arr[i]
                self.swaps += 1
                yield arr[:], (i, high), None, i
                yield from quick_sort(arr, low, i - 1)
                yield from quick_sort(arr, i + 1, high)

        yield from quick_sort(self.data, 0, len(self.data) - 1)
        yield self.data[:], (), list(range(len(self.data))), None

class MergeSort(SortAlgorithm):
    def sort_steps(self):
        def merge_sort(arr, l, r):
            if l < r:
                m = (l + r) // 2
                yield from merge_sort(arr, l, m)
                yield from merge_sort(arr, m + 1, r)
                yield from merge(arr, l, m, r)
        def merge(arr, l, m, r):
            L = arr[l:m + 1]
            R = arr[m + 1:r + 1]
            i = j = 0
            k = l
            while i < len(L) and j < len(R):
                self.comparisons += 1
                if L[i] <= R[j]:
                    arr[k] = L[i]
                    i += 1
                else:
                    arr[k] = R[j]
                    j += 1
                yield arr[:], (k,)
                k += 1
            while i < len(L):
                arr[k] = L[i]
                yield arr[:], (k,)
                i += 1
                k += 1
            while j < len(R):
                arr[k] = R[j]
                yield arr[:], (k,)
                j += 1
                k += 1
        yield from merge_sort(self.data, 0, len(self.data) - 1)
        yield self.data[:], (), list(range(len(self.data)))


class SorterVisualizer:
    def __init__(self):
        self.data = []
        self.original_data = []
        self.algorithm = None
        self.delay = 200
        self.seed = None

    def set_seed(self, seed):
        self.seed = seed

    def generate_list(self, size, min_val, max_val):
        if self.seed is not None:
            random.seed(self.seed)
        self.data = [random.randint(min_val, max_val) for _ in range(size)]
        self.original_data = copy.deepcopy(self.data)


    def shuffle_list(self):
        random.shuffle(self.data)

    def reset(self):
        self.data = copy.deepcopy(self.original_data)

    def choose_algorithm(self, name):
        algos = {
        "bubble": BubbleSort,
        "insertion": InsertionSort,
        "quick": QuickSort,
        "merge": MergeSort
    }

        self.data = copy.deepcopy(self.original_data)
        self.algorithm = algos[name.lower()](self.data)


    def set_speed(self, delay_ms):
        self.delay = delay_ms

    def color_by_state(self, i, highlights, sorted_indices, pivot):
        if sorted_indices and i in sorted_indices:
            return "green"
        if pivot is not None and i == pivot:
            return "orange"
        if i in highlights:
            return "red"
        return "skyblue"

    def visualize_bars(self, values, highlight_indices=(), sorted_indices=None, pivot=None):
        for i, bar in enumerate(self.bars):
            bar.set_height(values[i])
            bar.set_color(self.color_by_state(i, highlight_indices, sorted_indices, pivot))

    def update_display(self, step):
        data, highlights, *rest = step
        sorted_indices = rest[0] if rest else None
        pivot = rest[1] if len(rest) > 1 else None
        self.visualize_bars(data, highlights, sorted_indices, pivot)

    def count_operations(self):
        return self.algorithm.comparisons, self.algorithm.swaps

    def benchmark_algorithms(self, runs=5):
        algorithms = {
            "Bubble": BubbleSort,
            "Insertion": InsertionSort,
            "Quick": QuickSort,
            "Merge": MergeSort
        }
        results = {}
        for name, algo in algorithms.items():
            total_time = total_comp = total_swaps = 0
            for _ in range(runs):
                data_copy = copy.deepcopy(self.original_data)
                sorter = algo(data_copy)
                start = time.perf_counter()
                for _ in sorter.sort_steps():
                    pass
                total_time += time.perf_counter() - start
                total_comp += sorter.comparisons
                total_swaps += sorter.swaps
            results[name] = {
                "time": total_time / runs,
                "comparisons": total_comp // runs,
                "swaps": total_swaps // runs
            }
        return results

    def export_results_to_csv(self, filename, results):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Algorithm", "Time (s)", "Comparisons", "Swaps"])
            for algo, data in results.items():
                writer.writerow([algo, data["time"], data["comparisons"], data["swaps"]])

    def show_summary_table(self, results):
        print("\nAlgorithm Summary")
        print("-" * 50)
        for algo, d in results.items():
            print(f"{algo:<10} Time: {d['time']:.6f}s | Comp: {d['comparisons']} | Swaps: {d['swaps']}")


viz = SorterVisualizer()
viz.set_seed(50)
viz.generate_list(20, 1, 50)

print("Choose sorting algorithm:")
print("1 - Bubble Sort")
print("2 - Insertion Sort")
print("3 - Quick Sort")
print("4 - Merge Sort")

choice = input("Enter your choice: ")

if choice == "1":
    viz.choose_algorithm("bubble")
    title = "Bubble Sort Visualization"
elif choice == "2":
    viz.choose_algorithm("insertion")
    title = "Insertion Sort Visualization"
elif choice == "3":
    viz.choose_algorithm("quick")
    title = "Quick Sort Visualization"
elif choice == "4":
    viz.choose_algorithm("merge")
    title = "Merge Sort Visualization"
else:
    raise ValueError("Invalid choice")


fig, ax = plt.subplots()
ax.set_title(title)
ax.set_ylim(0, max(viz.data) + 5)

bars = ax.bar(range(len(viz.data)), viz.data)
viz.bars = bars

ani = animation.FuncAnimation(
    fig,
    viz.update_display,
    frames=viz.algorithm.sort_steps(),
    interval=viz.delay,
    repeat=False
)

results = viz.benchmark_algorithms()
viz.show_summary_table(results)
viz.export_results_to_csv("results.csv", results)

plt.show()
