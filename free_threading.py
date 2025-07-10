import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

height, width = 1200, 1200
image = np.zeros((height, width), dtype=int)

def complex_calculation(no_of_calculations, time_start):
    time_thread_start = time.time() - time_start
    for i in range(no_of_calculations):
        (i ** 2 + i ** 0.5) / (i + 1)

    return time_thread_start, time.time() - time_start

def compute_mandelbrot_block(y_range, time_start):
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    max_iter = 150
    time_thread_start = time.time() - time_start

    y_vals = np.linspace(y_min, y_max, height)[y_range[0]:y_range[1]]
    x_vals = np.linspace(x_min, x_max, width)

    for row_idx, y in enumerate(y_vals):
        for col_idx, re in enumerate(x_vals):
            c = re + 1j * y
            z = 0
            for i in range(max_iter):
                z = z * z + c
                if abs(z) > 2:
                    image[y_range[0] + row_idx, col_idx] = i
                    break

    return time_thread_start, time.time() - time_start

def run_job(func, args, executor_type, no_of_workers):
    time_start = time.time()
    with executor_type(max_workers=no_of_workers) as executor:
        res = executor.map(func, args, [time_start for _ in range(len(args))])
    return list(res)

def plot_chart(results, title):
    divs = [result[1] - result[0] for result in results]
    print(f"{title} - min, max, avg: {min(divs)}, {max(divs)}, {sum(divs)/len(divs)}")

    fig, ax = plt.subplots()
    start, stop = np.array(results).T

    ax.barh(range(len(start)), stop - start, height=0.7, left=start, color='k')
    ax.grid(axis='x')
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Worker")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def print_mandlebrot():
    plt.imshow(image, extent=[-2, 1, -1.5, 1.5], cmap="inferno")
    plt.title("Mandelbrot")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.colorbar(label="Iterations")
    plt.show()

if __name__ == "__main__":
    print(sys._is_gil_enabled())
    arithmetics = False
    mandelbrot = False
    no_of_cpus_to_work = 12

    if arithmetics:
        no_of_calculations = 10_000_000
        work_load = 12

        thread_arithmetics_benchmark = run_job(complex_calculation, [no_of_calculations] * work_load, ThreadPoolExecutor, no_of_cpus_to_work)
        process_arithmetics_benchmark = run_job(complex_calculation, [no_of_calculations] * work_load, ProcessPoolExecutor, no_of_cpus_to_work)

        print(thread_arithmetics_benchmark)
        print(process_arithmetics_benchmark)
        plot_chart(thread_arithmetics_benchmark, "Threads benchmark with GIL enabled - Arithmetics")
        plot_chart(process_arithmetics_benchmark, "Processes benchmark - Arithmetics")

    if mandelbrot:
        blocks_of_work = 12
        block_size = height // blocks_of_work
        divided_work = [(i * block_size, (i + 1) * block_size) for i in range(blocks_of_work)]

        process_mandlebrot_benchmark = run_job(compute_mandelbrot_block, divided_work, ProcessPoolExecutor, no_of_cpus_to_work)
        thread_mandlebrot_benchmark = run_job(compute_mandelbrot_block, divided_work, ThreadPoolExecutor, no_of_cpus_to_work)

        print(thread_mandlebrot_benchmark)
        print(process_mandlebrot_benchmark)
        print_mandlebrot()
        plot_chart(thread_mandlebrot_benchmark, "Threads benchmark with GIL enabled - Mandelbrot")
        plot_chart(process_mandlebrot_benchmark, "Processes Mandelbrot benchmark - Mandelbrot")
