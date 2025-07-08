import time
import matplotlib.pyplot as plt
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def complex_calculation(no_of_calculations, time_start):
    time_thread_start = time.time() - time_start
    for i in range(no_of_calculations):
        (i ** 2 + i ** 0.5) / (i + 1)

    return time_thread_start, time.time() - time_start


# def divide_work_per_workers(work, no_of_workers):
#     """ If the work in not enough for all of the workers duplicated the last data."""
#     leftovers = no_of_workers - (len(work) % no_of_workers)
#     work_ammount = len(work) // no_of_workers
#     work_divided = []
#
#     for _ in range(leftovers):
#         work.append(work[-1])
#
#     for worker in range(no_of_workers):
#         work_from = worker * work_ammount
#         work_to = ((worker + 1) * work_ammount) - 1
#         work_divided.append(work[work_from:work_to])
#
#     return work_divided

def run_job(func, args, executor_type, no_of_workers):
    time_start = time.time()
    with executor_type(max_workers=no_of_workers) as executor:
        res = executor.map(func, args, [time_start for _ in range(len(args))])
    return list(res)


def plot_chart(results, title):
    fig, ax = plt.subplots()
    start, stop = np.array(results).T

    ax.barh(range(len(start)), stop - start, height=0.7, left=start, color='k')
    ax.grid(axis='x')
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Worker")
    ax.set_title(title)
    plt.show()


# Example Usage
if __name__ == "__main__":
    # print(sys._is_gil_enabled())
    # print(os.cpu_count())
    no_of_calculations = 1_000_00
    thread_benchmark = run_job(complex_calculation, [no_of_calculations] * 12, ThreadPoolExecutor, 4)
    process_benchmark = run_job(complex_calculation, [no_of_calculations] * 12, ProcessPoolExecutor, 4)

    title_gil_suffix = " with GIL enabled" if sys._is_gil_enabled() else " with GIL disabled"
    plot_chart(thread_benchmark, "Thread benchmark" + title_gil_suffix)
    plot_chart(process_benchmark, "Process benchmark")
