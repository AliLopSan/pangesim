import time

import pandas as pd

if __name__ == "__main__":
    # Experimental grid
    gene_sizes = [10, 20, 50, 100, 200]
    replicates = 5
    alpha = 1.0
    gamma = 1.0

    benchmark_data = []

    for size in gene_sizes:
        for rep in range(1, replicates + 1):
            # Simulate random scenario

            # start clock
            t0 = time.perf_counter()

            t1 = time.perf_counter()

            t2 = time.perf_counter()

            duration_phases_1_3 = t1 - t0
            duration_phase_4 = t2 - t1
            total_duration = t2 - t0

            benchmark_data.append(
                {
                    "gene size": size,
                    "replicate": rep,
                    "runtime_phases_1-3": duration_phases_1_3,
                    "runtime_phase_4": duration_phase_4,
                    "total_runtime": total_duration,
                }
            )
    df = pd.DataFrame(benchmark_data)
