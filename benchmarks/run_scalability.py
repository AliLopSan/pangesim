"""Runtime analysis of the heuristic."""

import os

import pandas as pd
from pathlib import Path
from tqdm import tqdm

from benchmarks.runners import evaluate_scalability_run


def main() -> None:
    """Scalability test."""
    gene_sizes = [5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 500]
    replicates = 5
    benchmark_data = []

    print("\t Running Scalability test")
    with tqdm(total=len(gene_sizes) * replicates, desc="Total Benchmark Progress") as pbar:
        for size in gene_sizes:
            for rep in range(1, replicates + 1):
                # Run runner run
                results = evaluate_scalability_run(num_genes=size, replicate=rep)
                benchmark_data.append(results)

            pbar.set_postfix({"current_size": size})
            pbar.update(1)

    df = pd.DataFrame(benchmark_data)
    output_dir = Path("results/run_20260702")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "scalability_metrics.csv"   
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    main()
