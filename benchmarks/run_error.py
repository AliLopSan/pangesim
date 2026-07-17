"""MAPE analysis."""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from benchmarks.config import PARAM_GRID
from benchmarks.runners import evaluate_error_run


def main() -> None:
    """Scalability test."""
    gene_sizes = [50, 100, 150, 200, 250, 300]
    replicates = 5
    benchmark_data = []

    print("\t Running error test")
    with tqdm(total=len(gene_sizes) * replicates*len(PARAM_GRID),
              desc="Total Progress") as pbar:
        for size in gene_sizes:
            for rep in range(1, replicates + 1):
                for params in PARAM_GRID:
                    # Run runner run
                    results = evaluate_error_run(num_genes=size,
                                                 replicate=rep,
                                                 params=params)
                    benchmark_data.append(results)
                    pbar.set_postfix({"current_size": size})
                    pbar.update(1)

    df = pd.DataFrame(benchmark_data)
    output_dir = Path("results/run_20260713")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "error_metrics_ISMB_bounds.csv"
    df.to_csv(file_path, index=False)

    print("\n\nDONE :)\n")

if __name__ == "__main__":
    main()
