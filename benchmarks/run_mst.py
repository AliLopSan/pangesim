"""MAPE analysis."""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from benchmarks.runners import evaluate_mst_error_run


def main() -> None:
    """Scalability test."""
    gene_sizes = list(range(25,1001,25))
    replicates = 5
    benchmark_data = []

    print("\t Running error test")
    with tqdm(total=len(gene_sizes) * replicates, desc="Total Progress") as pbar:
        for size in gene_sizes:
            for rep in range(1, replicates + 1):
                # Run runner run
                results = evaluate_mst_error_run(num_genes=size, replicate=rep)
                benchmark_data.append(results)
                pbar.set_postfix({"current_size": size})
                pbar.update(1)

    df = pd.DataFrame(benchmark_data)
    output_dir = Path("results/run_20260903")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "metrics_mst.csv"
    df.to_csv(file_path, index=False)

    print("\n\nDONE :)\n")


if __name__ == "__main__":
    main()
