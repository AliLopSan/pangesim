"""MAPE analysis."""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from benchmarks.config import PARAM_GRID
from benchmarks.runners import evaluate_comparison_run


def main() -> None:
    """Scalability & refinement strategy comparison benchmark."""
    gene_sizes = [25, 50, 100, 150, 200, 250, 300, 350, 400]
    replicates = 5
    refinement_strategies = ["score", "cost"]
    benchmark_data = []

    total_runs = len(gene_sizes) * replicates * len(PARAM_GRID)

    print("\t Running error & refinement strategy comparison test")
    with tqdm(total=total_runs, desc="Total Progress") as pbar:
        for size in gene_sizes:
            for rep in range(1, replicates + 1):
                for params in PARAM_GRID:
                    run_results = evaluate_comparison_run(
                        num_genes=size,
                        replicate=rep,
                        params=params,
                        strategies=refinement_strategies,
                    )
                    benchmark_data.extend(run_results)

                    pbar.set_postfix({"current_size": size, "rep": rep})
                    pbar.update(1)

    df = pd.DataFrame(benchmark_data)

    output_dir = Path("results/run_20260817")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "cost_vs_score.csv"
    df.to_csv(file_path, index=False)

    print(f"\n\nDONE :) Saved {len(df)} records to {file_path}\n")


if __name__ == "__main__":
    main()
