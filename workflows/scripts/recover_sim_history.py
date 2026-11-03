from pathlib import Path

import pandas as pd
import ast
from tqdm import tqdm

import sys


def recover_sim_history(results: Path) -> pd.DataFrame:
    """Recovers simulation history from simulation results."""
    levels = ["low","medium","high"]
    gene_sizes = [25, 100, 200, 500, 800, 1000]
    n_replicates = 5
    summary_records = []
    events = [
        "insertion",
        "deletion",
        "rearrangement",
        "inversion",
        "fission",
        "fusion",
        "translocation",
        "gene_deletion",
    ]

    results.mkdir(parents=True, exist_ok=True)
    total_iterations = len(gene_sizes) * len(levels) * n_replicates

    with tqdm(total=total_iterations, desc="Total Progress") as pbar:
        for size in gene_sizes:
            size_dir = results / str(size)

            for level in levels:
                level_dir = size_dir / level
                for rep in range(n_replicates):
                    rep_dir = level_dir / f"rep_{rep}"
                    sim_history = pd.read_csv(rep_dir / "simulation_history.csv")

                    base_record = {
                        "size": size,
                        "level": level,
                        "rep": rep
                    }

                    # Convert each string in the column into a real Python dictionary
                    sim_history["event_counts"] = sim_history["event_counts"].apply(ast.literal_eval)

                    total_events = pd.DataFrame(
                        sim_history["event_counts"].tolist()
                    ).sum().to_dict()

                    for ev in events:
                        base_record[ev] = total_events[ev]


                    summary_records.append(base_record)
                    pbar.set_postfix({"level": level, "rep": rep})
                    pbar.update(1)

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(results / "simulation_events_summary.csv", index=False)
    print(f"\n Master CSV successfully saved to {results / 'experiment_summary.csv'}")
    print(summary_df.head())

    avg_events_by_level = summary_df.groupby("level")[events].mean()

    print(avg_events_by_level)


if __name__ == "__main__":
    out_dir = Path("results/run_20261022/")
    out_dir.mkdir(parents=True, exist_ok=True)
    recover_sim_history(out_dir)
