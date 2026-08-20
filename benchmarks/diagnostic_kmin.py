from pathlib import Path
from typing import Any
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from benchmarks.fixtures import random_simulated_pangenome
from pangesim.reconstruction import EulerianPathHeuristic
from pangesim.reconstruction.assignment import EulerianTrailAssignment
from pangesim.reconstruction.bounds import GreedyPairingISCB
from pangesim.reconstruction.bounds import GreedyPairing

def plot_bound_tightness_sweep(df: pd.DataFrame, x_col: str = "gene_size") -> None:
    """Plots k_min vs k_true across problem scale to locate tightness degradation.

    Args:
        df: Results df.
        x_col:Column to plot.
    """
    custom_rc = {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath}"}
    sns.set_theme(style="whitegrid", rc=custom_rc)
    plt.figure(figsize=(9, 6))

    # Plot ground truth baseline vs computed lower bound
    sns.lineplot(data=df, x=x_col, y="genomes_gt",
                 label=r"$k_{\text{true}}$ (Ground Truth)",
                 color="black",
                 linestyle="--",
                 errorbar="sd")
    sns.lineplot(data=df, x=x_col, y="k_min",
                 label=r"$k_{\min}$ (Lower Bound)",
                 color="crimson",
                 marker="o",
                 errorbar="sd")
    
    sns.lineplot(data=df, x=x_col, y="k_base",
                 label=r"$k_{base}$ (Eulerian Decomposition)",
                 color="fuchsia",
                 marker="D",
                 errorbar="sd")

    plt.xlabel(r"Number of genes on input")
    plt.ylabel(r"Genome Size ($k$)")
    plt.title(r"Lower Bound Tightness Diagnostic")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def kmin_run(num_genes: int, replicate:int) -> Dict[str, Any]:
    """Main runner to test kmin vs true k tightness bound.

    Args:
        num_genes: Number of genes per genome.
        replicate: Current replicate number.

    Returns:
        A dict with the computed kmin bound and also the base pangenome.
    """
    # Simulate random scenario
    ground_truth = random_simulated_pangenome(num_genes)
    matrix = ground_truth.compute_weighted_adjacencies()
    assignment = EulerianTrailAssignment()
    bounds = GreedyPairing
    heuristic = EulerianPathHeuristic(
        bounds_strategy=bounds, assignment_strategy=assignment
    )
    k_min, k_max, k_info = heuristic.bounds_strategy.compute_bounds(matrix,
                                                                    heuristic.params)
    base_pangenome = heuristic.assignment_strategy.assign_genomes(adjacencies=matrix,
                                                                  k=k_min)
    return {
        "gene_size": num_genes,
        "replicate": replicate,
        "genomes_gt": len(ground_truth),
        "k_min": k_min,
        "k_base":len(base_pangenome)
    }

def kmin_run(num_genes: int, replicate:int) -> Dict[str, Any]:
    """Main runner to test kmin vs true k tightness bound.

    Args:
        num_genes: Number of genes per genome.
        replicate: Current replicate number.

    Returns:
        A dict with the computed kmin bound and also the base pangenome.
    """
    # Simulate random scenario
    ground_truth = random_simulated_pangenome(num_genes)
    matrix = ground_truth.compute_weighted_adjacencies()
    assignment = EulerianTrailAssignment()
    bounds = GreedyPairingISCB()
    heuristic = EulerianPathHeuristic(
        bounds_strategy=bounds, assignment_strategy=assignment
    )
    k_min, k_max, k_info = heuristic.bounds_strategy.compute_bounds(matrix,
                                                                    heuristic.params)
    base_pangenome = heuristic.assignment_strategy.assign_genomes(adjacencies=matrix,
                                                                  k=k_min)
    return {
        "gene_size": num_genes,
        "replicate": replicate,
        "genomes_gt": len(ground_truth),
        "k_min": k_min,
        "k_base":len(base_pangenome)
    }


def main() -> None:
    """Scalability test."""
    gene_sizes = list(range(25, 1001, 25))
    benchmark_data = []
    repeats = 5

    print("\t Running kmin diagnostic")
    with tqdm(total=len(gene_sizes)*repeats, desc="Total Progress") as pbar:
        for size in gene_sizes:
            for r in range(1, repeats + 1):
                results = kmin_run(num_genes=size,replicate=r)
                benchmark_data.append(results)
                pbar.set_postfix({"current_size": size})
                pbar.update(1)

    df = pd.DataFrame(benchmark_data)
    output_dir = Path("results/run_20260820")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "kmin_diagnostic.csv"
    df.to_csv(file_path, index=False)
    plot_bound_tightness_sweep(df)

    print("\n\nDONE :)\n")


if __name__ == "__main__":
    main()
