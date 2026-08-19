"""Plotter script for individual results visualization."""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from benchmarks.config import PARAM_GRID

#from pangesim.visualization import ErrorVisualizer
from pangesim.visualization import RuntimeVisualizer


def plot_scalability(results_dir: Path, filename: Path) -> None:
    """Plots the results of the scalability test.

    Args:
        results_dir: Path where the results will be stored and where df is.
        filename: Path and name of the df
    """
    df = pd.read_csv(filename)

    out_dir = Path("results/run_20260715/runtime_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot total run-time
    vis_1 = RuntimeVisualizer()
    f1_name = out_dir / "total_runtime.pdf"
    vis_1.plot_total_runtime(df, f1_name)

    vis_2 = RuntimeVisualizer()
    f2_name = out_dir / "phase_1_3_runtime.pdf"
    vis_2.plot_phase_runtime(df, f2_name)

    vis_3 = RuntimeVisualizer()
    f3_name = out_dir / "phase_4_runtime.pdf"
    vis_3.plot_phase4_runtime(df, f3_name)

class ErrorVisualizer:
    """Handles comparative error visualization for pangenome refinement strategies."""

    def plot_genomes_mape(
        self, df: pd.DataFrame, params: Dict[str, float], output_path: Path
    ) -> None:
        """Plots MAPE across gene sizes compared between refinement strategies."""
        # Calculate MAPE (assuming 'genomes_gt' and 'genomes_inf' exist in df)
        # MAPE = |gt - inf| / gt * 100
        df = df.copy()
        df["mape"] = (
            (df["genomes_gt"] - df["genomes_inf"]).abs() / df["genomes_gt"]
        ) * 100

        # Filter for the specific parameter combination
        filtered_df = df[
            (df["alpha"] == params["alpha"]) & (df["gamma"] == params["gamma"])
        ]

        if filtered_df.empty:
            print(f"Warning: No data found for parameters {params}")
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        # Use seaborn lineplot with hue='strategy' to overlay score vs cost
        sns.lineplot(
            data=filtered_df,
            x="gene_size",
            y="mape",
            hue="strategy",
            style="strategy",
            markers=True,
            dashes=False,
            palette="viridis",
            ax=ax,
        )

        ax.set_title(
            f"MAPE Comparison (α={params['alpha']}, γ={params['gamma']})"
        )
        ax.set_xlabel("Number of genes")
        ax.set_ylabel("MAPE (%)")
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_mape(results_dir: Path, params: Dict[str, float], filename: Path) -> None:
    """Plots the MAPE for a given dict of parameters across refinement strategies.

    Args:
        results_dir: Path where results are stored.
        params: Dict containing "alpha" and "gamma".
        filename: Path to input results CSV.
    """
    df = pd.read_csv(filename)

    out_dir = results_dir / "mape_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Automated filename layout using params dict
    param_strings = [
        f"{key}_{str(value).replace('.', '_')}" for key, value in params.items()
    ]
    fname = f"mape_{'_'.join(param_strings)}.pdf"
    final_output_path = out_dir / fname

    vis = ErrorVisualizer()
    vis.plot_genomes_mape(df=df, params=params, output_path=final_output_path)


if __name__ == "__main__":
    print("\tRunning Harry Plotter ...")
    results = Path("results/run_20260817")
    df_file = results / "cost_vs_score.csv"

    for params in PARAM_GRID:
        plot_mape(results_dir=results, params=params, filename=df_file)

    print("Done! :)")
