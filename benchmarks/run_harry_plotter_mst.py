"""Plotter script for individual results visualization."""

from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pangesim.visualization.performance import BaseVisualizer

class MSTVisualizer(BaseVisualizer):
    """A class for performance visualizations of Naive algorithm."""
    def plot_phase_runtime(self, df: pd.DataFrame, output_path: str) -> None:
        """Plots the execution runtime across increasing gene sizes with error bands.

        Args:
            df: DataFrame containing columns ["gene size", "runtime_phases_1-3"].
            output_path: System path where the resulting PDF file will be saved.
        """
        fig, ax = plt.subplots(figsize=(8, 7))

        # sns.lineplot automatically groups replicates to calculate mean and variance
        sns.lineplot(
            data=df,
            x="gene size",
            y="runtime",
            ax=ax,
            marker="o",
            linewidth=2,
            errorbar="sd",  # Standard deviation band across the 5 replicates
            color="#1f77b4",
        )

        # ax.set_title(r"\textbf{Scalability Profile: Phases 1--3}")
        ax.set_xlabel(r"Input Scale (\textit{Number of Genes})")
        ax.set_ylabel(r"Execution Runtime (\textit{Seconds})")

        # Clean layout boundaries and saving as PDF for vector scaling in LaTeX
        plt.tight_layout()
        plt.savefig(output_path, format="pdf", dpi=300)
        plt.close()
        
    def plot_genomes_mape(
        self, df: pd.DataFrame, output_path: Path
    ) -> None:
        """Plots the mean absolute percentage error.

        Args:
            df: DataFrame containing columns ["genomes gt", "genomes inf"].
            params: Dictionary that specifies the alpha and gamma parameters
            output_path: System path where the resulting PDF file will be saved.
        """
        # Guard against empty data slices
        if df.empty:
            print(f"[Warning] No rows match the parameters: {params}. Skipping plot.")
            return

        # Row-level MAPE on the filtered subset
        df["MAPE"] = (
            (df["genomes gt"] - df["genomes inf"]).abs()
            / df["genomes gt"]
            * 100
        )
        # Main plotter
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.lineplot(
            data=df,
            x="gene size",
            y="MAPE",
            ax=ax,
            marker="o",
            linewidth=2.5,
            errorbar="sd",  # Calculates variance across the 5 replicates
            color="#FF2052",
        )

        # 5. Clean LaTeX Typography & Title Context
        #ax.set_title("Sequential Edge Insertion", pad=15)

        ax.set_xlabel(r"Input Scale (\textit{Number of Genes})")
        ax.set_ylabel(r"Mean Absolute Percentage Error (\textit{MAPE \%})")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylim(bottom=-0.50)

        plt.tight_layout()
        plt.savefig(output_path, format="pdf", dpi=400)
        plt.close()


    """A class for performance visualizations of Naive algorithm."""
    def plot_raw_k_difference(self, df: pd.DataFrame, output_path: str) -> None:
        """Plots the execution runtime across increasing gene sizes with error bands.

        Args:
            df: DataFrame containing columns ["gene size", "runtime_phases_1-3"].
            output_path: System path where the resulting PDF file will be saved.
        """
        fig, ax = plt.subplots(figsize=(8, 7))

        # Row-level MAPE on the filtered subset
        df["RAW"] = df["genomes gt"] - df["genomes inf"]

        # sns.lineplot automatically groups replicates to calculate mean and variance
        sns.lineplot(
            data=df,
            x="gene size",
            y="RAW",
            ax=ax,
            marker="o",
            linewidth=2,
            errorbar="sd",  # Standard deviation band across the 5 replicates
            color="#8DB600",
        )

        # 5. Clean LaTeX Typography & Title Context
        #ax.set_title("Sequential Edge Insertion", pad=15)

        ax.set_xlabel(r"Input Scale (\textit{Number of Genes})")
        ax.set_ylabel(r"$k_{true} - k_{inferred}$")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylim(bottom=-1,top=10)

        plt.tight_layout()
        plt.savefig(output_path, format="pdf", dpi=400)
        plt.close()

if __name__ == "__main__":
    print("\tRunning  Harry Plotter Naive version ...")
    results = Path("results/run_20260903")
    df_file = results / "metrics_mst.csv"
    df = pd.read_csv(df_file)
    vis = MSTVisualizer()
    out_error = results / "MAPE_mst.pdf"
    out_raw = results / "RAW_diff_mst.pdf"
    out_runtime = results / "runtime_mst.pdf"
    vis.plot_genomes_mape(df=df, output_path=out_error)
    vis.plot_phase_runtime(df=df, output_path=out_runtime)
    vis.plot_raw_k_difference(df=df, output_path=out_raw)
    print("Done! :)")
