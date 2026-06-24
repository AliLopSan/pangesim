"""Module for assessing optimization performance of pangenome reconstruction."""

from typing import Any
from typing import Dict
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class HeuristicPerformanceVisualizer:
    """Generates publication-quality diagnostic plots tracking heuristic convergence."""

    def __init__(self) -> None:
        """Initializes the visualizer with LaTeX typesetting preferences."""
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "text.latex.preamble": r"\usepackage{amsmath}",
            }
        )
        sns.set_theme(style="whitegrid")

    def plot_trajectory_dashboard(
        self, tracker_history: List[Dict[str, Any]], save_path: str | None = None
    ) -> None:
        """Creates a dual-panel dashboard plotting objective score and metrics.

        Args:
            tracker_history: The accumulated dictionary log list from PipelineTracker.
            save_path: Optional file path to export the resulting figure.
        """
        df = pd.DataFrame(tracker_history)

        # Create a clean multi-panel canvas
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # Panel 1: Optimization Objective Curve (Pan Score)
        sns.lineplot(
            data=df, x="Iteration", y="Score", ax=axes[0], color="#5B2C6F", marker="o", lw=2
        )
        axes[0].set_title(r"\textbf{Optimization Score Trajectory}", fontsize=13)
        axes[0].set_xlabel(r"Refinement Iterations ($t$)")
        axes[0].set_ylabel(r"Score($P$) $= \alpha k - \gamma \sum(m_uv - w_uv)^2$")

        # Panel 2: Topological Precision and Recall Divergence
        df_melted = df.melt(
            id_vars=["Iteration"],
            value_vars=["Number of Genomes Delta", "Number of Core Genes Delta"],
            var_name="Metric",
            value_name="Value",
        )

        # Clean up metric names for the LaTeX legend
        df_melted["Metric"] = df_melted["Metric"].map(
            {
                "Number of Genomes Delta": r"$|k_{true} - k_{inferred}|$",
                "Number of Core Genes Delta": r"$|C_{true} - C_{inferred}|$",
            }
        )

        sns.lineplot(
            data=df_melted, x="Iteration", y="Value", hue="Metric", ax=axes[1], lw=2.5, marker="s"
        )
        axes[1].set_title(r"\textbf{Global Differences}", fontsize=13)
        axes[1].set_xlabel(r"Refinement Iterations ($t$)")
        axes[1].set_ylabel(r"Value")
        # axes[1].set_ylim(-0.05, 1.05)
        axes[1].legend(loc="best")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[Visualizer] Saved performance report dashboard to: {save_path}")
        else:
            plt.show()
