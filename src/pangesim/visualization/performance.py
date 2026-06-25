"""Module for assessing optimization performance of pangenome reconstruction."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class BaseVisualizer:
    """Configures global LaTeX rendering and typography size preferences for plots."""

    def __init__(self, title_size: int = 16, label_size: int = 14, tick_size: int = 12) -> None:
        """Initializes global matplotlib runtime configuration (rc) parameters.

        Args:
            title_size: Font size for all plot headers.
            label_size: Font size for x and y axis titles.
            tick_size: Font size for numerical/categorical axis markers.
        """
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "text.latex.preamble": r"\usepackage{amsmath}",
                "axes.titlesize": title_size,  # Controls ax.set_title size
                "axes.labelsize": label_size,  # Controls x/ylabel size
                "xtick.labelsize": tick_size,  # Controls x-axis tick scale
                "ytick.labelsize": tick_size,  # Controls y-axis tick scale
                "legend.fontsize": tick_size,  # Controls internal legend text
            }
        )
        sns.set_theme(style="whitegrid")


class TrajectoryVisualizer(BaseVisualizer):
    """Handles single-scenario evaluation plots tracking optimization over iterations."""

    def __init__(self, **kwargs) -> None:
        """Forwards sizing arguments."""
        super().__init__(**kwargs)

    def plot_score(
        self, tracker_history: List[Dict[str, Any]], save_path: str | None = None
    ) -> None:
        """Plots the evolution of inferred pangenome score.

        Args:
            tracker_history: The accumulated dictionary log list from PipelineTracker.
            save_path: Optional file path to export the resulting figure.
        """
        df = pd.DataFrame(tracker_history)
        plt.figure(figsize=(8, 4))
        sns.lineplot(data=df, x="Iteration", y="Score", color="#5B2C6F", marker="o", lw=2)

        # 2. Identify where Phase 3 ends to locate the Phase 4 demarcation boundary
        p3_iters = [row["Iteration"] for row in tracker_history if "Phase 3" in row["Step"]]

        if p3_iters:
            p4_start_idx = max(p3_iters) + 1

            # Finding if Phase 4 actually happened or not
            if p4_start_idx in df["Iteration"].values:
                plt.axvline(
                    x=p4_start_idx,
                    color="#D35400",
                    linestyle=":",
                    lw=2,
                    label=r"\text{Phase 4 Activation}",
                )

                # Placing latex text next to the boundary line.
                y_max = df["Score"].max()
                plt.text(
                    p4_start_idx + 0.1,
                    y_max,
                    r"\textbf{Phase 4 start}",
                    color="#D35400",
                    fontsize=10,
                    rotation=90,
                    va="top",
                    ha="left",
                )
        plt.title(r"\textbf{Optimization Score Trajectory}")
        plt.xlabel(r"Iterations ($t$)")
        plt.ylabel(r"Score($P$) $= \alpha k - \gamma \sum(m_uv - w_uv)^2$")
        self._finalize_plot(save_path)

    def plot_k_bounds(
        self,
        tracker_history: List[Dict[str, Any]],
        kmin: int,
        kmax: int,
        save_path: str | None = None,
    ) -> None:
        """Tracks change on the number of genomes.

        Args:
            tracker_history: The accumulated dictionary log list from PipelineTracker.
            kmin: Lower bound computed by heuristic.
            kmax: Upper bound computed by heuristic.
            save_path: Optional file path to export the resulting figure.
        """
        df = pd.DataFrame(tracker_history)
        plt.figure(figsize=(8, 4))

        sns.lineplot(
            data=df, x="Iteration", y="Number of Genomes Delta",
            color="#1F618D", marker="s", lw=2
        )
        plt.axhline(y=kmin, color="crimson", linestyle="--", alpha=0.8)
        plt.axhline(y=kmax, color="crimson", linestyle="--", alpha=0.8)
        plt.text(
            0, kmin + 0.1, fr"$k_{{\min}} = {kmin}$",
            color="crimson", fontsize=11, va="bottom"
        )
        plt.text(
            0, kmax + 0.1, fr"$k_{{\max}} = {kmax}$",
            color="crimson", fontsize=11, va="bottom"
        )

        plt.title(r"\textbf{Genome Count Tracking}")
        plt.xlabel(r"Iterations ($t$)")
        plt.ylabel(r"$|k_{true} - k_{inferred}|$")

        self._finalize_plot(save_path)

    def plot_core_diff(
        self,
        tracker_history: List[Dict[str, Any]],
        save_path: str | None = None,
    ) -> None:
        """Plots difference in core size w.r.t. reference.

        Args:
            tracker_history: The accumulated dictionary log list from PipelineTracker.
            save_path: Optional file path to export the resulting figure.
        """
        df = pd.DataFrame(tracker_history)
        plt.figure(figsize=(8, 4))

        sns.lineplot(
            data=df, x="Iteration", y="Number of Core Genes Delta",
            color="#1F618D", marker="s", lw=2
        )

        plt.title(r"\textbf{Core Genes Count Tracking}")
        plt.xlabel(r"Iterations ($t$)")
        plt.ylabel(r"$|C_{true}| - |C_{inferred}|$")

        self._finalize_plot(save_path)

    def _finalize_plot(self, save_path: Optional[str]) -> None:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
