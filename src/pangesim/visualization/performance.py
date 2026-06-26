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

        # Identify where Phase 3 ends to locate the Phase 4 demarcation boundary
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

    def plot_score_from_dataframe(
        self,
        config_df: pd.DataFrame,
        strategy_label: str,
        config_label: str,
        save_path: str | None = None,
    ) -> None:
        """Plots the score evolution with an optional vertical line marking Phase 4 start.

        Args:
            config_df: DataFrame containing tracking rows for a single run configuration.
            strategy_label: The display string name of the active strategy.
            config_label: The LaTeX formatted string describing the current hyperparameters.
            save_path: Optional file path to export the resulting figure.
        """
        plt.figure(figsize=(8, 4))

        # Optimization trajectory curve
        sns.lineplot(data=config_df, x="Iteration", y="Score", color="#5B2C6F", marker="o", lw=2)

        y_min = config_df["Score"].min()
        y_max = config_df["Score"].max()
        y_range = y_max - y_min if y_max != y_min else 1.0
        text_baseline = y_min + (y_range * 0.05)

        # Identify where Phase 2 ends to mark Phase 3 Start
        p2_df = config_df[config_df["Step"].str.contains("Phase 2: Base Pangenome", na=False)]
        if not p2_df.empty:
            p3_start_idx = p2_df["Iteration"].max() + 1
            if p3_start_idx in config_df["Iteration"].values:
                plt.axvline(
                    x=p3_start_idx,
                    color="#D35400",
                    linestyle=":",
                    lw=1.5,
                    label=r"\text{Phase 3 Activation}",
                    alpha=0.6,
                )
                plt.text(
                    p3_start_idx - 0.1,
                    text_baseline,
                    r"\textbf{Phase 3 start}",
                    color="#D35400",
                    fontsize=5,
                    rotation=90,
                    va="bottom",
                    ha="left",
                )

        # Identify where Phase 3 ends to mark Phase 4 Start
        p3_df = config_df[config_df["Step"].str.contains("Phase 3", na=False)]
        if not p3_df.empty:
            p4_start_idx = p3_df["Iteration"].max() + 1
            if p4_start_idx in config_df["Iteration"].values:
                plt.axvline(
                    x=p4_start_idx,
                    color="#D35400",
                    linestyle=":",
                    lw=2,
                    label=r"\text{Phase 4 Activation}",
                    alpha=0.6,
                )
                plt.text(
                    p4_start_idx + 0.1,
                    text_baseline,
                    r"\textbf{Phase 4 start}",
                    color="#D35400",
                    fontsize=5,
                    rotation=90,
                    va="bottom",
                    ha="left",
                )

        # 3. Clean styling configurations
        # clean_config = config_label.replace("$", "")
        title_text = rf"\textbf{{Score Tracking: {strategy_label}}} ({config_label})"
        plt.title(title_text, fontsize=11)
        plt.xlabel(r"Iterations ($t$)")
        plt.ylabel(r"Score($P$) $= \alpha k - \gamma \sum(m_{uv} - w_{uv})^2$")
        plt.tight_layout()

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
            data=df, x="Iteration", y="Number of Genomes Delta", color="#1F618D", marker="s", lw=2
        )
        plt.axhline(y=kmin, color="crimson", linestyle="--", alpha=0.8)
        plt.axhline(y=kmax, color="crimson", linestyle="--", alpha=0.8)
        plt.text(
            0, kmin + 0.1, rf"$k_{{\min}} = {kmin}$", color="crimson", fontsize=11, va="bottom"
        )
        plt.text(
            0, kmax + 0.1, rf"$k_{{\max}} = {kmax}$", color="crimson", fontsize=11, va="bottom"
        )

        plt.title(r"\textbf{Genome Count Tracking}")
        plt.xlabel(r"Iterations ($t$)")
        plt.ylabel(r"$|k_{true} - k_{inferred}|$")

        self._finalize_plot(save_path)

    def plot_k_bounds_from_dataframe(
        self,
        config_df: pd.DataFrame,
        strategy_label: str,
        config_label: str,
        save_path: str | None = None,
    ) -> None:
        """Tracks changes on the number of genomes against computed heuristic bounds.

        Args:
            config_df: DataFrame containing tracking rows for a single run configuration.
            strategy_label: The display string name of the active strategy.
            config_label: The LaTeX formatted string describing the current hyperparameters.
            save_path: Optional file path to export the resulting figure.
        """
        plt.figure(figsize=(8, 4))

        sns.lineplot(
            data=config_df,
            x="Iteration",
            y="Number of Genomes Delta",
            color="#1F618D",
            marker="s",
            lw=2,
        )

        kmin = int(config_df["kmin"].iloc[0])
        kmax = int(config_df["kmax"].iloc[0])

        plt.axhline(y=kmin, color="crimson", linestyle="--", alpha=0.8)
        plt.axhline(y=kmax, color="crimson", linestyle="--", alpha=0.8)

        plt.text(
            0, kmin + 0.1, rf"$k_{{\min}} = {kmin}$", color="crimson", fontsize=11, va="bottom"
        )
        plt.text(
            0, kmax + 0.1, rf"$k_{{\max}} = {kmax}$", color="crimson", fontsize=11, va="bottom"
        )

        plt.title(rf"\textbf{{Genome Count Tracking: {strategy_label}}} ({config_label})")
        plt.xlabel(r"Iterations ($t$)")
        plt.ylabel(r"$|k_{true} - k_{inferred}|$")
        plt.tight_layout()

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
            data=df,
            x="Iteration",
            y="Number of Core Genes Delta",
            color="#1F618D",
            marker="s",
            lw=2,
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

    def plot_strategy_parameters(
        self,
        strategy_df: pd.DataFrame,
        strategy_label: str,
        metric: str = "Score",
        save_path: str | None = None,
    ) -> None:
        """Plots parameter configuration curves for a single reconstruction strategy.

        Args:
            strategy_df: DataFrame containing execution rows filtered for one strategy.
            strategy_label: The clear display string name of the target strategy.
            metric: The target column key to plot on the y-axis (defaults to 'Score').
            save_path: The file path where the generated figure should be exported.
        """
        plt.figure(figsize=(9, 5))

        # Distinct parameter configurations map directly to line colors
        sns.lineplot(
            data=strategy_df,
            x="Iteration",
            y=metric,
            hue="Config",
            marker="o",
            lw=2,
        )

        title_text = rf"\textbf{{{strategy_label}: Parameter Sensitivity}}"

        plt.title(title_text)
        plt.xlabel(r"Iterations ($t$)")
        plt.ylabel(rf"\text{{{metric}}}")

        # Configure a clean LaTeX legend box anchored to the right side
        plt.legend(
            loc="best",
            title=r"\textbf{Parameters}",
        )
        plt.tight_layout()

        self._finalize_plot(save_path)
