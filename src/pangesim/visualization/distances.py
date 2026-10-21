from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class MWMComparisonVisualizer():
    """Handles comparative MWM distance visualization between MST and SEQ methods."""

    def __init__(self, title_size: int = 50, label_size: int = 25, tick_size: int = 30) -> None:
        """Initializes global matplotlib runtime configuration (rc) parameters.

        Args:
            title_size: Font size for all plot headers.
            label_size: Font size for x and y axis titles.
            tick_size: Font size for numerical/categorical axis markers.
        """
        # Latex-compatible configuration for papers
        custom_rc = {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath}",
            "axes.titlesize": 18,  # Controls ax.set_title size
            "axes.labelsize": 16,  # Controls x/ylabel size
            "xtick.labelsize": 14,  # Controls x-axis tick scale
            "ytick.labelsize": 14,  # Controls y-axis tick scale
            "legend.fontsize": 14,  # Controls internal legend text
        }
        sns.set_theme(style="whitegrid", rc=custom_rc)

    def _finalize_plot(self, save_path: Path) -> None:
        """Function to store or show finalized plot."""
        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the actual file safely
            fname = save_path / "MWM_dumbbell_comparison.pdf"
            plt.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_dumbbell_comparison(
        self,
        mst_matching: dict[tuple[str, str], float],
        seq_matching: dict[tuple[str, str], float],
        sim_labels: list[str],
        save_path: Path | None = None,
    ) -> None:
        """Generates a professional comparative dumbbell stem plot for MST vs SEQ match distances.

        Args:
            mst_matching: Dictionary mapping (sim_name, mst_name) to distance.
            seq_matching: Dictionary mapping (sim_name, seq_name) to distance.
            sim_labels: List of simulated ground-truth genome labels.
            save_path: Optional file path to export the resulting publication-ready figure.
        """
        plt.figure(figsize=(12, 8))

        x_indices = np.arange(len(sim_labels))
        mst_distances = []
        seq_distances = []

        for sim in sim_labels:
            mst_dist = next((dist for (s, _), dist in mst_matching.items() if s == sim), np.nan)
            mst_distances.append(mst_dist)
            seq_dist = next((dist for (s, _), dist in seq_matching.items() if s == sim), np.nan)
            seq_distances.append(seq_dist)

        # Portfolio color palette
        mst_color = "#5003C0"
        seq_color = "#FF467A"

        for i, (mst_dist, seq_dist) in enumerate(zip(mst_distances, seq_distances)):
            # Draw the connecting vertical stem (dumbbell bridge) between MST and SEQ
            plt.plot(
                [i, i],
                [mst_dist, seq_dist],
                color="#B0B0B0",
                linestyle="-",
                linewidth=2.5,
                zorder=1,
            )

            # Plot prominent markers for MST and SEQ endpoints (labels added once for legend)
            plt.scatter(
                i,
                mst_dist,
                color=mst_color,
                s=120,
                zorder=2,
                label=r"\textbf{MST}" if i == 0 else "",
                edgecolors="black",
                linewidth=0.5,
            )
            plt.scatter(
                i,
                seq_dist,
                color=seq_color,
                s=120,
                zorder=2,
                label=r"\textbf{SEQ}" if i == 0 else "",
                edgecolors="black",
                linewidth=0.5,
            )

        plt.xticks(x_indices, [str(i) for i in x_indices])
        plt.xlabel(r"Simulated Genome Index $(G_i)$", fontsize=14)
        plt.ylabel(r"Minimum Weight Matching Distance ($d_{\text{MWM}}$)", fontsize=14)
        #plt.title(r"\textbf{Comparative Reconstruction Performance: MST vs SEQ}", fontsize=16)

        plt.legend(
            loc="best",
            title=r"\textbf{Method}",
            frameon=True,
            fancybox=True,
            shadow=True,
        )
        plt.grid(True, linestyle="--", alpha=0.6)

        # Utilize the robust base class finalizer for directory creation and saving
        self._finalize_plot(save_path)
