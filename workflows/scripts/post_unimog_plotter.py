"""Harry Plotter version UNIMOG."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pangesim.io.unimog import parse_dcj_id_matrix
from pangesim.visualization.distances import MWMComparisonVisualizer
from workflows.scripts.unit_mwm_comparison import compute_mwm_evaluation


def enrich_summary_with_mwm(results_root: Path) -> pd.DataFrame:
    """Adds mwm results to master dataframe, computes mwm dumbell plots."""
    summary_csv = results_root / "experiment_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary file not found at {summary_csv}")

    df_summary = pd.read_csv(summary_csv)

    # Initialize MWM metric columns if they don't exist
    for col in ["mwm_min", "mwm_max", "mwm_average"]:
        if col not in df_summary.columns:
            df_summary[col] = np.nan

    for idx, row in df_summary.iterrows():
        size = row["gene_size"]
        level = row["complexity"]
        rep = row["replicate"]
        method = row["method"]

        rep_dir = results_root / str(size) / str(level) / f"rep_{rep}"

        dcj_file = rep_dir / "dcj_id_results.txt"
        if not dcj_file.exists():
            dcj_file = rep_dir / "dcj_id_matrix.pkl"
        if not dcj_file.exists():
            dcj_file = rep_dir / "dcj_matrix.pkl"

        if dcj_file.exists():
            try:
                if dcj_file.suffix == ".pkl":
                    df_dcj = pd.read_pickle(dcj_file)
                else:
                    df_dcj = parse_dcj_id_matrix(dcj_file)

                mst_matching, seq_matching, sim_labels = compute_mwm_evaluation(df_dcj)
                matching = mst_matching if method == "MST" else seq_matching

                vals = list(matching.values())
                if vals:
                    df_summary.at[idx, "mwm_min"] = float(min(vals))
                    df_summary.at[idx, "mwm_max"] = float(max(vals))
                    df_summary.at[idx, "mwm_average"] = float(sum(vals) / len(vals))

                # Generate MWM visualizer comparison plots per replicate
                vis_dir = rep_dir / "plots"
                vis_dir.mkdir(parents=True, exist_ok=True)
                vis = MWMComparisonVisualizer()
                vis.plot_dumbbell_comparison(
                    mst_matching=mst_matching,
                    seq_matching=seq_matching,
                    sim_labels=sim_labels,
                    save_path=vis_dir,
                )

            except Exception as e:
                print(f"Warning: Could not process MWM for {dcj_file}: {e}")

    # Save enriched master dataframe
    df_summary.to_csv(summary_csv, index=False)
    print(f"Successfully enriched {summary_csv} with MWM metrics.")
    return df_summary


def plot_dcj_distance_distribution(
    df_matrix: pd.DataFrame,
    group_a_prefix: str = "simulated_",
    group_b_prefix: str = "MST_",
    mst_color: str = "#5003C0",
    seq_color: str = "#FF467A",
    figsize: tuple[int, int] = (12, 7),
) -> tuple[plt.Figure, plt.Axes]:
    """Extracts pairwise distances and plots their distribution."""
    custom_rc = {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}",
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
    sns.set_theme(style="whitegrid", rc=custom_rc)

    rows = [idx for idx in df_matrix.index if idx.startswith(group_a_prefix)]
    cols = [col for col in df_matrix.columns if col.startswith(group_b_prefix)]

    if not rows or not cols:
        raise ValueError(
            f"No matching genome labels found for: '{group_a_prefix}' or '{group_b_prefix}'"
        )

    sub_matrix = df_matrix.loc[rows, cols]
    valid_distances = sub_matrix.to_numpy().flatten()
    valid_distances = valid_distances[~np.isnan(valid_distances)]

    if len(valid_distances) == 0:
        raise ValueError("No valid distance comparisons found (all values are NaN).")

    fig, ax = plt.subplots(figsize=figsize)
    color = mst_color if group_b_prefix == "MST_" else seq_color

    sns.histplot(
        valid_distances,
        discrete=True,
        kde=False,
        edgecolor="white",
        color=color,
        ax=ax,
    )

    ax.set_xlabel(r"DCJ Distance", fontsize=15)
    ax.set_ylabel(r"Pairwise Count", fontsize=15)

    plt.tight_layout()
    return fig, ax


def plot_paired_dcj_range_lollipops(
    df_matrix: pd.DataFrame,
    gt_prefix: str = "simulated_",
    mst_prefix: str = "MST_",
    seq_prefix: str = "SEQ_",
    mst_color: str = "#5003C0",
    seq_color: str = "#FF467A",
    title: str = r"DCJ Distance Range (Min--Max) per Genome",
    figsize: tuple[int, int] = (16, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Plots paired range lollipops showing min to max DCJ distances."""
    custom_rc = {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}",
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
    sns.set_theme(style="whitegrid", rc=custom_rc)
    gt_rows = [idx for idx in df_matrix.index if idx.startswith(gt_prefix)]
    mst_cols = [col for col in df_matrix.columns if col.startswith(mst_prefix)]
    seq_cols = [col for col in df_matrix.columns if col.startswith(seq_prefix)]

    if not gt_rows or not mst_cols or not seq_cols:
        raise ValueError(
            f"Could not find matching labels for prefixes: "
            f"'{gt_prefix}', '{mst_prefix}', or '{seq_prefix}'"
        )

    mst_min = df_matrix.loc[gt_rows, mst_cols].min(axis=1)
    mst_max = df_matrix.loc[gt_rows, mst_cols].max(axis=1)
    seq_min = df_matrix.loc[gt_rows, seq_cols].min(axis=1)
    seq_max = df_matrix.loc[gt_rows, seq_cols].max(axis=1)

    x_indices = np.arange(1, len(gt_rows) + 1)
    offset = 0.15

    fig, ax = plt.subplots(figsize=figsize)

    ax.vlines(
        x=x_indices - offset,
        ymin=mst_min,
        ymax=mst_max,
        colors=mst_color,
        linewidth=2.5,
        alpha=0.85,
    )
    ax.scatter(
        x=x_indices - offset,
        y=mst_min,
        color=mst_color,
        s=40,
        zorder=3,
        facecolors="white",
        edgecolors=mst_color,
        linewidth=1.5,
    )
    ax.scatter(
        x=x_indices - offset,
        y=mst_max,
        color=mst_color,
        s=80,
        zorder=3,
        label=r"MST Range (Min--Max)",
    )

    ax.vlines(
        x=x_indices + offset,
        ymin=seq_min,
        ymax=seq_max,
        colors=seq_color,
        linewidth=2.5,
        alpha=0.85,
    )
    ax.scatter(
        x=x_indices + offset,
        y=seq_min,
        color=seq_color,
        s=40,
        zorder=3,
        facecolors="white",
        edgecolors=seq_color,
        linewidth=1.5,
    )
    ax.scatter(
        x=x_indices + offset,
        y=seq_max,
        color=seq_color,
        s=80,
        zorder=3,
        label=r"SEQ Range (Min--Max)",
    )

    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(i) for i in x_indices])
    ax.set_xlim(0.5, len(gt_rows) + 0.5)

    overall_max = max(mst_max.max(), seq_max.max())
    padding = overall_max * 0.20
    ax.set_ylim(bottom=-5, top=overall_max + padding)

    ax.set_title(title, pad=20)
    ax.set_xlabel(r"Ground Truth Genome Index")
    ax.set_ylabel(r"DCJ Distance Range")
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig, ax


def plot_summary_mwm(
    df_summary: pd.DataFrame,
    mst_color: str = "#5003C0",
    seq_color: str = "#FF467A",
    title: str = r"Average MWM Distance Across Gene Sizes and Complexity",
    figsize: tuple[int, int] = (14, 7),
) -> tuple[plt.Figure, plt.Axes]:
    """Plots average MWM evaluation distance from the experiment summary DataFrame."""
    custom_rc = {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}",
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
    sns.set_theme(style="whitegrid", rc=custom_rc)
    fig, ax = plt.subplots(figsize=figsize)

    palette = {"MST": mst_color, "SEQ": seq_color}

    sns.lineplot(
        data=df_summary,
        x="gene_size",
        y="mwm_average",
        hue="method",
        style="complexity",
        markers=True,
        dashes=False,
        palette=palette,
        linewidth=2,
        ax=ax,
    )

    ax.set_title(title, pad=20)
    ax.set_xlabel(r"Gene Size")
    ax.set_ylabel(r"Average MWM Distance")
    ax.legend(loc="upper left", title=r"\textbf{Method / Complexity}")

    plt.tight_layout()
    return fig, ax


def plot_genome_sizes(
    df: pd.DataFrame,
    genome_col: str = "genome id",
    size_col: str = "genome size",
    color: str = "#5003C0",
    figsize: tuple[int, int] = (14, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a bar chart of genome sizes directly using Pandas/Matplotlib."""
    custom_rc = {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}",
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
    sns.set_theme(style="whitegrid", rc=custom_rc)
    sizes_per_genome = df.groupby(genome_col)[size_col].first()

    fig, ax = plt.subplots(figsize=figsize)
    sizes_per_genome.plot(kind="bar", color=color, ax=ax, width=0.7)

    n_genomes = len(sizes_per_genome)
    ax.set_xticks(range(n_genomes))
    ax.set_xticklabels(range(1, n_genomes + 1), rotation=0)

    plt.tick_params(axis="x", direction="out", rotation=45)

    ax.set_title(r"Genome Size per Genome", pad=15)
    ax.set_xlabel(r"Genome")
    ax.set_ylabel(r"Genome Size (Gene Count)")
    plt.tight_layout()

    return fig, ax


def plot_pangenome_metrics(results_dir: Path, source_dir: Path, pname: str) -> None:
    """Wrapper function to plot individual pangenome metrics."""
    gen = pname + "_genome.csv"
    csv_gen = source_dir / gen
    outdir = results_dir / pname
    outdir.mkdir(parents=True, exist_ok=True)

    if csv_gen.exists():
        genomes = pd.read_csv(csv_gen)
        fig, ax = plot_genome_sizes(df=genomes)
        plt.savefig(outdir / "genome_size_distribution.pdf", dpi=350)
        plt.close()


if __name__ == "__main__":
    results_root = Path("results/run_20261022/")

    # Add MWM calculations to master dataframe from UniMoG output
    if (results_root / "experiment_summary.csv").exists():
        df_summary = enrich_summary_with_mwm(results_root)

        # 2. Render summary plot for average MWM distances
        summary_plots_dir = results_root / "summary_plots"
        summary_plots_dir.mkdir(parents=True, exist_ok=True)

        fig_mwm, ax_mwm = plot_summary_mwm(df_summary)
        fig_mwm.savefig(summary_plots_dir / "summary_mwm_average.pdf", dpi=350)
        plt.close(fig_mwm)

    # 3. Process individual pangenome metric plots
    levels = ["low", "medium", "high"]
    gene_sizes = [25, 100, 200, 500, 800, 1000]

    for size in gene_sizes:
        size_dir = results_root / str(size)
        if not size_dir.exists():
            continue

        for level in levels:
            level_dir = size_dir / level
            if not level_dir.exists():
                continue

            plots_dir = level_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            for p in ["simulated", "mst", "seq"]:
                plot_pangenome_metrics(
                    results_dir=plots_dir,
                    source_dir=level_dir,
                    pname=p,
                )
