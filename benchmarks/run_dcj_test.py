"""DCJ test with UNIMOG."""

from pangesim.visualization.performance import BaseVisualizer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from pathlib import Path

from benchmarks.fixtures import random_simulated_pangenome
from pangesim.io.unimog import export_to_unimog
from pangesim.io.unimog import parse_dcj_matrix
from pangesim.reconstruction import SequentialEdgeInsertion
from pangesim.reconstruction.assignment import MSTAssignment
from pangesim.reconstruction.base import AdjacencyMatrix
from pangesim.reconstruction.refining import SequentialEdgeRefinement
from pangesim.reconstruction.utils import GlobalGenomePool



def run_seq_inference(m:AdjacencyMatrix):
    """Sequential edge addition method."""
    heuristic = SequentialEdgeInsertion()
    inf_pangenome = heuristic.reconstruct(m)
    return inf_pangenome


def run_mst_inference(m:AdjacencyMatrix):
    """Maximum Spanning Tree inference."""
    assign = MSTAssignment()
    id_pool = GlobalGenomePool(start_id=1)
    base_pangenome = assign.assign_genomes(m, id_pool)
    refiner = SequentialEdgeRefinement(id_pool)
    inf_pangenome = refiner.refine(source=m, target=base_pangenome)
    return inf_pangenome

def generate_pan():
    """UNIMOG experiment."""
    num_genes = 1500
    results = Path("results/run_20260908")
    results.mkdir(parents=True, exist_ok=True)

    #Ground Truth generation
    ground_truth = random_simulated_pangenome(num_genes)
    f = results / "ground_truth.unimog"
    export_to_unimog(f, ground_truth)

    #MST inference
    matrix = ground_truth.compute_weighted_adjacencies()

    #MST inference
    mst = run_mst_inference(matrix)
    mst.check_integrity()
    f2 = results / "mst_pangenome.unimog"
    export_to_unimog(f2, mst)

    #Sequential inference
    seq = run_seq_inference(matrix)
    seq.check_integrity()
    f3 = results / "seq_pangenome.unimog"
    export_to_unimog(f3, seq)


def plot_dcj_heatmap(
    df_matrix: pd.DataFrame,
    group_a_prefix: str = "simulated_",
    group_b_prefix: str = "MST_",
    title: str = r"\textbf{DCJ Distances: Simulated vs MST}",
    cbar_label: str = "DCJ Distance",
    figsize: tuple[int, int] = (14, 10),
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a seaborn DCJ heatmap using simple numeric ticks and BaseVisualizer styling."""
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"})

    nan_color = "white"
    # Filter rows and columns matching the group prefixes
    rows = [idx for idx in df_matrix.index if idx.startswith(group_a_prefix)]
    cols = [col for col in df_matrix.columns if col.startswith(group_b_prefix)]

    if not rows or not cols:
        raise ValueError(
            f"No matching genome labels found for prefixes: '{group_a_prefix}' or '{group_b_prefix}'"
        )

    sub_matrix = df_matrix.loc[rows, cols]

    fig, ax = plt.subplots(figsize=figsize)

    #NaN will appear in this color
    ax.set_facecolor(nan_color)

    #Keeping viridis + white
    current_cmap = plt.cm.get_cmap("viridis").copy()
    current_cmap.set_bad(color=nan_color)

    # Render heatmap with Viridis
    cbar_kws = {"label": cbar_label}
    sns.heatmap(
        sub_matrix,
        cmap=current_cmap,
        annot=False,
        cbar_kws=cbar_kws,
        ax=ax,
        linewidths=0.25,
        linecolor='white',
        mask=sub_matrix.isna(),
    )

    # Override text tick labels with clean numerical indices (1, 2, 3...)
    x_indices = np.arange(1, len(cols) + 1)
    y_indices = np.arange(1, len(rows) + 1)

    # Set integer ticks at every item (or step if matrix is dense)
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(x_indices, rotation=0)

    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels(y_indices, rotation=0)

    # Axis Labels (formatted for LaTeX rendering enabled by BaseVisualizer)
    #ax.set_title(title, pad=20)
    if group_b_prefix == "MST_":
        ax.set_xlabel(r"MST Genomes")
    if group_b_prefix == "SEQ_":
        ax.set_xlabel(r"SEQ Genomes")
    ax.set_ylabel(r"Simulated Genomes")

    plt.tight_layout()
    return fig, ax


def plot_dcj_distance_distribution(
    df_matrix: pd.DataFrame,
    group_a_prefix: str = "simulated_",
    group_b_prefix: str = "MST_",
    mst_color: str = "#5003C0",
    seq_color: str = "#FF467A",
    figsize: tuple[int, int] = (12, 7)
) -> tuple[plt.Figure, plt.Axes]:
    """Extracts pairwise distances between two groups, removes NaNs, and plots their distribution."""
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

    tick_step = 1
    # Filter rows and columns for target prefixes
    rows = [idx for idx in df_matrix.index if idx.startswith(group_a_prefix)]
    cols = [col for col in df_matrix.columns if col.startswith(group_b_prefix)]

    if not rows or not cols:
        raise ValueError(
            f"No matching genome labels found for: '{group_a_prefix}' or '{group_b_prefix}'"
        )

    # Extract sub-matrix and flatten, dropping NaNs
    sub_matrix = df_matrix.loc[rows, cols]
    valid_distances = sub_matrix.to_numpy().flatten()
    valid_distances = valid_distances[~np.isnan(valid_distances)]

    if len(valid_distances) == 0:
        raise ValueError("No valid distance comparisons found (all values are NaN).")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram with KDE curve using Viridis color
    viridis_color = plt.cm.viridis(0.3)  # Distinct dark teal/blue from viridis

    if group_b_prefix == "MST_":
        color = mst_color
    else:
        color = seq_color

    sns.histplot(
        valid_distances,
        discrete=True,
        kde=False,
        edgecolor="white",
        color=color,
        ax=ax,
    )

    # Explicitly force integer tick labels on the x-axis
    min_val = int(np.floor(valid_distances.min()))
    max_val = int(np.ceil(valid_distances.max()))
    
    # Generate integer ticks
    """
    x_ticks = np.arange(min_val, max_val + 1, tick_step)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks], rotation=0, fontsize=5)
    """
    # Axis Labels & Formatting
    #ax.set_title(title, pad=20)
    ax.set_xlabel(r"DCJ Distance",fontsize=15)
    ax.set_ylabel(r"Pairwise Count",fontsize=15)

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
    """Plots paired range lollipops showing the min (start) to max (end) DCJ distances

    from MST and SEQ to each ground truth genome.
    """
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
    # Filter genome labels
    gt_rows = [idx for idx in df_matrix.index if idx.startswith(gt_prefix)]
    mst_cols = [col for col in df_matrix.columns if col.startswith(mst_prefix)]
    seq_cols = [col for col in df_matrix.columns if col.startswith(seq_prefix)]

    if not gt_rows or not mst_cols or not seq_cols:
        raise ValueError(
            f"Could not find matching labels for prefixes: "
            f"'{gt_prefix}', '{mst_prefix}', or '{seq_prefix}'"
        )

    # Compute min and max distances per ground truth genome
    mst_min = df_matrix.loc[gt_rows, mst_cols].min(axis=1)
    mst_max = df_matrix.loc[gt_rows, mst_cols].max(axis=1)

    seq_min = df_matrix.loc[gt_rows, seq_cols].min(axis=1)
    seq_max = df_matrix.loc[gt_rows, seq_cols].max(axis=1)

    # Setup positions and offsets
    x_indices = np.arange(1, len(gt_rows) + 1)
    offset = 0.15

    fig, ax = plt.subplots(figsize=figsize)

    # --- 1. MST Range Stems & Markers ---
    # Range Stem (Min to Max)
    ax.vlines(
        x=x_indices - offset,
        ymin=mst_min,
        ymax=mst_max,
        colors=mst_color,
        linewidth=2.5,
        alpha=0.85,
    )
    # Min Marker (Smaller dot)
    ax.scatter(
        x_indices - offset,
        mst_min,
        color=mst_color,
        s=40,
        zorder=3,
        facecolors="white",
        edgecolors=mst_color,
        linewidth=1.5,
    )
    # Max Marker (Solid cap/dot)
    ax.scatter(
        x_indices - offset,
        mst_max,
        color=mst_color,
        s=80,
        zorder=3,
        label=r"MST Range (Min--Max)",
    )

    # --- 2. SEQ Range Stems & Markers ---
    # Range Stem (Min to Max)
    ax.vlines(
        x=x_indices + offset,
        ymin=seq_min,
        ymax=seq_max,
        colors=seq_color,
        linewidth=2.5,
        alpha=0.85,
    )
    # Min Marker (Hollow circle)
    ax.scatter(
        x_indices + offset,
        seq_min,
        color=seq_color,
        s=40,
        zorder=3,
        facecolors="white",
        edgecolors=seq_color,
        linewidth=1.5,
    )
    # Max Marker (Solid circle)
    ax.scatter(
        x_indices + offset,
        seq_max,
        color=seq_color,
        s=80,
        zorder=3,
        label=r"SEQ Range (Min--Max)",
    )

    # Formatting x-axis
    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(i) for i in x_indices])
    ax.set_xlim(0.5, len(gt_rows) + 0.5)

    # Find global maximum distance across both methods
    overall_max = max(mst_max.max(), seq_max.max())

    # Add 20% headroom above overall_max to make room for the legend
    padding = overall_max * 0.20
    ax.set_ylim(bottom=-5,
                top=overall_max + padding)

    # Labels and Legend
    ax.set_title(title, pad=20)
    ax.set_xlabel(r"Ground Truth Genome Index")
    ax.set_ylabel(r"DCJ Distance Range")
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig, ax

def plot_paired_dcj_boxplots(
    df_matrix: pd.DataFrame,
    gt_prefix: str = "simulated_",
    mst_prefix: str = "MST_",
    seq_prefix: str = "SEQ_",
    mst_color: str = "#5003C0",
    seq_color: str = "#FF467A",
    title: str = r"\textbf{DCJ Distance Distributions per Ground Truth Genome}",
    figsize: tuple[int, int] = (16, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Plots paired boxplots comparing the distribution of DCJ distances from MST and SEQ

    to each individual ground truth genome.
    """
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

    # Filter labels
    gt_rows = [idx for idx in df_matrix.index if idx.startswith(gt_prefix)]
    mst_cols = [col for col in df_matrix.columns if col.startswith(mst_prefix)]
    seq_cols = [col for col in df_matrix.columns if col.startswith(seq_prefix)]

    if not gt_rows or not mst_cols or not seq_cols:
        raise ValueError(
            f"Could not find matching labels for prefixes: "
            f"'{gt_prefix}', '{mst_prefix}', or '{seq_prefix}'"
        )

    # Melt sub-matrices into a long-format DataFrame suitable for Seaborn boxplot
    records = []
    for i, gt_id in enumerate(gt_rows, start=1):
        # Collect MST distances
        mst_vals = df_matrix.loc[gt_id, mst_cols].dropna().values
        for val in mst_vals:
            records.append({"GT_Index": i, "Distance": val, "Method": "MST"})

        # Collect SEQ distances
        seq_vals = df_matrix.loc[gt_id, seq_cols].dropna().values
        for val in seq_vals:
            records.append({"GT_Index": i, "Distance": val, "Method": "SEQ"})

    df_long = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=figsize)

    # Custom palette matching exact hexadecimal inputs
    palette = {"MST": mst_color, "SEQ": seq_color}

    # Render paired boxplots
    sns.boxplot(
        data=df_long,
        x="GT_Index",
        y="Distance",
        hue="Method",
        palette=palette,
        width=0.6,
        fliersize=3,
        linewidth=1.2,
        ax=ax,
    )

    # Axis Labels & Title
    ax.set_title(title, pad=20)
    ax.set_xlabel(r"Ground Truth Genome Index")
    ax.set_ylabel(r"DCJ Distance")

    # Clean legend formatting
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles,
        labels=[rf"\textbf{{{lbl}}}" for lbl in labels],
        loc="upper right",
    )

    plt.tight_layout()
    return fig, ax

def get_matrix():
    
    results = Path("results/run_20260908")
    f = results / "dcj_results.txt"
    df = parse_dcj_matrix(f)

    print(df.head())



    fig, ax = plot_paired_dcj_range_lollipops(df)
    pname = results / "range_lollipop.pdf"
    plt.savefig(pname, dpi=350)
    plt.show()
    plt.close()

    fig1, ax1 = plot_paired_dcj_boxplots(df)
    pname = results / "paired_boxplot.pdf"
    plt.savefig(pname, dpi=350)
    plt.show()
    plt.close()

    fig2, ax2 = plot_dcj_distance_distribution(df,
                               group_a_prefix="simulated_",
                               group_b_prefix="MST_")
    pname = results / "mst_distribution.pdf"
    plt.savefig(pname, dpi=350)
    plt.show()
    plt.close()

    fig2, ax2 = plot_dcj_distance_distribution(df,
                               group_a_prefix="simulated_",
                               group_b_prefix="SEQ_")
    pname = results / "seq_distribution.pdf"
    plt.savefig(pname, dpi=350)
    plt.show()
    plt.close()
    


if __name__ == "__main__":
    #generate_pan()
    get_matrix()
