"""Harry Plotter version UNIMOG."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pangesim.io.unimog import parse_dcj_matrix


def plot_dcj_distance_distribution(
    df_matrix: pd.DataFrame,
    group_a_prefix: str = "simulated_",
    group_b_prefix: str = "MST_",
    mst_color: str = "#5003C0",
    seq_color: str = "#FF467A",
    figsize: tuple[int, int] = (12, 7)) -> tuple[plt.Figure, plt.Axes]:
    """Extracts pairwise distances and plots their distribution."""
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
    """Plots paired range lollipops showing min to max DCJ distances."""
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
    title: str = r"DCJ Distance Distributions per Ground Truth Genome",
    figsize: tuple[int, int] = (16, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Plots paired boxplots comparing the distribution of DCJ distances."""
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
            "axes.titlesize": 18,  # Controls ax.set_title size
            "axes.labelsize": 18,  # Controls x/ylabel size
            "xtick.labelsize": 14,  # Controls x-axis tick scale
            "ytick.labelsize": 14,  # Controls y-axis tick scale
            "legend.fontsize": 14,  # Controls internal legend text
        }
    sns.set_theme(style="whitegrid", rc=custom_rc)
    # Group by genome ID and get the size
    sizes_per_genome = df.groupby(genome_col)[size_col].first()

    fig, ax = plt.subplots(figsize=figsize)

    sizes_per_genome.plot(kind="bar", color=color, ax=ax, width=0.7)

    # Set tick positions and customize tick labels to 1...n
    n_genomes = len(sizes_per_genome)
    ax.set_xticks(range(n_genomes))
    ax.set_xticklabels(range(1, n_genomes + 1), rotation=0)

    
    plt.tick_params(axis="x", direction='out', rotation=45)

    ax.set_title(r"Genome Size per Genome", pad=15)
    ax.set_xlabel(r"Genome")
    ax.set_ylabel(r"Genome Size (Gene Count)")
    plt.tight_layout()

    return fig, ax

def plot_path_counts_per_genome(
    df: pd.DataFrame,
    pan_col: str = "pan",
    genome_col: str = "genome id",
    path_col: str = "path_id",
    title: str = r"Number of Paths per Genome",
    figsize: tuple[int, int] = (12, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a bar chart showing the count of paths for each genome ID."""
    custom_rc = {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath}",
            "axes.titlesize": 18,  # Controls ax.set_title size
            "axes.labelsize": 18,  # Controls x/ylabel size
            "xtick.labelsize": 14,  # Controls x-axis tick scale
            "ytick.labelsize": 14,  # Controls y-axis tick scale
            "legend.fontsize": 14,  # Controls internal legend text
        }
    sns.set_theme(style="whitegrid", rc=custom_rc)
    # Count unique paths per genome (and per pangenome if pan_col exists)
    group_cols = [col for col in [pan_col, genome_col] if col in df.columns]

    path_counts = (
        df.groupby(group_cols)[path_col]
        .nunique()
        .reset_index(name="path_count")
    )
    

    fig, ax = plt.subplots(figsize=figsize)

    # Render barplot (grouped by pangenome source if available)
    hue_param = pan_col if pan_col in df.columns else None

    sns.barplot(
        data=path_counts,
        x=genome_col,
        y="path_count",
        hue=hue_param,
        palette="viridis" if hue_param else None,
        color="#5003C0" if not hue_param else None,
        ax=ax,
    )

    # Set tick positions and customize tick labels to 1...n
    n_genomes = len(path_counts)
    ax.set_xticks(range(n_genomes))
    ax.set_xticklabels(range(1, n_genomes + 1), rotation=0)

    # Format axis and labels
    ax.set_title(title, pad=15)
    ax.set_xlabel(r"Genome")
    ax.set_ylabel(r"Number of paths")

    
    plt.tick_params(axis="x", direction='out', rotation=45)

    # Ensure integer ticks on y-axis for counts
    ax.yaxis.get_major_locator().set_params(integer=True)
    plt.tight_layout()

    return fig, ax

def plot_path_length_distribution(
    df: pd.DataFrame,
    pan_col: str = "pan",
    genome_col: str = "genome id",
    length_col: str = "length",
    title: str = r"Path Length Distribution per Genome",
    figsize: tuple[int, int] = (14, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plots boxplots showing the distribution of path lengths per genome ID."""
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
    fig, ax = plt.subplots(figsize=figsize)
    hue_param = pan_col if pan_col in df.columns else None

    sns.boxplot(
        data=df,
        x=genome_col,
        y=length_col,
        hue=hue_param,
        palette="viridis" if hue_param else None,
        color="#5003C0" if not hue_param else None,
        fliersize=3,
        linewidth=1.2,
        ax=ax,
    )

    # Formatting
    ax.set_title(title, pad=15)
    ax.set_xlabel(r"Genome ID")
    ax.set_ylabel(r"Path Length")

    # Ensure clean integer ticks on y-axis
    ax.yaxis.get_major_locator().set_params(integer=True)

    plt.tick_params(axis="x", direction='out', rotation=45)

    if hue_param:
        ax.legend(title=r"\textbf{Pangenome}", loc="upper right")

    plt.tight_layout()
    return fig, ax

def plot_edge_weight_distribution(
    df_edges: pd.DataFrame,
    weight_col: str = "weight",
    pan_col: str = "pan",
    title: str = r"Edge Weight Distribution",
    is_discrete: bool = True,
    figsize: tuple[int, int] = (12, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a histogram showing the distribution of edge weights across the graph."""
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

    fig, ax = plt.subplots(figsize=figsize)

    hue_param = pan_col if pan_col in df_edges.columns else None

    sns.histplot(
        data=df_edges,
        x=weight_col,
        hue=hue_param,
        discrete=is_discrete,
        kde=False,
        palette="viridis" if hue_param else None,
        color="#5003C0" if not hue_param else None,
        edgecolor="black",
        alpha=0.5,
        ax=ax,
    )

    # Formatting
    ax.set_title(title, pad=15)
    ax.set_xlabel(r"Edge Weight")
    ax.set_ylabel(r"Count")

    if hue_param:
        sns.move_legend(ax, "upper right", title=r"\textbf{Pangenome}")

    plt.tight_layout()
    return fig, ax

def plot_all_dcjs(results_dir:Path, dcj_file:Path) -> None:
    """Wrapper function.

    Args:
        results_dir: Where you want to put the results.
        dcj_file: The name of the UNIMOG output, should be .txt.
    """
    df = pd.read_pickle(dcj_file)
    #Range lollipop plots
    fig, ax = plot_paired_dcj_range_lollipops(df)
    pname = results_dir / "range_lollipop.pdf"
    plt.savefig(pname, dpi=350)
    #plt.show()
    plt.close()

    #Paired DCJ boxplots
    fig1, ax1 = plot_paired_dcj_boxplots(df)
    pname = results_dir / "paired_boxplot.pdf"
    plt.savefig(pname, dpi=350)
    #plt.show()
    plt.close()

    mst_dir = results_dir / "mst"
    mst_dir.mkdir(parents=True, exist_ok=True)

    fig2, ax2 = plot_dcj_distance_distribution(df,
                               group_a_prefix="simulated_",
                               group_b_prefix="MST_")
    pname = results_dir / "dcj_distribution.pdf"
    plt.savefig(pname, dpi=350)
    #plt.show()
    plt.close()

    mst_dir = results_dir / "seq"
    mst_dir.mkdir(parents=True, exist_ok=True)

    fig2, ax2 = plot_dcj_distance_distribution(df,
                               group_a_prefix="simulated_",
                               group_b_prefix="SEQ_")
    pname = results_dir / "dcj_distribution.pdf"
    plt.savefig(pname, dpi=350)
    #plt.show()
    plt.close()

def plot_all_dcjs_id(results_dir:Path, dcj_file:Path) -> None:
    """Wrapper function.

    Args:
        results_dir: Where you want to put the results.
        dcj_file: The name of the UNIMOG output, should be .txt.
    """
    df = pd.read_pickle(dcj_file)
    #Range lollipop plots
    fig, ax = plot_paired_dcj_range_lollipops(df)
    pname = results_dir / "range_lollipop_id.pdf"
    plt.savefig(pname, dpi=350)
    #plt.show()
    plt.close()

    #Paired DCJ boxplots
    fig1, ax1 = plot_paired_dcj_boxplots(df)
    pname = results_dir / "paired_boxplot_id.pdf"
    plt.savefig(pname, dpi=350)
    #plt.show()
    plt.close()

    mst_dir = results_dir / "mst"
    mst_dir.mkdir(parents=True, exist_ok=True)

    fig2, ax2 = plot_dcj_distance_distribution(df,
                               group_a_prefix="simulated_",
                               group_b_prefix="MST_")
    pname = results_dir / "dcj_distribution_id.pdf"
    plt.savefig(pname, dpi=350)
    #plt.show()
    plt.close()

    mst_dir = results_dir / "seq"
    mst_dir.mkdir(parents=True, exist_ok=True)

    fig2, ax2 = plot_dcj_distance_distribution(df,
                               group_a_prefix="simulated_",
                               group_b_prefix="SEQ_")
    pname = results_dir / "dcj_distribution_id.pdf"
    plt.savefig(pname, dpi=350)
    #plt.show()
    plt.close()


def plot_pangenome_metrics(results_dir:Path, source_dir:Path, pname:str)->None:
    """Wrapper function to plot everything.

    Args:
        results_dir: Where you want to put the plots.
        source_dir: Where the dataframes are.
        pname: Prefix of the file.
    """
    gen = pname + "_genome.csv"
    csv_gen = source_dir / gen
    wei = pname + "_weights.csv"
    csv_wei = source_dir / wei
    outdir = results_dir / pname
    outdir.mkdir(parents=True, exist_ok=True)
    genomes = pd.read_csv(csv_gen)
    #Genome sizes
    fig, ax = plot_genome_sizes(df=genomes)
    pname = outdir / "genome_size_distribution.pdf"
    plt.savefig(pname, dpi=350)
    #plt.show()
    plt.close()

    #Path length
    fig1, ax2 =  plot_path_counts_per_genome(df=genomes)
    pname = outdir / "path_length_distribution.pdf"
    plt.savefig(pname, dpi=350)
    #plt.show()
    plt.close()

    #Weight distribution
    weights = pd.read_csv(csv_wei)
    fig2, ax2 =plot_edge_weight_distribution(df_edges=weights)
    pname = outdir / "edge_weight_distribution.pdf"
    plt.savefig(pname, dpi=350)
    #plt.show()
    plt.close()


if __name__ == "__main__":
    results_root = Path("results/run_20260909/")
    results_root.mkdir(parents=True, exist_ok=True)

    levels = ["low", "medium", "high"]
    gene_sizes = [25, 100, 200, 500, 1000, 1500]

    for size in gene_sizes:
        size_dir = results_root / str(size)
        size_dir.mkdir(parents=True, exist_ok=True)

        for level in levels:
            level_dir = size_dir / level
            level_dir.mkdir(parents=True, exist_ok=True)

            plots_dir = level_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            pans = ["simulated", "mst", "seq"]

            # General pangenome metrics
            for p in pans:
                plot_pangenome_metrics(
                    results_dir=plots_dir,
                    source_dir=level_dir,
                    pname=p,
                )

            # Plot dcj distances
            dcj_csv = level_dir / "dcj_matrix.pkl"
            plot_all_dcjs(plots_dir, dcj_csv)

            dcj_id_csv = level_dir / "dcj_id_matrix.pkl"
            plot_all_dcjs_id(plots_dir, dcj_id_csv)
