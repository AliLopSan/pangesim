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
    num_genes = 10
    results = Path("results/run_20260907")
    results.mkdir(parents=True, exist_ok=True)

    #Ground Truth generation
    ground_truth = random_simulated_pangenome(num_genes)
    f = results / "ground_truth.unimog"
    export_to_unimog(f, ground_truth)

    #MST inference
    matrix = ground_truth.compute_weighted_adjacencies()

    #MST inference
    mst = run_mst_inference(matrix)
    f2 = results / "mst_pangenome.unimog"
    export_to_unimog(f2, mst)

    #Sequential inference
    seq = run_seq_inference(matrix)
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
    figsize: tuple[int, int] = (12, 7),
    visualizer_kwargs: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Extracts pairwise distances between two groups, removes NaNs, and plots their distribution."""
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })

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
        color = "#FF467A"
    else:
        color = "#AB03A9"

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
    x_ticks = np.arange(min_val, max_val + 1, tick_step)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks], rotation=0)

    # Axis Labels & Formatting
    #ax.set_title(title, pad=20)
    ax.set_xlabel(r"DCJ Distance",fontsize=15)
    ax.set_ylabel(r"Pairwise Count",fontsize=15)

    plt.tight_layout()
    return fig, ax


def get_matrix():
    
    results = Path("results/run_20260907")
    f = results / "dcj_results_v2.txt"
    df = parse_dcj_matrix(f)

    print(df.head())

    fig, ax = plot_dcj_heatmap(df,
                               group_a_prefix="simulated_",
                               group_b_prefix="MST_")
    pname = results / "mst.png"
    plt.savefig(pname, dpi=300)
    plt.show()
    plt.close()

    fig2, ax2 = plot_dcj_distance_distribution(df,
                               group_a_prefix="simulated_",
                               group_b_prefix="MST_")
    pname = results / "mst_distribution.png"
    plt.savefig(pname, dpi=300)
    plt.show()
    plt.close()

    fig, ax = plot_dcj_heatmap(df,
                               group_a_prefix="simulated_",
                               group_b_prefix="SEQ_")
    pname = results / "seq.png"
    plt.savefig(pname, dpi=300)
    plt.show()
    plt.close()

    fig2, ax2 = plot_dcj_distance_distribution(df,
                               group_a_prefix="simulated_",
                               group_b_prefix="SEQ_")
    pname = results / "seq_distribution.png"
    plt.savefig(pname, dpi=300)
    plt.show()
    plt.close()

    


if __name__ == "__main__":
    #generate_pan()
    get_matrix()
