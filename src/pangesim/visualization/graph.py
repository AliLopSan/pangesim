"""Module for visualizing pangenome graphs and constituent genome paths.

This module provides visualization utilities for the pangenome class.
"""

import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx

from pangesim import Pangenome


class PangenomeVisualizer:
    """Handles the rendering of pangenomes."""

    __slots__ = "pangenome"

    def __init__(self, pangenome: Pangenome | None = None) -> None:
        """Initializes the visualizer with a pangenome object.

        Args:
            pangenome: A pangenome object.
        """
        self.pangenome = pangenome

        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "text.latex.preamble": r"\usepackage{amsmath}",
            }
        )

    def _build_graph(self) -> nx.Graph:
        """Constructs a NetworkX Graph from the weighted adjacency list."""
        graph = nx.Graph()
        adjacencies: List[Tuple[Any, Any, float]] = self.pangenome.compute_weighted_adjacencies()
        for u, v in adjacencies:
            graph.add_edge(u, v, weight=adjacencies[(u, v)])
        return graph

    def _compute_layout(self, graph: nx.Graph, method: str) -> Dict[Any, Tuple[float, float]]:
        """Computes geometric node coordinates based on the selected layout style.

        Args:
            graph: The networkx graph structure to evaluate.
            method: The naming string corresponding to the layout engine type
              ('dot', 'planar', 'kamada_kawai', or 'spectral').

        Returns:
            A dictionary mapping node keys to explicit (x, y) coordinates.
        """
        if method == "dot":
            return nx.nx_agraph.graphviz_layout(graph, prog="dot")
        elif method == "planar":
            try:
                return nx.planar_layout(graph)
            except nx.NetworkXException:
                # Fallback safely if the simulated graph is non-planar
                return nx.kamada_kawai_layout(graph)
        elif method == "kamada_kawai":
            return nx.kamada_kawai_layout(graph)
        elif method == "spectral":
            return nx.spectral_layout(graph)
        else:
            return nx.spring_layout(graph)

    def plot_pangenome_grid(
        self,
        layout_method: str = "dot",
        figsize: Tuple[int, int] = (16, 8),
        output_path: Optional[str] = None,
        filename: str = "pangenome_grid.pdf",
    ) -> None:
        """Plots the aggregate pangenome graph alongside individual genome paths.

        Args:
            layout_method: The pre-defined graph layout, either
                          "dot", "planar", "kamada-kawai" or "spectral".
            figsize: A tuple containing the width and height of the canvas.
            output_path: The path where the plots will be stored.
            filename: The file name and type of the created plot.
        """
        graph = self._build_graph()
        num_genomes = len(self.pangenome)

        # Compute the user-defined layout
        pos = self._compute_layout(graph, method=layout_method)

        fig = plt.figure(figsize=figsize)

        # Create a GridSpec layout: 2 columns total (Left column = 50% width)
        # The right column is partitioned evenly into a sub-grid for the genomes
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

        # Left Panel: Main Aggregate Graph
        ax_main = fig.add_subplot(gs[0, 0])
        weights: List[float] = [graph[u][v]["weight"] for u, v in graph.edges()]

        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax_main,
            node_color="#A0CBE8",
            node_size=700,
            edgecolors="black",
        )
        nx.draw_networkx_edges(graph, pos, ax=ax_main, width=weights, edge_color="#4E79A7")
        nx.draw_networkx_labels(graph, pos, ax=ax_main, font_size=10, font_weight="bold")
        ax_main.set_title(
            "Adjacency Graph\n(Edge width = Weight)",
            fontsize=14,
            fontweight="bold",
        )
        ax_main.axis("off")

        # Right Panel: Dynamically partition the remaining space for subplots
        gs_genomes = gs[0, 1].subgridspec(1, num_genomes)

        for i, genome in enumerate(self.pangenome.genomes):
            ax_sub = fig.add_subplot(gs_genomes[0, i])

            active_nodes = list(genome.gene_set)
            active_edges = list(genome.get_adjacency_tuples())

            # Define node visibility arrays based on usage
            node_colors = ["#C13383" if n in active_nodes else "#D3D3D3" for n in graph.nodes()]
            node_alphas = [1.0 if n in active_nodes else 0.5 for n in graph.nodes()]

            # Draw background faded nodes first to safely simulate individual cell alpha
            for node, color, alpha in zip(graph.nodes(), node_colors, node_alphas):
                nx.draw_networkx_nodes(
                    graph,
                    pos,
                    nodelist=[node],
                    ax=ax_sub,
                    node_color=color,
                    node_size=400,
                    alpha=alpha,
                    edgecolors="black" if node in active_nodes else "#A9A9A9",
                )

            # Draw active edges (Solid, 100% alpha)
            if active_edges:
                nx.draw_networkx_edges(
                    graph,
                    pos,
                    edgelist=list(active_edges),
                    ax=ax_sub,
                    width=2.5,
                    edge_color="#E15759",
                    alpha=1.0,
                )

            # Draw inactive edges (Dotted, 50% alpha)
            inactive_edges = [e for e in graph.edges() if e not in active_edges]
            if inactive_edges:
                nx.draw_networkx_edges(
                    graph,
                    pos,
                    edgelist=inactive_edges,
                    ax=ax_sub,
                    width=1.0,
                    edge_color="#D3D3D3",
                    style="dotted",
                    alpha=0.5,
                )

            # Node labels rendered inside nodes (font size scaled to fit nicely)
            nx.draw_networkx_labels(
                graph, pos, ax=ax_sub, font_size=8, font_weight="bold", font_color="white"
            )

            # Apply indexed LaTeX formula string to subplot header titles
            ax_sub.set_title(f"$G_{{{i + 1}}}$", fontsize=14)
            ax_sub.axis("off")

        plt.tight_layout()

        # Handle File I/O Serialization Logic
        save_dir = output_path if output_path else "."
        os.makedirs(save_dir, exist_ok=True)
        full_save_path = os.path.join(save_dir, filename)
        plt.savefig(full_save_path, dpi=400, bbox_inches="tight")

        plt.show()


class RefinementVisualizerV1(PangenomeVisualizer):
    """Subclass tracking and rendering genome-by-genome grid refinement comparisons."""

    def __init__(self, ground_truth_pangenome: Any) -> None:
        """Initializes the environment with a fixed ground truth model."""
        super().__init__(pangenome=ground_truth_pangenome)
        self.gt_pangenome = ground_truth_pangenome.copy()
        self.gt_graph = self._build_graph()

    def _get_combined_layout(
        self, inferred_graph: nx.Graph, layout_method: str
    ) -> Dict[Any, Tuple[float, float]]:
        """Computes a static node layout using the union of both graphs."""
        combined = nx.Graph()
        combined.add_nodes_from(self.gt_graph.nodes())
        combined.add_nodes_from(inferred_graph.nodes())
        combined.add_edges_from(self.gt_graph.edges())
        combined.add_edges_from(inferred_graph.edges())
        return self._compute_layout(combined, method=layout_method)

    def _plot_genome_grid(
        self,
        pangenome: Any,
        graph: nx.Graph,
        pos: Dict[Any, Tuple[float, float]],
        subgrid: Any,
        fig: plt.Figure,
    ) -> None:
        """Helper to render a row of constituent genome graphs or a truncation alert."""
        total_genomes = len(pangenome.genomes)

        # Enforce our strict max threshold rule
        if total_genomes <= 10:
            num_slots = total_genomes
            truncate = False
        else:
            num_slots = 10
            truncate = True

        gs_genomes = subgrid.subgridspec(1, num_slots)

        for idx in range(num_slots):
            ax_sub = fig.add_subplot(gs_genomes[0, idx])

            # If we're on the last slot and truncation is active, inject text panel
            if truncate and idx == 9:
                remaining = total_genomes - 9
                ax_sub.text(
                    0.5,
                    0.5,
                    f"... and {remaining}\nother genomes",
                    color="#555555",
                    style="italic",
                    weight="bold",
                    ha="center",
                    va="center",
                    fontsize=10,
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="#F5F5F5",
                        edgecolor="#D3D3D3",
                    ),
                )
                ax_sub.axis("off")
                continue

            # Standard active rendering loop for genomes 1 through 9
            genome = pangenome.genomes[idx]
            active_nodes = list(genome.gene_set)
            active_edges = list(genome.get_adjacency_tuples())

            node_colors = ["#C13383" if n in active_nodes else "#D3D3D3" for n in graph.nodes()]
            node_alphas = [1.0 if n in active_nodes else 0.5 for n in graph.nodes()]

            for node, color, alpha in zip(graph.nodes(), node_colors, node_alphas):
                nx.draw_networkx_nodes(
                    graph,
                    pos,
                    nodelist=[node],
                    ax=ax_sub,
                    node_color=color,
                    node_size=250,
                    alpha=alpha,
                    edgecolors="black" if node in active_nodes else "#A9A9A9",
                )

            if active_edges:
                nx.draw_networkx_edges(
                    graph,
                    pos,
                    edgelist=active_edges,
                    ax=ax_sub,
                    width=2.0,
                    edge_color="#E15759",
                    alpha=1.0,
                )

            inactive_edges = [e for e in graph.edges() if e not in active_edges]
            if inactive_edges:
                nx.draw_networkx_edges(
                    graph,
                    pos,
                    edgelist=inactive_edges,
                    ax=ax_sub,
                    width=2.0,
                    edge_color="#D3D3D3",
                    style="dotted",
                    alpha=0.6,
                )

            ax_sub.set_title(rf"$G_{idx + 1}$", fontsize=9)
            ax_sub.axis("off")

    def plot_refinement_step(
        self,
        inferred_pangenome: Any,
        step: int,
        layout_method: str = "kamada_kawai",
        output_path: Optional[str] = None,
    ) -> None:
        """Generates aligned stacked subplots comparing GT and inferred genome rows."""
        self.pangenome = inferred_pangenome
        inf_graph = self._build_graph()

        pos = self._get_combined_layout(inf_graph, layout_method)

        fig = plt.figure(figsize=(16, 9))
        gs_main = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)

        # 1. Render Top Track: Ground Truth Reference Genome Row
        fig.text(0.02, 0.72, "GROUND\nTRUTH", fontsize=12, weight="bold")
        self._plot_genome_grid(
            pangenome=self.gt_pangenome,
            graph=self.gt_graph,
            pos=pos,
            subgrid=gs_main[0, 0],
            fig=fig,
        )

        # 2. Render Bottom Track: Inferred Refinement Step Genome Row
        title_str = f"INFERRED\n(STEP {step})"
        fig.text(0.02, 0.28, title_str, fontsize=12, weight="bold")
        self._plot_genome_grid(
            pangenome=inferred_pangenome,
            graph=inf_graph,
            pos=pos,
            subgrid=gs_main[1, 0],
            fig=fig,
        )

        plt.suptitle(
            rf"\textbf{{Genome Alignment Refinement Tracking --- Frame {step:03d}}}",
            fontsize=14,
            y=0.96,
        )

        if output_path:
            os.makedirs(output_path, exist_ok=True)
            filename = f"refinement_step_{step:03d}.png"
            plt.savefig(
                os.path.join(output_path, filename),
                format="png",
                bbox_inches="tight",
            )
        plt.close()


class SmartRefinementVisualizer(PangenomeVisualizer):
    """Subclass tracking genome alignments using a stacked multi-row dashboard layout."""

    def __init__(
        self,
        ground_truth_pangenome: Any,
        layout_method: str = "kamada_kawai",
    ) -> None:
        """Initializes tracking environment and layout options."""
        super().__init__(pangenome=ground_truth_pangenome)
        self.gt_pangenome = ground_truth_pangenome
        self.gt_graph = self._build_graph()
        self.layout_method = layout_method
        self.cached_pos = None

    def precompute_global_layout(self, final_inferred_pangenome: Any) -> None:
        """Pre-computes coordinate positions using final structural states."""
        self.pangenome = final_inferred_pangenome
        final_graph = self._build_graph()

        combined = nx.Graph()
        combined.add_nodes_from(self.gt_graph.nodes())
        combined.add_nodes_from(final_graph.nodes())
        combined.add_edges_from(self.gt_graph.edges())
        combined.add_edges_from(final_graph.edges())

        self.cached_pos = self._compute_layout(combined, method=self.layout_method)

    def _plot_aggregate_panel(
        self,
        graph: nx.Graph,
        pos: Dict[Any, Tuple[float, float]],
        ax: plt.Axes,
        title: str,
        node_color: str,
    ) -> None:
        """Renders the primary structural adjacency graph with edge counts."""
        nx.draw_networkx_nodes(
            graph, pos, ax=ax, node_color=node_color, node_size=700, edgecolors="black"
        )
        nx.draw_networkx_edges(graph, pos, ax=ax, width=1.5, edge_color="#4E79A7")
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=12, font_weight="bold")

        labels = nx.get_edge_attributes(graph, "weight")
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=labels,
            ax=ax,
            font_size=14,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

    def _plot_genome_stacked_grid(
        self,
        pangenome: Any,
        graph: nx.Graph,
        pos: Dict[Any, Tuple[float, float]],
        subgrid: Any,
        fig: plt.Figure,
        base_node_color: str,
        prev_pangenome: Optional[Any] = None,
    ) -> Tuple[str, set]:
        """Renders individual genomes into a stacked 2-row subgrid (Max 5x2)."""
        total_genomes = len(pangenome.genomes)
        truncate = total_genomes > 9

        gs_genomes = subgrid.subgridspec(2, 5, hspace=0.3, wspace=0.25)
        delta_msg = ""
        highlighted_genomes = set()

        prev_genomes_count = len(prev_pangenome.genomes) if prev_pangenome else 0
        if prev_pangenome and total_genomes > prev_genomes_count:
            delta_msg = f"Added {total_genomes - prev_genomes_count} new genome"
            highlighted_genomes = set(range(prev_genomes_count, total_genomes))

        max_visible_slots = 10 if truncate else total_genomes

        for slot_idx in range(max_visible_slots):
            row = slot_idx // 5
            col = slot_idx % 5
            ax_sub = fig.add_subplot(gs_genomes[row, col])

            if truncate and slot_idx == 9:
                remaining = total_genomes - 9
                ax_sub.text(
                    0.5,
                    0.5,
                    f"... and {remaining}\nother genomes",
                    color="#555555",
                    style="italic",
                    weight="bold",
                    ha="center",
                    va="center",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#F5F5F5", ec="#D3D3D3"),
                )
                ax_sub.axis("off")
                continue

            genome = pangenome.genomes[slot_idx]
            active_nodes = list(genome.gene_set)
            active_edges = list(genome.get_adjacency_tuples())

            new_edges = set()
            removed_edges = set()

            if prev_pangenome and slot_idx < prev_genomes_count:
                prev_genome = prev_pangenome.genomes[slot_idx]
                prev_edges = set(prev_genome.get_adjacency_tuples())
                curr_edges = set(active_edges)

                new_edges = curr_edges - prev_edges
                removed_edges = prev_edges - curr_edges

                if new_edges:
                    delta_msg = f"Genome {slot_idx + 1}: Added {list(new_edges)[0]}"
                elif removed_edges:
                    delta_msg = f"Genome {slot_idx + 1}: Pruned {list(removed_edges)[0]}"

            is_new_genome = slot_idx in highlighted_genomes
            face_color = "#EAEAEA" if is_new_genome else "white"
            ax_sub.set_facecolor(face_color)

            # Keep #FF55A3 if it's a new genome; otherwise use the track's custom base color
            active_node_color = "#FF55A3" if is_new_genome else base_node_color
            node_colors = [
                active_node_color if n in active_nodes else "#D3D3D3" for n in graph.nodes()
            ]
            node_alphas = [1.0 if n in active_nodes else 0.4 for n in graph.nodes()]

            for node, color, alpha in zip(graph.nodes(), node_colors, node_alphas):
                nx.draw_networkx_nodes(
                    graph,
                    pos,
                    nodelist=[node],
                    ax=ax_sub,
                    node_color=color,
                    node_size=200,
                    alpha=alpha,
                    edgecolors="black" if node in active_nodes else "#A9A9A9",
                )

            normal_active = [e for e in active_edges if e not in new_edges]
            if normal_active:
                nx.draw_networkx_edges(
                    graph,
                    pos,
                    edgelist=normal_active,
                    ax=ax_sub,
                    width=1.2,
                    edge_color="black",
                    alpha=1.0,
                )

            if new_edges:
                nx.draw_networkx_edges(
                    graph,
                    pos,
                    edgelist=list(new_edges),
                    ax=ax_sub,
                    width=4,
                    edge_color="#FF2052",
                    alpha=1.0,
                )

            if removed_edges:
                nx.draw_networkx_edges(
                    graph,
                    pos,
                    edgelist=list(removed_edges),
                    ax=ax_sub,
                    width=1.8,
                    edge_color="#FF4B4B",
                    style="dashed",
                    alpha=0.7,
                )

            all_special = set(active_edges).union(removed_edges)
            inactive_edges = [e for e in graph.edges() if e not in all_special]
            if inactive_edges:
                nx.draw_networkx_edges(
                    graph,
                    pos,
                    edgelist=inactive_edges,
                    ax=ax_sub,
                    width=0.6,
                    edge_color="#D3D3D3",
                    style="dotted",
                    alpha=0.3,
                )

            title_suffix = " (NEW)" if is_new_genome else ""
            ax_sub.set_title(f"G {slot_idx + 1}{title_suffix}", fontsize=14)
            ax_sub.axis("off")

        return delta_msg, highlighted_genomes

    def plot_refinement_step(
        self,
        inferred_pangenome: Any,
        step: int,
        prev_pangenome: Optional[Any] = None,
        output_path: Optional[str] = None,
    ) -> None:
        """Plots stacked dashboard tracks: Ground Truth and Inferred flow."""
        self.pangenome = inferred_pangenome
        inf_graph = self._build_graph()

        if self.cached_pos is None:
            self.cached_pos = self._get_combined_layout(inf_graph, self.layout_method)

        fig = plt.figure(figsize=(20, 12))
        gs_main = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.35)

        # --- TOP SECTION: GROUND TRUTH TRACK (#007FFF) ---
        gs_top = gs_main[0, 0].subgridspec(1, 2, width_ratios=[1, 1.8], wspace=0.15)
        ax_gt_agg = fig.add_subplot(gs_top[0, 0])
        self._plot_aggregate_panel(
            graph=self.gt_graph,
            pos=self.cached_pos,
            ax=ax_gt_agg,
            title="Benchmark Pangenome",
            node_color="#007FFF",
        )
        self._plot_genome_stacked_grid(
            pangenome=self.gt_pangenome,
            graph=self.gt_graph,
            pos=self.cached_pos,
            subgrid=gs_top[0, 1],
            fig=fig,
            base_node_color="#007FFF",
        )

        # --- BOTTOM SECTION: INFERRED STATE TRACK (#BD33A4) ---
        gs_inf = gs_main[1, 0].subgridspec(1, 2, width_ratios=[1, 1.8], wspace=0.15)
        ax_inf_agg = fig.add_subplot(gs_inf[0, 0])
        self._plot_aggregate_panel(
            graph=inf_graph,
            pos=self.cached_pos,
            ax=ax_inf_agg,
            title=f"Inferred Pangenome (Step {step})",
            node_color="#BD33A4",
        )
        delta_text, _ = self._plot_genome_stacked_grid(
            pangenome=inferred_pangenome,
            graph=inf_graph,
            pos=self.cached_pos,
            subgrid=gs_inf[0, 1],
            fig=fig,
            base_node_color="#BD33A4",
            prev_pangenome=prev_pangenome,
        )

        plt.suptitle(
            rf"\textbf{{Pangenome Refinement --- Frame {step:03d}}}"
            f"\n\\small\\textit{{Event Log: {delta_text}}}",
            fontsize=20,
            y=0.97,
        )

        if output_path:
            os.makedirs(output_path, exist_ok=True)
            filename = f"refinement_step_{step:03d}.png"
            plt.savefig(os.path.join(output_path, filename), format="png", bbox_inches="tight")
        plt.close()
