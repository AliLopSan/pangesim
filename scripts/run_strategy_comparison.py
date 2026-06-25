"""Script to compare different strategies used in the Heuristic."""

from typing import Any
from typing import Dict
from typing import List

import pandas as pd

from pangesim import Genome
from pangesim import Pangenome
from pangesim.metrics.classification import CoreContingencyTable
from pangesim.metrics.classification import EdgeContingencyTable
from pangesim.metrics.summary import PangenomeSummaryMetrics
from pangesim.reconstruction import EulerianPathHeuristic
from pangesim.reconstruction.assignment import DummyAssignment
from pangesim.reconstruction.assignment import EulerianTrailAssignment
from pangesim.reconstruction.base import AdjacencyMatrix
from pangesim.reconstruction.bounds import GreedyPairingISCB
from pangesim.reconstruction.operators import MergeOperator
from pangesim.reconstruction.operators import SplitOperator
from pangesim.reconstruction.sorting import WeightSorting
from pangesim.reconstruction.utils import pan_score
from pangesim.visualization import TrajectoryVisualizer


class PipelineTracker:
    """Accumulates structural and accuracy statistics across reconstruction phases."""

    def __init__(self) -> None:
        """Constructor function for the PipelineTracker."""
        self.history: List[Dict[str, Any]] = []
        self.summary_metric = PangenomeSummaryMetrics()
        self.classification_edges = EdgeContingencyTable()
        self.classification_core = CoreContingencyTable()

    def __call__(
        self,
        step_name: str,
        pangenome: Pangenome,
        iteration: int,
        ground_truth: Pangenome,
        alpha: float,
        gamma: float,
    ) -> None:
        """Evaluates the pangenome at the current phase against ground truth baseline.

        Args:
            step_name: Phase name.
            pangenome: The inferred pangenome.
            iteration: Iteration number.
            ground_truth: The benchmark pangenome.
            alpha: The per-genome reward in the score.
            gamma: The weight-error penalty coefficient.
        """
        gt_matrix = ground_truth.compute_weighted_adjacencies()
        # Calculate scores using your pan score calculation
        current_score = pan_score(target=pangenome, source=gt_matrix, alpha=alpha, gamma=gamma)

        # Run global and classification metrics
        global_stats = self.summary_metric.evaluate(ground_truth, pangenome)
        core_stats = self.classification_core.evaluate(ground_truth, pangenome)
        edge_stats = self.classification_edges.evaluate(ground_truth, pangenome)

        # Consolidate results into a unified log entry
        log_entry = {
            "Step": step_name,
            "Iteration": iteration,
            "Score": current_score,
            "Number of Genomes Delta": global_stats["k_diff"],
            "Number of Core Genes Delta": global_stats["c_diff"],
            "True Positive Edges": edge_stats["tp"],
            "Edge Accuracy": edge_stats["accuracy"],
            "Edge Precision": edge_stats["precision"],
            "Edge Recall": edge_stats["recall"],
            "True Positive Core Genes": core_stats["tp"],
            "Core Gene Accuracy": core_stats["accuracy"],
            "Core Gene Precision": core_stats["precision"],
            "Core Gene Recall": core_stats["recall"],
        }
        self.history.append(log_entry)


def sample_pangenome():
    """Fixture that provisions a pangenome mock."""
    # Ground-truth genomes
    gt_g1 = Genome(genome_id=0)
    g1_edges = [(1, 2), (2, 6), (6, 7), (7, 9), (9, 11), (3, 10), (3, 4), (4, 5)]
    for edge in g1_edges:
        gt_g1.add_edge(edge)

    gt_g2 = Genome(genome_id=1)
    g2_edges = [(2, 3), (3, 10), (5, 4), (4, 8)]
    for edge in g2_edges:
        gt_g2.add_edge(edge)

    gt_g3 = Genome(genome_id=2)
    g3_edges = [(2, 3), (3, 10)]
    for edge in g3_edges:
        gt_g3.add_edge(edge)

    gt_g4 = Genome(genome_id=3)
    g4_edges = [(1, 2), (2, 3), (3, 4), (4, 8), (10, 9), (9, 11)]
    for edge in g4_edges:
        gt_g4.add_edge(edge)

    gt_g5 = Genome(genome_id=4)
    g5_edges = [(1, 2), (2, 3), (3, 10), (10, 9), (5, 4), (4, 8)]
    for edge in g5_edges:
        gt_g5.add_edge(edge)

    ground_truth = Pangenome(pangenome_id="ground_truth")
    ground_truth.add_genome(gt_g1)
    ground_truth.add_genome(gt_g2)
    ground_truth.add_genome(gt_g3)
    ground_truth.add_genome(gt_g4)
    ground_truth.add_genome(gt_g5)

    return ground_truth


def optimize_with_operators(
    pangenome: Pangenome,
    matrix: AdjacencyMatrix,
    ground_truth: Pangenome,
    k_min: int,
    k_max: int,
    tracker: PipelineTracker,
    alpha: float,
    gamma: float,
) -> Pangenome:
    """Orchestrates Phase 4: Optimizes pangenome using split/merge operators within bounds."""
    improved = True
    pangenome = pangenome.copy()

    current_score = pan_score(target=pangenome, source=matrix, alpha=alpha, gamma=gamma)
    print(f"\nStarting Phase 4 Optimization. Initial Score: {current_score:.4f}")

    while improved:
        improved = False
        current_k = len(pangenome.genomes)

        # --- Operator 1: Merge Genomes ---
        if current_k > k_min:
            candidate_pan = pangenome.copy()
            m_1 = MergeOperator()

            while len(candidate_pan) > k_max:
                m_1.improve(candidate_pan)

            merge_score = pan_score(target=candidate_pan, source=matrix, alpha=alpha, gamma=gamma)
            if merge_score > current_score:
                pangenome = candidate_pan
                current_score = merge_score
                improved = True
                tracker(
                    step_name="Phase 4: Successful Merge",
                    pangenome=pangenome,
                    iteration=len(tracker.history),
                    ground_truth=ground_truth,
                    alpha=alpha,
                    gamma=gamma,
                )

        # --- Operator 2: Split Genomes ---
        if current_k <= k_max:
            candidate_pan = pangenome.copy()
            # Execute your edge-splitting strategy wrapper
            s_1 = SplitOperator()
            s_1.improve(candidate_pan)

            split_score = pan_score(target=candidate_pan, source=matrix, alpha=alpha, gamma=gamma)
            if split_score > current_score:
                pangenome = candidate_pan
                current_score = split_score
                improved = True
                tracker(
                    step_name="Phase 4: Successful Split",
                    pangenome=pangenome,
                    iteration=len(tracker.history),
                    ground_truth=ground_truth,
                    alpha=alpha,
                    gamma=gamma,
                )

    print(f"Phase 4 Complete. Final Optimized Score: {current_score:.4f}")
    return pangenome


def run_evaluation_trial() -> None:
    """Executes evaluation pipeline on mock data and displays comparative diagnostics."""
    true_pangenome = sample_pangenome()
    mock_matrix = true_pangenome.compute_weighted_adjacencies()

    # Set up our tracking listener instance
    tracker = PipelineTracker()

    # Initialize orchestrator
    heuristic = EulerianPathHeuristic()

    print("Running heuristic pipeline reconstruction...")
    inf_pangenome = heuristic.reconstruct(
        matrix=mock_matrix, ground_truth=true_pangenome, callbacks=[tracker]
    )
    optimize_with_operators(
        pangenome=inf_pangenome,
        matrix=mock_matrix,
        ground_truth=true_pangenome,
        k_min=heuristic.k_min,
        k_max=heuristic.k_max,
        tracker=tracker,
        alpha=heuristic.params["alpha"],
        gamma=heuristic.params["gamma"],
    )

    # Convert logged history into a scannable DataFrame for quick interpretation
    df_results = pd.DataFrame(tracker.history)

    viz = TrajectoryVisualizer()
    viz.plot_score(
        tracker_history=tracker.history, save_path="../data/score_full_pipeline.pdf"
    )
    viz.plot_k_bounds(
        tracker_history=tracker.history,kmin=heuristic.k_min,
        kmax=heuristic.k_max,
        save_path="../data/k_bounds_full_pipeline.pdf"
    )
    viz.plot_core_diff(
        tracker_history=tracker.history, save_path="../data/core_full_pipeline.pdf"
    )

    print(df_results.head())


def compare_strategies() -> None:
    """Plots comparisons vs the different strategies."""
    true_pangenome = sample_pangenome()
    mock_matrix = true_pangenome.compute_weighted_adjacencies()

    # Edge Assignment
    print("\t Running Dummy Edge assignement ...")
    tracker_1 = PipelineTracker()
    strategy_1 = EulerianPathHeuristic(assignment_strategy=DummyAssignment())
    strategy_1.reconstruct(matrix=mock_matrix, ground_truth=true_pangenome, callbacks=[tracker_1])
    viz = TrajectoryVisualizer()
    viz.plot_score(
        tracker_history=tracker_1.history, save_path="../data/s1_score.pdf"
    )
    viz.plot_k_bounds(
        tracker_history=tracker_1.history,kmin=strategy_1.k_min,
        kmax=strategy_1.k_max,
        save_path="../data/s1_k_bounds.pdf"
    )
    viz.plot_core_diff(
        tracker_history=tracker_1.history, save_path="../data/s1_core.pdf"
    )

    # Eulerian by length Assignment
    print("\t Running Eulerian by length Assignment ...")
    tracker_2 = PipelineTracker()
    strategy_2 = EulerianPathHeuristic(
        bounds_strategy=GreedyPairingISCB(), assignment_strategy=EulerianTrailAssignment()
    )
    strategy_2.reconstruct(matrix=mock_matrix, ground_truth=true_pangenome, callbacks=[tracker_2])
    viz = TrajectoryVisualizer()
    viz.plot_score(
        tracker_history=tracker_2.history, save_path="../data/s2_score.pdf"
    )
    viz.plot_k_bounds(
        tracker_history=tracker_2.history,kmin=strategy_2.k_min,
        kmax=strategy_2.k_max,
        save_path="../data/s2_k_bounds.pdf"
    )
    viz.plot_core_diff(
        tracker_history=tracker_2.history, save_path="../data/s2_core.pdf"
    )

    # Eulerian by weight Assignment
    print("\t Running Eulerian by weight Assignment ...")
    tracker_3 = PipelineTracker()
    assign = EulerianTrailAssignment(trail_sorting=WeightSorting())
    strategy_3 = EulerianPathHeuristic(
        bounds_strategy=GreedyPairingISCB(), assignment_strategy=assign
    )
    strategy_3.reconstruct(matrix=mock_matrix, ground_truth=true_pangenome, callbacks=[tracker_3])
    viz = TrajectoryVisualizer()
    viz.plot_score(
        tracker_history=tracker_3.history, save_path="../data/s3_score.pdf"
    )
    viz.plot_k_bounds(
        tracker_history=tracker_3.history,kmin=strategy_3.k_min,
        kmax=strategy_3.k_max,
        save_path="../data/s3_k_bounds.pdf"
    )
    viz.plot_core_diff(
        tracker_history=tracker_3.history, save_path="../data/s3_core.pdf"
    )


if __name__ == "__main__":
    run_evaluation_trial()
    compare_strategies()
