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
from pangesim.reconstruction.bounds import GreedyPairingISCB
from pangesim.reconstruction.sorting import WeightSorting
from pangesim.reconstruction.utils import pan_score
from pangesim.visualization import HeuristicPerformanceVisualizer


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


def run_evaluation_trial() -> None:
    """Executes evaluation pipeline on mock data and displays comparative diagnostics."""
    true_pangenome = sample_pangenome()
    mock_matrix = true_pangenome.compute_weighted_adjacencies()

    # Set up our tracking listener instance
    tracker = PipelineTracker()

    # Initialize orchestrator
    heuristic = EulerianPathHeuristic()

    print("Running heuristic pipeline reconstruction...")
    # Execute with hooks activated
    heuristic.reconstruct(matrix=mock_matrix, ground_truth=true_pangenome, callbacks=[tracker])

    # Convert logged history into a scannable DataFrame for quick interpretation
    df_results = pd.DataFrame(tracker.history)

    viz = HeuristicPerformanceVisualizer()
    viz.plot_trajectory_dashboard(
        tracker_history=tracker.history, save_path="../data/default_heuristic_k_genes.pdf"
    )

    print(df_results.head())


def compare_strategies() -> None:
    """Plots comparisons vs the different strategies."""
    true_pangenome = sample_pangenome()
    mock_matrix = true_pangenome.compute_weighted_adjacencies()

    # Edge Assignment
    print("\t Running Dummy Edge assignement ...")
    tracker_1 = PipelineTracker()
    strategy1 = EulerianPathHeuristic(assignment_strategy=DummyAssignment())
    strategy1.reconstruct(matrix=mock_matrix, ground_truth=true_pangenome, callbacks=[tracker_1])
    viz_1 = HeuristicPerformanceVisualizer()
    viz_1.plot_trajectory_dashboard(
        tracker_history=tracker_1.history, save_path="../data/strategy_1.pdf"
    )

    # Eulerian by length Assignment
    print("\t Running Eulerian by length Assignment ...")
    tracker_2 = PipelineTracker()
    strategy_2 = EulerianPathHeuristic(
        bounds_strategy=GreedyPairingISCB(), assignment_strategy=EulerianTrailAssignment()
    )
    strategy_2.reconstruct(matrix=mock_matrix, ground_truth=true_pangenome, callbacks=[tracker_2])
    viz_2 = HeuristicPerformanceVisualizer()
    viz_2.plot_trajectory_dashboard(
        tracker_history=tracker_2.history, save_path="../data/strategy_2.pdf"
    )

    # Eulerian by weight Assignment
    print("\t Running Eulerian by weight Assignment ...")
    tracker_3 = PipelineTracker()
    assign = EulerianTrailAssignment(trail_sorting=WeightSorting())
    strategy_3 = EulerianPathHeuristic(
        bounds_strategy=GreedyPairingISCB(), assignment_strategy=assign
    )
    strategy_3.reconstruct(matrix=mock_matrix, ground_truth=true_pangenome, callbacks=[tracker_3])
    viz_3 = HeuristicPerformanceVisualizer()
    viz_3.plot_trajectory_dashboard(
        tracker_history=tracker_3.history, save_path="../data/strategy_3.pdf"
    )


def run_evaluation():
    """Tests on different strategies over a simple ground-truth graph."""
    ground_truth = sample_pangenome()
    sample_matrix = ground_truth.compute_weighted_adjacencies()

    # Eulerian paths by length
    bounds = GreedyPairingISCB()
    assign = EulerianTrailAssignment()
    heuristic = EulerianPathHeuristic(bounds_strategy=bounds, assignment_strategy=assign)
    pangenome = heuristic.reconstruct(sample_matrix)
    summary = PangenomeSummaryMetrics()
    results_1 = summary.evaluate(ground_truth, pangenome)
    metric = CoreContingencyTable()
    class_1 = metric.evaluate(ground_truth, pangenome)
    print("Results of Eulerian paths by length.")
    print(results_1)
    print(class_1)

    # Eulerian paths by weight
    assign = EulerianTrailAssignment(trail_sorting=WeightSorting())
    heuristic = EulerianPathHeuristic(bounds_strategy=bounds, assignment_strategy=assign)
    pangenome = heuristic.reconstruct(sample_matrix)
    summary = PangenomeSummaryMetrics()
    results_2 = summary.evaluate(ground_truth, pangenome)
    metric = CoreContingencyTable()
    class_2 = metric.evaluate(ground_truth, pangenome)
    print("Results of Eulerian paths by weight.")
    print(results_2)
    print(class_2)


if __name__ == "__main__":
    run_evaluation_trial()
    compare_strategies()
