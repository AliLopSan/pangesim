"""Instrumentation tools for recording optimization run state histories."""

from typing import Any
from typing import Dict
from typing import List

from pangesim import Pangenome
from pangesim.metrics.classification import CoreContingencyTable
from pangesim.metrics.classification import EdgeContingencyTable
from pangesim.metrics.summary import PangenomeSummaryMetrics
from pangesim.reconstruction.utils import pan_score


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
            "True Genomes": len(ground_truth),
            "Inferred Genomes": len(pangenome),
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


class StructuralVisualizerTracker(PipelineTracker):
    """Callback tracker that securely records intermediate graph states."""

    def __call__(
        self,
        step_name: str,
        pangenome: Any,
        iteration: int,
        ground_truth: Any,
        alpha: float,
        gamma: float,
    ) -> None:
        """Intercepts the tracking execution to store deep-copied graph topologies."""
        # 1. Run the base class evaluation logic to populate self.history
        super().__call__(
            step_name=step_name,
            pangenome=pangenome,
            iteration=iteration,
            ground_truth=ground_truth,
            alpha=alpha,
            gamma=gamma,
        )

        # 2. Safely grab the log entry that super().__call__ just appended
        latest_log_entry = self.history[-1]

        # 3. Inject the explicit deep copy of the pangenome graph topology
        latest_log_entry["inferred_pangenome_state"] = pangenome.copy()
