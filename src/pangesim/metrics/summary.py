"""Global inference metrics."""

from typing import Dict

from pangesim import Pangenome
from pangesim.metrics.base import PangenomeMetric


class PangenomeSummaryMetrics(PangenomeMetric):
    """Calculates macroscopic delta metrics between two pangenomes."""

    @property
    def name(self) -> str:
        """Prints metric name."""
        return "Pangenome Summary Metrics"

    def evaluate(self, ground_truth: Pangenome, inferred: Pangenome) -> Dict[str, float]:
        """Computes number of inferred genomes, core size and score.

        Args:
        ground_truth: The benchmark pangenome.
        inferred: The reconstructed pangenome.

        Returns:
        A dictionary with the global metrics.
        """
        k = abs(len(ground_truth) - len(inferred))
        c = abs(len(ground_truth.core) - len(inferred.core))

        return {"k_diff": k, "c_diff": c}
