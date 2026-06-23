"""Blueprint for performace metrics."""

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict

from pangesim import Genome
from pangesim import Pangenome


class BaseMetric(ABC):
    """Abstract Base Class acting as the structural blueprint for all evaluation metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the human-readable name of the metric for reporting and logging."""
        pass

    @abstractmethod
    def evaluate(self, ground_truth: Any, inferred: Any) -> Dict[str, float]:
        """Executes the quantitative comparison between ground truth and inferred objects.

        Args:
            ground_truth: The benchmark structure.
            inferred: The algorithmically reconstructed structure.

        Returns:
            A dictionary mapping metric sub-score names to their float values.
        """
        pass


class GenomeMetric(BaseMetric):
    """Blueprint for metrics evaluating individual path forests (Genomes)."""

    @abstractmethod
    def evaluate(self, ground_truth: Genome, inferred: Genome) -> Dict[str, float]:
        """Performs comparison strictly between two individual Genomes."""
        pass


class PangenomeMetric(BaseMetric):
    """Blueprint for metrics evaluating global graph topology populations (Pangenomes)."""

    @abstractmethod
    def evaluate(self, ground_truth: Pangenome, inferred: Pangenome) -> Dict[str, float]:
        """Performs comparison strictly between two Pangenomes."""
        pass
