"""Main orchestration engine executing the 4-phase Eulerian path heuristic."""
from typing import Any
from typing import Dict

from pangesim import Pangenome
from pangesim.reconstruction import AdjacencyMatrix
from pangesim.reconstruction import AssignmentStrategy
from pangesim.reconstruction import BoundsStrategy
from pangesim.reconstruction.assignment import DummyAssignment
from pangesim.reconstruction.bounds import DummyBounds


class EulerianPathHeuristic:
    """Orchestrates pangenome reconstruction using a 4-phase pipeline."""

    def __init__(
        self,
            params: Dict[str, Any] | None = None,
            bounds_strategy: BoundsStrategy | None = None,
            assignment_strategy: AssignmentStrategy | None = None,
    ) -> None:
        """Constructor for the Eulerian path heuristic.

        Args:
           params: A dict of optional parameters.
           bounds_strategy: Strategy used to compute genome bounds.
           assignment_strategy: Strategy used to assign genomes.
        """
        self.params = params or {"alpha":1.0,"gamma":1.0}
        self.bounds_strategy = (
            bounds_strategy
            if bounds_strategy is not None
            else DummyBounds()
        )

        self.assignment_strategy = (
            assignment_strategy
            if assignment_strategy is not None
            else DummyAssignment()
        )

    def reconstruct(self, matrix: AdjacencyMatrix) -> Pangenome:
        """Executes full pipeline.

        Args:
            matrix: weighted adjacency matrix.

        Returns:
            A candidate pangenome.
        """
        # Phase 1: Compute bounds
        k_min, k_max, info = self.bounds_strategy.compute_bounds(matrix)
        print("\t Successfully computed bounds kmin : ", k_min,
              "k_max: ", k_max)
        # Phase 2: Paths assignment
        pangenome = self.assignment_strategy.assign_genomes(adjacencies=matrix,
                                                            k=k_max)
        return pangenome
