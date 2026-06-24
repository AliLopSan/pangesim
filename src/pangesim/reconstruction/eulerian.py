"""Main orchestration engine executing the 4-phase Eulerian path heuristic."""

from typing import Any
from typing import Callable
from typing import Dict
from typing import List

from pangesim import Pangenome
from pangesim.reconstruction import AdjacencyMatrix
from pangesim.reconstruction import AssignmentStrategy
from pangesim.reconstruction import BoundsStrategy
from pangesim.reconstruction import RefinementStrategy
from pangesim.reconstruction.assignment import DummyAssignment
from pangesim.reconstruction.bounds import DummyBounds
from pangesim.reconstruction.refining import ResidualsRefinement


class EulerianPathHeuristic:
    """Orchestrates pangenome reconstruction using a 4-phase pipeline."""

    def __init__(
        self,
        params: Dict[str, Any] | None = None,
        bounds_strategy: BoundsStrategy | None = None,
        assignment_strategy: AssignmentStrategy | None = None,
        refine_strategy: RefinementStrategy | None = None,
    ) -> None:
        """Constructor for the Eulerian path heuristic.

        Args:
           params: A dict of optional parameters.
           bounds_strategy: Strategy used to compute genome bounds.
           assignment_strategy: Strategy used to assign genomes.
           refine_strategy: Strategy used to refine base pangenome.
        """
        self.params = params or {"alpha": 1.0, "gamma": 1.0}
        self.bounds_strategy = bounds_strategy if bounds_strategy is not None else DummyBounds()

        self.assignment_strategy = (
            assignment_strategy if assignment_strategy is not None else DummyAssignment()
        )
        self.refine_strategy = (
            refine_strategy if refine_strategy is not None else ResidualsRefinement()
        )

    def reconstruct(
        self,
        matrix: AdjacencyMatrix,
        ground_truth: Pangenome | None = None,
        callbacks: List[Callable] | None = None,
    ) -> Pangenome:
        """Executes full pipeline.

        Args:
            matrix: weighted adjacency matrix.
            ground_truth: Benchmark to compare with.
            callbacks: Event Callback Observer pattern.

        Returns:
            A candidate pangenome.
        """
        callbacks = callbacks or []

        # Phase 1: Compute bounds
        k_min, k_max, info = self.bounds_strategy.compute_bounds(matrix, self.params)

        # Phase 2: Paths assignment
        base_pangenome = self.assignment_strategy.assign_genomes(adjacencies=matrix, k=k_min)
        # Event listener for Base Pangenome State
        for callback in callbacks:
            callback(
                step_name="Phase 2: Base Assignment",
                iteration=0,
                pangenome=base_pangenome,
                ground_truth=ground_truth,
                alpha=self.params["alpha"],
                gamma=self.params["gamma"],
            )

        # Phase 3: Refinement
        pangenome = self.refine_strategy.refine(
            source=matrix, target=base_pangenome, ground_truth=ground_truth, callbacks=callbacks
        )

        return pangenome
