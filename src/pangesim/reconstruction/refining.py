from typing import Dict

from pangesim import Pangenome
from pangesim.reconstruction import AdjacencyMatrix
from pangesim.reconstruction import RefinementStrategy
from pangesim.reconstruction.utils import build_residuals


class ResidualsRefinement(RefinementStrategy):
    """Refines a given pangenome using scoring function.

    Use a custom score function to define whether new genomes should be added.
    """

    __slots__ = ("params", "max_iter")

    def __init__(self, params: Dict[str, float] | None = None, max_iter: int | None = None) -> None:
        """Constructor for ResidualsRefinement.

        Args:
           params: Dictionary of parameters.
           max_iter: Failsafe nuumber of iterations.
        """
        self.params = params or {"alpha": 1.0, "gamma": 1.0}
        # Refinement follows until convergence or max_iter
        self.max_iter = max_iter or 100

    def refine(self, source: AdjacencyMatrix, target: Pangenome) -> Pangenome:
        """Main refinement method.

        Args:
           source: Input Adjacency Matrix.
           target: Initial pangenome to refine
        Returns:
           A refined pangenome.
        """
        residuals: AdjacencyMatrix = build_residuals(target=target, source=source)

        # Find the keys with the max and min values
        under_edge = max(residuals, key=residuals.get)
        over_edge = min(residuals, key=residuals.get)

        print("Under-represented edge: ", under_edge)
        print("Over-represented edge: ", over_edge)

        return target
