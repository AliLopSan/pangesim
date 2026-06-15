"""Phase 1: Computing bounds on the number of genomes."""

from collections.abc import Iterator
from typing import Dict
from typing import List
from typing import Tuple

from pangesim.reconstruction import AdjacencyList
from pangesim.reconstruction import AdjacencyMatrix
from pangesim.reconstruction import BoundsStrategy
from pangesim.reconstruction import matrix_to_list


class DummyBounds(BoundsStrategy):
    """Calculates k bounds."""

    def compute_bounds(self, matrix: AdjacencyMatrix, params: Dict | None = None):
        """Takes an adjacency matrix and computes node degrees.

        Args:
          matrix: Weighted Adjacency Graph
          params: Optional Parameters
        """
        k_min = 1
        k_max = max(1, len(matrix))
        degrees = dict()
        ad_list = matrix_to_list(matrix)
        for v in ad_list:
            degrees[v] = len(ad_list[v])
        return k_min, k_max, degrees


class GreedyPairingISCB(BoundsStrategy):
    """Calculates k bounds using greedy multiplicity pairing."""

    def compute_node_bound(self, node: int, neighbors: List[Tuple]) -> int:
        """Computes the bound associated with a graph node.

        Args:
           node: Node identifier.
           neighbors: Adjacent nodes and their corresponding multiplicity.

        Returns:
           k_v bound value for the node.
        """
        # Collect neighbor's multiplicities
        multiplicities: List[int] = []
        for u, m in neighbors:
            if m > 0:
                multiplicities.append(m)
        multiplicities.sort(reverse=True)
        k_v = 0

        while len(multiplicities) >= 2:
            a, b = multiplicities[0], multiplicities[1]
            paired_min = min(a, b)
            k_v += paired_min
            multiplicities[0] -= paired_min
            multiplicities[1] -= paired_min
            k_v += multiplicities[0] + multiplicities[1]
            multiplicities = sorted(multiplicities[2:], reverse=True)

        if len(multiplicities) == 1:
            k_v += multiplicities[0]

        return k_v

    def iter_bounds(
        self,
        ad_list: AdjacencyList,
    ) -> Iterator[Tuple[int, int]]:
        """Yields node bounds one at a time."""
        for node, neighbors in ad_list.items():
            yield (
                node,
                self.compute_node_bound(node, neighbors),
            )

    def compute_k_max(
        self,
        matrix: AdjacencyMatrix,
        kmin: int,
        alpha: float,
        gamma: float,
    ) -> int:
        """Computes the score-derived upper bound.

        Args:
            matrix: Weighted adjacency matrix.
            kmin: minimum per node k.
            alpha: per-genome reward in the score.
            gamma: weight-error penalty coefficient.

        Returns:
           The structural upper bound.
        """
        rho = alpha / gamma
        budget = sum(v * v for v in matrix.values()) / rho
        return kmin + int(budget)

    def _get_positive_parameter(
        self,
        params: dict[str, float],
        name: str,
        default: float,
    ) -> float:
        """Returns a positive parameter value.

        Args:
            params: Parameter dictionary.
            name: Parameter name.
            default: Default value.

        Returns:
            Parameter value.

        Raises:
            ValueError: If the parameter is not strictly positive.
        """
        value = params.get(name, default)
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0. Received {value}.")

        return value

    def compute_bounds(
        self,
        matrix: AdjacencyMatrix,
        params: Dict[str, float],
    ) -> Tuple[int, int, Dict[int, int]]:
        """Computes lower and upper bounds for the greedy pairing problem.

        Args:
           matrix: Adjacency matrix representation of the graph.
           params: Optional heuristic parameters.

        Returns:
           A tuple containing:
            - k_min: Minimum feasible bound.
            - k_max: Maximum feasible bound.
            - node_bounds: Per-node bound values.

        Raises:
            ValueError: If alpha or gamma are not strictly positive.
        """
        adjacency_list = matrix_to_list(matrix)

        node_bounds = {node: value for node, value in self.iter_bounds(adjacency_list)}

        k_min = max(node_bounds.values()) if node_bounds else 1

        alpha = self._get_positive_parameter(params, "alpha", 1.0)
        gamma = self._get_positive_parameter(params, "gamma", 1.0)
        k_max = self.compute_k_max(matrix, k_min, alpha, gamma)

        return k_min, k_max, node_bounds
