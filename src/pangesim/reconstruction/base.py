"""Abstract definitions ensuring interchangeable execution strategies."""
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Tuple

import networkx as nx

from pangesim import Pangenome

# Type alias for structural clarity
AdjacencyMatrix = Dict[Tuple[int, int], int]
AdjacencyList = Dict[int,List[Tuple[int,int]]]

def matrix_to_list(matrix: AdjacencyMatrix,
                   directed: bool = False) -> AdjacencyList:
    """Transforms a weighted adjacency matrix representation into a list.

    Args:
        matrix: A weighted adjacency graph.
        directed: If True, treats edges as one-way. If False, treats edges
            as undirected, duplicating entries for both nodes.

    Returns:
        An adjacency list with weights representation.
    """
    adj_list: AdjacencyList = {}
    for source, target in matrix:
        if source not in adj_list:
            adj_list[source] = []
        adj_list[source].append((target,matrix[(source,target)]))

        if not directed:
            if target not in adj_list:
                adj_list[target] = []
            adj_list[target].append((source,matrix[(source,target)]))
    return adj_list

class BoundsStrategy(ABC):
    """Bounds strategy blueprint for computing k bounds."""
    @abstractmethod
    def compute_bounds(self,
                       adjacencies: AdjacencyMatrix,
                       params: Dict | None = None,
                       ) -> Tuple[int, int,Dict[int, int]]:
        """Computes the genomes bounds.

        Args:
           adjacencies: The weighted adjacency matrix.
           params: An optional dictionary of parameters.

        Returns (k_min, k_max_structural, per_vertex_k) where:
           k_min               = max_v LocalK(v, H)
           k_max_structural    = max_{uv} m_{uv}   (sharp upper bound)
           per_vertex_k        = {v: LocalK(v, H)} (or any other usefl info).
        """
        pass


class AssignmentStrategy(ABC):
    """Assignment strategy blueprint for computing genome assignments."""
    @abstractmethod
    def assign_genomes(self,
                      adjacencies: AdjacencyMatrix,
                      k: int) -> Pangenome:
        """Decomposes graph into a list of genomes.

        Args:
           adjacencies : The weighted adjacency matrix.
           k: the final number of genomes.

        Returns:
           A pangenome object with k genomes.
        """
        pass


class RefinementStrategy(ABC):
    """Refinement Strategy blueprint."""
    @abstractmethod
    def reconcile(self, pangenome:Pangenome,
                  observed: AdjacencyMatrix,
                  alpha: float,
                  gamma: float,
                  max_iter: int = 100
                  ) -> Pangenome:
        """Computes residuals and materializes the unified Pangenome object.

        Args:
            pangenome: The intial pangenome object to be refined.
            observed: The weighted adjacencies.
            alpha: per-genome reward in the score.
            gamma: weight-error penalty coefficient.
            max_iter: If convergence is not reached, then iteratively fix
                    the last residual until max_iter rounds.

        Returns:
            The refined pangenome object.
        """
        pass

class OddPairingStrategy(ABC):
    """Abstract base class for pairing odd-degree vertices."""

    @abstractmethod
    def pair_vertices(self,graph: nx.MultiGraph,
                      odd_vertices: list[int]) -> list[tuple[int, int]]:
        """Computes new edges to make a component Eulerian.

        Args:
           graph: a networkx multi graph.
           odd_vertices: All odd vertices in the graph.
        """
        pass
