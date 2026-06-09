"""Concrete strategies for pairing odd-degree nodes in a graph."""

import random

import networkx as nx

from pangesim.reconstruction.base import OddPairingStrategy


class RandomOddPairing(OddPairingStrategy):
    """Pairs odd vertices completely at random (Baseline heuristic)."""

    def pair_vertices(self, graph: nx.Graph,
                      odd_vertices: list[int]) -> list[tuple[int, int]]:
        """Random pairing of vertices.

        Args:
           graph: A network x graph.
           odd_vertices: A list of odd vertices to be paired.

        Returns:
           A list of pairs of odd vertices.
        """
        nodes = odd_vertices.copy()
        random.shuffle(nodes)
        return [(nodes[i], nodes[i + 1]) for i in range(0, len(nodes), 2)]

class MinimumWeightPerfectMatching(OddPairingStrategy):
    """Pairs vertices by minimizing edge weights (Advanced optimization)."""

    def pair_vertices(self, graph: nx.Graph,
                      odd_vertices: list[int]) -> list[tuple[int, int]]:
        """MWPM pairing of vertices.

        Args:
           graph: A network x graph.
           odd_vertices: A list of odd vertices to be paired.

        Returns:
           A list of pairs of odd vertices.
        """
        # Future implementation goes here
        pass
