"""Sorting strategies for Eulerian trails."""

from itertools import pairwise
from typing import List

from pangesim.reconstruction.base import AdjacencyMatrix
from pangesim.reconstruction.base import TrailSortingStrategy


class LengthSorting(TrailSortingStrategy):
    """Sorts trails based on the number of edges they contain."""

    def __init__(self, descending: bool = True):
        """Constructor of the sorting strategy.

        Args:
           descending: Choose from descending to ascending order.
        """
        self.descending = descending

    def sort(self, trails: List[List[int]]) -> List[List[int]]:
        """Main sorting function.

        Args:
           trails: elements to be sorted.

        Returns:
           A sorted list of trails.
        """
        return sorted(trails, key=len, reverse=self.descending)


class WeightSorting(TrailSortingStrategy):
    """Sorts trails based on the cumulative weight of their edges."""

    def __init__(self, descending: bool = True):
        """Constructor of the sorting strategy.

        Args:
           descending: Choose from descending to ascending order.
        """
        self.descending = descending

    def get_trail_weight(self, trail: List[int], matrix: AdjacencyMatrix) -> int:
        """Computes the weight of a trail.

        Args:
            trail: Trail to compute as list of ints.
            matrix: Input weighted adjacencies.
        """
        weight = 0

        if len(trail) > 1:
            for u, v in pairwise(trail):
                edge = (u, v) if u < v else (v, u)
                weight += matrix.get(edge, 0)
        return weight

    def sort(self, trails: List[List[int]], matrix: AdjacencyMatrix) -> List[List[int]]:
        """Main sorting function.

        Args:
           trails: elements to be sorted.
           matrix: the adjacency matrix to draw the weights from.

        Returns:
           A sorted list of trails.
        """
        sorted_trails = sorted(
            trails, key=lambda t: self.get_trail_weight(t, matrix), reverse=self.descending
        )
        # We calculate the sum of weights for each trail
        return sorted_trails
