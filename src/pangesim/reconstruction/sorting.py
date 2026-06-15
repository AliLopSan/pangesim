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

    def sort(self, trails):
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
    #TODO:recheck this
    def sort(self,
             trails,
             matrix:AdjacencyMatrix):
        """Main sorting function.

        Args:
           trails: elements to be sorted.
           matrix: the adjacency matrix to draw the weights from.

        Returns:
           A sorted list of trails.
        """
        # We calculate the sum of weights for each trail
        return sorted(
            trails,
            key=lambda t: sum(matrix[u][v] for u, v in t),
            reverse=self.descending
        )
