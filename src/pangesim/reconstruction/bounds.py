"""Phase 1: Computing bounds on the number of genomes."""
from pangesim.reconstruction import AdjacencyMatrix
from pangesim.reconstruction import BoundsStrategy
from pangesim.reconstruction import matrix_to_list


class DummyBounds(BoundsStrategy):
    """Calculates k bounds."""
    def compute_bounds(self,matrix:AdjacencyMatrix):
        """Takes a dummy adjacency matrix and computes them."""
        k_min = 1
        k_max = max(1, len(matrix))  # Toy placeholder logic
        degrees = dict()
        ad_list = matrix_to_list(matrix)
        for v in ad_list:
            degrees[v] = len(ad_list[v])
        return k_min,k_max,degrees
