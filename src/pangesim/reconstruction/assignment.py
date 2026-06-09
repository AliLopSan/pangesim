"""Phase 2: Assignment of Paths to Genomes."""
from typing import List

from tralda.datastructures.doubly_linked import DLList
from tralda.datastructures.doubly_linked import DLListNode

from pangesim import Genome
from pangesim import Pangenome
from pangesim.reconstruction import AdjacencyMatrix
from pangesim.reconstruction import AssignmentStrategy
from pangesim.reconstruction import matrix_to_list
from pangesim.reconstruction.utils import TopologicalExplorer
from pangesim.reconstruction.utils import component_to_networkx


class DummyAssignment(AssignmentStrategy):
    """Assigns every edge to a genome."""
    def assign_genomes(self,
                      adjacencies: AdjacencyMatrix,
                      k: int) -> Pangenome:
        """Decomposes graph into a list of genomes.

        Every edge in the adjacency matrix is a genome.

        Args:
        adjacencies: The weighted adjacency matrix.
        k: The final number of genomes.

        Returns:
        A pangenome object with k genomes.
        """
        genomes: List[Genome] = []
        i = 0

        for u,v in adjacencies:
            genome = Genome(genome_id = str(i))
            u_node = DLListNode(value=u)
            v_node = DLListNode(value=v)
            new_path = DLList()
            new_path.append(u_node)
            new_path.append(v_node)
            genome.add_path(new_path)
            genomes.append(genome)
            i+=1

        return Pangenome(pangenome_id=f"dummy_with_{k}",genomes=genomes)


class EulerianTrailAssignment(AssignmentStrategy):
    """Decomposes an adjacency matrix into genome trails using localized components.

    This strategy leverages a lightweight TopologicalExplorer to partition the
    global graph into independent subgraphs, minimizing NetworkX overhead by
    restricting circuit calculations strictly to isolated, active components.
    """
    __slots__ = ("directed")

    def __init__(self, directed: bool = False) -> None:
        """Initializes the assignment engine.

        Args:
            directed: If True, treats edges as directed vectors. Defaults to False.
        """
        self.directed = directed

    def assign_genomes(self, adjacencies: AdjacencyMatrix, k: int) -> Pangenome:
        """Decomposes an adjacency matrix into a reconstructed Pangenome object.

        Args:
            adjacencies: The global weighted adjacency matrix.
            k: The target number of genomes to reconstruct.

        Returns:
            A  Pangenome containing k genomes built by Eulerian Path decomposition.
        """
        adj_list = matrix_to_list(adjacencies, directed=self.directed)
        explorer = TopologicalExplorer(adj_list, directed=self.directed)
        components = explorer.extract_components()

        for component in components:
            nx_component = component_to_networkx(nodes=component.nodes,
                                                 adj_list=adj_list,
                                                 directed=self.directed)
            if nx_component.is_eulerian:
                print("Component is eulerian")

            else:
                print("Component is not eulerian")

        # TODO: Transform all_reconstructed_paths into a Pangenome(genomes=...) list
        return Pangenome()


