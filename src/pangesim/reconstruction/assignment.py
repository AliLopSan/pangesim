"""Phase 2: Assignment of Paths to Genomes."""
from typing import List

from tralda.datastructures.doubly_linked import DLList
from tralda.datastructures.doubly_linked import DLListNode

from pangesim import Genome
from pangesim import Pangenome
from pangesim.reconstruction import AdjacencyMatrix
from pangesim.reconstruction import AssignmentStrategy


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
        genomes = List[Genome]
        i = 0
        for u,v in AdjacencyMatrix:
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





