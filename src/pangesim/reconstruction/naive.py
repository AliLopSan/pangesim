"""Sequential edge insertion."""
from typing import Callable
from typing import List
from typing import Tuple

from pangesim import Genome
from pangesim import Pangenome
from pangesim.reconstruction import AdjacencyMatrix


class SequentialEdgeInsertion:
    """Orchestrates the naive pangenome reconstruction."""
    def build_sorted_residuals(self, matrix:AdjacencyMatrix) -> List:
        """Sorts the AdjacencyMatrix's tuples in decreasing order.

        Args:
            matrix: Input adjacency matrix in the form ((node,node),weight)
        """
        residuals = [
            (element, matrix[element])
            for element in matrix
            if element[0] != element[1]
        ]
        residuals = sorted(residuals, key=lambda x:x[1], reverse=True)
        return residuals

    def find_slack(self, pan:Pangenome, edge:Tuple[int,int])->List:
        """Finds a slack genome.

        Args:
            pan: the current pangenome.
            edge: the target edge.
        """
        slack_list: List[Genome] = []

        if len(pan) == 0:
            return slack_list
        else:
            u = edge[0]
            v = edge[1]
            for i in range(len(pan)):
                genome = pan.genomes[i]
                if u in genome.gene_set and v in genome.gene_set:
                    if not genome.has_edge(edge):
                        if genome.degree(u) < 2 and genome.degree(v) < 2:
                            if not genome.would_break_path_forest(edge):
                                slack_list.append(i)
                else:
                    if u in genome.gene_set and genome.degree(u) < 2:
                        slack_list.append(i)
                    if v in genome.gene_set and genome.degree(v) < 2:
                        slack_list.append(i)
                    if u not in genome.gene_set and v not in genome.gene_set:
                        slack_list.append(i)
            return slack_list

    def new_genome_with_edge(self, pan:Pangenome, edge:Tuple) -> Genome:
        """Creates a brand new genome with the given edge.

        Args:
            pan: Current pangenome.
            edge: Edge to be added.
        """
        i = len(pan)
        new_genome = Genome(genome_id=i)
        if new_genome.add_edge(edge):
            return new_genome
        return None

    def reconstruct(
            self,
            matrix: AdjacencyMatrix,
            ground_truth: Pangenome | None = None,
            callbacks: List[Callable] | None = None) -> Pangenome:
        """Executes full pipeline.

        Args:
            matrix: weighted adjacency matrix.
            ground_truth: Benchmark to compare with.
            callbacks: Event Callback Observer pattern.

        Returns:
            A pangenome that explains matrix.
        """
        pan = Pangenome(pangenome_id="SEQ")
        callbacks = callbacks or []

        r = self.build_sorted_residuals(matrix)

        for edge,weight in r:
            #Compute slack set of edge
            target = weight
            slack_list = self.find_slack(pan, edge)

            while target > 0:
                success = False
                if slack_list:
                    index = slack_list.pop()
                    candidate_genome = pan.genomes[index].copy()

                    if candidate_genome.add_edge(edge):
                        pan.replace_genome(candidate_genome._genome_id, candidate_genome)
                        success = True

                if not success:
                    new_genome = self.new_genome_with_edge(pan, edge)
                    if new_genome is not None:
                        pan.add_genome(new_genome)
                        success = True

                if success:
                    target -= 1
                else:
                    break

        return pan
