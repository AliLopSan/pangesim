"""Classes for Pangenome refinment."""
from typing import Dict
from typing import List
from typing import Tuple

from pangesim import Genome
from pangesim import Pangenome
from pangesim.reconstruction import AdjacencyMatrix
from pangesim.reconstruction import RefinementStrategy
from pangesim.reconstruction.utils import build_residuals
from pangesim.reconstruction.utils import pan_score


class ResidualsRefinement(RefinementStrategy):
    """Refines a given pangenome using scoring function.

    Use a custom score function to define whether new genomes should be added.
    """

    __slots__ = ("params", "max_iter")

    def __init__(self, params: Dict[str, float] | None = None,
                 max_iter: int | None = None) -> None:
        """Constructor for ResidualsRefinement.

        Args:
           params: Dictionary of parameters.
           max_iter: Failsafe nuumber of iterations.
        """
        self.params = params or {"alpha": 1.0, "gamma": 1.0}
        # Refinement follows until convergence or max_iter
        self.max_iter = max_iter or 100



    def new_genome_with_edge(self,pan:Pangenome,edge:Tuple) -> Genome:
        """Creates a new genome with same core edges and new edge.

        Args:
            pan: current pangenome.
            edge: under-represented edge.

        Returns:
           A genome with the same core edges and a new edge.
        """
        i = len(pan)
        new_genome = Genome(genome_id=i)
        new_genome.add_edge(edge)

        #Add only core edges that would not break path forest condition
        if len(pan.core_edges) > 0:
            for core_edge in pan.core_edges:
                if not new_genome.would_break_path_forest(core_edge):
                    new_genome.add_edge(core_edge)
        return new_genome

    def find_slack(self,pan:Pangenome,
                   edge:Tuple[int,int]) -> List:
        """Find a list of slack genomes.

        Args:
            pan: Pangenome under study.
            edge: Edge to locate.

        Returns:
            A list of pangenomes where the endpoints of edge are free.
        """
        slack_list: List[Genome] = []
        u = edge[0]
        v = edge[1]
        for i in range(len(pan.genomes)):
            genome = pan._genomes[i]
            if u in genome.gene_set and v in genome.gene_set:
                if genome.degree(u) < 2 and genome.degree(v) < 2:
                    if not genome.has_edge(edge) and not genome.would_break_path_forest(edge):
                        slack_list.append(i)
        return slack_list

    def fix_under_edge(self,pan:Pangenome,
                       adj:AdjacencyMatrix,
                       edge:Tuple[int,int],
                       score:float) -> None:
        """Fix an under-represented edge.

        Args:
           pan: Current Pangenome.
           adj: Input weighted adjacency graph.
           edge: Edge to be corrected.
           score: The current score of the Pangenome.

        Returns:
           A corrected Pangenome.
        """
        slack = self.find_slack(pan,edge)
        new_pan = pan.copy()

        if slack:
            #There can be more than 1 candidate, choose the best score
            best_ix = slack.pop()
            best_genome = pan.genomes[best_ix].copy()
            best_genome.add_edge(edge)
            new_pan.replace_genome(best_genome._genome_id,
                                   best_genome)
            best_score = pan_score(new_pan,adj,
                                   self.params["alpha"],self.params["gamma"])
            while slack:
                candidate_pan = pan.copy()
                candidate_ix = slack.pop()
                candidate_genome = pan.genomes[candidate_ix].copy()
                candidate_genome.add_edge(edge)
                candidate_pan.replace_genome(candidate_genome._genome_id,
                                             candidate_genome)
                candidate_score = pan_score(candidate_pan,adj,
                                   self.params["alpha"],self.params["gamma"])
                if candidate_score > best_score:
                    best_ix = candidate_ix
                    best_genome = candidate_genome
                    best_score = candidate_score
            #Replace old slack candidate with its modified version
            pan.replace_genome(genome_id=best_genome._genome_id,
                               new_genome=best_genome)

        else:
            #Cost logic will come afterwards
            g = self.new_genome_with_edge(pan,edge)
            pan.add_genome(g)

    def fix_over_edge(self,pan:Pangenome,
                      adj:AdjacencyMatrix,
                      edge:Tuple[int,int],
                      score:float) -> Pangenome:
        """Fix an over-represented edge.

        Args:
           pan: Current Pangenome.
           adj: Input weighted adjacency graph.
           edge: Edge to be corrected.
           score: The current score of the Pangenome.

        Returns:
           A corrected Pangenome.
        """
        best_pan = pan.copy()
        g_list = list(pan._genomes)
        best_genome = g_list.pop()
        if best_genome.remove_edge(edge):
            best_pan.replace_genome(best_genome._genome_id,
                                   best_genome)
            best_score = pan_score(best_pan,adj,
                                   self.params["alpha"],self.params["gamma"])
        else:
            best_score = score

        while g_list:
            candidate_genome = g_list.pop()
            candidate_pan = pan.copy()
            if candidate_genome.remove_edge(edge):
                candidate_pan.replace_genome(candidate_genome._genome_id,
                                             candidate_genome)
                candidate_score = pan_score(candidate_pan,adj,
                                   self.params["alpha"],self.params["gamma"])
                if candidate_score > best_score:
                    best_genome = candidate_genome
                    best_score = candidate_score

        pan.replace_genome(genome_id=best_genome._genome_id,
                               new_genome=best_genome)


    def refine(self, source: AdjacencyMatrix, target: Pangenome) -> Pangenome:
        """Main refinement method.

        Refines a pangenome by iteratively minimizing edge residuals
        using a greedy hill-climbing optimization driven by pan_score.

        Args:
           source: Input Adjacency Matrix.
           target: Initial pangenome to refine
        Returns:
           A refined pangenome.
        """
        residuals: AdjacencyMatrix = build_residuals(target=target, source=source)
        current_score = pan_score(target,
                                  source,
                                  self.params["alpha"],
                                  self.params["gamma"])
        converged = False
        iters = 0
        pangenome = target.copy()

        while not converged:
            prev_score = current_score
            if all(r == 0 for r in residuals.values()) or iters == self.max_iter:
                break

            worst_edge, worst_residual = max(
                residuals.items(), key=lambda item: abs(item[1])
            )

            # Step 3: Branching strategy depending on the residual type
            if worst_residual > 0:
                # Under-represented edge
                self.fix_under_edge(pangenome,source,worst_edge,current_score)
            else:
                # Over-represented edge
                self.fix_over_edge(pangenome, source,worst_edge,current_score)

            #recalculte residuals
            residuals = build_residuals(target=pangenome,source=source)
            current_score = pan_score(pangenome,
                                  source,
                                  self.params["alpha"],
                                  self.params["gamma"])
            #If there was no score change, get out
            if prev_score == current_score:
                converged = True

            iters += 1


        return pangenome
