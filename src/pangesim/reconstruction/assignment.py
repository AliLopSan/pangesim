"""Phase 2: Assignment of Paths to Genomes."""
from typing import Any
from typing import List
from typing import Set
from typing import Tuple

import networkx as nx
from tralda.datastructures.doubly_linked import DLList
from tralda.datastructures.doubly_linked import DLListNode

from pangesim import Genome
from pangesim import Pangenome
from pangesim.reconstruction import AdjacencyList
from pangesim.reconstruction import AdjacencyMatrix
from pangesim.reconstruction import AssignmentStrategy
from pangesim.reconstruction import TrailSortingStrategy
from pangesim.reconstruction import matrix_to_list
from pangesim.reconstruction.pairing import IterativeOddPairing
from pangesim.reconstruction.sorting import LengthSorting
from pangesim.reconstruction.utils import ComponentTopology
from pangesim.reconstruction.utils import TopologicalExplorer
from pangesim.reconstruction.utils import build_dll_from_list
from pangesim.reconstruction.utils import component_to_networkx
from pangesim.reconstruction.utils import is_graph_a_path
from pangesim.reconstruction.utils import print_adj_list


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
    __slots__ = ("directed", "eulerize_strategy","path")

    def __init__(self,
                 directed: bool = False,
                 eulerize_strategy: IterativeOddPairing | None = None,
                 trail_sorting: TrailSortingStrategy | None = None,
                 path: bool = False,
                 ) -> None:
        """Initializes the assignment engine.

        Args:
            directed: If True, treats edges as directed vectors. Defaults to False.
            eulerize_strategy: If the graph is not eulerian, use this strategy.
            trail_sorting: Sorting strategy to visit trails. Default is by length.
            path: If True looks for eulerian path instead of trail fro each cc.
        """
        self.directed = directed
        self.eulerize_strategy = (
            eulerize_strategy
            if eulerize_strategy is not None
            else IterativeOddPairing()
        )
        self.trail_sorting = (
            trail_sorting
            if trail_sorting is not None
            else LengthSorting()
        )
        self.path = path

    def graph_to_trails(self,
                        eulerian_edges:List[Any],
                        added_edges:List[Tuple[int,int]],
                        graph:nx.MultiGraph):
        """Given the eulerian edges, return a list of trails.

        Args:
           eulerian_edges: The list of edges in the eulerian path.
           added_edges: Edges that were added.
           graph: The current component.

        Returns:
           A list of lists of trails.
        """
        trails_list: List[List[int]] = []

        #Get the nodes of the eulerian path
        nodes: Set[int] = set()
        node_sequence: List[int] = []
        for u,v,k in eulerian_edges:
            nodes.add(u)
            node_sequence.append(u)

        if len(added_edges) == 0:
            trails_list.append(node_sequence)
        else:
            # split circuit at temporary edges to recover real trails
            # build vertex sequence from circuit edges
            vertex_seq: List[int] = [eulerian_edges[0][0]]
            temp_positions: List[int] = []  # positions (in vertex_seq) of temp edges

            for idx, (u, v, key) in enumerate(eulerian_edges):
                vertex_seq.append(v)
                if not graph.edges[u,v,key]["native"]:
                    temp_positions.append(len(vertex_seq) - 1)

            # cut the circular sequence at every temp-edge seam
            # temp_positions mark indices in vertex_seq where the temp edge ends
            cuts = sorted(set(temp_positions))
            # rotate so first cut is at position 0
            offset = cuts[0]
            rotated = vertex_seq[offset:-1] + vertex_seq[:offset]
            # new cut positions after rotation
            new_cuts = sorted((c - offset) % (len(rotated)) for c in cuts)
            prev = 0
            for cut in new_cuts[1:]:  # first cut is at 0 (seam start)
                segment = rotated[prev:cut]
                if len(segment) >= 2:
                    trails_list.append(segment)
                prev = cut

            # last segment
            segment = rotated[prev:]
            if len(segment) >= 2:
                trails_list.append(segment)
        return trails_list

    def direct_trail(self,component:ComponentTopology,
                     graph:nx.Graph,
                     adj_list:AdjacencyList) -> List[int]:
        """When component is a path, return the path.

        Args:
           component: Current component.
           graph: the component as nx Graph.
           adj_list: The adjacency list.

        Returns:
          The component as a list of ints.
        """
        trail: List[int] = []

        leaves = [n for n, d in graph.degree() if d == 1]

        for v in nx.dfs_preorder_nodes(graph,leaves[0]):
            trail.append(v)
        return trail


    def compute_component_trails(self,
                                 component:ComponentTopology,
                                 adj_list:AdjacencyList
                                 ) -> List[List[int]]:
        """Computes eulerian trails using the given eulerization strategy.

        Args:
            component: The current component.
            adj_list: The weighted adjacency list.

        Returns:
            A Tuple that contains:
            - A list of Eulerian trails as lists of integers
            - The set of edges that were added.
        """
        eulerian_edges: List[Tuple[int, int,int]] = []
        added_edges: List[Tuple[int,int]] = []
        trails_list: List[List[int]] = []
        nx_component = component_to_networkx(nodes=component.nodes,
                                                 adj_list=adj_list,
                                                 directed=self.directed)

        if is_graph_a_path(nx_component):
            trail = self.direct_trail(component,nx_component,adj_list)
            trails_list.append(trail)
        else:
            #if we chose to find eulerian path first, then
            if self.path:
                if nx.has_eulerian_path(nx_component):
                    eulerian_edges = list(nx.eulerian_path(nx_component, keys=True))
                else:
                    added_edges=self.eulerize_strategy.pair_vertices(
                        graph=nx_component,
                        odd_vertices=component.odd_vertices)
                    nx_component.add_edges_from(added_edges,native=False)
                    print_adj_list(nx_component)
                    if nx.has_eulerian_path(nx_component):
                        eulerian_edges = list(nx.eulerian_path(nx_component, keys=True))
                    else:
                        eulerian_edges = list(nx.eulerian_circuit(nx_component))
            #go to eulerian circuit directly
            else:
                if nx.is_eulerian(nx_component):
                    eulerian_edges = list(nx.eulerian_circuit(nx_component, keys=True))
                else:
                    added_edges=self.eulerize_strategy.pair_vertices(
                        graph=nx_component,
                        odd_vertices=component.odd_vertices)
                    nx_component.add_edges_from(added_edges,native=False)
                    eulerian_edges = list(nx.eulerian_circuit(nx_component, keys=True))

            trails_list = self.graph_to_trails(eulerian_edges=eulerian_edges,
                                               added_edges=added_edges,
                                               graph=nx_component)

        return trails_list


    def compute_trails(self,
                       matrix:AdjacencyMatrix,
                       ) -> List[List[int]]:
        """Computes eulerian trails using the given eulerization strategy.

        Args:
            matrix: The weighted adjacency list.

        Returns:
            A list of Eulerian trails as lists of integers.
        """
        adj_list = matrix_to_list(matrix=matrix, directed=self.directed)
        explorer = TopologicalExplorer(adj_list, directed=self.directed)
        components = explorer.extract_components()
        trails: List[List[int]] = []
        #Note for future implementations, what is the graph is directed?
        for component in components:
            comp_trails = self.compute_component_trails(component=component,
                                                   adj_list=adj_list)
            for t in comp_trails:
                trails.append(t)
        return trails

    def _split_trail_conflicts(self,trail: List[int],
                               used_genes: set) -> List[List[int]]:
        """Splits a trail into subtrails.

        The resulting subtrails don't revisit the genes of used_genes.

        Args:
            trail: initial trail
            used_genes: Genes that should not be repeated.

        Returns:
           A list of maximal subtrails that do not repeat nodes.
        """
        fragments: List[List[int]] = []
        current: List[int] = []
        for v in trail:
            if v in used_genes:
                if len(current) >= 2:
                    fragments.append(current)
                current = []
            else:
                current.append(v)
        if len(current) >= 2:
            fragments.append(current)
        return fragments

    def build_genomes(self,trails:List[List[int]], k:int) -> List[Genome]:
        """Assigns sorted trails to k genomes.

        Args:
            trails: list of trails to assign.
            k: number of desired genomes.

        Returns:
            A list of k genomes.
        """
        def build_fragment(fragment:List[int]) -> None:
            """Recursively build fragments.

            If conflict vertices exist, then split further.

            Args:
                fragment: a fragment as a list of ints.
            """
            # build_dll_from_list <- list to DLList
            genes_in_f = set(fragment)
            best_i = min(range(k), key=lambda i: len(genomes[i].gene_set & genes_in_f))
            used_genes = genomes[best_i].gene_set

            if used_genes & genes_in_f:
                #if conflicts exist, we split the fragment
                sub_fragments = self._split_trail_conflicts(fragment,used_genes)
                for sub in sub_fragments:
                    build_fragment(sub)
            else:
                new_path = build_dll_from_list(fragment)
                genomes[best_i].add_path(new_path)

        genomes: List[Genome] = [Genome(genome_id=i) for i in range(k)]

        for trail in trails:
            build_fragment(trail)

        return genomes

    def assign_genomes(self, adjacencies: AdjacencyMatrix, k: int) -> Pangenome:
        """Decomposes an adjacency matrix into a reconstructed Pangenome object.

        Args:
            adjacencies: The global weighted adjacency matrix.
            k: The target number of genomes to reconstruct.

        Returns:
            A  Pangenome containing k genomes built by Eulerian Path decomposition.
        """
        trails = self.compute_trails(adjacencies)
        trails_sorted = self.trail_sorting.sort(trails)
        genomes = self.build_genomes(trails_sorted, k)

        return Pangenome(pangenome_id="Euler",genomes=genomes)


