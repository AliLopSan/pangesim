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
from pangesim.reconstruction import matrix_to_list
from pangesim.reconstruction.pairing import IterativeOddPairing
from pangesim.reconstruction.utils import ComponentTopology
from pangesim.reconstruction.utils import TopologicalExplorer
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
                 path: bool = False,
                 ) -> None:
        """Initializes the assignment engine.

        Args:
            directed: If True, treats edges as directed vectors. Defaults to False.
            eulerize_strategy: If the graph is not eulerian, use this strategy.
            path: If True looks for eulerian path instead of trail fro each cc.
        """
        self.directed = directed
        self.eulerize_strategy = (
            eulerize_strategy
            if eulerize_strategy is not None
            else IterativeOddPairing()
        )
        self.path = path

    def edges_to_trails_list(self,
                            edges:List[Any],
                            temp_edges:List[Tuple[int,int]],
                            )-> List[List[int]]:
        """Transforms a list of edges to trail, removes added edges.

        Args:
            edges: The list of edges to transform.
            temp_edges: All the added temporal edges.

        Returns:
            A trail as a list of integers.
        """
        node_sequence: List[int] = []
        trails_list: List[List[int]] = []
        temp_as_set: Set[int] = set(temp_edges)

        if len(temp_as_set) == 0:
            for u,v in edges:
                node_sequence.append(u)
            trails_list.append(node_sequence)
        else:
            temp_trail: List[int] = []
            for u,v in edges:
                if (u,v) in temp_as_set or (v,u) in temp_as_set:
                    temp_trail.append(u)
                    trails_list.append(temp_trail)
                    temp_trail = []
                else:
                    temp_trail.append(u)
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
        eulerian_edges: List[Tuple[int, int]] = []
        added_edges: List[Tuple[int,int]] = []
        trails_list: List[List[int]] = []
        nx_component = component_to_networkx(nodes=component.nodes,
                                                 adj_list=adj_list,
                                                 directed=self.directed)

        print("\tAnalysis of component: ",component.nodes)
        print_adj_list(nx_component)

        if is_graph_a_path(nx_component):
            trail = self.direct_trail(component,nx_component,adj_list)
            trails_list.append(trail)
        else:
            if not self.path:
                print("\t Finding Eulerian path first")
                if not nx.is_eulerian(nx_component):
                    added_edges=self.eulerize_strategy.pair_vertices(
                        graph=nx_component,
                        odd_vertices=component.odd_vertices)
                    print("Edges to add: ", added_edges)
                    nx_component.add_edges_from(added_edges)
                    print_adj_list(nx_component)
                    print("\t Adjacency list of component AFTER addition")
                if nx.has_eulerian_path(nx_component):
                    eulerian_edges = list(nx.eulerian_path(nx_component))
                else:
                    eulerian_edges = list(nx.eulerian_circuit(nx_component))
            else:
                print("\t Finding Eulerian circuit first")
                if nx.has_eulerian_path(nx_component):
                    eulerian_edges = list(nx.eulerian_path(nx_component))
                else:
                    added_edges=self.eulerize_strategy.pair_vertices(
                        graph=nx_component,
                        odd_vertices=component.odd_vertices)
                    nx_component.add_edges_from(added_edges)
                    eulerian_edges = list(nx.eulerian_circuit(nx_component))
            print("\t \t eulerian edges: ", eulerian_edges)

            trails_list = self.edges_to_trails_list(edges=eulerian_edges,
                                                temp_edges=added_edges)

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




    def assign_genomes(self, adjacencies: AdjacencyMatrix, k: int) -> Pangenome:
        """Decomposes an adjacency matrix into a reconstructed Pangenome object.

        Args:
            adjacencies: The global weighted adjacency matrix.
            k: The target number of genomes to reconstruct.

        Returns:
            A  Pangenome containing k genomes built by Eulerian Path decomposition.
        """
        trails = self.compute_trails(adjacencies)

        print("Found ", len(trails), " trails.")
        for trail in trails:
            print("\t",trail)


        # TODO: Transform all_reconstructed_paths into a Pangenome(genomes=...) list
        return Pangenome(pangenome_id="Euler")


