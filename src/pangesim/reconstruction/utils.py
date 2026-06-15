"""Topological processing utilities for Eulerian pangenome reconstruction."""

from collections import deque
from typing import Dict
from typing import List
from typing import NamedTuple

import networkx as nx
from tralda.datastructures.doubly_linked import DLList
from tralda.datastructures.doubly_linked import DLListNode

from pangesim.reconstruction.base import AdjacencyList


class ComponentTopology(NamedTuple):
    """Encapsulates the structural characteristics of a connected component."""
    nodes: set[int]
    odd_vertices: list[int]
    is_eulerian: bool


class TopologicalExplorer:
    """A high-performance scratchpad that segments pre-built adjacency lists.

    This class runs a fast BFS traversal over a custom AdjacencyList structure
    to isolate connected subgraphs and identify topological source/sink imbalances.
    """
    __slots__ = ("adj_list", "directed","degrees")

    def __init__(self,
                 adj_list: AdjacencyList,
                 directed: bool = False) -> None:
        """Initializes the explorer with a pre-constructed adjacency list.

        Args:
            adj_list: A pre-built AdjacencyList dict mapping nodes to (neighbor, weight).
            directed: If True, treats the connections as directed edges.
        """
        self.adj_list = adj_list
        self.directed = directed
        self.degrees: Dict[int, int] = {}

        # Calculate degrees from the adjacency list representation
        if self.directed:
            # Directed graph: track balance as (out_degree - in_degree)
            for source, neighbors in self.adj_list.items():
                self.degrees[source] = self.degrees.get(source, 0) + len(neighbors)
                for target, _ in neighbors:
                    self.degrees[target] = self.degrees.get(target, 0) - 1
        else:
            # Undirected graph: the length of the neighbor list IS the degree
            # since matrix_to_list already handled duplicating entries!
            for node, neighbors in self.adj_list.items():
                self.degrees[node] = len(neighbors)

    def extract_components(self) -> list[ComponentTopology]:
        """Segments the graph into isolated connected components using BFS.

        Returns:
            A list of ComponentTopology tuples detailing each component.
        """
        visited: set[int] = set()
        components: list[ComponentTopology] = []

        for root_node in self.adj_list:
            if root_node in visited:
                continue

            component_nodes: set[int] = set()
            odd_vertices: list[int] = []
            queue: deque[int] = deque([root_node])
            visited.add(root_node)

            while queue:
                current = queue.popleft()
                component_nodes.add(current)

                # Identify imbalances using our pre-calculated lookup map
                # (Handle missing nodes safely if they have no outward edges in directed mode)
                balance = self.degrees.get(current, 0)
                if self.directed:
                    if balance != 0:
                        odd_vertices.append(current)
                else:
                    if balance % 2 != 0:
                        odd_vertices.append(current)

                # Traverse neighbors from the (neighbor, weight) tuple layout
                for neighbor, _ in self.adj_list.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(
                ComponentTopology(
                    nodes=component_nodes,
                    odd_vertices=odd_vertices,
                    is_eulerian=len(odd_vertices) == 0
                )
            )

        return components

def component_to_networkx(
        nodes: set[int],
        adj_list: AdjacencyList,
        directed: bool = False ) -> nx.MultiGraph:
    """Transforms a specific isolated connected component into a NetworkX graph.

    This utility isolates edge translation to a subset of nodes, ensuring
    NetworkX graph construction scales efficiently with large datasets.

    Args:
        nodes: The set of node IDs belonging strictly to this component.
        adj_list: The global pre-built AdjacencyList representation.
        directed: If True, returns an nx.DiGraph; otherwise returns an nx.Graph.

    Returns:
        A NetworkX graph object with the component's topology and weights.
    It also stores a native tag, to see if the edge was native or added.
    """
    graph = nx.MultiDiGraph() if directed else nx.MultiGraph()

    for node in nodes:
        # Safely fetch neighbors from the adjacency list
        for neighbor, weight in adj_list.get(node, []):
            # Avoid duplicating undirected edges in the NetworkX layer
            if not directed and node > neighbor:
                continue
            graph.add_edge(node, neighbor, weight=weight,native=True)

    return graph

def is_graph_a_path(graph: nx.MultiGraph) -> bool:
    """Checks if given graph is a path.

    Args:
        graph: A networkx graph.
    """
    if len(graph) <= 1:
        return True
    if not nx.is_tree(graph):
        return False
    return max(dict(graph.degree()).values()) <= 2

def print_adj_list(graph: nx.MultiGraph)->None:
    """Prints the adjacency list of the given graph.

    Args:
        graph: A networkx graph.
    """
    for node in graph:
        neighbors = [v for v in graph.neighbors(node)]
        print("\t", node, "\t ", neighbors)

def build_dll_from_list(some_list:List[int]) -> List[DLList]:
    """Builds a DLList from a list of ints.

    Args:
       some_list: A list of integers.

    Returns:
       A double linked list with the given nodes and order.
    """
    new_path = DLList()

    for v in some_list:
        v_node = DLListNode(value=v)
        new_path.append(v_node)

    return new_path

