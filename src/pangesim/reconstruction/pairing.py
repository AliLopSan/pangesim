"""Concrete strategies for Eulerization (transforming a non-eulerian graph into one)."""

import random as rd
from itertools import combinations
from typing import Dict
from typing import List

import networkx as nx

from pangesim.reconstruction.base import OddPairingStrategy


class RandomOddPairing(OddPairingStrategy):
    """Pairs odd vertices completely at random (Baseline heuristic)."""

    def pair_vertices(self, graph: nx.MultiGraph, odd_vertices: list[int]) -> list[tuple[int, int]]:
        """Random pairing of vertices.

        Args:
           graph: A network x graph.
           odd_vertices: A list of odd vertices to be paired.

        Returns:
           A list of pairs of odd vertices.
        """
        nodes = odd_vertices.copy()
        rd.shuffle(nodes)

        edges_to_add = []
        for i in range(0, len(nodes), 2):
            edges_to_add.append((nodes[i], nodes[i + 1]))

        return edges_to_add


class IterativeOddPairing(OddPairingStrategy):
    """Pairs odd vertices using an iterative state-evaluation loop.

    This strategy re-evaluates the graph's degree sequence after adding
    each individual edge, preventing blind edge allocations that leave
    topological anomalies.
    """

    def odd_adj_list(self, graph: nx.MultiGraph, odd_vertices: list[int]) -> Dict[int, List[int]]:
        """Current adjacency list of the odd nodes.

        Args:
            graph: the current component.
            odd_vertices: the list of odd vertices

        Returns:
            The adjacency list of all odd vertices.
        """
        ad_list: Dict[int, List[int]] = {}
        for node in odd_vertices:
            neighbors = [v for v in graph.neighbors(node)]
            ad_list[node] = list(neighbors)
        return ad_list

    def pair_vertices(self, graph: nx.MultiGraph, odd_vertices: list[int]) -> list[tuple[int, int]]:
        """Iteratively pairs odd vertices, checking graph state at each step.

        Args:
            graph: The current NetworkX MultiGraph component layer.
            odd_vertices: Initial list of node IDs with odd degrees.

        Returns:
            A list of edge tuples that successfully eulerized the graph.
        """
        edges_to_add: list[tuple[int, int]] = []
        ad_list = self.odd_adj_list(graph, odd_vertices)
        odd_tracker: list = list(odd_vertices)

        rd.shuffle(odd_tracker)

        # Dynamic optimization loop
        while len(odd_tracker) > 0:
            u = odd_tracker.pop()
            v = odd_tracker.pop()

            new_edge = (u, v)
            edges_to_add.append(new_edge)

            # Step 3: Update the working state immediately
            ad_list[u].append(v)
            ad_list[v].append(u)

            odd_tracker = [n for n in ad_list if len(ad_list[n]) % 2 != 0]

        return edges_to_add


class MinimumWeightMatching(OddPairingStrategy):
    """Pairs vertices by minimizing edge weights (Advanced optimization)."""

    def pair_vertices(self, graph: nx.MultiGraph, odd_vertices: list[int]) -> list[tuple[int, int]]:
        """MWPM pairing of vertices.

        Args:
           graph: A network x graph.
           odd_vertices: A list of odd vertices to be paired.

        Returns:
           A list of edges to add.
        """
        if len(odd_vertices) == 0:
            return graph

        #get all shortest paths between vertices of odd degree
        odd_deg_pairs_paths = [
            (m, {n:nx.shortest_path(graph, source=m, target=n)})
            for m, n in combinations(odd_vertices, 2)
        ]
        upper_bound_on_max_path_length = len(graph) + 1

        aux_graph = nx.Graph()
        for n, ps in odd_deg_pairs_paths:
            for m, p in ps.items():
                if n!= m:
                    aux_graph.add_edge(
                        m,n,weight=upper_bound_on_max_path_length - len(p), path=p)
        best_matching = nx.Graph(list(nx.max_weight_matching(aux_graph)))
        edges = []

        for m,n in best_matching.edges():
            path = aux_graph[m][n]["path"]
            for edge in list(nx.utils.pairwise(path)):
                edges.append(edge)
        return edges


