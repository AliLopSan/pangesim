"""Core datastructures representing individual genomes, and pangenomes."""

from collections import Counter
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Set
from typing import Tuple

from tralda.datastructures.doubly_linked import DLList
from tralda.datastructures.doubly_linked import DLListNode


def unwrap_node_value(node: DLListNode) -> Any:
    """Safely extracts the raw primitive value from a tralda node structure."""
    if hasattr(node, "_value") and hasattr(node._value, "_value"):
        return node._value._value
    if hasattr(node, "item"):
        return node.item
    return node


class Genome:
    """A genome modeled as a path forest."""

    __slots__ = ("_genome_id", "heads", "gene_set")

    def __init__(
        self,
        genome_id: Any | None = None,
        heads: List[DLListNode] | None = None,
        gene_set: Set | None = None,
    ) -> None:
        """Initializes an empty Genome container."""
        self._genome_id = genome_id
        self.heads = heads if heads is not None else []
        self.gene_set = gene_set if gene_set is not None else set()

    def __len__(self) -> int:
        """Returns the total number of nodes across all paths in the genome."""
        return sum(1 for _ in self.iter_nodes())

    def __str__(self) -> str:
        """Returns the number of paths and a human-reable list of paths."""
        summary = [
            f"Genome {self._genome_id}:",
            f"├── Number of Paths: {self.path_count}",
            f"├── Number of Genes: {len(self)}",
            "└── Assigned Paths:",
        ]
        paths = self.get_path_sequences()
        for i, path in enumerate(paths[:5]):
            if len(path)< 10:
                summary.append(f"\t{i + 1}) {path} ")
            else:
                summary.append(f"\t{i + 1}) {path[:3]} ...")
        if self.path_count > 5:
            summary.append(f"\t... and {self.path_count - 5} more paths.")
        return "\n".join(summary)

    @property
    def path_count(self) -> int:
        """Returns the number of distinct path fragments in this forest."""
        return len(self.heads)

    def add_path(self, path: DLList) -> None:
        """Adds an individual path segment by archiving its head node pointer.

        Updates gene set.
        """
        start_node = path.first_node()

        if start_node is not None:
            self.heads.append(start_node)
            current = start_node
            while current is not None:
                value = unwrap_node_value(current)
                if value is not None:
                    self.gene_set.add(value)
                current = current._next

    def iter_nodes(self) -> Generator[DLListNode, None, None]:
        """Yields all genes (DLListNodes) sequentially from path heads.

        Yields:
             DLListNode present per path.
        """
        for head in self.heads:
            current = head
            while current is not None:
                if isinstance(current, DLListNode):
                    yield current
                current = current._next

    def iter_edges(self) -> Generator[Tuple, None, None]:
        """Yields all edges present in the current genome.

        Yields:
           Tuple of DLListNodes in the genome.
        """
        for node in self.iter_nodes():
            if node._next is not None:
                yield (node, node._next)

    def has_edge(self, edge:Tuple[int,int]) -> bool:
        """Checks whether the genome contains a given edge.

        Args:
           edge: the target edge.

        Returns:
           True if edge is in the given genome.
        """
        u = edge[0]
        v = edge[1]
        if u not in self.gene_set or v not in self.gene_set:
            return False

        flag = False
        key = (u, v) if u <= v else (v, u)
        for x,y in self.iter_edges():
            x_val = unwrap_node_value(x)
            y_val = unwrap_node_value(y)
            target = (x_val, y_val) if x_val <= y_val else (y_val, x_val)
            if key[0] == target[0] and key[1] == target[1]:
                flag = True
                break
        return flag


    def get_path_sequences(self) -> List[List[Any]]:
        """Traverse every individual path sequentially from its head pointer.

        Returns:
           A list of paths, where each path is a list of node values.
        """
        all_paths: List[List[Any]] = []
        current_path_nodes: List[Any] = []

        for node in self.iter_nodes():
            value = unwrap_node_value(node)
            current_path_nodes.append(value)
            if node._next is None:
                all_paths.append(current_path_nodes)
                current_path_nodes = []

        return all_paths

    def get_adjacency_tuples(self) -> Set[Tuple[Any, Any]]:
        """Calculates all adjacency tuples inside this genome.

        Returns:
            A set of tuples representing (from node, to node) with sorted values
        """
        adjacencies: Set[Tuple[Any, Any]] = set()

        for head in self.heads:
            current = head

            while current and current._next:
                v1 = unwrap_node_value(current)
                v2 = unwrap_node_value(current._next)
                temp = (v1, v2)
                adjacencies.add(tuple(sorted(temp)))
                current = current._next
        return adjacencies

    def as_adjacency_list(self) -> Dict[Any, Set]:
        """Transforms genome into adjacency list.

        Returns:
           A dictionary with nodes as values a list of neighbors.
        """
        adjacency_list = dict()

        for node in self.iter_nodes():
            n = unwrap_node_value(node)
            adjacency_list[n] = set()

        ad_tuples = self.get_adjacency_tuples()

        for u, v in ad_tuples:
            adjacency_list[u].add(v)
            adjacency_list[v].add(u)

        return adjacency_list

    # Jiazhen's code
    def get_a_path(self, head: DLListNode) -> DLList:
        """Given a head node, reconstruct the path as a list.

        Returns:
            The original list (DLList) of the path with the given head.
        """
        new_dllist = DLList()

        new_dllist._first = head

        node = head
        count = 0

        while node is not None:
            count = count + 1
            new_dllist._last = node
            node = node._next

        new_dllist._count = count

        return new_dllist

    def would_break_path_forest(self,edge:Tuple[int,int]) -> bool:
        """Checks whether inserting (u,v) would break the path forest condition.

        Args:
            edge: the edge to test.

        Returns:
            True if there exists a possible violation created by the edge.
        """
        u = edge[0]
        v = edge[1]
        # If they do not exist, we are safe
        if u not in self.gene_set and v not in self.gene_set:
            return False

        #Case 1: They already exist there
        if self.has_edge(edge):
            return True

        # If they exist, we will find them
        u_node: DLListNode | None = None
        v_node: DLListNode | None = None

        for node in self.iter_nodes():
            val = unwrap_node_value(node)
            if val == u:
                u_node = node
            if val == v:
                v_node = node
            if u_node and v_node:
                break


        # Case 2: Possible branching
        # If either of them is an internal node, then it's not possible
        if u_node and u_node._prev is not None and u_node._next is not None:
            return True

        if v_node and v_node._prev is not None and v_node._next is not None:
            return True

        # Case 3: Cycle detection
        # If both nodes exist in the forest, make sure v is not upstream of u
        if u_node and v_node:
            curr = v_node
            while curr is not None:
                if unwrap_node_value(curr) == u:
                    return True  # Connecting u -> v closes a cycle!
                curr = curr._next

        return False



    def check_integrity(self) -> bool:
        """Checks for path forest conditions.

        Returns:
           True if this is a valid path forest.
        """
        seen_nodes: Set[Any] = set()
        for node in self.iter_nodes():
            value = unwrap_node_value(node)

            # Cycle condition check
            if value in seen_nodes:
                return False
            seen_nodes.add(value)

            # Isolated node check
            if node._prev is None and node._next is None:
                return False

        ad = self.as_adjacency_list()

        for node in ad:
            neighbors = ad[node]
            if len(neighbors) > 2:
                return False

        return True


class Pangenome:
    """A set of genomes."""

    __slots__ = ("_pangenome_id", "_genomes")

    def __init__(self, pangenome_id: Any, genomes: List[Genome] | None = None) -> None:
        """Pangenome constructor.

        Args:
            pangenome_id: Unique identifier for this pangenome collection.
            genomes: Optional initial list of Genome instances.
        """
        self._pangenome_id: Any = pangenome_id
        self._genomes: List[Genome] = genomes if genomes is not None else []

    def __len__(self) -> int:
        """Returns the total number of genomes contained in the pangenome."""
        return len(self._genomes)

    @property
    def genomes(self) -> List[Genome]:
        """Returns the list of genomes within this pangenome."""
        return self._genomes

    @property
    def universal_gene_set(self) -> Set[Any]:
        """Returns the universal gene set (the union of all gene sets)."""
        if not self._genomes:
            return set()

        return set().union(*(genome.gene_set for genome in self._genomes))

    @property
    def total_gene_count(self) -> int:
        """Returns the total number of unique genes present in the pangenome."""
        return len(self.universal_gene_set)

    def add_genome(self, genome: Genome) -> None:
        """Adds a single genome to the pangenome collection."""
        self._genomes.append(genome)

    def compute_core_genes(self) -> Set[Any]:
        """Computes the core gene set using an intersection loop.

        Returns:
            A set of raw gene identifiers present across all genomes.
        """
        if not self._genomes:
            return set()

        # Initialize with the first genome's gene set
        core_genes = self._genomes[0].gene_set

        # Iteratively intersect with the remaining genomes
        for genome in self._genomes[1:]:
            # Short-circuit optimization: if core is already empty, stop checking
            if not core_genes:
                break
            core_genes.intersection_update(genome.gene_set)

        return core_genes

    def compute_weighted_adjacencies(self) -> Dict[Tuple[Any, Any], int]:
        """Calculates frequencies of all adjacencies.

        Returns:
            A dictionary mapping sorted tuple edges to their global frequency counts.
        """
        adjacency_counter: Counter[Tuple[Any, Any]] = Counter()

        for genome in self._genomes:
            # Leverage the Genome's internal adjacency calculations
            genome_edges = genome.get_adjacency_tuples()
            adjacency_counter.update(genome_edges)

        return dict(adjacency_counter)

    def check_integrity(self) -> bool:
        """Validates that all underlying genomes match the global graph topology.

        This method acts as a defensive pipeline gate, ensuring individual
        genome paths do not contain orphaned nodes or disjoint edges.

        Returns:
            bool: True if all integrity checks pass cleanly.

        Raises:
            ValueError: If a genome path breaks topological constraints.
        """
        if not self._genomes:
            raise ValueError("Integrity Failure: Pangenome contains no member genomes.")

        for genome in self._genomes:
            if hasattr(genome, "check_integrity") and not genome.check_integrity():
                raise ValueError(f"Genome '{genome.id}' failed internal validation.")

        return True

    def summary(self) -> str:
        """Overview of the pangenome object."""
        core_genes = list(self.compute_core_genes())
        summary_info = [
            f"Pangenome {self._pangenome_id}:",
            f"├── Constituent genomes: {len(self)}",
            f"└── Core genes: {len(core_genes)}",
        ]
        if len(core_genes) > 0:
            summary_info.append(f"> {core_genes[:5]}")
        return "\n".join(summary_info)
