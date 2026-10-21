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
    if isinstance(node.get(), int):
        return node.get()
    if hasattr(node, "item"):
        return node.item
    return node


class Genome:
    """A genome modeled as a path forest."""

    __slots__ = ("_genome_id", "heads", "gene_set", "_node_cache")

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
        self._node_cache: Dict[Any, DLListNode] = {}

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
            if len(path) < 10:
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
                    self._node_cache[value] = current
                current = current._next

    def degree(self, node_identifier: Any) -> int:
        """Computes the degree of a node.

        Accepts either a primitive value (searched via cache) or a DLListNode.

        Args:
            node_identifier: input node to verify or id.
        """
        node = node_identifier
        if not isinstance(node, DLListNode):
            node = self._node_cache.get(node_identifier)

        if not node:
            return 0

        neighbors = 0

        if node._prev is not None:
            neighbors += 1
        if node._next is not None:
            neighbors += 1
        return neighbors

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

    def has_edge(self, edge: Tuple[int, int]) -> bool:
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
        for x, y in self.iter_edges():
            x_val = unwrap_node_value(x)
            y_val = unwrap_node_value(y)
            target = (x_val, y_val) if x_val <= y_val else (y_val, x_val)
            if key[0] == target[0] and key[1] == target[1]:
                flag = True
                break
        return flag

    def _reverse_path(self, node: DLListNode) -> DLListNode:
        """Reverses the entire path containing the given node and updates heads.

        Args:
            node: Any node belonging to the target path.

        Returns:
            The new head of the reversed path.
        """
        # 1. Find the current head of this path
        curr = node
        while curr._prev is not None:
            curr = curr._prev
        old_head = curr

        # 2. Traverse and reverse _next and _prev pointers
        prev_node = None
        curr = old_head
        new_head = None
        while curr is not None:
            next_node = curr._next
            curr._next = prev_node
            curr._prev = next_node
            prev_node = curr
            new_head = curr
            curr = next_node

        # 3. Update self.heads reference
        if old_head in self.heads:
            idx = self.heads.index(old_head)
            self.heads[idx] = new_head
        else:
            self.heads.append(new_head)

        return new_head

    def add_edge(self, edge: Tuple[int, int]) -> bool:
        """Inserts a given edge to genome.

        Now handles path extension, creation and joining paths.

        Args:
           edge: Edge to be added

        Returns:
           True if it is possible to add edge, False otherwise.
        """
        u, v = edge[0], edge[1]

        if self.has_edge((u, v)):
            return True

        # Check path forest constraints before addition
        if self.would_break_path_forest((u, v)):
            return False

        u_is_new = u not in self.gene_set
        v_is_new = v not in self.gene_set

        # Dynamically instantiate missing genes
        if u_is_new:
            self.gene_set.add(u)
            self._node_cache[u] = DLListNode(value=u)
        if v_is_new:
            self.gene_set.add(v)
            self._node_cache[v] = DLListNode(value=v)

        #Updating node cache
        u_n = self._node_cache[u]
        v_n = self._node_cache[v]

        # Verify degree conditions
        if self.degree(u_n) >= 2 or self.degree(v_n) >= 2:
            return False

        # Case 1: Both nodes are new -> Create a new 2-node path
        if u_is_new and v_is_new:
            u_n._next = v_n
            v_n._prev = u_n
            self.heads.append(u_n)  # u becomes the head of this new path
            return True

        # Case 2: Exactly one node is new -> Extend an existing path
        if u_is_new or v_is_new:
            if u_is_new and not v_is_new:
                # Attach u to v
                if v_n._next is None:  # v is tail, attach u after (v -> u)
                    v_n._next = u_n
                    u_n._prev = v_n
                    return True
                elif v_n._prev is None:  # v is head, attach u before (u -> v)
                    u_n._next = v_n
                    v_n._prev = u_n
                    # u is now the new head of this path
                    if v_n in self.heads:
                        idx = self.heads.index(v_n)
                        self.heads[idx] = u_n
                    else:
                        self.heads.append(u_n)
                    return True
            else:  # v is new and u exists
                if u_n._next is None:  # u is tail, attach v after (u -> v)
                    u_n._next = v_n
                    v_n._prev = u_n
                    return True
                elif u_n._prev is None:  # u is head, attach v before (v -> u)
                    v_n._next = u_n
                    u_n._prev = v_n
                    if u_n in self.heads:
                        idx = self.heads.index(u_n)
                        self.heads[idx] = v_n
                    else:
                        self.heads.append(v_n)
                    return True
            return False
        # Case 3: Both nodes exist -> Merging two distinct paths
        u_is_tail = u_n._next is None
        u_is_head = u_n._prev is None
        v_is_tail = v_n._next is None
        v_is_head = v_n._prev is None

        # Subcase A: u is tail, v is head (u -> v) -> Direct merge
        if u_is_tail and v_is_head:
            u_n._next = v_n
            v_n._prev = u_n
            if v_n in self.heads:
                self.heads.remove(v_n)
            return True

        # Subcase B: v is tail, u is head (v -> u) -> Direct merge
        if v_is_tail and u_is_head:
            v_n._next = u_n
            u_n._prev = v_n
            if u_n in self.heads:
                self.heads.remove(u_n)
            return True

        # Subcase C: u is tail, v is tail -> Reverse path v so v becomes head, then merge
        if u_is_tail and v_is_tail:
            new_v_head = self._reverse_path(v_n)
            u_n._next = new_v_head
            new_v_head._prev = u_n
            if new_v_head in self.heads:
                self.heads.remove(new_v_head)
            return True

        # Subcase D: u is head, v is head -> Reverse path u so u becomes tail, then merge
        if u_is_head and v_is_head:
            new_u_head = self._reverse_path(u_n)
            # Find tail of reversed u path
            curr = new_u_head
            while curr._next is not None:
                 curr = curr._next
            curr._next = v_n
            v_n._prev = curr
            if v_n in self.heads:
                self.heads.remove(v_n)
            return True

        return False

    def remove_edge(self, edge: Tuple[int, int]) -> bool:
        """Removes a given edge from the genome.

        Args:
           edge: Edge to be removed

        Returns:
           True if the edge existed.
        """
        if self.has_edge(edge):
            u, v = edge[0], edge[1]
            u_node = self._node_cache.get(u)
            v_node = self._node_cache.get(v)

            # Protection against isolated vertices
            if self.degree(u_node) <= 1 or self.degree(v_node) <= 1:
                return False

            # Check Direction A: u -> v
            if u_node._next == v_node:
                u_node._next = None
                v_node._prev = None
                self.heads.append(v_node)
                return True

            # Check Direction B: v -> u
            if v_node._next == u_node:
                v_node._next = None
                u_node._prev = None
                self.heads.append(u_node)
                return True
        else:
            return False

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

    def would_break_path_forest(self, edge: Tuple[int, int]) -> bool:
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

        # Case 1: They already exist there
        if self.has_edge(edge):
            return True

        # If they exist, we will find them
        u_node = self._node_cache.get(u)
        v_node = self._node_cache.get(v)

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

    def copy(self) -> "Genome":
        """Returns a deep copy of the genome structure."""
        new_genome = Genome(genome_id=self._genome_id)
        # Re-build independent path copies using your get_path_sequences structure
        for path_seq in self.get_path_sequences():
            new_list = DLList()
            for val in path_seq:
                new_list.append(DLListNode(value=val))
            new_genome.add_path(new_list)
        return new_genome

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

    __slots__ = ("_pangenome_id", "_genomes", "core", "core_edges")

    def __init__(self, pangenome_id: Any, genomes: List[Genome] | None = None) -> None:
        """Pangenome constructor.

        Args:
            pangenome_id: Unique identifier for this pangenome collection.
            genomes: Optional initial list of Genome instances.
        """
        self._pangenome_id: Any = pangenome_id
        self._genomes: List[Genome] = genomes if genomes is not None else []
        self.core = set()
        self.core_edges = set()

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

    def compute_core_genes(self) -> Set[Any]:
        """Computes the core gene set using an intersection loop.

        Returns:
            A set of raw gene identifiers present across all genomes.
        """
        if not self._genomes:
            return set()

        # Initialize with the first genome's gene set
        core_genes = self._genomes[0].gene_set.copy()

        # Iteratively intersect with the remaining genomes
        for genome in self._genomes[1:]:
            # Short-circuit optimization: if core is already empty, stop checking
            if not core_genes:
                break
            core_genes.intersection_update(genome.gene_set)

        return core_genes

    def compute_core_edges(self) -> Set[Any]:
        """Obtains the core edges.

        Returns:
               The set of core edges, if any.
        """
        core_edges = set()

        if len(self.core) > 1:
            for g in self.genomes:
                for u, v in g.iter_edges():
                    u_val = unwrap_node_value(u)
                    v_val = unwrap_node_value(v)
                    if u_val in self.core and v_val in self.core:
                        key = (u_val, v_val) if u_val <= v_val else (v_val, u_val)
                        core_edges.add(key)
        return core_edges

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

    def add_genome(self, genome: Genome) -> None:
        """Adds a single genome to the pangenome collection."""
        self._genomes.append(genome)
        self.core = self.compute_core_genes()
        self.core_edges = self.compute_core_edges()

    def remove_genome(self, genome_id: Any) -> None:
        """Removes a genome from the pangenome collection by its identifier.

        After removal, the core gene and core edge states are completely
        recomputed to maintain global sync.

        Args:
            genome_id: The unique identifier (_genome_id) of the genome to remove.

        Raises:
            KeyError: If a genome with the provided genome_id does not exist.
        """
        target_index = -1

        for i, genome in enumerate(self._genomes):
            if genome._genome_id == genome_id:
                target_index = i
                break

        # Defensive failure:
        if target_index == -1:
            raise KeyError(f"Genome with ID '{genome_id}' not found in pangenome.")

        self._genomes.pop(target_index)

        self.core = self.compute_core_genes()
        self.core_edges = self.compute_core_edges()

    def replace_genome(self, genome_id: Any, new_genome: Genome) -> None:
        """Replaces an existing genome instance."""
        for i, g in enumerate(self._genomes):
            if g._genome_id == genome_id:
                self._genomes[i] = new_genome
                self.core = self.compute_core_genes()
                self.core_edges = self.compute_core_edges()
                return
        raise KeyError(f"Genome with ID '{genome_id}' not found in pangenome.")

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
                raise ValueError(f"Genome '{genome._genome_id}' failed internal validation.")

        return True

    def copy(self) -> "Pangenome":
        """Returns a deep copy of the pangenome structure."""
        new_pangenome = Pangenome(pangenome_id=self._pangenome_id)
        for genome in self._genomes:
            new_genome = genome.copy()
            new_pangenome.add_genome(new_genome)

        return new_pangenome

    def summary(self) -> str:
        """Overview of the pangenome object."""
        core_genes = list(self.core)
        summary_info = [
            f"Pangenome {self._pangenome_id}:",
            f"├── Constituent genomes: {len(self)}",
            f"└── Core genes: {len(core_genes)}",
        ]
        if len(core_genes) > 0:
            summary_info.append(f"> {core_genes[:5]}")
        return "\n".join(summary_info)

    def print_full_paths(self) -> None:
        """Prints all paths of each genome in the pangenome fully without truncation.

        Useful for visual comparison of small-scale example runs in the terminal.
        """
        lines = [
            "========================================",
            f" {self._pangenome_id} full paths",
            "========================================",
        ]
        for genome in self._genomes:
            lines.append(
                f"Genome {genome._genome_id}  of {len(self)} genomes"
                f"\n(Total Genes: {len(genome)}, Paths: {genome.path_count})"
            )
            paths = genome.get_path_sequences()
            for i, path in enumerate(paths):
                prefix = "└──" if i == len(paths) - 1 else "├──"
                lines.append(f"    {prefix} Path {i + 1}: {path}")
            lines.append("-" * 40)
        print("\n".join(lines))
