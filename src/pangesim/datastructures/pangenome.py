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
    """Safely extracts the raw primitive value from a tralda node structure or raw int."""
    if isinstance(node, int):
        return node
    if hasattr(node, "get") and callable(node.get):
        try:
            val = node.get()
            if isinstance(val, int):
                return val
        except Exception:
            pass
    if hasattr(node, "_value"):
        return unwrap_node_value(node._value)
    if hasattr(node, "item"):
        return node.item
    return node


class Genome:
    """A genome modeled as a path forest."""

    __slots__ = ("_genome_id", "heads", "gene_set", "_node_cache","head_set","tail_set")

    def __init__(
        self,
        genome_id: Any | None = None,
        heads: List[DLListNode] | None = None,
        gene_set: Set | None = None,
    ) -> None:
        self._genome_id = genome_id
        self.heads = heads if heads is not None else []
        self.gene_set = gene_set if gene_set is not None else set()
        self._node_cache: Dict[Any, DLListNode] = {}

        self.head_set: Set[int] = set()
        self.tail_set: Set[int] = set()

        # Auto-sync cache, gene set, and head/tail integer sets from heads
        for head in self.heads:
            head_val = unwrap_node_value(head)
            if head_val is not None:
                self.head_set.add(head_val)

            curr = head
            prev_val = None
            while curr is not None:
                val = unwrap_node_value(curr)
                if val is not None:
                    self.gene_set.add(val)
                    node_obj = curr if isinstance(curr, DLListNode) else DLListNode(value=val)
                    self._node_cache[val] = node_obj
                prev_val = val
                curr = curr._next if hasattr(curr, "_next") else None

            if prev_val is not None:
                self.tail_set.add(prev_val)

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
        """Adds an individual path segment, updating gene set, cache, and head/tail sets."""
        start_node = path.first_node()
        if start_node is not None:
            node_to_add = (
                start_node if isinstance(start_node, DLListNode) else DLListNode(value=start_node)
            )
            self.heads.append(node_to_add)

            head_val = unwrap_node_value(start_node)
            if head_val is not None:
                self.head_set.add(head_val)

            current = start_node
            prev_val = None
            while current is not None:
                value = unwrap_node_value(current)
                if value is not None:
                    self.gene_set.add(value)
                    node_obj = (
                        current if isinstance(current, DLListNode) else DLListNode(value=value)
                    )
                    self._node_cache[value] = node_obj
                prev_val = value
                current = current._next if hasattr(current, "_next") else None

            if prev_val is not None:
                self.tail_set.add(prev_val)

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
        """Reverses a path and updates head_set and tail_set accordingly."""
        curr = node
        while curr._prev is not None:
            curr = curr._prev
        old_head = curr
        old_head_val = unwrap_node_value(old_head)

        # Find old tail before reversing
        old_tail = old_head
        while old_tail._next is not None:
            old_tail = old_tail._next
        old_tail_val = unwrap_node_value(old_tail)

        # Reverse pointers
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

        # Update heads list reference
        if old_head in self.heads:
            idx = self.heads.index(old_head)
            self.heads[idx] = new_head
        else:
            self.heads.append(new_head)

        # Swap head/tail integer roles in the sets
        if old_head_val in self.head_set:
            self.head_set.remove(old_head_val)
        self.tail_set.add(old_head_val)

        if old_tail_val in self.tail_set:
            self.tail_set.remove(old_tail_val)
        self.head_set.add(old_tail_val)

        return new_head

    def add_edge(self, edge: Tuple[int, int]) -> bool:
        """Inserts a given edge to genome by coordinating path creation, extension, or merging."""
        if not edge or edge[0] is None or edge[1] is None:
            return False

        u, v = edge[0], edge[1]
        key = (u, v) if u <= v else (v, u)

        if self.has_edge(key):
            return True

        if self.would_break_path_forest(key):
            return False

        u_is_new = u not in self._node_cache
        v_is_new = v not in self._node_cache

        # Dynamically instantiate missing nodes
        if u_is_new:
            self.gene_set.add(u)
            self._node_cache[u] = DLListNode(value=u)
        if v_is_new:
            self.gene_set.add(v)
            self._node_cache[v] = DLListNode(value=v)

        u_n = self._node_cache[u]
        v_n = self._node_cache[v]

        if self.degree(u_n) >= 2 or self.degree(v_n) >= 2:
            return False

        # Dispatch to clean modular helper methods based on node states
        if u_is_new and v_is_new:
            #print("\t\t Handling both new case ", edge)
            return self._handle_both_new(u_n, v_n)
        elif u_is_new or v_is_new:
            #print("\t\t Handling one is new case ", edge)
            return self._handle_single_new(u_n, v_n, u_is_new)
        else:
            #print("\t\t Handling path merge case ", edge)
            return self._handle_path_merge(u_n, v_n)

    def _handle_both_new(self, u_n: DLListNode, v_n: DLListNode) -> bool:
        """Case 1: Both nodes are new -> Create a new 2-node path."""
        u_val = unwrap_node_value(u_n)
        v_val = unwrap_node_value(v_n)

        u_n._next = v_n
        v_n._prev = u_n
        self.heads.append(u_n)

        self.head_set.add(u_val)
        self.tail_set.add(v_val)
        return True

    def _handle_single_new(self, u_n: DLListNode, v_n: DLListNode, u_is_new: bool) -> bool:
        """Case 2: Exactly one node is new -> Extend an existing path."""
        new_node, existing_node = (u_n, v_n) if u_is_new else (v_n, u_n)
        new_val = unwrap_node_value(new_node)
        existing_val = unwrap_node_value(existing_node)

        # Attach after tail
        if existing_node._next is None:
            existing_node._next = new_node
            new_node._prev = existing_node

            # existing_node is no longer a tail; new_node is the new tail
            if existing_val in self.tail_set:
                self.tail_set.remove(existing_val)
            self.tail_set.add(new_val)
            return True

        # Attach before head
        elif existing_node._prev is None:
            new_node._next = existing_node
            existing_node._prev = new_node

            # Find and update heads list reference
            if existing_node in self.heads:
                idx = self.heads.index(existing_node)
                self.heads[idx] = new_node
            else:
                self.heads.append(new_node)

            # existing_node is no longer a head; new_node is the new head
            if existing_val in self.head_set:
                self.head_set.remove(existing_val)
            self.head_set.add(new_val)
            return True

        return False

    def _handle_path_merge(self, u_n: DLListNode, v_n: DLListNode) -> bool:
        """Case 3: Both nodes exist -> Merging two distinct paths using integer sets."""
        u_val = unwrap_node_value(u_n)
        v_val = unwrap_node_value(v_n)

        #Find the head of u path
        u_head = u_n
        while u_head._prev is not None:
            u_head = u_head._prev
        u_head_val = unwrap_node_value(u_head)

        #Find the tail of u path
        u_tail = u_n
        while u_tail._next is not None:
            u_tail = u_tail._next
        u_tail_val = unwrap_node_value(u_tail)

        #Find the head of v path
        v_head = v_n
        while v_head._prev is not None:
            v_head = v_head._prev
        v_head_val = unwrap_node_value(v_head)

        #Find the tail of the v path
        v_tail = v_n
        while v_tail._next is not None:
            v_tail = v_tail._next
        v_tail_val = unwrap_node_value(v_tail)

        if u_head_val == v_head_val:
            return False  # Same path, cycle check

        u_is_tail = u_val in self.tail_set
        u_is_head = u_val in self.head_set
        v_is_tail = v_val in self.tail_set
        v_is_head = v_val in self.head_set

        # Subcase A: u is tail, v is head (u -> v)
        if u_is_tail and v_is_head:
            self.head_set.remove(u_head_val)
            if u_head in self.heads:
                self.heads.remove(u_head)
            self.tail_set.remove(v_tail_val)

            v_tail._next = u_head
            u_head._prev = v_tail
            return True

        # Subcase B: v is tail, u is head (v -> u)
        if v_is_tail and u_is_head:
            self.head_set.remove(v_head_val)
            if v_head in self.heads:
                self.heads.remove(v_head)
            self.tail_set.remove(u_tail_val)

            u_tail._next = v_head
            v_head._prev = u_tail
            return True

        # Subcase C: u is tail, v is tail -> Reverse v's path so v becomes head, then join u to v
        if u_is_tail and v_is_tail:
            new_v_head = self._reverse_path(v_n)
            u_n._next = new_v_head
            new_v_head._prev = u_n
            self.tail_set.discard(u_val)
            self.head_set.discard(v_head_val)
            if v_n in self.heads:
                self.heads.remove(v_n)
            self.head_set.add(v_val)
            return True

        # Subcase D: u is head, v is head -> remove u from heads, join with v's tail
        if u_is_head and v_is_head:
            # Connect v's tail to u
            v_tail._next = u_n
            u_n._prev = v_tail

            # u was a head, but now it has a predecessor (v_tail). Remove u as a head!
            self.head_set.discard(u_val)
            if u_n in self.heads:
                self.heads.remove(u_n)

            # Update tail set: v_tail is no longer a tail, u_tail is still the tail
            self.tail_set.discard(v_tail_val)
            return True

        return False

    def remove_edge(self, edge: Tuple[int, int]) -> bool:
        """Removes a given edge from the genome and updates integer head/tail sets."""
        if self.has_edge(edge):
            u, v = edge[0], edge[1]
            u_node = self._node_cache.get(u)
            v_node = self._node_cache.get(v)

            # Protection against isolated vertices
            if self.degree(u_node) <= 1 or self.degree(v_node) <= 1:
                return False

            # Check Direction A: u -> v was the edge (splitting path into u (new tail) and v (new head))
            if u_node._next == v_node:
                u_node._next = None
                v_node._prev = None

                # v_node becomes a new path head
                self.heads.append(v_node)
                self.head_set.add(v)
                # u_node becomes a new path tail
                self.tail_set.add(u)
                return True

            # Check Direction B: v -> u was the edge (splitting path into v (new tail) and u (new head))
            if v_node._next == u_node:
                v_node._next = None
                u_node._prev = None

                # u_node becomes a new path head
                self.heads.append(u_node)
                self.head_set.add(u)
                # v_node becomes a new path tail
                self.tail_set.add(v)
                return True

        return False

    def get_path_sequences(self) -> List[List[int]]:
        """Traverse every individual path sequentially from its head pointer.

        Returns:
           A list of paths, where each path is a list of node values.
        """
        all_paths: List[List[int]] = []

        for head in self.heads:
            path = self.get_a_path(head)
            current_path_nodes: List[DLListNode] = []
            for node in path:
                value = unwrap_node_value(node)
                current_path_nodes.append(value)
            all_paths.append(current_path_nodes)
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
        """Checks whether inserting (u,v) would break the path forest condition."""
        u = edge[0]
        v = edge[1]
        if u not in self.gene_set and v not in self.gene_set:
            return False

        if self.has_edge(edge):
            return True

        u_node = self._node_cache.get(u)
        v_node = self._node_cache.get(v)

        # Branching check (degree cannot exceed 2)
        if u_node and u_node._prev is not None and u_node._next is not None:
            return True
        if v_node and v_node._prev is not None and v_node._next is not None:
            return True

        if u_node and v_node:
            u_head = u_node
            while u_head._prev is not None:
                u_head = u_head._prev

            v_head = v_node
            while v_head._prev is not None:
                v_head = v_head._prev

            if u_head == v_head:
                curr = u_node
                while curr is not None:
                    if curr == v_node:
                        return True
                    curr = curr._next
                curr = v_node
                while curr is not None:
                    if curr == u_node:
                        return True
                    curr = curr._next
                return True

        return False

    def copy(self) -> "Genome":
        """Returns a robust deep copy of the genome, preserving exact node chains and integer sets."""
        new_genome = Genome(genome_id=self._genome_id)
        new_genome.gene_set = self.gene_set.copy()
        new_genome.head_set = self.head_set.copy()
        new_genome.tail_set = self.tail_set.copy()

        new_heads: List[DLListNode] = []
        new_node_cache: Dict[Any, DLListNode] = {}

        for head in self.heads:
            curr = head
            new_head = None
            prev_new_node = None

            while curr is not None:
                val = unwrap_node_value(curr)
                new_node = DLListNode(value=val)

                if new_head is None:
                    new_head = new_node

                if prev_new_node is not None:
                    prev_new_node._next = new_node
                    new_node._prev = prev_new_node

                new_node_cache[val] = new_node
                prev_new_node = new_node
                curr = curr._next

            if new_head is not None:
                new_heads.append(new_head)

        new_genome.heads = new_heads
        new_genome._node_cache = new_node_cache
        return new_genome

    def check_integrity(self) -> bool:
        """Checks for path forest conditions.

        Returns:
           True if this is a valid path forest.
        """
        if len(self.gene_set) < 2:
            print("\t\t Gene set: :", self.gene_set)
            return False

        seen_nodes: Set[Any] = set()
        for node in self.iter_nodes():
            value = unwrap_node_value(node)

            # Cycle condition check
            if value in seen_nodes:
                print(f"\t\t Cycle condition failed with node {value}")
                return False
            seen_nodes.add(value)

            # Isolated node check
            if node._prev is None and node._next is None:
                print(f"\t\t Isolated node found {node.value}")
                return False

        ad = self.as_adjacency_list()

        if len(ad) == 0:
            print("\t\t Gene set: :", self.gene_set)
            print("\t\t Heads: :", self.heads_set)
            print("\t\t But no adjacency list :( ")
            return False

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
            for path in genome.get_path_sequences():
                if len(path) < 2:
                    raise ValueError(f"Genome '{genome._genome_id}' failed path validation.")

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
