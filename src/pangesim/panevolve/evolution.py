import random
from typing import Any
from typing import Dict
from typing import List

import asymmetree.treeevolve as te
import numpy as np
from asymmetree.utils.phylogenetic_trees import sorted_edges
from tralda.datastructures.doubly_linked import DLList
from tralda.datastructures.doubly_linked import DLListNode

from pangesim import Genome
from pangesim import Pangenome
from pangesim.datastructures.pangenome import unwrap_node_value


class MutationModel:
    """Abstract base class for genome evolution events.

    Subclass this to implement more advanced or biologically realistic events
    (e.g., specific inversion parameters or multi-gene duplications).
    """

    def apply_events(self, genome: Genome, num_events: int, global_pool: Any) -> None:
        """Applies a specific number of structural events to the target genome.

        Args:
            genome: The Genome instance to modify in-place.
            num_events: Total number of mutation actions to execute.
            global_pool: A stateful counter or tracker to draw unique new gene IDs.
        """
        raise NotImplementedError


class SimpleModel(MutationModel):
    """A baseline mutation engine handling basic structural operations."""

    def _apply_insertions(self, genome: Genome, num_events: int, global_pool: Any) -> None:
        """Safely inserts a brand new gene into a random path in the forest."""
        for _ in range(num_events):
            new_gene_id = global_pool.get_next_id()
            new_node = DLListNode(value=new_gene_id)

            if not genome.heads:
                # If empty, create an entire new single-edge path
                new_gene_id_neighbor = global_pool.get_next_id()
                new_node_neighbor = DLListNode(value=new_gene_id_neighbor)
                new_path = DLList()
                new_path.append(new_node)
                new_path.append(new_node_neighbor)
                genome.add_path(new_path)
                return

            # Otherwise, find a random path head and insert it right after it
            random_head = random.choice(genome.heads)
            old_next = random_head._next

            random_head._next = new_node
            new_node._prev = random_head
            new_node._next = old_next
            if old_next is not None:
                old_next._prev = new_node

            genome.gene_set.add(new_gene_id)

    def _apply_deletions(self, genome: Genome, num_events: int) -> None:
        """Simulates chromosome breakage by deleting adjacency edges.

        Splits existing paths into smaller fragments, creating new path heads.

        Args:
            genome: The Genome instance to mutate in-place.
            num_events: The number of edge-breakage events to execute.
        """
        for _ in range(num_events):
            # 1. Gather all valid internal edges that can be broken
            # An edge is valid if a node has a valid downstream neighbor (_next)
            valid_edges = []
            for u, v in genome.iter_edges():
                if u._prev is not None and v._next is not None:
                    valid_edges.append((u, v))

            # If all paths are already 1 edge paths, stop
            if len(valid_edges) < 1:
                break

            # 2. Randomly select an upstream node whose outgoing edge will be broken
            upstream_node, downstream_node = random.choice(valid_edges)

            # 3. Cut the bonds to split the path
            if downstream_node is not None:
                # The downstream segment loses its backward link
                downstream_node._prev = None

                # The downstream segment head is registered as a new independent path
                genome.heads.append(downstream_node)

            # The upstream node loses its forward link, terminating that fragment
            upstream_node._next = None

    def _apply_rearrangements(self, genome: Genome, num_events: int) -> None:
        """Rearranges paths by swapping the heads of two distinct path segments.

        This changes the structural adjacencies without losing or isolating genes.
        """
        if len(genome.heads) < 2:
            return

        for _ in range(num_events):
            # Choose two path entries at random
            idx_a, idx_b = random.sample(range(len(genome.heads)), 2)
            head_a = genome.heads[idx_a]
            head_b = genome.heads[idx_b]

            # Swap their trailing lineages
            head_a._next, head_b._next = head_b._next, head_a._next

            if head_a._next is not None:
                head_a._next._prev = head_a
            if head_b._next is not None:
                head_b._next._prev = head_b

    def _apply_inversions(self, genome: Genome, num_events: int) -> None:
        """Invert the segment of the path from node i to the node j (any btw 2 ends).

        The length of the path should >= 2.

        This changes the structural adjacencies without losing or isolating genes.
        """
        for _ in range(num_events):
            candidate_heads = []

            for head in genome.heads:
                path = genome.get_a_path(head)

                if path._count >= 2:
                    candidate_heads.append(head)

            if not candidate_heads:
                break

            random_head = random.choice(candidate_heads)
            path = genome.get_a_path(random_head)

            ind1 = random.randint(0, path._count - 2)
            ind2 = random.randint(ind1 + 1, path._count - 1)

            start = path.node_at(ind1)
            end = path.node_at(ind2)

            before = start._prev
            after = end._next

            current = start

            while True:
                old_next = current._next

                current._next, current._prev = current._prev, current._next

                if current is end:
                    break

                current = old_next

            if before is not None:
                before._next = end
                end._prev = before
            else:  # start is the head, also update heads
                head_index = genome.heads.index(random_head)
                genome.heads[head_index] = end
                end._prev = None

            start._next = after

            if after is not None:
                after._prev = start

    def _apply_fissions(self, genome: Genome, num_events: int) -> None:
        """Break the path at a randomly selected node, add them to the genome.

        The length of the path should >= 4, so the paths after fission both have >= 2 nodes.

        This changes the structural adjacencies without losing or isolating genes.
        """
        for _ in range(num_events):
            candidate_heads = []

            for head in genome.heads:
                path = genome.get_a_path(head)

                if path._count >= 4:
                    candidate_heads.append(head)

            if not candidate_heads:
                break

            random_head = random.choice(candidate_heads)
            path = genome.get_a_path(random_head)

            split_index = random.randint(2, path._count - 2)

            downstream_node = path.node_at(split_index)
            upstream_node = downstream_node._prev
            downstream_node._prev = None
            upstream_node._next = None

            # Get the path of made of the tail portion
            path_new = genome.get_a_path(downstream_node)

            # Add path_new to the genome (including putting genes in the set)
            genome.add_path(path_new)

    def _apply_fusions(self, genome: Genome, num_events: int) -> None:
        """Fuse randomly selected path i and j, i as the head portion.

        Args:
            genome: The Genome instance to mutate in-place.
            num_events: The number of edge-breakage events to execute.
        """
        for _ in range(num_events):
            if len(genome.heads) < 2:
                break

            # Randomly choose two different paths
            random_head1, random_head2 = random.sample(genome.heads, 2)

            # Get path1: the head portion of the fused path
            path1 = genome.get_a_path(random_head1)

            # Fusion
            path1._last._next = random_head2
            random_head2._prev = path1._last

            # path2 is fused to path1, remove it from heads
            genome.heads.remove(random_head2)

    def _detach_segment(self, d_start: DLListNode, d_end: DLListNode) -> None:
        """Detach a contiguous segment from its current path.

        The remaining parts of the path are reconnected after detaching the
        segment from d_start through d_end.

        Args:
            d_start (DLListNode): First node of the segment to detach.
            d_end (DLListNode): Last node of the segment to detach.

        Assumption:
            d_start and d_end belong to the same path, and d_start occurs
            before or is equal to d_end.
        """
        before = d_start._prev
        after = d_end._next

        d_start._prev = None
        d_end._next = None

        if before is not None:
            before._next = after

        if after is not None:
            after._prev = before

    def _insert_segment(
        self, i_start: DLListNode, i_end: DLListNode | None, d_start: DLListNode, d_end: DLListNode
    ) -> None:
        """Insert a detached segment into a path.

        The segment from d_start through d_end is inserted between i_start
        and i_end.

        Args:
            i_start (DLListNode): Node after which the detached segment is inserted.
            i_end (DLListNode | None): Node following the inserted segment, or None.
            d_start (DLListNode): First node of the detached segment.
            d_end (DLListNode): Last node of the detached segment.

        Assumption:
            d_start._prev is None.
            d_end._next is None.
            i_start and i_end belong to the receiving path.
        """
        d_start._prev = i_start
        d_end._next = i_end

        i_start._next = d_start
        if i_end is not None:
            i_end._prev = d_end

    def _apply_translocations(self, genome: Genome, num_events: int) -> None:
        """Randomly choose two paths and break points, then exchange their tail portions.

        Both reciprocal and non-reciprocal translocation are allowed.

        This changes the structural adjacencies without losing or isolating genes.

        Args:
            genome: The Genome instance to mutate in-place.
            num_events: The number of edge-breakage events to execute.
        """
        if len(genome.heads) > 1:  # The genome should have at least 2 paths
            for _ in range(num_events):
                # Randomly choose two different paths
                random_head1, random_head2 = random.sample(genome.heads, 2)

                # Get path1 and path2
                path1 = genome.get_a_path(random_head1)
                path2 = genome.get_a_path(random_head2)

                # In case we have a path with only one node
                if path1._count <= 1 or path2._count <= 1:
                    continue

                # Determine which translocation types are possible
                # Reciprocal is always possible
                possible_events = ["reciprocal"]

                if path2._count > 2:  # path2 gives and path1 takes
                    possible_events.append("path1_takes")

                if path1._count > 2:  # path1 gives and path2 takes
                    possible_events.append("path2_takes")

                # Randomly choose one valid event
                event = random.choice(possible_events)

                if event == "reciprocal":
                    # reciprocal translocation, exchange segments
                    # Get insertion points and detached segments in path 1 and 2
                    ind1 = random.randint(1, path1._count - 1)
                    ind2 = random.randint(1, path2._count - 1)

                    len1 = random.randint(1, path1._count - ind1)
                    len2 = random.randint(1, path2._count - ind2)

                    d_start1 = path1.node_at(ind1)
                    d_start2 = path2.node_at(ind2)
                    i_start1 = d_start1._prev
                    i_start2 = d_start2._prev

                    d_end1 = path1.node_at(ind1 + len1 - 1)
                    d_end2 = path2.node_at(ind2 + len2 - 1)
                    i_end1 = d_end1._next
                    i_end2 = d_end2._next

                    # Translocation
                    self._detach_segment(d_start1, d_end1)
                    self._detach_segment(d_start2, d_end2)

                    self._insert_segment(i_start1, i_end1, d_start2, d_end2)
                    self._insert_segment(i_start2, i_end2, d_start1, d_end1)

                elif event == "path1_takes":
                    # Choose insertion point (can be head) in path1
                    ind1 = random.randint(0, path1._count - 1)

                    i_start1 = path1.node_at(ind1)
                    i_end1 = i_start1._next

                    # Choose detached segment (after head) from path2
                    ind2 = random.randint(1, path2._count - 1)
                    if ind2 == 1:  # Make sure after translocation, it remains at least 2 nodes
                        len2 = random.randint(1, path2._count - ind2 - 1)
                    else:
                        len2 = random.randint(1, path2._count - ind2)

                    d_start2 = path2.node_at(ind2)
                    d_end2 = path2.node_at(ind2 + len2 - 1)

                    # Detach from path2
                    self._detach_segment(d_start2, d_end2)

                    # Insert into path1
                    self._insert_segment(i_start1, i_end1, d_start2, d_end2)

                else:  # path2_takes
                    # Choose insertion point (can be head) in path2
                    ind2 = random.randint(0, path2._count - 1)

                    i_start2 = path2.node_at(ind2)
                    i_end2 = i_start2._next

                    # Choose detached segment (after head) from path1
                    ind1 = random.randint(1, path1._count - 1)

                    if ind1 == 1:  # Make sure after translocation, it remains at least 2 nodes
                        len1 = random.randint(1, path1._count - ind1 - 1)
                    else:
                        len1 = random.randint(1, path1._count - ind1)

                    d_start1 = path1.node_at(ind1)
                    d_end1 = path1.node_at(ind1 + len1 - 1)

                    # Detach from path1
                    self._detach_segment(d_start1, d_end1)

                    # Insert into path2
                    self._insert_segment(i_start2, i_end2, d_start1, d_end1)


class IDPool:
    """Manages unique identifiers across evolutionary lineages."""

    def __init__(self, starting_id: int = 0) -> None:
        """Constructor of an id pool."""
        self.current_id = starting_id

    def get_next_id(self) -> int:
        """Function to generate a new unique id."""
        uid = self.current_id
        self.current_id += 1
        return uid


class PangenomeSimulator:
    """Class for simulating a pangenome along a given dated species tree."""

    def __init__(
        self,
        insertion_rate: float = 0.0,
        deletion_rate: float = 0.0,
        rearrangement_rate: float = 0.0,
        inversion_rate: float = 0.0,
        fission_rate: float = 0.0,
        fusion_rate: float = 0.0,
        translocation_rate: float = 0.0,
        mutation_model: MutationModel | None = None,
    ) -> None:
        """Initializes the pangenome simulator with the given rates and models."""
        self.insertion_rate = insertion_rate
        self.deletion_rate = deletion_rate
        self.rearrangement_rate = rearrangement_rate
        self.inversion_rate = inversion_rate
        self.fission_rate = fission_rate
        self.fusion_rate = fusion_rate
        self.translocation_rate = translocation_rate

        self.model = mutation_model if mutation_model is not None else SimpleModel()
        self.gene_id_pool = IDPool()
        self.genome_id_pool = IDPool()

    def _draw_poisson_events(self, rate: float, branch_length: float) -> int:
        """Draws a random sample of mutation events."""
        if rate <= 0.0:
            return 0

        rate_parameter = rate * max(branch_length, 0.001)

        # Returns an exact, C-accelerated Poisson integer immune to underflow/overflow
        return int(np.random.poisson(lam=rate_parameter))

    def generate_pangenome(self, k: int = 2, length: int = 10) -> Pangenome:
        """Generates a dated species tree and evolves a pangenome down its lineages.

        Args:
            k: The desired number of genomes (leaves in the generated tree).
            length: The starting number of genes in the ancestral root genome.

        Returns:
            A finalized Pangenome collection containing k evolved genomes.
        """
        # Parameter checkups
        if not isinstance(k, int) or k < 1:
            raise ValueError("Number of genomes k must be an int >=1")
        elif k == 0:
            return Pangenome(None)

        if not isinstance(length, int) or length < 1:
            raise ValueError("Number of genes length must be an int >=1")
        elif length == 0:
            return Pangenome(None)

        # 1. Generation of species tree using AsymmeTree
        # We simulate a dated species tree with exactly k leaves
        species_tree = te.species_tree_n(k, model="yule", innovations=True, planted=False)

        # 2. We instantiate the ancestral genome at the root with the given length
        root_path = DLList()
        for _ in range(length):
            root_path.append(DLListNode(value=self.gene_id_pool.get_next_id()))

        root_genome = Genome(genome_id=self.genome_id_pool.get_next_id())
        root_genome.add_path(root_path)

        # Registry map tracking genomes alive at each active tree node object/ID
        node_genomes: Dict[Any, Genome] = {species_tree.root: root_genome}
        leaf_genomes: List[Genome] = []

        # 3. Top-down traversal of the species tree's edges
        for u, v in sorted_edges(species_tree):
            # Fetch the parent genome snapshot
            parent_genome = node_genomes[u]

            # Calculate the branch length using timestamps
            branch_len = abs(u.tstamp - v.tstamp)

            # Reconstruct an independent structural duplicate for the child lineage
            child_genome = Genome(genome_id=self.genome_id_pool.get_next_id())
            for head in parent_genome.heads:
                new_path = DLList()
                curr = head
                while curr is not None:
                    val = unwrap_node_value(curr)
                    new_path.append(DLListNode(value=val))
                    curr = curr._next
                child_genome.add_path(new_path)

            # Draw mutation events along this specific edge timeline
            num_ins = self._draw_poisson_events(self.insertion_rate, branch_len)
            num_del = self._draw_poisson_events(self.deletion_rate, branch_len)
            num_rea = self._draw_poisson_events(self.rearrangement_rate, branch_len)
            num_inv = self._draw_poisson_events(self.inversion_rate, branch_len)
            num_fis = self._draw_poisson_events(self.fission_rate, branch_len)
            num_fus = self._draw_poisson_events(self.fusion_rate, branch_len)
            num_trsl = self._draw_poisson_events(self.translocation_rate, branch_len)

            # Apply mutations
            if num_ins > 0:
                self.model._apply_insertions(child_genome, num_ins, self.gene_id_pool)
            if num_del > 0:
                self.model._apply_deletions(child_genome, num_del)
            if num_rea > 0:
                self.model._apply_rearrangements(child_genome, num_rea)
            if num_inv > 0:
                self.model._apply_inversions(child_genome, num_inv)
            if num_fis > 0:
                self.model._apply_fissions(child_genome, num_fis)
            if num_fus > 0:
                self.model._apply_fusions(child_genome, num_fus)
            if num_trsl > 0:
                self.model._apply_translocations(child_genome, num_trsl)

            # Cache the child state for downstream branches or final leaf processing
            node_genomes[v] = child_genome

            # If the child node is a terminal leaf, keep it
            if v.is_leaf():
                leaf_genomes.append(child_genome)

        return Pangenome(pangenome_id=f"simulated_k{k}_l{length}", genomes=leaf_genomes)
