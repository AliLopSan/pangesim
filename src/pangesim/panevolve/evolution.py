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
        """Detach the contiguous segment [d_start, ..., d_end] from its current path.

        Then reconnect the remaining path.

        Assumption:
            start and end belong to the same path,
            and start occurs before or is equal to end.
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
        """Insert the detached segment [d_start, ..., d_end].

        Insert it between i_start and i_end.

        Assumption:
            d_start._prev is None
            d_end._next is None
            i_start and i_end belongs to the receiving path.
        """
        d_start._prev = i_start
        d_end._next = i_end

        i_start._next = d_start
        if i_end is not None:
            i_end._prev = d_end

    def _apply_translocations(self, genome: Genome, num_events: int) -> None:
        """Exchange possibly empty segments between two distinct paths.

        Both segments nonempty gives a reciprocal translocation; exactly one
        empty segment gives a non-reciprocal translocation. Both paths retain
        at least two nodes, and no genes are lost or duplicated.

        Args:
            genome: The Genome instance to modify in-place.
            num_events: Number of translocations to attempt.
        """
        if len(genome.heads) < 2:
            return

        for _ in range(num_events):
            # Randomly get two paths and their lengths
            head1, head2 = random.sample(genome.heads, 2)
            path1 = genome.get_a_path(head1)
            path2 = genome.get_a_path(head2)
            n1, n2 = path1._count, path2._count

            if n1 < 2 or n2 < 2:
                continue

            # Draw lengths satisfying both final-path size constraints.
            # l1 = 0 is feasible only if path2 can give up a node.
            # l1: the segment length from 1 to 2, l2: the segment length from 2 to 1
            # constraint1: n1 - l1 + l2 >= 2
            # constraint2: n2 - l2 + l1 >= 2
            # constraint3: l1 + l2 >= 1
            l1 = random.randint(0 if n2 >= 3 else 1, n1 - 1)
            min_l2 = max(0, 2 - n1 + l1)
            max_l2 = min(n2 - 1, n2 + l1 - 2)
            if l1 == 0:
                min_l2 = max(min_l2, 1)  # Do not exchange two empty segments.
            l2 = random.randint(min_l2, max_l2)

            # ind is the first segment index; for len = 0 it is a gap index.
            # Restrict to indices >= 1 so registered heads do not change.
            ind1 = random.randint(1, n1 - l1)
            ind2 = random.randint(1, n2 - l2)

            # nodes before and after the segments for exchange
            before1 = path1.node_at(ind1 - 1)
            after1 = path1.node_at(ind1 + l1) if ind1 + l1 < n1 else None
            before2 = path2.node_at(ind2 - 1)
            after2 = path2.node_at(ind2 + l2) if ind2 + l2 < n2 else None

            # start and end of the segments to exchange
            if l1 > 0:
                start1 = path1.node_at(ind1)
                end1 = path1.node_at(ind1 + l1 - 1)
            if l2 > 0:
                start2 = path2.node_at(ind2)
                end2 = path2.node_at(ind2 + l2 - 1)

            # Detach both nonempty segments before inserting either one.
            if l1 > 0:
                self._detach_segment(start1, end1)
            if l2 > 0:
                self._detach_segment(start2, end2)

            # The path receiving an empty segment is already reconnected
            # by _detach_segment; only nonempty incoming segments need insertion.
            if l2 > 0:
                self._insert_segment(before1, after1, start2, end2)
            if l1 > 0:
                self._insert_segment(before2, after2, start1, end1)


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

            pan = Pangenome(pangenome_id=f"simulated_k{k}_l{length}", genomes=leaf_genomes)
            pan.core = pan.compute_core_genes()
            pan.core_edges = pan.compute_core_edges()

        return pan

class LinearSimulator:
    """Class for simulating a pangenome using isolated events and no evolutionary tree."""

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

    def generate_pangenome(
        self,
        k: int = 2,
        length: int = 10,
        complexity: str = "medium")-> Pangenome:
        """Tree-less pangenome generation function.

        Generates an ancestral root genome and evolves k-1 descendant genomes
        from it using Poisson mutation events whose rates are dynamically scaled
        based on a stochastic frequency tier.

        Args:
           k: The desired total number of genomes.
           length: The starting number of genes in the ancestral root genome.
           complexity: Mutation frequency string ("low", "medium", "high").

        Returns:
           A finalized Pangenome collection containing k evolved genomes.
        """
        if not isinstance(k, int) or k < 1:
            raise ValueError("Number of genomes k must be an int >=1")
        if not isinstance(length, int) or length < 1:
            raise ValueError("Number of genes length must be an int >=1")

        # Map frequency tiers to rate multiplier ranges
        frequency_ranges = {
            "low": (0.0, 0.5),
            "medium": (0.5, 2.0),
            "high": (3.0, 5.0),
        }
        if complexity not in frequency_ranges:
            raise ValueError("Frequency must be one of 'low', 'medium', or 'high'.")

        min_scale, max_scale = frequency_ranges[complexity]

        # 1. Instantiate the ancestral root genome
        root_path = DLList()
        for _ in range(length):
            root_path.append(DLListNode(value=self.gene_id_pool.get_next_id()))

        root_genome = Genome(genome_id=self.genome_id_pool.get_next_id())
        root_genome.add_path(root_path)

        genomes: List[Genome] = [root_genome]

        for _ in range(k-1):
            scale_factor = np.random.uniform(min_scale, max_scale)
            branch_len = 1.0 * scale_factor
            child_genome = Genome(genome_id=self.genome_id_pool.get_next_id())
            for head in root_genome.heads:
                new_path = DLList()
                curr = head
                while curr is not None:
                    val = unwrap_node_value(curr)
                    new_path.append(DLListNode(value=val))
                    curr = curr._next
                child_genome.add_path(new_path)

            # Draw Poisson mutation events using base rates scaled by the stochastic factor
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

            genomes.append(child_genome)

        pan = Pangenome(pangenome_id=f"simulated_k{k}_l{length}_{complexity}", genomes=genomes)
        pan.core = pan.compute_core_genes()
        pan.core_edges = pan.compute_core_edges()

        return pan

