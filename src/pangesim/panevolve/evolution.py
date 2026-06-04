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
        mutation_model: MutationModel | None = None,
    ) -> None:
        """Initializes the pangenome simulator with the given rates and models."""
        self.insertion_rate = insertion_rate
        self.deletion_rate = deletion_rate
        self.rearrangement_rate = rearrangement_rate

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

            # Apply mutations
            if num_ins > 0:
                self.model._apply_insertions(child_genome, num_ins, self.gene_id_pool)
            if num_del > 0:
                self.model._apply_deletions(child_genome, num_del)
            if num_rea > 0:
                self.model._apply_rearrangements(child_genome, num_rea)

            # Cache the child state for downstream branches or final leaf processing
            node_genomes[v] = child_genome

            # If the child node is a terminal leaf, keep it
            if v.is_leaf():
                leaf_genomes.append(child_genome)

        return Pangenome(pangenome_id=f"simulated_k{k}_l{length}", genomes=leaf_genomes)
