"""Operators for local pangenome optimization."""

import random as rd
from itertools import combinations
from itertools import pairwise
from typing import List
from typing import Tuple

from pangesim import Genome
from pangesim import Pangenome
from pangesim.reconstruction import RefinementOperator


class MergeOperator(RefinementOperator):
    """Merges two genomes."""

    def __init__(self, random: bool = True, genomes: List[Genome] | None = None):
        """Constructor of the merging operator.

        Args:
           random: Choose two genomes randomly.
           genomes: If not random, merges the genomes in this list.
        """
        self.random = random
        if genomes is None:
            self.genomes = []
        else:
            self.genomes = genomes

    def merge(self, genomes: Tuple[Genome, Genome]) -> Genome | None:
        """Main function to merge a tuple of genomes.

        Args:
           genomes: A tuple of Genomes.

        Returns:
           The merged genome, if possible if not returns None.
        """
        genome_a, genome_b = genomes

        # If their gene set symmetric difference is empty, return None
        if not (genome_a.gene_set ^ genome_b.gene_set):
            return None

        # Safely marge the genomes:
        new_genome = genome_a.copy()
        b_adjacencies = genome_b.get_adjacency_tuples()

        for u, v in b_adjacencies:
            success = new_genome.add_edge((u, v))
            if not success:
                return None

        return new_genome

    def improve(self, pangenome: Pangenome) -> bool:
        """Merges all the given genomes.

        Args:
           pangenome: Input pangenome

        Returns:
           True if merging is possible, merges genomes on the fly.
        """
        if len(pangenome) < 2:
            return False

        if self.random:
            all_pairs = list(combinations(pangenome._genomes, 2))
        else:
            all_pairs = list(combinations(self.genomes, 2))

        rd.shuffle(all_pairs)

        for genome_a, genome_b in all_pairs:
            pair = genome_a, genome_b
            new_genome = self.merge(pair)

            if new_genome:
                id_a = genome_a._genome_id
                id_b = genome_b._genome_id

                pangenome.remove_genome(id_a)
                pangenome.remove_genome(id_b)

                pangenome.add_genome(new_genome)
                return True

        return False


class SplitOperator(RefinementOperator):
    """Splits a genome."""

    def __init__(self, random: bool = True, genome: Genome | None = None):
        """Constructor of the splitting operator.

        Args:
           random: Choose one genome randomly.
           genome: If not random, provide a specific genome.
        """
        self.random = random
        if genome is None:
            self.genome = None
        else:
            self.genome = genome

    def split(self, core: set, k: int) -> Tuple[Genome, Genome] | None:
        """Splits a given genome according to core.

        Args:
           core: The core set of the pangenome
           k: the number of genomes in the given pangenome.

        Returns:
           Two genomes if split is possible, None otherwise.
        """
        if self.genome.path_count <= 1:
            return None
        else:
            genome_a = Genome(genome_id=self.genome._genome_id)
            genome_b = Genome(genome_id=k)
            paths = self.genome.get_path_sequences()

            for i, path in enumerate(paths):
                path_genes = set(path)
                if path_genes & core:
                    for edge in pairwise(path):
                        genome_a.add_edge(edge)
                        genome_b.add_edge(edge)
                else:
                    if i % 2 == 0:
                        for edge in pairwise(path):
                            genome_a.add_edge(edge)
                    else:
                        for edge in pairwise(path):
                            genome_b.add_edge(edge)
            return (genome_a, genome_b)

    def improve(self, pangenome: Pangenome) -> bool:
        """Splits a random or given genome.

        Args:
           pangenome: Input pangenome

        Returns:
           True if merging is possible, merges genomes on the fly.
        """
        if len(pangenome) == 0:
            return False

        if self.random:
            self.genome = rd.choice(pangenome._genomes)

        pair = self.split(pangenome.core, len(pangenome))

        if pair:
            pangenome.replace_genome(self.genome._genome_id, pair[0])
            pangenome.add_genome(pair[1])
            return True

        return False
