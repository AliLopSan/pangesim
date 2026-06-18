"""Tests verifying pangenome generation and parameter scaling."""

import random as rd

from pangesim.panevolve import PangenomeSimulator


def test_creation_no_evolution_rates():
    """Validates that specifying k and l outputs correct structural targets."""
    # Initialize with no mutations so we can precisely track sizes
    sim = PangenomeSimulator(insertion_rate=0.0, deletion_rate=0.0)

    # Request a pangenome with 3 genomes (k=3) starting with 12 genes (l=12)
    pangenome = sim.generate_pangenome(k=3, length=12)

    # Assertions
    assert len(pangenome) == 3  # Confirms tree was generated with 3 leaves
    assert pangenome.total_gene_count == 12  # Confirms ancestral length was 12
    assert len(pangenome.compute_core_genes()) == 12


def test_extreme_deletion():
    """Checks integrity of the resulting genomes."""
    # Generate a simulation with a high deletion rate
    sim = PangenomeSimulator(deletion_rate=100.0)

    # Request a pangenome with 1 genome starting with 12 genes
    pangenome = sim.generate_pangenome(k=2, length=4)

    # Assertions
    for genome in pangenome.genomes:
        # Total node count MUST still be exactly 10! Edges were deleted, not genes.
        assert len(genome) == 4

        # Because deletion rate was so high, every edge should be broken.
        assert genome.path_count > 1

        # Check for non-trivial path forest condition
        assert genome.check_integrity() is True


def test_combination_events():
    """Checks integrity of the resulting genomes."""
    del_rates = [0, 1, 5, 100]
    rea_rates = [0, 1, 5, 100]

    for d in del_rates:
        for r in rea_rates:
            sim = PangenomeSimulator(deletion_rate=d, rearrangement_rate=r)

            genomes = rd.randint(2, 15)
            genes = rd.randint(2, 200)

            # Request a pangenome with 3 genome starting with 12 genes
            pangenome = sim.generate_pangenome(k=genomes, length=genes)
            # Check for non-trivial path forest condition
            for genome in pangenome.genomes:
                assert genome.check_integrity() is True

def test_get_a_path():
    """Check if the path can be created from a given head in the pangenome."""
    
    # Initialize with no mutations so we can precisely track sizes
    sim = PangenomeSimulator(insertion_rate=0.0, deletion_rate=0.0)

    # Request a pangenome with 3 genomes (k=3) starting with 12 genes (l=12)
    pangenome = sim.generate_pangenome(k=3, length=12)
    
    # Randomly choose a genome from the pangenome
    new_genome = random.choice(pangenome._genomes)

    # Randomly choose a head from the genome
    new_head = random.choice(new_genome.heads)

    # Get the path (chromosome) associated with the head
    new_path = new_genome.get_a_path(new_head)
