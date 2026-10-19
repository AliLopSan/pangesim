"""Tests verifying pangenome generation and parameter scaling."""

import random as rd

import numpy as np
from tralda.datastructures.doubly_linked import DLList

from pangesim.panevolve import LinearSimulator
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
    new_genome = rd.choice(pangenome._genomes)

    # Randomly choose a head from the genome
    new_head = rd.choice(new_genome.heads)

    # Get the path (chromosome) associated with the head
    new_path = new_genome.get_a_path(new_head)

    # The assertion check
    assert isinstance(new_path, DLList)

    # Assertion (the path should be longer than 1)
    assert new_path._count > 1


def test_mutations():
    """Test the added mutation functions: inv, fiss, fuss, and trsl.

    The code is similar to test_combination_events.
    """
    del_rate = 0
    rea_rate = 0

    inv_rates = [0, 1, 5, 100]
    fis_rates = [0, 1, 5, 100]
    fus_rates = [0, 1, 5, 100]
    trsl_rates = [0, 1, 5, 100]

    for d1 in inv_rates:
        for d2 in fis_rates:
            for d3 in fus_rates:
                for d4 in trsl_rates:
                    sim = PangenomeSimulator(
                        deletion_rate=del_rate,
                        rearrangement_rate=rea_rate,
                        inversion_rate=d1,
                        fission_rate=d2,
                        fusion_rate=d3,
                        translocation_rate=d4,
                    )

                    genomes = rd.randint(2, 15)
                    genes = rd.randint(2, 200)

                    # Request a pangenome with k genome starting with l genes
                    pangenome = sim.generate_pangenome(k=genomes, length=genes)

                    # Check for non-trivial path forest condition
                    for genome in pangenome.genomes:
                        assert genome.check_integrity() is True

    # Check integrity of the pangenome
    assert pangenome.check_integrity() is True

    # Test of edge cases
    # Inversion case 1: 6 genomes, each has four genes
    sim = PangenomeSimulator(inversion_rate=100.0)
    genomes = 6
    genes = 2
    pangenome = sim.generate_pangenome(k=genomes, length=genes)
    for genome in pangenome.genomes:
        assert genome.check_integrity() is True

    # Fission case 1: 6 genomes, each has four genes
    sim = PangenomeSimulator(fission_rate=100.0)
    genomes = 6
    genes = 4
    pangenome = sim.generate_pangenome(k=genomes, length=genes)
    for genome in pangenome.genomes:
        assert genome.check_integrity() is True

    # Fusion case 1: 20 genomes, each has four genes
    sim = PangenomeSimulator(fusion_rate=100.0)
    genomes = 10
    genes = 2
    pangenome = sim.generate_pangenome(k=genomes, length=genes)
    for genome in pangenome.genomes:
        assert genome.check_integrity() is True

    # Translocation case 1: 2 genomes, each has two genes
    sim = PangenomeSimulator(translocation_rate=100.0)
    genomes = 2
    genes = 2
    pangenome = sim.generate_pangenome(k=genomes, length=genes)
    for genome in pangenome.genomes:
        assert genome.check_integrity() is True

    # Translocation case 2: 2 genomes, each has two or three genes
    sim = PangenomeSimulator(translocation_rate=100.0)
    genomes = 2
    genes = rd.randint(2, 3)
    pangenome = sim.generate_pangenome(k=genomes, length=genes)
    for genome in pangenome.genomes:
        assert genome.check_integrity() is True

    # Check integrity of the pangenome
    assert pangenome.check_integrity() is True


def test_linear_simulator():
    """Checks integrity of the resulting genomes."""
    rates = (0.0,10.0)

    genomes = rd.randint(2, 15)
    genes = rd.randint(2, 200)

    r1 = np.random.uniform(rates[0], rates[1])
    r2 = np.random.uniform(rates[0], rates[1])
    r3 = np.random.uniform(rates[0], rates[1])
    r4 = np.random.uniform(rates[0], rates[1])
    r5 = np.random.uniform(rates[0], rates[1])
    r6 = np.random.uniform(rates[0], rates[1])
    r7 = np.random.uniform(rates[0], rates[1])

    #Tests for the different complexity levels
    sim1 = LinearSimulator(insertion_rate=r1,
        deletion_rate=r2,
        rearrangement_rate=r3,
        inversion_rate=r4,
        fission_rate=r5,
        fusion_rate=r6,
        translocation_rate=r7)
    pan1 = sim1.generate_pangenome(k=genomes,
        length=genes, complexity="low")

    print("\t Simulated pangenome: \n", pan1.summary())

    assert pan1.check_integrity() is True

    #Tests for the different complexity levels
    sim2 = LinearSimulator(insertion_rate=r1,
                          deletion_rate=r2,
                          rearrangement_rate=r3,
                          inversion_rate=r4,
                          fission_rate=r5,
                          fusion_rate=r6,
                          translocation_rate=r7)
    pan2 = sim2.generate_pangenome(k=genomes,
                                       length=genes)

    print("\t Simulated pangenome: \n", pan2.summary())

    assert pan2.check_integrity() is True

    # Tests for the different complexity levels
    sim3 = LinearSimulator(
        insertion_rate=r1,
        deletion_rate=r2,
        rearrangement_rate=r3,
        inversion_rate=r4,
        fission_rate=r5,
        fusion_rate=r6,
        translocation_rate=r7,
    )
    pan3 = sim3.generate_pangenome(k=genomes, length=genes, complexity="high")

    print("\t Simulated pangenome: \n", pan3.summary())

    assert pan3.check_integrity() is True

