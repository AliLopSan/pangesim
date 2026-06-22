"""Tests for refinement operators."""

from pangesim import Genome
from pangesim import Pangenome
from pangesim.reconstruction.operators import MergeOperator


def test_merge_genomes():
    """Tests the merging of genomes."""
    # Genome 1: 1 - 2 - 3 - 4
    genome_1 = Genome(genome_id=1)
    genome_1.add_edge((1, 2))
    genome_1.add_edge((3, 4))

    # Genome 2: 4 - 5
    genome_2 = Genome(genome_id=2)
    genome_2.add_edge((4, 5))

    # Genome 3: 2 - 4
    genome_3 = Genome(genome_id=3)
    genome_3.add_edge((2, 4))

    pangenome = Pangenome(pangenome_id="test_pan")
    pangenome.add_genome(genome_1)
    pangenome.add_genome(genome_2)
    pangenome.add_genome(genome_3)

    print("\n Pangenome before merge:")
    print(pangenome.summary())

    # Testing defaults:
    m_1 = MergeOperator()
    success = m_1.improve(pangenome)
    assert success

    print("\t Pangenome after merge: ")
    print(pangenome.summary())

    # Testing Cycle creation
    impossible = [genome_1, genome_3]
    m_2 = MergeOperator(random=False, genomes=impossible)
    success = m_2.improve(pangenome)

    assert not success
