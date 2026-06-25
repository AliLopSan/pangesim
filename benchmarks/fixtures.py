"""Pre-configured mock pangenome for benchmarking."""

from pangesim import Genome
from pangesim import Pangenome


def get_mock_pangenome_scenario() -> Pangenome:
    """Returns a stable 5-genome mock pangenome."""
    # Ground-truth genomes
    gt_g1 = Genome(genome_id=0)
    g1_edges = [(1, 2), (2, 6), (6, 7), (7, 9), (9, 11), (3, 10), (3, 4), (4, 5)]
    for edge in g1_edges:
        gt_g1.add_edge(edge)

    gt_g2 = Genome(genome_id=1)
    g2_edges = [(2, 3), (3, 10), (5, 4), (4, 8)]
    for edge in g2_edges:
        gt_g2.add_edge(edge)

    gt_g3 = Genome(genome_id=2)
    g3_edges = [(2, 3), (3, 10)]
    for edge in g3_edges:
        gt_g3.add_edge(edge)

    gt_g4 = Genome(genome_id=3)
    g4_edges = [(1, 2), (2, 3), (3, 4), (4, 8), (10, 9), (9, 11)]
    for edge in g4_edges:
        gt_g4.add_edge(edge)

    gt_g5 = Genome(genome_id=4)
    g5_edges = [(1, 2), (2, 3), (3, 10), (10, 9), (5, 4), (4, 8)]
    for edge in g5_edges:
        gt_g5.add_edge(edge)

    ground_truth = Pangenome(pangenome_id="ground_truth")
    ground_truth.add_genome(gt_g1)
    ground_truth.add_genome(gt_g2)
    ground_truth.add_genome(gt_g3)
    ground_truth.add_genome(gt_g4)
    ground_truth.add_genome(gt_g5)

    return ground_truth
