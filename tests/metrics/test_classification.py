"""Set of tests for the classification metrics."""

import pytest

from pangesim import Genome
from pangesim import Pangenome
from pangesim.metrics.classification import CoreContingencyTable
from pangesim.metrics.classification import EdgeContingencyTable


@pytest.fixture
def sample_pangenomes():
    """Fixture that provisions clean ground truth and inferred Pangenome mocks."""
    # Ground-truth genomes
    gt_g1 = Genome(genome_id="gt_g1")
    gt_g1.add_edge((1, 2))
    gt_g1.add_edge((2, 3))
    gt_g2 = Genome(genome_id="gt_g2")
    gt_g2.add_edge((1, 2))
    gt_g2.add_edge((4, 5))

    ground_truth = Pangenome(pangenome_id="truth_set")
    ground_truth.add_genome(gt_g1)
    ground_truth.add_genome(gt_g2)

    # Inferred genomes
    inf_g1 = Genome(genome_id="in_g1")
    inf_g1.add_edge((1, 2))
    inf_g1.add_edge((2, 3))
    inf_g2 = Genome(genome_id="in_g1")
    inf_g2.add_edge((1, 2))
    inf_g2.add_edge((3, 4))
    inf_g2.add_edge((4, 5))

    inferred = Pangenome(pangenome_id="inferred_set")
    inferred.add_genome(inf_g1)
    inferred.add_genome(inf_g2)

    return ground_truth, inferred


# ==============================================================================
# Unit Tests
# ==============================================================================


def test_core_contingency_table_evaluation(sample_pangenomes):
    """Verifies TP, FP, FN, TN math for Core gene classifications."""
    ground_truth, inferred = sample_pangenomes
    metric = CoreContingencyTable()
    results = metric.evaluate(ground_truth, inferred)

    assert results["tp"] == 2
    assert results["fp"] == 1
    assert results["fn"] == 0
    assert results["tn"] == 2


def test_edge_contingency_table_evaluation(sample_pangenomes):
    """Verifies classification math for graph edge adjustments."""
    ground_truth, inferred = sample_pangenomes
    metric = EdgeContingencyTable()
    results = metric.evaluate(ground_truth, inferred)

    assert results["tp"] == 3
    assert results["fp"] == 1
    assert results["fn"] == 0
