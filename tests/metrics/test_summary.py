"""Set of tests for the summary metrics."""

from pangesim.metrics.summary import PangenomeSummaryMetrics
from pangesim.panevolve import PangenomeSimulator
from pangesim.reconstruction import EulerianPathHeuristic


def test_summary():
    """Test for summary metrics."""
    # Simulated ground-truth
    sim = PangenomeSimulator(deletion_rate=0.0, rearrangement_rate=5)
    h_pan = sim.generate_pangenome(k=3, length=12)
    h_graph = h_pan.compute_weighted_adjacencies()

    # Test #1: Sanity check
    summary = PangenomeSummaryMetrics()
    results = summary.evaluate(h_pan, h_pan)

    # what happens when it's the same
    assert results["k_diff"] == 0
    assert results["c_diff"] == 0

    # Test #2: Reconstruction differences
    # Reconstructed Pangenome
    heuristic = EulerianPathHeuristic()
    pangenome = heuristic.reconstruct(h_graph)
    summary = PangenomeSummaryMetrics()
    results = summary.evaluate(h_pan, pangenome)

    k_ans = abs(len(h_pan) - len(pangenome))
    c_ans = abs(len(h_pan.core) - len(pangenome.core))

    assert results["k_diff"] == k_ans
    assert results["c_diff"] == c_ans
