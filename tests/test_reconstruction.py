"""Tests for reconstruction algorithms."""
from pangesim.panevolve import PangenomeSimulator
from pangesim.reconstruction import EulerianPathHeuristic


def test_dummy_pipeline():
    """Naive Pangenome reconstruction."""
    sim = PangenomeSimulator(deletion_rate=2, rearrangement_rate=3)
    p = sim.generate_pangenome(k=3, length=15)

    #Input for reconstruction algorithm
    h_ground = p.compute_weighted_adjacencies()
    print("\t Created ground truth : ", type(h_ground), "with ",
          len(h_ground),"edges")
    heuristic = EulerianPathHeuristic()
    p_recons  = heuristic.reconstruct(h_ground)

    assert len(p_recons) == len(h_ground)
