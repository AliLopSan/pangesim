"""Tests for reconstruction algorithms and utilities."""
from pangesim.panevolve import PangenomeSimulator
from pangesim.reconstruction import EulerianPathHeuristic
from pangesim.reconstruction.assignment import EulerianTrailAssignment
from pangesim.reconstruction.base import matrix_to_list
from pangesim.reconstruction.bounds import GreedyPairingISCB
from pangesim.reconstruction.utils import TopologicalExplorer


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

    greedy = GreedyPairingISCB()
    greedy_heuristic = EulerianPathHeuristic(bounds_strategy=greedy)
    inferred = greedy_heuristic.reconstruct(h_ground)

    assert len(inferred) == len(h_ground)

def test_topological_explorer_undirected():
    """Validates component isolation and parity mapping on an undirected graph."""
    # Graph structure:
    # Component 1 (Triangle, all even): 1-2, 2-3, 3-1
    # Component 2 (Line, two odd nodes): 4-5
    sample_adjacencies = {
        (1, 2): 1, (2, 3): 1, (3, 1): 1,
        (4, 5): 1
    }
    sample_list = matrix_to_list(sample_adjacencies)

    explorer = TopologicalExplorer(sample_list, directed=False)
    components = explorer.extract_components()

    assert len(components) == 2

    # Find the triangle component
    triangle = next(c for c in components if 1 in c.nodes)
    assert triangle.nodes == {1, 2, 3}
    assert triangle.is_eulerian is True
    assert len(triangle.odd_vertices) == 0

    # Find the line component
    line = next(c for c in components if 4 in c.nodes)
    assert line.nodes == {4, 5}
    assert line.is_eulerian is False
    assert set(line.odd_vertices) == {4, 5}

def test_eulerian_assignments():
    """Test the basics of eulerian assignments."""
    sample_matrix = {
        (1, 2): 1, (2, 3): 1, (3, 1): 1,
        (4, 5): 1
    }
    assign = EulerianTrailAssignment()
    pan = assign.assign_genomes(sample_matrix,3)

    assert len(pan) == 0
