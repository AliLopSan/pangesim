"""Tests for reconstruction algorithms and utilities."""
from pangesim.panevolve import PangenomeSimulator
from pangesim.reconstruction import EulerianPathHeuristic
from pangesim.reconstruction.assignment import EulerianTrailAssignment
from pangesim.reconstruction.base import matrix_to_list
from pangesim.reconstruction.bounds import GreedyPairingISCB
from pangesim.reconstruction.utils import TopologicalExplorer

def test_topological_explorer_undirected():
    """Validates component isolation and parity mapping on an undirected graph."""
    # Graph structure:
    # Component 1 (Triangle, all even): 1-2, 2-3, 3-1
    # Component 2 (Line, two odd nodes): 4-5
    sample_adjacencies = {(1, 2): 1, (2, 3): 1, (3, 1): 1, (4, 5): 1}
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
    # Graph structure:
    # Component 1 (Triangle, all even): 1-2, 2-3, 3-1
    # Component 2 (Line, two odd nodes): 4-5
    # Component 3 (Unrooted tree): ((6,7)8,(10,11)9);
    sample_matrix = {
        (1, 2): 1,
        (2, 3): 1,
        (3, 1): 1,
        (4, 5): 1,
        (6, 8): 3,
        (8, 7): 2,
        (8, 9): 4,
        (9, 10): 3,
        (9, 11): 3,
    }
    assign = EulerianTrailAssignment()
    pangenome = assign.assign_genomes(sample_matrix, 3)

    assert pangenome.check_integrity() is True


def test_defaults():
    """Tests the heuristic's defaults methods."""
    # Graph structure:
    # Component 1 (Triangle, all even): 1-2, 2-3, 3-1
    # Component 2 (Line, two odd nodes): 4-5
    # Component 3 (Unrooted tree): ((6,7)8,(10,11)9);
    sample_matrix = {
        (1, 2): 1,
        (2, 3): 1,
        (3, 1): 1,
        (4, 5): 1,
        (6, 8): 3,
        (8, 7): 2,
        (8, 9): 4,
        (9, 10): 3,
        (9, 11): 3,
    }
    heuristic = EulerianPathHeuristic()
    pangenome = heuristic.reconstruct(sample_matrix)
    assert pangenome.check_integrity() is True
    

def test_full_heuristic():
    """Test of the full pipeline."""
    # Graph structure:
    # Component 1 (Triangle, all even): 1-2, 2-3, 3-1
    # Component 2 (Line, two odd nodes): 4-5
    # Component 3 (Unrooted tree): ((6,7)8,(10,11)9);
    sample_matrix = {
        (1, 2): 1,
        (2, 3): 1,
        (3, 1): 1,
        (4, 5): 1,
        (6, 8): 3,
        (8, 7): 2,
        (8, 9): 4,
        (9, 10): 3,
        (9, 11): 3,
    }
    bounds = GreedyPairingISCB()
    assign = EulerianTrailAssignment()
    heuristic = EulerianPathHeuristic(bounds_strategy=bounds, assignment_strategy=assign)
    pangenome = heuristic.reconstruct(sample_matrix)
    assert pangenome.check_integrity() is True


def test_roboust_example():
    """Testing each step of the pipeline."""
    sample_matrix = {
        (1, 2): 3,
        (2, 3): 4,
        (2, 6): 1,
        (3, 4): 2,
        (3, 10): 3,
        (4, 5): 3,
        (4, 8): 3,
        (6, 7): 1,
        (7, 9): 1,
        (10, 9): 3,
        (9, 11): 2,
    }
    params = {"alpha": 0.5, "gamma": 1.0}
    bounds = GreedyPairingISCB()
    assign = EulerianTrailAssignment()
    heuristic = EulerianPathHeuristic(
        params=params, bounds_strategy=bounds, assignment_strategy=assign
    )
    pangenome = heuristic.reconstruct(sample_matrix)
    assert pangenome.check_integrity() is True
