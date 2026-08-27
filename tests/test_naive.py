"""Tests for naive reconstruction algorithm."""

from pangesim.reconstruction import SequentialEdgeInsertion


def test_build_residuals():
    """Test for the sorting phase of the pipeline."""
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
    algorithm = SequentialEdgeInsertion()
    r = algorithm.build_sorted_residuals(sample_matrix)

    first_value = r[0]
    assert first_value[1] == 4


def test_reconstruction():
    """Tests the full pipeline."""
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
    algorithm = SequentialEdgeInsertion()

    pangenome = algorithm.reconstruct(sample_matrix)
    
    assert pangenome.check_integrity() is True
    
