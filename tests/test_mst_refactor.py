"""Test for MST and MSTAssignment."""
__author__ = "AliLopsan"
__status__ = "Development"

from pangesim.reconstruction.assignment import MSTAssignment
from pangesim.reconstruction.refining import SequentialEdgeRefinement
from pangesim.reconstruction.utils import GlobalGenomePool


def test_mst_connected():
    """Test for MST and MSTAssignment."""
    print("\t Test Matrix")
    sample_matrix = {
        (3, 8):1,
    (2, 9):1,
    (0, 5):2,
    (1, 6):1,
    (4, 7):1,
    (1, 2):2,
    (1, 4):1,
    (2, 3):2,
    (6, 7):2,
    (8, 9):2,
    (5, 6):2,
    (7, 8):2,
    (0, 1):1,
    (4, 5):1
    }
    for edge in sample_matrix:
        print("\t ",edge,"\t",sample_matrix[edge])
    assign = MSTAssignment()
    id_pool = GlobalGenomePool(start_id=1)
    pangenome = assign.assign_genomes(sample_matrix, id_pool)

    print("\t For MST connected: ")

    for comp in assign.component_trees:
        tree = assign.component_trees[comp]
        tree.print_tree()

    print("\t\tBase Pangenome: \n",pangenome.summary())
    for genome in pangenome.genomes:
        print("\t\t Genome ", genome._genome_id,"\n",genome)
    assert pangenome.check_integrity() is True

    refiner = SequentialEdgeRefinement(id_pool)

    inferred = refiner.refine(source=sample_matrix, target=pangenome)

    assert inferred.check_integrity() is True
    print("\t\t Inferred Pangenome: \n", inferred.summary())


if __name__ == "__main__":
    test_mst_connected()
