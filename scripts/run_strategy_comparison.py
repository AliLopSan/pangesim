"""Script to compare different strategies used in the Heuristic."""
from pangesim.reconstruction import EulerianPathHeuristic
from pangesim.reconstruction.assignment import EulerianTrailAssignment
from pangesim.reconstruction.bounds import GreedyPairingISCB
from pangesim.reconstruction.sorting import WeightSorting
from pangesim.visualization import PangenomeVisualizer


def run_evaluation():
    """Tests on different strategies over a simple ground-truth graph."""
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
    #Results directories:
    out_path = "../data/"

    #Eulerian paths by length
    bounds = GreedyPairingISCB()
    assign = EulerianTrailAssignment()
    heuristic = EulerianPathHeuristic(bounds_strategy=bounds,
                                      assignment_strategy=assign)
    pangenome = heuristic.reconstruct(sample_matrix)
    vis = PangenomeVisualizer(pangenome)
    vis.plot_pangenome_grid(output_path=out_path,
                            filename="by_length.pdf")

    #Eulerian paths by weight
    assign = EulerianTrailAssignment(trail_sorting=WeightSorting())
    heuristic = EulerianPathHeuristic(bounds_strategy=bounds,
                                      assignment_strategy=assign)
    pangenome = heuristic.reconstruct(sample_matrix)
    vis = PangenomeVisualizer(pangenome)
    vis.plot_pangenome_grid(output_path=out_path,
                            filename="by_weight.pdf")

if __name__ == "__main__":
    run_evaluation()
