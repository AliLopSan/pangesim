"""Unit test for Maximum Weight matching comparisons."""
__author__ = "AliLopSan"

from pathlib import Path

import numpy as np

from pangesim.io.unimog import export_to_unimog
from pangesim.panevolve import LinearSimulator
from pangesim.reconstruction import SequentialEdgeInsertion
from pangesim.reconstruction.assignment import MSTAssignment
from pangesim.reconstruction.base import AdjacencyMatrix
from pangesim.reconstruction.refining import SequentialEdgeRefinement
from pangesim.reconstruction.utils import GlobalGenomePool


def run_seq_inference(m:AdjacencyMatrix):
    """Sequential edge addition method."""
    heuristic = SequentialEdgeInsertion()
    inf_pangenome = heuristic.reconstruct(m)
    return inf_pangenome


def run_mst_inference(m:AdjacencyMatrix):
    """Maximum Spanning Tree inference."""
    assign = MSTAssignment()
    id_pool = GlobalGenomePool(start_id=1)
    base_pangenome = assign.assign_genomes(m, id_pool)
    refiner = SequentialEdgeRefinement(id_pool)
    inf_pangenome = refiner.refine(source=m, target=base_pangenome)
    return inf_pangenome

def run_inferences(results: Path):
    """UNIMOG input generation.

    Args:
        results: The results path for the output.
    """
    rates = (0.0, 10.0)
    r1 = np.random.uniform(rates[0], rates[1])
    r2 = np.random.uniform(rates[0], rates[1])
    r3 = np.random.uniform(rates[0], rates[1])
    r4 = np.random.uniform(rates[0], rates[1])
    r5 = np.random.uniform(rates[0], rates[1])
    r6 = np.random.uniform(rates[0], rates[1])
    r7 = np.random.uniform(rates[0], rates[1])
    r8 = np.random.uniform(rates[0], rates[1])

    # Tests for the different complexity levels
    sim = LinearSimulator(
        insertion_rate=r1,
        deletion_rate=r2,
        rearrangement_rate=r3,
        inversion_rate=r4,
        fission_rate=r5,
        fusion_rate=r6,
        translocation_rate=r7,
        gene_deletion_rate=r8,
    )
    #GT generation
    print("Running MST inference...")
    print("\t Generating ground truth...")
    ground_truth = sim.generate_pangenome(k=20, length=1000, complexity="medium")
    matrix = ground_truth.compute_weighted_adjacencies()
    filename = "ground_truth.unimog"
    f = results / filename
    export_to_unimog(f, ground_truth)

    #ground_truth.print_full_paths()

    # MST inference
    print("\t Generating MST Pangenome...")
    mst = run_mst_inference(matrix)
    mst.check_integrity()
    f2 = results / "mst_pangenome.unimog"
    export_to_unimog(f2, mst)

    #mst.print_full_paths()

    # Sequential inference
    print("\t Generating SEQ Pangenome...")
    seq = run_seq_inference(matrix)
    seq.check_integrity()
    f3 = results / "seq_pangenome.unimog"
    export_to_unimog(f3, seq)

    #seq.print_full_paths()



if __name__ == "__main__":
    out_dir = Path("results/run_20261020/unit_test/")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_inferences(out_dir)
