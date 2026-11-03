"""Main script for running inference strategies."""
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pangesim.io.unimog import export_to_unimog
from pangesim.metrics.summary import PangenomeDistribution
from pangesim.reconstruction import SequentialEdgeInsertion
from pangesim.reconstruction.assignment import MSTAssignment
from pangesim.reconstruction.base import AdjacencyMatrix
from pangesim.reconstruction.refining import SequentialEdgeRefinement
from pangesim.reconstruction.utils import GlobalGenomePool
from workflows.scripts.generate_sim import generate_pan


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

def run_inference(results: Path)-> None:
    """UNIMOG input generation.

    Args:
        results: The results path for the output.
    """
    levels = ["low","medium","high"]
    gene_sizes = [25, 100, 200, 500, 800, 1000]
    n_replicates = 5
    summary_records = []

    results.mkdir(parents=True, exist_ok=True)
    total_iterations = len(gene_sizes) * len(levels) * n_replicates

    with tqdm(total=total_iterations, desc="Total Progress") as pbar:
        for size in gene_sizes:
            size_dir = results / str(size)
            size_dir.mkdir(parents=True, exist_ok=True)

            for level in levels:
                level_dir = size_dir / level
                level_dir.mkdir(parents=True, exist_ok=True)

                for rep in range(n_replicates):
                    rep_dir = level_dir / f"rep_{rep}"
                    rep_dir.mkdir(parents=True, exist_ok=True)

                    # Ground Truth generation (passing replicate as seed component if supported)
                    sim_history, ground_truth = generate_pan(num_genes=size, evol=level)
                    filename = "ground_truth.unimog"
                    export_to_unimog(rep_dir / filename, ground_truth)
                    df_history = pd.DataFrame(sim_history)
                    df_history.to_csv(rep_dir / "simulation_history.csv", index=False)

                    gt_dist = PangenomeDistribution(pangenome=ground_truth, name="simulated")
                    gt_g = gt_dist.get_genome_summary()
                    gt_g.to_csv(rep_dir / "simulated_genome.csv", index=False)
                    gt_w = gt_dist.get_weight_distribution()
                    gt_w.to_csv(rep_dir / "simulated_weights.csv", index=False)

                    matrix = ground_truth.compute_weighted_adjacencies()
                    gt_core = ground_truth.core

                    #MST inference
                    mst = run_mst_inference(matrix)
                    mst.check_integrity()
                    export_to_unimog(rep_dir / "mst_pangenome.unimog", mst)

                    mst_dist = PangenomeDistribution(pangenome=mst, name="MST")
                    mst_g = mst_dist.get_genome_summary()
                    mst_g.to_csv(rep_dir / "mst_genome.csv", index=False)
                    mst_w = mst_dist.get_weight_distribution()
                    mst_w.to_csv(rep_dir / "mst_weights.csv", index=False)

                    # MST Core metrics
                    mst_core = mst.core
                    mst_sd = gt_core.symmetric_difference(mst_core)

                    summary_records.append(
                        {
                            "gene_size": size,
                            "complexity": level,
                            "replicate": rep,
                            "method": "MST",
                            "genomes_gt": len(ground_truth),
                            "genomes_inf": len(mst),
                            "core_gt": len(gt_core),
                            "core_inf": len(mst_core),
                            "core_sd": len(mst_sd)
                        }
                    )

                    # Sequential inference
                    seq = run_seq_inference(matrix)
                    seq.check_integrity()
                    export_to_unimog(rep_dir / "seq_pangenome.unimog", seq)

                    seq_dist = PangenomeDistribution(pangenome=seq, name="SEQ")
                    seq_g = seq_dist.get_genome_summary()
                    seq_g.to_csv(rep_dir / "seq_genome.csv", index=False)
                    seq_w = seq_dist.get_weight_distribution()
                    seq_w.to_csv(rep_dir / "seq_weights.csv", index=False)

                    # SEQ Core metrics
                    seq_core = seq.core
                    seq_sd = gt_core.symmetric_difference(seq_core)

                    summary_records.append(
                        {
                            "gene_size": size,
                            "complexity": level,
                            "replicate": rep,
                            "method": "SEQ",
                            "genomes_gt": len(ground_truth),
                            "genomes_inf": len(seq),
                            "core_gt": len(gt_core),
                            "core_inf": len(seq_core),
                            "core_sd": len(seq_sd)
                        }
                    )

                    pbar.set_postfix({"size": size, "level": level, "rep": rep})
                    pbar.update(1)

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(results / "experiment_summary.csv", index=False)
    print(f"\n Master summary CSV successfully saved to {results / 'experiment_summary.csv'}")


if __name__ == "__main__":
    out_dir = Path("results/run_20261022/")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_inference(out_dir)
