"""Main script for running inference strategies."""

import random as rd
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from pangesim.io.unimog import export_to_unimog
from pangesim.metrics.summary import PangenomeDistribution
from pangesim.panevolve import LinearSimulator
from pangesim.reconstruction import SequentialEdgeInsertion
from pangesim.reconstruction.assignment import MSTAssignment
from pangesim.reconstruction.base import AdjacencyMatrix
from pangesim.reconstruction.refining import SequentialEdgeRefinement
from pangesim.reconstruction.utils import GlobalGenomePool


def run_seq_inference(m: AdjacencyMatrix):
    """Sequential edge addition method."""
    heuristic = SequentialEdgeInsertion()
    inf_pangenome = heuristic.reconstruct(m)
    return inf_pangenome


def run_mst_inference(m: AdjacencyMatrix):
    """Maximum Spanning Tree inference."""
    assign = MSTAssignment()
    id_pool = GlobalGenomePool(start_id=1)
    base_pangenome = assign.assign_genomes(m, id_pool)
    refiner = SequentialEdgeRefinement(id_pool)
    inf_pangenome = refiner.refine(source=m, target=base_pangenome)
    return inf_pangenome


def generate_pan(num_genes: int, rate_range: Tuple) -> Tuple:
    """Returns a random simulated pangenome.

    Args:
    num_genes: Number of desired genes.
    rate_range: The inferior and superior rate ranges for event generation.

    Raises:
       TypeError: If num_genes is not an integer.
       ValueError: If num_genes is less than 2 or evol not in options.
    """
    num_genomes = rd.randint(10, 31)
    r1 = np.random.uniform(rate_range[0], rate_range[1])
    r2 = np.random.uniform(rate_range[0], rate_range[1])
    r3 = np.random.uniform(rate_range[0], rate_range[1])
    r4 = np.random.uniform(rate_range[0], rate_range[1])
    r5 = np.random.uniform(rate_range[0], rate_range[1])
    r6 = np.random.uniform(rate_range[0], rate_range[1])
    r7 = np.random.uniform(rate_range[0], rate_range[1])
    r8 = np.random.uniform(rate_range[0], rate_range[1])

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

    ground_truth = sim.generate_pangenome(k=num_genomes, length=num_genes, complexity="medium")

    return sim.simulation_history, ground_truth


def run_inference_with_events(results: Path) -> None:
    """UNIMOG input generation.

    Args:
        results: The results path for the output.
    """
    rates = [
        (0.0, 0.5),
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, 4.0),
        (5.0, 6.0),
        (6.0, 7.0),
        (7.0, 8.0),
        (8.0, 9.0),
        (9.0, 10.0),
    ]
    n_replicates = 5
    events = ["insertion","deletion",
              "rearrangement","inversion",
              "fission","fusion","translocation",
              "gene_deletion"]
    summary_records = []

    results.mkdir(parents=True, exist_ok=True)
    total_iterations = len(rates) * n_replicates

    batch_dir = results / "batch_results"
    batch_dir.mkdir(parents=True, exist_ok=True)
    sim_id = 0

    with tqdm(total=total_iterations, desc="Total Progress") as pbar:
        for rate in rates:
            for rep in range(n_replicates):
                rep_dir = batch_dir / f"sim_{sim_id}"
                rep_dir.mkdir(parents=True, exist_ok=True)

                # 1. Ground Truth generation & simulation history
                sim_history, ground_truth = generate_pan(num_genes=500, rate_range=rate)
                filename = "ground_truth.unimog"
                export_to_unimog(rep_dir / filename, ground_truth)
                df_history = pd.DataFrame(sim_history)
                df_history.to_csv(rep_dir / "simulation_history.csv", index=False)

                #
                total_events = {
                    "insertion": 0,
                    "deletion": 0,
                    "rearrangement": 0,
                    "inversion": 0,
                    "fission": 0,
                    "fusion": 0,
                    "translocation": 0,
                    "gene_deletion": 0,
                }
                for entry in sim_history:
                    counts = entry.get("event_counts", {})
                    for ev_key, ev_val in counts.items():
                        if ev_key in total_events:
                            total_events[ev_key] += ev_val

                gt_dist = PangenomeDistribution(pangenome=ground_truth, name="simulated")
                gt_g = gt_dist.get_genome_summary()
                gt_g.to_csv(rep_dir / "simulated_genome.csv", index=False)
                gt_w = gt_dist.get_weight_distribution()
                gt_w.to_csv(rep_dir / "simulated_weights.csv", index=False)

                matrix = ground_truth.compute_weighted_adjacencies()
                gt_core = ground_truth.core

                # MST run
                mst = run_mst_inference(matrix)
                mst.check_integrity()
                export_to_unimog(rep_dir / "mst_pangenome.unimog", mst)

                mst_dist = PangenomeDistribution(pangenome=mst, name="MST")
                mst_g = mst_dist.get_genome_summary()
                mst_g.to_csv(rep_dir / "mst_genome.csv", index=False)
                mst_w = mst_dist.get_weight_distribution()
                mst_w.to_csv(rep_dir / "mst_weights.csv", index=False)

                mst_core = mst.core
                mst_sd = gt_core.symmetric_difference(mst_core)
                mst_int = gt_core.intersection(mst_core)

                # Build base record with shared parameters and aggregated event counts
                base_record = {
                    "sim_id": sim_id,
                    "total_genes_gt":ground_truth.total_gene_count,
                    "genomes_gt": len(ground_truth),
                    "core_gt": len(gt_core),
                }

                for ev in events:
                    base_record[f"event_{ev}"] = total_events[ev]

                    # Append MST record
                mst_record = base_record.copy()
                mst_record.update(
                    {
                        "method": "MST",
                        "genomes_inf": len(mst),
                        "core_inf": len(mst_core),
                        "core_sd": len(mst_sd),
                        "core_int": len(mst_int),
                    }
                )
                summary_records.append(mst_record)

                # 3. Sequential inference
                seq = run_seq_inference(matrix)
                seq.check_integrity()
                export_to_unimog(rep_dir / "seq_pangenome.unimog", seq)

                seq_dist = PangenomeDistribution(pangenome=seq, name="SEQ")
                seq_g = seq_dist.get_genome_summary()
                seq_g.to_csv(rep_dir / "seq_genome.csv", index=False)
                seq_w = seq_dist.get_weight_distribution()
                seq_w.to_csv(rep_dir / "seq_weights.csv", index=False)

                seq_core = seq.core
                seq_sd = gt_core.symmetric_difference(seq_core)
                seq_int = gt_core.intersection(seq_core)

                # Append SEQ record
                seq_record = base_record.copy()
                seq_record.update(
                    {
                        "method": "SEQ",
                        "genomes_inf": len(seq),
                        "core_inf": len(seq_core),
                        "core_sd": len(seq_sd),
                        "core_int": len(seq_int),
                    }
                )
                summary_records.append(seq_record)

                pbar.set_postfix({"sim_id": sim_id})
                pbar.update(1)

                sim_id += 1

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(results / "experiment_summary.csv", index=False)
    print(f"\n Master CSV successfully saved to {results / 'experiment_summary.csv'}")
    print(summary_df.head())

if __name__ == "__main__":
    out_dir = Path("results/run_20261028/")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_inference_with_events(out_dir)
