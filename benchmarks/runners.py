"""Orchestrates pipeline execution profiles across different strategies."""

import time
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from benchmarks import PipelineTracker
from benchmarks.fixtures import random_simulated_pangenome
from pangesim import Pangenome
from pangesim.reconstruction import EulerianPathHeuristic
from pangesim.reconstruction.assignment import DummyAssignment
from pangesim.reconstruction.assignment import EulerianTrailAssignment
from pangesim.reconstruction.bounds import GreedyPairingISCB
from pangesim.reconstruction.base import AdjacencyMatrix
from pangesim.reconstruction.operators import MergeOperator
from pangesim.reconstruction.operators import SplitOperator
from pangesim.reconstruction.sorting import WeightSorting
from pangesim.reconstruction.utils import pan_score


def optimize_with_operators(
    pangenome: Pangenome,
    matrix: AdjacencyMatrix,
    ground_truth: Pangenome,
    k_min: int,
    k_max: int,
    tracker: PipelineTracker,
    alpha: float,
    gamma: float,
) -> Pangenome:
    """Phase 4: Optimize pangenome using split/merge operators within bounds."""
    improved = True
    max_iters = 100
    pangenome = pangenome.copy()

    current_score = pan_score(target=pangenome, source=matrix, alpha=alpha, gamma=gamma)
    #print(f"\nStarting Phase 4 Optimization. Initial Score: {current_score:.4f}")
    i = 0

    while improved:
        improved = False
        current_k = len(pangenome.genomes)

        # --- Operator 1: Merge Genomes ---
        if current_k > k_min:
            candidate_pan = pangenome.copy()
            m_1 = MergeOperator()
            m_1.improve(candidate_pan)
            merge_score = pan_score(target=candidate_pan,
                                    source=matrix, alpha=alpha, gamma=gamma)
            if merge_score > current_score:
                pangenome = candidate_pan
                current_score = merge_score
                improved = True
                tracker(
                    step_name="Phase 4: Successful Merge",
                    pangenome=pangenome,
                    iteration=len(tracker.history),
                    ground_truth=ground_truth,
                    alpha=alpha,
                    gamma=gamma,
                )

        # --- Operator 2: Split Genomes ---
    
        if current_k < k_max:
            candidate_pan = pangenome.copy()
            s_1 = SplitOperator()
            s_1.improve(candidate_pan)

            split_score = pan_score(target=candidate_pan,
                                    source=matrix, alpha=alpha, gamma=gamma)
            if split_score > current_score:
                pangenome = candidate_pan
                current_score = split_score
                improved = True
                tracker(
                    step_name="Phase 4: Successful Split",
                    pangenome=pangenome,
                    iteration=len(tracker.history),
                    ground_truth=ground_truth,
                    alpha=alpha,
                    gamma=gamma,
                )
      
        i += 1
        if i == max_iters:
            improved = False

    #print(f"Phase 4 Complete. Final Optimized Score: {current_score:.4f}")
    return pangenome


def evaluate_strategy_run(
    strategy_key: str,
    matrix: AdjacencyMatrix,
    ground_truth: Pangenome,
    params: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Configures an orchestrator, runs the pipeline, and returns tracking outputs.

    Args:
        strategy_key: Unique identifier selecting the specific path-assignment strategy.
        matrix: Input weighted adjacency matrix representing edge frequencies.
        ground_truth: The benchmark pangenome used for comparative accuracy metrics.
        params: Hyperparameters dict containing alpha and gamma.

    Returns:
        A tuple containing:
            - history: List of optimization dictionaries recorded at every step.
            - k_min: The calculated integer lower bound limit for number of genomes.
            - k_max: The calculated integer upper bound limit for number of genomes.
    """
    #bounds = GreedyPairingISCB()
    if strategy_key == "edge_assignment":
        assignment = DummyAssignment()
    elif strategy_key == "eulerian_length":
        assignment = EulerianTrailAssignment()
    else:
        assignment = EulerianTrailAssignment(trail_sorting=WeightSorting())

    tracker = PipelineTracker()
    bounds = GreedyPairingISCB()
    heuristic = EulerianPathHeuristic(
        bounds_strategy=bounds,
        params=params,
        assignment_strategy=assignment,
    )

    # Execute core reconstruction pipeline
    inf_pangenome = heuristic.reconstruct(
        matrix=matrix, ground_truth=ground_truth, callbacks=[tracker]
    )
    """
    optimize_with_operators(
        pangenome=inf_pangenome,
        matrix=matrix,
        ground_truth=ground_truth,
        k_min=heuristic.k_min,
        k_max=heuristic.k_max,
        tracker=tracker,
        alpha=params["alpha"],
        gamma=params["gamma"],
    )
    """

    # Extract the runtime bounds metadata safely before the instance scope terminates
    k_min = heuristic.k_min if heuristic.k_min is not None else 0
    k_max = heuristic.k_max if heuristic.k_max is not None else 0

    return tracker.history, k_min, k_max


def evaluate_scalability_run(num_genes: int, replicate: int) -> Dict[str, Any]:
    """Main runner to test scalability.

    Args:
        num_genes: Number of genes per genome.
        replicate: Current replicate number.

    Returns:
        A dict with the total runtime, as well as phases 1-3 and phase 4.
    """
    # Simulate random scenario
    tracker = PipelineTracker()
    ground_truth = random_simulated_pangenome(num_genes)
    matrix = ground_truth.compute_weighted_adjacencies()
    params = {"alpha": 1.0, "gamma": 1.0}
    assignment = EulerianTrailAssignment()
    heuristic = EulerianPathHeuristic(
        params=params, assignment_strategy=assignment)

    # start clock
    t0 = time.perf_counter()
    inf_pangenome = heuristic.reconstruct(matrix=matrix,
                                          ground_truth=ground_truth,
                                          callbacks=[tracker])

    # End of phases 1-3
    t1 = time.perf_counter()

    optimize_with_operators(
        pangenome=inf_pangenome,
        matrix=matrix,
        ground_truth=ground_truth,
        k_min=heuristic.k_min,
        k_max=heuristic.k_max,
        tracker=tracker,
        alpha=params["alpha"],
        gamma=params["gamma"],
    )

    # End of phase 4
    t2 = time.perf_counter()

    duration_phases_1_3 = t1 - t0
    duration_phase_4 = t2 - t1
    total_duration = t2 - t0

    return {
        "gene size": num_genes,
        "replicate": replicate,
        "genomes gt":len(ground_truth),
        "genomes inf":len(inf_pangenome),
        "runtime_phases_1-3": duration_phases_1_3,
        "runtime_phase_4": duration_phase_4,
        "total_runtime": total_duration,
    }

def evaluate_error_run(num_genes: int, replicate: int,
                             params:Dict[str, float]) -> Dict[str, Any]:
    """Main runner to test rsme.

    Args:
        num_genes: Number of genes per genome.
        replicate: Current replicate number.
        params: Hyperparameters dict containing alpha and gamma.

    Returns:
        A dict with the total runtime, as well as phases 1-3 and phase 4.
    """
    # Simulate random scenario
    tracker = PipelineTracker()
    ground_truth = random_simulated_pangenome(num_genes)
    matrix = ground_truth.compute_weighted_adjacencies()
    assignment = EulerianTrailAssignment()
    bounds = GreedyPairingISCB()
    heuristic = EulerianPathHeuristic(bounds_strategy=bounds,
        params=params, assignment_strategy=assignment)

    t0 = time.perf_counter()
    inf_pangenome = heuristic.reconstruct(matrix=matrix,
                                          ground_truth=ground_truth,
                                          callbacks=[tracker])

    t1 = time.perf_counter()
    """
    optimize_with_operators(
        pangenome=inf_pangenome,
        matrix=matrix,
        ground_truth=ground_truth,
        k_min=heuristic.k_min,
        k_max=heuristic.k_max,
        tracker=tracker,
        alpha=params["alpha"],
        gamma=params["gamma"],
    )
    """
    t2 = time.perf_counter()

    duration_phases_1_3 = t1 - t0
    duration_phase_4 = t2 - t1
    total_duration = t2 - t0

    return {
        "gene size": num_genes,
        "replicate": replicate,
        "genomes gt":len(ground_truth),
        "genomes inf":len(inf_pangenome),
        "runtime_phases_1-3": duration_phases_1_3,
        "runtime_phase_4": duration_phase_4,
        "total_runtime": total_duration,
        "alpha": params["alpha"],
        "gamma": params["gamma"]
    }
