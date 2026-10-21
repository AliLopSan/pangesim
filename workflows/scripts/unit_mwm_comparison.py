from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from pangesim.io.unimog import parse_dcj_id_matrix
from pangesim.visualization.distances import MWMComparisonVisualizer


def compute_mwm_evaluation(dist_matrix: pd.DataFrame) -> tuple[dict, dict, list]:
    """Computes MWM bipartite matchings.

    Returns:
        A tuple containing (mst_matching_dict, seq_matching_dict, sim_labels).
    """
    # 1. Fill symmetric distance matrix
    full_matrix = dist_matrix.fillna(dist_matrix.T)

    # 2. Identify groups
    all_labels = list(full_matrix.index)
    sim_labels = [label for label in all_labels if label.startswith("simulated_")]
    mst_labels = [label for label in all_labels if label.startswith("MST_")]
    seq_labels = [label for label in all_labels if label.startswith("SEQ_")]

    # Extract sub-matrices
    mat_sim_mst = full_matrix.loc[sim_labels, mst_labels]
    mat_sim_seq = full_matrix.loc[sim_labels, seq_labels]

    # 3. Helper to solve MWM and return a dictionary of matches
    def get_mwm_dictionary(sub_df: pd.DataFrame, method_name: str) -> dict:
        cost_array = sub_df.to_numpy()
        if np.isnan(cost_array).any():
            max_val = np.nanmax(cost_array) if not np.isnan(cost_array).all() else 100.0
            cost_array = np.nan_to_num(cost_array, nan=max_val)

        row_ind, col_ind = linear_sum_assignment(cost_array)

        matching_dict = {}
        for r_idx, c_idx in zip(row_ind, col_ind):
            sim_name = sub_df.index[r_idx]
            inf_name = sub_df.columns[c_idx]
            dist = sub_df.iloc[r_idx, c_idx]
            # Store tuple as key and distance as value
            matching_dict[(sim_name, inf_name)] = float(dist)

        return matching_dict

    # Get dictionaries for both methods
    mst_matching = get_mwm_dictionary(mat_sim_mst, "MST")
    seq_matching = get_mwm_dictionary(mat_sim_seq, "SEQ")

    """
    print("--- MST MWM Dictionary ---")
    for edge in mst_matching:
        print("\t ", edge, "\t", mst_matching[edge])
    print("\n--- SEQ MWM Dictionary ---")
    for edge in seq_matching:
        print("\t ", edge, "\t", seq_matching[edge])
    """

    return mst_matching, seq_matching, sim_labels

if __name__ == "__main__":
    results_root = Path("results/run_20261020/unit_test/")
    results_root.mkdir(parents=True, exist_ok=True)
    print("Using pandas version: ", pd.__version__)
    print("Using numpy version: ", np.__version__)
    dcj_file = results_root / "dcj_id_results.txt"
    df = parse_dcj_id_matrix(dcj_file)
    print(df.head())

    mst_distances, seq_distances, sim_labels = compute_mwm_evaluation(df)

    vis = MWMComparisonVisualizer()
    vis.plot_dumbbell_comparison(mst_matching=mst_distances,
                                 seq_matching=seq_distances,
                                 sim_labels=sim_labels,
                                 save_path=results_root)
