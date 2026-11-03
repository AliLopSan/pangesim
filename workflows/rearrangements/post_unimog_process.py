from pathlib import Path

import numpy as np
import pandas as pd

from pangesim.io.unimog import parse_dcj_id_matrix
from workflows.scripts.unit_mwm_comparison import compute_mwm_evaluation


def enrich_summary_with_mwm(results_root: Path) -> pd.DataFrame:
    """Adds mwm results to master dataframe."""
    summary_csv = results_root / "experiment_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary file not found at {summary_csv}")

    df_summary = pd.read_csv(summary_csv)

    # Initialize MWM metric columns if they don't exist
    for col in ["mwm_min", "mwm_max", "mwm_average"]:
        if col not in df_summary.columns:
            df_summary[col] = np.nan

    for idx, row in df_summary.iterrows():
        idx = row["sim_id"]
        method = row["method"]

        rep_dir = results_root / f"sim_{idx}"
        dcj_file = rep_dir / "dcj_id_results.txt"
        if dcj_file.exists():
            try:
                if dcj_file.suffix == ".pkl":
                    df_dcj = pd.read_pickle(dcj_file)
                else:
                    df_dcj = parse_dcj_id_matrix(dcj_file)

                mst_matching, seq_matching, sim_labels = compute_mwm_evaluation(df_dcj)
                matching = mst_matching if method == "MST" else seq_matching

                vals = list(matching.values())
                if vals:
                    df_summary.at[idx, "mwm_min"] = float(min(vals))
                    df_summary.at[idx, "mwm_max"] = float(max(vals))
                    df_summary.at[idx, "mwm_average"] = float(sum(vals) / len(vals))

            except Exception as e:
                print(f"Warning: Could not process MWM for {dcj_file}: {e}")

    # Save enriched master dataframe
    df_summary.to_csv(summary_csv, index=False)
    print(f"Successfully enriched {summary_csv} with MWM metrics.")
    print(df_summary.describe())
    return df_summary


if __name__ == "__main__":
    results_root = Path("results/run_20261028/")

    # Add MWM calculations to master dataframe from UniMoG output
    if (results_root / "experiment_summary.csv").exists():
        df_summary = enrich_summary_with_mwm(results_root)
