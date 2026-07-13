"""Plotter script for individual results visualization."""
from pathlib import Path
from typing import Dict

import pandas as pd

from benchmarks.config import PARAM_GRID
from pangesim.visualization import ErrorVisualizer
from pangesim.visualization import RuntimeVisualizer

def plot_scalability(results_dir:Path, filename:Path)->None:
    """Plots the results of the scalability test.

    Args:
        results_dir: Path where the results will be stored and where df is.
        filename: Path and name of the df
    """
    df = pd.read_csv(filename)

    out_dir = Path("results/run_20260713/runtime_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    #Plot total run-time
    vis_1 = RuntimeVisualizer()
    f1_name = out_dir / "total_runtime.pdf"
    vis_1.plot_total_runtime(df,f1_name)

    vis_2 = RuntimeVisualizer()
    f2_name = out_dir / "phase_1_3_runtime.pdf"
    vis_2.plot_phase_runtime(df, f2_name)

    vis_3 = RuntimeVisualizer()
    f3_name = out_dir / "phase_4_runtime.pdf"
    vis_3.plot_phase4_runtime(df, f3_name)

def plot_mape(results_dir:Path, params:Dict[str,float],filename:Path )->None:
    """Plots the mape for a given dict of parameters.

    Args:
        results_dir: Path where the results will be stored and where df is.
        params: A dict of parameters that should contain "alpha" and "gamma"
        filename: Path and name of the dataframe
    """
    df = pd.read_csv(filename)

    out_dir = Path("results/run_20260713/mape_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Automated filename layout using components from the params dict
    param_strings = []
    for key, value in params.items():
        clean_val = str(value).replace(".", "_")
        param_strings.append(f"{key}_{clean_val}")

    fname = f"mape_{'_'.join(param_strings)}.pdf"
    final_output_path = out_dir / fname
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    vis = ErrorVisualizer()
    vis.plot_genomes_mape(df=df,params=params,output_path=final_output_path)

if __name__ == "__main__":
    print("\tRunning  Harry Plotter ...")
    results = Path("results/run_20260713")
    df_file = results / "error_metrics_ISMB_bounds.csv"
    plot_scalability(results_dir=results, filename=df_file)
    for params in PARAM_GRID:
        plot_mape(results_dir=results, params=params, filename=df_file)
    print("Done! :)")
