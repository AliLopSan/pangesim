"""Plotter script for individual results visualization."""
import pandas as pd
from pathlib import Path
from pangesim.visualization import RuntimeVisualizer

def plot_scalability()->None:
    """ Plots the results of the scalability test."""
    results_dir = Path("results/run_20260706/")
    file_to_read = results_dir / "scalability_metrics.csv"
    df = pd.read_csv(file_to_read)

    out_dir = Path("results/run_20260706/runtime_plots")
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

    print(df.head())

if __name__ == "__main__":
    plot_scalability()
