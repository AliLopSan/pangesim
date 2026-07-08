
from pathlib import Path

import pandas as pd

from benchmarks.config import PARAM_GRID
from benchmarks.config import STRATEGIES
from benchmarks.fixtures import random_simulated_pangenome
from benchmarks.runners import evaluate_strategy_run
from pangesim.visualization import TrajectoryVisualizer


def main() -> None:
    """Experiment run for a simple mock pangenome."""
    print("Loading experimental scenario fixtures...")
    true_pangenome = random_simulated_pangenome(300)
    mock_matrix = true_pangenome.compute_weighted_adjacencies()

    all_results = []

    # Run every strategy across every parameter set
    for strat_key, strat_label in STRATEGIES.items():
        for params in PARAM_GRID:
            print(f"Evaluating {strat_label} (alpha={params['alpha']}, gamma={params['gamma']})...")

            run_history, calculated_kmin, calculated_kmax = evaluate_strategy_run(
                strategy_key=strat_key,
                matrix=mock_matrix,
                ground_truth=true_pangenome,
                params=params,
            )

            # Convert to DataFrame and tag it with structural metadata
            df_run = pd.DataFrame(run_history)
            df_run["Strategy"] = strat_label
            df_run["Strategy_Key"] = strat_key
            df_run["Config"] = rf"$\alpha = {params['alpha']},\, \gamma = {params['gamma']}$"
            df_run["kmin"] = calculated_kmin
            df_run["kmax"] = calculated_kmax

            all_results.append(df_run)

    master_df = pd.concat(all_results, ignore_index=True)

    print("\nMaster execution dataframe compiled successfully.")
    print(master_df.head())

    out_dir = Path("results/run_20260707/single_scenario")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_output_path = out_dir / "master_evaluation_metrics.csv"
    master_df.to_csv(csv_output_path, index=False)

    # 2. Slice dataframe by strategy and generate individual parameter plots
    for strat_key, strat_label in STRATEGIES.items():
        print(f"Generating parameter trajectory visualization for {strat_label}...")

        # Filter rows belonging only to this specific algorithmic strategy
        strat_df = master_df[master_df["Strategy_Key"] == strat_key]
        strategy_dir = out_dir / strat_key
        strategy_dir.mkdir(parents=True, exist_ok=True)

        output_file = strategy_dir / "all_score.pdf"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        viz = TrajectoryVisualizer()
        viz.plot_strategy_parameters(
            strategy_df=strat_df,
            strategy_label=strat_label,
            metric="Score",
            save_path=output_file,
        )
        output_file = strategy_dir/ "all_genomes_difference.pdf"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        viz = TrajectoryVisualizer()
        viz.plot_strategy_parameters(
            strategy_df=strat_df,
            strategy_label=strat_label,
            metric="Number of Genomes Delta",
            save_path=output_file,
        )
        output_file = strategy_dir / "all_core_difference.pdf"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        viz = TrajectoryVisualizer()
        viz.plot_strategy_parameters(
            strategy_df=strat_df,
            strategy_label=strat_label,
            metric="Number of Core Genes Delta",
            save_path=output_file,
        )
        # For each parameter, compute the bounds:
        for config_label, config_df in strat_df.groupby("Config"):
            strategy_dir = Path(strategy_dir)
            strategy_dir.mkdir(parents=True, exist_ok=True)
            clean_filename = (
                config_label.replace("$", "")
                .replace("\\alpha = ", "alpha")
                .replace("\\gamma = ", "gamma")
                .replace(",\\, ", "_")
                .strip()
            )
            output_file_bounds = strategy_dir / f"bounds_{clean_filename}.pdf"
            output_file_score = strategy_dir / f"score_{clean_filename}.pdf"

            viz.plot_k_bounds_from_dataframe(
                config_df=config_df,
                strategy_label=strat_label,
                config_label=str(config_label),
                save_path=output_file_bounds,
            )
            viz.plot_score_from_dataframe(
                config_df=config_df,
                strategy_label=strat_label,
                config_label=str(config_label),
                save_path=output_file_score,
            )


if __name__ == "__main__":
    main()
