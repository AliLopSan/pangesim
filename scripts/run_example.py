"""Script to generate a toy pangenome simulation and its graph plot.

This serves as a quick-start example and smoke test for the panevolve package.
"""

import argparse
from pathlib import Path

from pangesim.panevolve import PangenomeSimulator
from pangesim.visualization import PangenomeVisualizer


def main():
    """Parses CLI arguments, executes simulation, and outputs visualization layout.

    Returns:
        The visualization of the simulated pangenome and its adjacency graph.
    """
    # Using either user-defined arguments or defaults for simulation
    parser = argparse.ArgumentParser(
        description="Generate a simulated pangenome and save its visualization."
    )
    parser.add_argument("--deletion-rate", type=float, default=2.0, help="Rate of gene deletion")
    parser.add_argument(
        "--rearrangement-rate", type=float, default=3.0, help="Rate of genome rearrangement"
    )
    parser.add_argument("-k", type=int, default=3, help="Number of desired genomes")
    parser.add_argument("--length", type=int, default=15, help="Number of genes/nodes per genome")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save plots")

    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Simulating pangenome (k={args.k}, length={args.length})...")

    # Initialize and run simulation
    sim = PangenomeSimulator(
        deletion_rate=args.deletion_rate, rearrangement_rate=args.rearrangement_rate
    )
    p1 = sim.generate_pangenome(k=args.k, length=args.length)

    print("Generating visualization...")
    vis = PangenomeVisualizer(p1)

    # Save the plot instead of just showing it
    vis.plot_pangenome_grid(output_path=output_path, filename="pangenome_example.png")

    print("Done!")


if __name__ == "__main__":
    main()
