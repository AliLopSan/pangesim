"""Tests verifying visualization layouts."""

from pathlib import Path

from pangesim.panevolve import PangenomeSimulator
from pangesim.visualization import PangenomeVisualizer


def test_visualization(tmp_path: Path) -> None:
    """Verify that pangenome layout methods execute and generate valid file artifacts."""
    # Minimal ground truth simulation
    sim = PangenomeSimulator(deletion_rate=2, rearrangement_rate=3)
    p1 = sim.generate_pangenome(k=3, length=15)
    vis = PangenomeVisualizer(p1)

    # Layout scenarios to iterate over
    layouts = ["dot", "planar", "kamada_kawai", "spectral"]

    for layout in layouts:
        # Construct an absolute path inside the isolated temporary directory
        output_file = f"p_{layout}_layout.pdf"

        # Execute the layout pipeline
        vis.plot_pangenome_grid(
            layout_method=layout, output_path=tmp_path, filename=str(output_file)
        )
        o_file = tmp_path / f"p_{layout}_layout.pdf"

        # 3. Assertions: Assert that the engine actually created a non-empty file
        assert o_file.exists(), f"Visualization artifact for {layout} layout was not generated."
        assert o_file.stat().st_size > 0, f"Generated PDF for {layout} layout is empty."
