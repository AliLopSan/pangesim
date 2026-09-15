import textwrap
from pathlib import Path

import pytest
from tralda.datastructures.doubly_linked import DLList
from tralda.datastructures.doubly_linked import DLListNode

from pangesim import Genome
from pangesim import Pangenome
from pangesim.io.unimog import export_to_unimog


@pytest.fixture
def sample_pangenome() -> Pangenome:
    """Fixture providing a deterministic two-genome Pangenome instance."""
    g1 = Genome(genome_id=0)
    g2 = Genome(genome_id=1)

    # Genome 0 path: 1 -> 2
    p1 = DLList()
    p1.append(DLListNode(value=1))
    p1.append(DLListNode(value=2))
    g1.add_path(p1)

    # Genome 1 path: 1 -> 2 -> 3
    p2 = DLList()
    p2.append(DLListNode(value=1))
    p2.append(DLListNode(value=2))
    p2.append(DLListNode(value=3))
    g2.add_path(p2)

    pangenome = Pangenome(pangenome_id="test")
    pangenome.add_genome(g1)
    pangenome.add_genome(g2)

    return pangenome

def test_export_to_unimog_format(tmp_path: Path, sample_pangenome: Pangenome) -> None:
    """Validates export_to_unimog writes the exact UniMoG format expected for linear chromosomes."""
    # Arrange
    output_file = tmp_path / "output.unimog"

    # Act
    export_to_unimog(output_file, sample_pangenome)

    # Assert
    assert output_file.exists(), "Output file was not created."

    expected_content = textwrap.dedent(
        """\
        >test_0
         1 2 |
        >test_1
         1 2 3 |
        """
    )

    actual_content = output_file.read_text(encoding="utf-8")
    assert actual_content == expected_content

