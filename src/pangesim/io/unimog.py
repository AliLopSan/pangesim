"""Exporters for dcj tools (UniMoG)."""
from pathlib import Path

from pangesim import Pangenome

#-----------------------------------------------------
# UNIMOG input
#-----------------------------------------------------

def export_to_unimog(file_path: Path | str, pan: Pangenome) -> None:
     """Write sequences to a fasta file.

     Args:
        file_path: The path to the output fasta file.
        pan: A pangenome.
     """
     file_path = Path(file_path)
     terminator = "|"
     with file_path.open("w") as f:
          for genome in pan.genomes:
               label = str(pan._pangenome_id) + "_" + str(genome._genome_id)
               f.write(f">{label}\n")
               for path in genome.get_path_sequences():
                    as_string = " ".join(str(n) for n in path)
                    f.write(f" {as_string} {terminator}")
               f.write("\n")
