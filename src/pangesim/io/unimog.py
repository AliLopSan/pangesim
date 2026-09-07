"""Exporters for dcj tools (UniMoG)."""
import pandas as pd
import numpy as np

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

#--------------------------------------------------------------
# UNIMOG output
#--------------------------------------------------------------
def parse_dcj_matrix(file_path: Path | str) -> pd.DataFrame:
     """ Extracts the dcj matrix from UNIMOG's output text."""
     matrix_lines = []
     capture = False
     with open(file_path, 'r') as f:
          for line in f:
               if "DCJ distance comparisons:" in line:
                    capture = True
                    continue
               if capture:
                    # Stop if we hit an empty line or a non-table divider line
                    stripped = line.strip()
                    if not stripped or stripped.startswith("___"):
                         continue
                    if line.startswith("|") or "\t" in line or "simulated_" in line:
                         matrix_lines.append(stripped)
                    elif len(matrix_lines) > 0 and not line.startswith("|"):
                         break
     # Clean the raw pipe-separated lines
     cleaned_data = []
     for line in matrix_lines:
          # Split by '|' and strip whitespace off each cell
          cells = [c.strip() for c in line.split('|')]
          if cells and cells[0] == '':
               cells.pop(0)
          if cells and cells[-1] == '':
               cells.pop()
          cleaned_data.append(cells)
          
     if not cleaned_data:
          raise ValueError("No matrix data found under 'DCJ distance comparisons:'")
     headers = cleaned_data[0][1:]
     row_labels = []
     matrix_values = []
     for row in cleaned_data[1:]:
          if len(row) < 2:
               continue
          row_labels.append(row[0])
          matrix_values.append([float(val) if val != '-' else np.nan for val in row[1:]])
     df = pd.DataFrame(matrix_values, index=row_labels, columns=headers)
     return df
