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
    """Extracts the DCJ distance matrix from UNIMOG output."""
    path = Path(file_path)

    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")

    matrix_lines = []
    capture = False

    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()

            if "DCJ distance comparisons:" in line:
                capture = True
                continue

            if capture:
                # Stop if we hit a new section header
                if (
                    stripped
                    and not stripped.startswith("|")
                    and not stripped.startswith("_")
                    and stripped.endswith(":")
                ):
                    break

                if not stripped or stripped.startswith("___"):
                    continue

                if stripped.startswith("|"):
                    matrix_lines.append(stripped)

    if not matrix_lines:
        raise ValueError(
            f"No matrix data found under 'DCJ distance comparisons:' in {path}"
        )

    # Convert table blocks into explicit (row_label, col_label, value) tuples
    records = []
    current_headers = []
    all_rows = set()
    all_cols = set()

    for line in matrix_lines:
        cells = [c.strip() for c in line.split("|")]
        if cells and cells[0] == "":
            cells.pop(0)
        if cells and cells[-1] == "":
            cells.pop()

        if not cells:
            continue

        # Header row check (first cell is empty)
        if cells[0] == "":
            current_headers = cells[1:]
            for col in current_headers:
                all_cols.add(col)
        else:
            row_label = cells[0]
            all_rows.add(row_label)
            values = cells[1:]

            # Map values to current chunk headers
            for col_label, val_str in zip(current_headers, values):
                val = np.nan if val_str == "-" else float(val_str)
                records.append(
                    {"Row": row_label, "Column": col_label, "Value": val}
                )

    # Build DataFrame safely via pivot
    records_df = pd.DataFrame(records)

    # De-duplicate entries if UNIMOG repeated any cell coordinates
    records_df = records_df.drop_duplicates(subset=["Row", "Column"])

    df = records_df.pivot(index="Row", columns="Column", values="Value")

    # Maintain original order of appearance
    ordered_rows = [r for r in all_rows if r in df.index]
    ordered_cols = [c for c in all_cols if c in df.columns]

    return df.reindex(index=ordered_rows, columns=ordered_cols)

def parse_dcj_id_matrix(file_path: Path | str) -> pd.DataFrame:
     """ Extracts the dcj indel matrix from UNIMOG's output text."""
     matrix_lines = []
     capture = False
     with open(file_path, 'r') as f:
          for line in f:
               if "DCJ-indel distance comparisons:" in line:
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
