"""
convert_dataset.py
──────────────────
Converts the OpenPOM / GoodScents-Leffingwell CSV dataset into the JSON format
expected by OdorGNN:

    { "data": [ {"SMILES": [0.0, 1.0, 0.0, ...]}, ... ] }

Usage:
    python convert_dataset.py \
        --input  data/curated_GS_LF_merged_4983.csv \
        --output data/data.json

The script also prints the 138 odor label names in order so you can update
ODOR_DIM and ODOR_BASIS in model.py.
"""

import argparse
import json
import csv
from pathlib import Path

# The SMILES column name in the CSV
SMILES_COL = "nonStereoSMILES"

def convert(input_path: Path, output_path: Path):
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # All columns except the SMILES column are odor labels
    all_cols   = reader.fieldnames
    odor_cols  = [c for c in all_cols if c != SMILES_COL]

    print(f"Found {len(odor_cols)} odor descriptors:")
    print(odor_cols)
    print()

    data = []
    skipped = 0

    for row in rows:
        smiles = row[SMILES_COL].strip()
        if not smiles:
            skipped += 1
            continue

        # Convert binary label columns to a list of floats
        odor_vec = []
        for col in odor_cols:
            val = row[col].strip()
            try:
                odor_vec.append(float(val))
            except ValueError:
                odor_vec.append(0.0)

        data.append({smiles: odor_vec})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"data": data}, f)

    print(f"Converted {len(data)} molecules  ({skipped} skipped)  →  {output_path}")
    print()
    print("─" * 60)
    print("Paste this into src/model.py to replace ODOR_BASIS and ODOR_DIM:")
    print("─" * 60)
    print(f"ODOR_BASIS = {odor_cols}")
    print(f"ODOR_DIM   = {len(odor_cols)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/curated_GS_LF_merged_4983.csv")
    parser.add_argument("--output", default="data/data.json")
    args = parser.parse_args()

    convert(Path(args.input), Path(args.output))