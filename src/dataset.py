from __future__ import annotations
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.utils import smiles_to_tensors


class OdorDataset(Dataset):
    """
    Loads labelled odor data from a JSON file with the structure:

        {
            "data": [
                {"CC(=O)O": [0.0, 0.8, 0.0, ...]},
                ...
            ]
        }

    Each entry maps one SMILES string to an odor vector of any length —
    6 for the hand-curated basis, 138 for the full OpenPOM dataset.

    Unparseable SMILES (e.g. metals with non-standard valence) are skipped
    with a warning rather than crashing.
    """

    def __init__(self, json_path: str | Path):
        self.samples: list[tuple[str, list[float]]] = []
        self.odor_dim: int = -1

        with open(Path(json_path), "r") as f:
            raw = json.load(f)

        for entry in raw["data"]:
            smiles, odor_vec = next(iter(entry.items()))

            # Infer odor_dim from the first entry, then enforce consistency
            if self.odor_dim == -1:
                self.odor_dim = len(odor_vec)
            elif len(odor_vec) != self.odor_dim:
                raise ValueError(
                    f"Inconsistent odor vector length for '{smiles}': "
                    f"expected {self.odor_dim}, got {len(odor_vec)}"
                )

            self.samples.append((smiles, odor_vec))

        # Validate all SMILES upfront — skip unparseable entries (e.g. metals)
        valid = []
        skipped = []
        for i, (smiles, odor_vec) in enumerate(self.samples):
            try:
                smiles_to_tensors(smiles)
                valid.append((smiles, odor_vec))
            except Exception as e:
                skipped.append((i, smiles, str(e)))

        if skipped:
            print(f"Skipping {len(skipped)} unparseable SMILES:")
            for i, s, e in skipped:
                print(f"  [{i}] {s!r}: {e}")

        self.samples = valid
        print(f"Loaded {len(self.samples)} molecules with {self.odor_dim}-dim odor vectors.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        smiles, odor_vec = self.samples[idx]

        try:
            x, A, L = smiles_to_tensors(smiles)
        except Exception as e:
            raise ValueError(f"Failed to parse SMILES at index {idx}: '{smiles}'") from e

        y = torch.tensor(odor_vec, dtype=torch.float32)

        return x, A, L, y


def collate_single(batch):
    """
    Custom collate function for DataLoader.

    Because every molecule has a different number of atoms the tensors cannot
    be stacked into a single batch tensor. We use batch_size=1 and this
    collate function simply unwraps the single-element list so the training
    loop receives plain tensors rather than lists-of-tensors.
    """
    assert len(batch) == 1, "OdorDataset requires batch_size=1"
    x, A, L, y = batch[0]
    return x, A, L, y