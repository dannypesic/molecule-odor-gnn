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
                {"CC(=O)O": [0.0, 0.8, 0.0, 0.2, 0.0, 0.0]},
                ...
            ]
        }

    Each entry maps one SMILES string to a 6-float odor vector:
        [flowery, fruity, spicy, resinous, burnt, putrid]

    SMILES → (x, A, L) conversion is done lazily on __getitem__ so startup
    is fast even for large datasets.  Precomputing and caching the tensors
    is straightforward if needed later.
    """

    ODOR_LABELS = ["flowery", "fruity", "spicy", "resinous", "burnt", "putrid"]

    def __init__(self, json_path: str | Path):
        self.samples: list[tuple[str, list[float]]] = []

        with open(Path(json_path), "r") as f:
            raw = json.load(f)

        for entry in raw["data"]:
            smiles, odor_vec = next(iter(entry.items()))
            if len(odor_vec) != 6:
                raise ValueError(
                    f"Expected 6-element odor vector for '{smiles}', "
                    f"got {len(odor_vec)}"
                )
            self.samples.append((smiles, odor_vec))

        bad = []
        for i, (smiles, _) in enumerate(self.samples):
            try:
                smiles_to_tensors(smiles)
            except Exception as e:
                bad.append((i, smiles, str(e)))

        if bad:
            msg = "\n".join(f"  [{i}] {s!r}: {e}" for i, s, e in bad)
            raise ValueError(f"Dataset contains {len(bad)} unparseable SMILES:\n{msg}")
            

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        smiles, odor_vec = self.samples[idx]

        try:
            x, A, L = smiles_to_tensors(smiles)
        except Exception as e:
            raise ValueError(f"Failed to parse SMILES at index {idx}: '{smiles}'") from e

        y = torch.tensor(odor_vec, dtype=torch.float32)  # [6]

        return x, A, L, y


def collate_single(batch):
    """
    Custom collate function for DataLoader.

    Because every molecule has a different number of atoms the tensors cannot
    be stacked into a single batch tensor.  We use batch_size=1 and this
    collate function simply unwraps the single-element list so the training
    loop receives plain tensors rather than lists-of-tensors.
    """
    assert len(batch) == 1, "OdorDataset requires batch_size=1"
    x, A, L, y = batch[0]
    return x, A, L, y