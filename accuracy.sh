#!/usr/bin/env bash
# Per-label (Hamming) accuracy of best_model.pth on 10 molecules verified absent
# from the training set.
#
# ./accuracy.sh
# THRESHOLD=0.3 ./accuracy.sh
set -euo pipefail

cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate

THRESHOLD="${THRESHOLD:-0.5}" python - <<'PY'
import csv, os, warnings
warnings.filterwarnings("ignore")
import torch
from src.utils import smiles_to_tensors
from src.model import OdorGNN

THRESH = float(os.environ["THRESHOLD"])

with open("data/curated_GS_LF_merged_4983.csv") as f:
    labels = [h for h in next(csv.reader(f)) if h != "nonStereoSMILES"]

model = OdorGNN(output_dim=len(labels))
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

TEST = [
    ("Muscone",              "CC1CCCCCCCCCCCCC1=O",             ["musk", "powdery", "animal", "sweet"]),
    ("Ambrettolide",         "O=C1CCCCCCC=CCCCCCCO1",           ["musk", "sweet", "fatty"]),
    ("Habanolide",           "O=C1CCCCCCCC=CCCCCO1",            ["musk", "sweet", "powdery"]),
    ("Musk ketone",          "CC(C)(C)c1c(C)c(N(=O)=O)c(C(C)=O)c(C)c1N(=O)=O", ["musk", "sweet", "powdery"]),
    ("cis-Jasmone",          "CC=CCC1=C(C)CCC1=O",              ["jasmin", "floral", "herbal", "woody"]),
    ("Nootkatone",           "CC1CC(=O)C=C2C1(CCC(C2)C(=C)C)C", ["grapefruit", "citrus", "woody"]),
    ("Geosmin",              "CC12CCCC(C)(O)C1CCCC2",           ["earthy", "musty", "mushroom"]),
    ("Dihydromyrcenol",      "CC(C)=CCCC(C)(C)O",               ["citrus", "fresh", "floral"]),
    ("2-Acetyl-1-pyrroline", "CC(=O)C1=NCCC1",                  ["popcorn", "roasted", "nutty"]),
    ("alpha-Phellandrene",   "CC(C)C1CC=CC(C)=C1",              ["citrus", "terpenic", "herbal", "green"]),
]

correct = total = 0
for _, smiles, truth in TEST:
    x, A, L = smiles_to_tensors(smiles)
    with torch.no_grad():
        probs = torch.sigmoid(model(x, L, A)).tolist()
    truth = set(truth)
    for label, p in zip(labels, probs):
        correct += (p >= THRESH) == (label in truth)
        total   += 1

acc = 100 * correct / total
print(f"Accuracy: {acc:.1f}%; threshold {THRESH}")
PY
