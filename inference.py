import torch
from src.utils import smiles_to_tensors
from src.model import OdorGNN
import csv

# Load label names directly from the CSV
with open("data/curated_GS_LF_merged_4983.csv") as f:
    headers = next(csv.reader(f))
labels = [h for h in headers if h != "nonStereoSMILES"]

model = OdorGNN(output_dim=len(labels))
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

molecules = [
    ("Geraniol",     "CC(C)=CCC/C(C)=C/CO"),
    ("Limonene",     "C=C(C)C1CCC(=CC1)C"),
    ("Allicin",      "C=CCS(=O)SCC=C"),
    ("Hedione",      "COC(=O)CC1CCC(=O)C1CCCC"),
    ("Butyric acid", "CCCC(=O)O")
]

for name, smiles in molecules:
    print(f"\n{name} — {smiles}")
    print("-" * 40)
    try:
        x, A, L = smiles_to_tensors(smiles)
        with torch.no_grad():
            probs = torch.sigmoid(model(x, L, A))
        results = sorted(zip(labels, probs.tolist()), key=lambda t: t[1], reverse=True)
        for label, prob in results:
            if prob > 0.15:
                print(f"  {label:<20} {prob:.3f}")
    except Exception as e:
        print(f"  Error: {e}")