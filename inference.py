import torch
from src.utils import smiles_to_tensors
from src.model import OdorGNN

model = OdorGNN()
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

x, A, L = smiles_to_tensors("CC(C)=CCC/C(C)=C/CO")  # geraniol
print(model.predict_named(x, L, A))