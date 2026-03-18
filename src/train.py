from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import csv

from src.dataset import OdorDataset, collate_single
from src.model import OdorGNN, ODOR_BASIS

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH      = Path(__file__).parent.parent / "data" / "data.json"
EPOCHS         = 100
LR             = 1e-3
HIDDEN_DIM     = 32
CONV_CHANNELS  = 64
NUM_GNN_LAYERS = 2
POLY_DEGREE    = 2
VAL_SPLIT      = 0.2
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)

# ── Data ───────────────────────────────────────────────────────────────────────
dataset    = OdorDataset(DATA_PATH)
val_size   = max(1, int(len(dataset) * VAL_SPLIT))
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])

# batch_size=1 because molecules differ in atom count — collate_single unwraps
train_loader = DataLoader(
    train_set, batch_size=1, shuffle=True,  collate_fn=collate_single
)
val_loader = DataLoader(
    val_set,   batch_size=1, shuffle=False, collate_fn=collate_single
)

print(f"Train: {train_size} molecules   Val: {val_size} molecules")

# ── Model ──────────────────────────────────────────────────────────────────────
# ODOR_DIM is read from the dataset so it works for both 6-label and 138-label data
model = OdorGNN(
    hidden_dim=HIDDEN_DIM,
    num_gnn_layers=NUM_GNN_LAYERS,
    poly_degree=POLY_DEGREE,
    conv_channels=CONV_CHANNELS,
    output_dim=dataset.odor_dim,
).to(DEVICE)

print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

# BCEWithLogitsLoss is correct for multilabel binary classification
# (each odor label is independently 0 or 1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)

# ── Training loop ──────────────────────────────────────────────────────────────
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):

    # Train
    model.train()
    train_loss = 0.0
    for x, A, L, y in train_loader:
        x, A, L, y = x.to(DEVICE), A.to(DEVICE), L.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        pred = model(x, L, A)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validate
    if epoch % 1 == 0 or epoch == EPOCHS:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, A, L, y in val_loader:
                x, A, L, y = x.to(DEVICE), A.to(DEVICE), L.to(DEVICE), y.to(DEVICE)
                pred      = model(x, L, A)
                val_loss += criterion(pred, y).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        flag = "  ← best" if val_loss < best_val_loss else ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

        print(
            f"Epoch {epoch:4d}/{EPOCHS}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}{flag}"
        )

print(f"\nBest val loss: {best_val_loss:.4f}  (saved to best_model.pth)")

# ── Sample predictions ─────────────────────────────────────────────────────────
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

with open(DATA_PATH.parent / "../data/curated_GS_LF_merged_4983.csv") as f:
    headers = next(csv.reader(f))
labels = [h for h in headers if h != "nonStereoSMILES"]

print("\nSample predictions on validation set (showing labels with pred > 0.5):")
print("-" * 60)

with torch.no_grad():
    for i, (x, A, L, y) in enumerate(val_loader):
        if i >= 5:
            break
        x, A, L = x.to(DEVICE), A.to(DEVICE), L.to(DEVICE)
        logits = model(x, L, A).cpu()
        probs  = torch.sigmoid(logits)

        smiles = val_set.dataset.samples[val_set.indices[i]][0]
        pred_labels = [labels[j] for j, p in enumerate(probs) if p > 0.5]
        true_labels = [labels[j] for j, v in enumerate(y.squeeze()) if v > 0.5]

        print(f"SMILES: {smiles[:50]}")
        print(f"  pred: {pred_labels}")
        print(f"  true: {true_labels}")
        print()