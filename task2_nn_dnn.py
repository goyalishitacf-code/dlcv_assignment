"""
task2_nn_dnn.py
---------------
Task 2 : Image Retrieval using NN and Deep NN
fi = NN(Ii)   /  fi = DNN(Ii)

A shallow NN (1 hidden layer) and a deeper MLP are trained as
classifiers; the penultimate layer activations are used as feature
vectors for retrieval.

Key insight:
  Learned features ARE better than handcrafted ones on complex data
  because they adapt to the training distribution.  However, plain
  MLPs see pixels as a flat vector — they cannot capture spatial
  hierarchy (translation invariance, edge→shape→object hierarchy).
  CNNs fix this via convolutions.  NN fails where CNNs succeed
  because images of the same class can look very different at the
  pixel level (different positions, lighting, backgrounds).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


# ── helpers ───────────────────────────────────────────────────────────────────

def _preprocess(images: np.ndarray) -> torch.Tensor:
    """Flatten (N,H,W,3) → (N, H*W*3), normalise to [0,1]."""
    N = images.shape[0]
    x = images.reshape(N, -1).astype(np.float32) / 255.0
    return torch.tensor(x)


def _make_loader(x: torch.Tensor, y: torch.Tensor,
                 batch_size: int = 256) -> DataLoader:
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


# ── Model definitions ─────────────────────────────────────────────────────────

class ShallowNN(nn.Module):
    """Single hidden layer MLP."""
    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat), feat


class DeepNN(nn.Module):
    """4-layer MLP — deeper representation but still no spatial awareness."""
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024,  512),  nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,   256),  nn.BatchNorm1d(256),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,   128),  nn.BatchNorm1d(128),  nn.ReLU(),
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat), feat


# ── Training ──────────────────────────────────────────────────────────────────

def _train(model, loader, epochs: int = 10, device: str = "cpu"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (ep + 1) % 5 == 0:
            print(f"  Epoch {ep+1}/{epochs}  loss={total_loss/len(loader):.4f}")
    model.eval()
    return model


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def _extract_features(model, x_tensor: torch.Tensor,
                       device: str = "cpu", batch_size: int = 512) -> np.ndarray:
    model.eval()
    feats = []
    for i in range(0, len(x_tensor), batch_size):
        xb = x_tensor[i:i+batch_size].to(device)
        _, feat = model(xb)
        feats.append(feat.cpu().numpy())
    return np.concatenate(feats)


# ── Public API ────────────────────────────────────────────────────────────────

def train_and_extract_nn(images: np.ndarray,
                          labels: np.ndarray,
                          epochs: int = 15,
                          hidden: int = 512,
                          device: str = "cpu") -> np.ndarray:
    """Train shallow NN classifier and return penultimate-layer features."""
    x = _preprocess(images)
    y = torch.tensor(labels, dtype=torch.long)
    n_classes = len(np.unique(labels))
    in_dim = x.shape[1]
    loader = _make_loader(x, y)
    model = ShallowNN(in_dim, hidden, n_classes)
    print("[NN] Training shallow NN ...")
    _train(model, loader, epochs=epochs, device=device)
    feats = _extract_features(model, x, device=device)
    print(f"[NN] Feature shape: {feats.shape}")
    return feats


def train_and_extract_dnn(images: np.ndarray,
                           labels: np.ndarray,
                           epochs: int = 15,
                           device: str = "cpu") -> np.ndarray:
    """Train deep MLP classifier and return penultimate-layer features."""
    x = _preprocess(images)
    y = torch.tensor(labels, dtype=torch.long)
    n_classes = len(np.unique(labels))
    in_dim = x.shape[1]
    loader = _make_loader(x, y)
    model = DeepNN(in_dim, n_classes)
    print("[DNN] Training deep NN ...")
    _train(model, loader, epochs=epochs, device=device)
    feats = _extract_features(model, x, device=device)
    print(f"[DNN] Feature shape: {feats.shape}")
    return feats


def retrieve_nn(query_feat: np.ndarray,
                gallery_feats: np.ndarray,
                top_k: int = 5) -> tuple:
    """L2 nearest-neighbour retrieval on learned features."""
    from scipy.spatial.distance import cdist
    dists = cdist(query_feat[None, :], gallery_feats, metric="euclidean")[0]
    idx   = np.argsort(dists)[:top_k]
    return idx, dists[idx]
