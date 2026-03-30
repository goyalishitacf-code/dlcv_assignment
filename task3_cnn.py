"""
task3_cnn.py
------------
Task 3 : CNN-based Image Retrieval
fi = CNN(Ii)

Uses a pretrained ResNet-18 (ImageNet weights) as a feature extractor.
The global-average-pooled representation (512-d) captures hierarchical
spatial features without retraining.

Advantages over NN / DNN:
  • Spatial feature learning : convolutional kernels slide over the image
    → position-invariant edge / texture / shape detectors.
  • Robustness : pooling layers provide translation tolerance; deeper
    layers learn semantic concepts.
  • Transfer generalisation : ImageNet features transfer well to CIFAR /
    MNIST even without fine-tuning.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial.distance import cdist


# ── Preprocessing ─────────────────────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _preprocess_batch(images: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 → (N, 3, 64, 64) normalised tensor."""
    return torch.stack([_transform(img) for img in images])


# ── Model ─────────────────────────────────────────────────────────────────────

def build_cnn_extractor(device: str = "cpu") -> nn.Module:
    """
    ResNet-18 backbone truncated before the final FC layer.
    Output: 512-dimensional feature vector per image.
    """
    base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Remove final classification head
    extractor = nn.Sequential(*list(base.children())[:-1])   # → (N, 512, 1, 1)
    extractor.eval()
    extractor.to(device)
    for p in extractor.parameters():
        p.requires_grad = False
    return extractor


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_cnn_features(images: np.ndarray,
                          device: str = "cpu",
                          batch_size: int = 128) -> np.ndarray:
    """
    Extract CNN features from a batch of images.

    Parameters
    ----------
    images    : (N, H, W, 3) uint8
    device    : 'cpu' or 'cuda'
    batch_size: processing chunk size

    Returns
    -------
    features  : (N, 512) float32
    """
    extractor = build_cnn_extractor(device)
    feats = []
    print("[CNN] Extracting ResNet-18 features ...")
    for i in range(0, len(images), batch_size):
        batch = _preprocess_batch(images[i:i+batch_size]).to(device)
        out   = extractor(batch)           # (B, 512, 1, 1)
        out   = out.squeeze(-1).squeeze(-1)  # (B, 512)
        feats.append(out.cpu().numpy())
    feats = np.concatenate(feats)
    print(f"[CNN] Feature shape: {feats.shape}")
    return feats.astype(np.float32)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_cnn(query_feat: np.ndarray,
                 gallery_feats: np.ndarray,
                 top_k: int = 5,
                 metric: str = "cosine") -> tuple:
    """
    Retrieve top-K images using cosine similarity on CNN features.

    Returns
    -------
    indices   : (top_k,)
    distances : (top_k,)  cosine distance (0=identical, 2=opposite)
    """
    dists = cdist(query_feat[None, :], gallery_feats, metric=metric)[0]
    idx   = np.argsort(dists)[:top_k]
    return idx, dists[idx]
