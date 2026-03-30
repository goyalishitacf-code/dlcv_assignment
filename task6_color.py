"""
task6_color.py
--------------
Task 6 : Inter- and Intra-Channel Color Feature Analysis

Intra-channel features capture the distribution of individual colour channels:
    fintra = [μR, σR, skewR, μG, σG, skewG, μB, σB, skewB,
              μH, σH, μS, σS, μV, σV]   (15-d)

Inter-channel features capture cross-channel relationships:
    finter = [corr(R,G), corr(G,B), corr(R,B),
              EMD(hist_R, hist_G), EMD(hist_G, hist_B)]   (5-d)

We also build a richer 64-bin histogram per channel and combine all into
a 192+15+5 = 212-d colour descriptor, then compare colour-only retrieval
against LBP-fused and CNN-fused variants.
"""

import numpy as np
from scipy.stats import skew, pearsonr, wasserstein_distance
from scipy.spatial.distance import cdist


# ── Intra-channel statistics ──────────────────────────────────────────────────

def _channel_stats(channel: np.ndarray) -> np.ndarray:
    """Mean, std, skewness of a single channel (H,W) uint8."""
    flat = channel.ravel().astype(np.float64)
    return np.array([flat.mean(), flat.std(), skew(flat)], dtype=np.float32)


def _hsv_stats(image: np.ndarray) -> np.ndarray:
    """HSV mean and std (H,S,V each) — 6-d."""
    import cv2
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float64)
    stats = []
    for c in range(3):
        ch = hsv[:, :, c].ravel()
        stats.extend([ch.mean(), ch.std()])
    return np.array(stats, dtype=np.float32)


def extract_intra(image: np.ndarray) -> np.ndarray:
    """
    Intra-channel colour features.

    Parameters
    ----------
    image : (H, W, 3) uint8

    Returns
    -------
    feat : (15,) float32  [μ,σ,skew per RGB channel + μ,σ per HSV channel]
    """
    rgb_stats = np.concatenate([_channel_stats(image[:, :, c]) for c in range(3)])
    hsv_stats = _hsv_stats(image)
    return np.concatenate([rgb_stats, hsv_stats]).astype(np.float32)


# ── Inter-channel statistics ──────────────────────────────────────────────────

def _channel_hist(channel: np.ndarray, bins: int = 64) -> np.ndarray:
    hist, _ = np.histogram(channel.ravel(), bins=bins, range=(0, 255), density=True)
    return hist.astype(np.float32)


def extract_inter(image: np.ndarray) -> np.ndarray:
    """
    Inter-channel colour features.

    Returns
    -------
    feat : (5,) float32
        [corr(R,G), corr(G,B), corr(R,B), EMD(H_R,H_G), EMD(H_G,H_B)]
    """
    channels = [image[:, :, c].ravel().astype(np.float64) for c in range(3)]
    corr_rg, _ = pearsonr(channels[0], channels[1])
    corr_gb, _ = pearsonr(channels[1], channels[2])
    corr_rb, _ = pearsonr(channels[0], channels[2])

    hists = [_channel_hist(image[:, :, c]) for c in range(3)]
    # Wasserstein (Earth Mover) distance between histograms
    emd_rg = wasserstein_distance(hists[0], hists[1])
    emd_gb = wasserstein_distance(hists[1], hists[2])

    feat = np.array([
        corr_rg if np.isfinite(corr_rg) else 0.0,
        corr_gb if np.isfinite(corr_gb) else 0.0,
        corr_rb if np.isfinite(corr_rb) else 0.0,
        emd_rg, emd_gb,
    ], dtype=np.float32)
    return feat


# ── Rich histogram descriptor ─────────────────────────────────────────────────

def extract_color_hist(image: np.ndarray, bins: int = 64) -> np.ndarray:
    """64-bin normalised histogram per RGB channel → 192-d."""
    hists = [_channel_hist(image[:, :, c], bins) for c in range(3)]
    return np.concatenate(hists).astype(np.float32)


# ── Full colour descriptor ────────────────────────────────────────────────────

def extract_color_full(image: np.ndarray) -> np.ndarray:
    """
    Combine intra + inter + histogram → 212-d colour descriptor.
    L2-normalised.
    """
    feat = np.concatenate([
        extract_intra(image),       # 15
        extract_inter(image),       #  5
        extract_color_hist(image),  # 192
    ])
    norm = np.linalg.norm(feat) + 1e-10
    return (feat / norm).astype(np.float32)


def extract_color_batch(images: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) → (N, 212) colour features."""
    feats = []
    print("[Color] Extracting colour features ...")
    for img in images:
        feats.append(extract_color_full(img))
    feats = np.stack(feats)
    print(f"[Color] Feature shape: {feats.shape}")
    return feats


# ── Fused variants ────────────────────────────────────────────────────────────

def fuse_color_lbp(color_feats: np.ndarray, lbp_feats: np.ndarray) -> np.ndarray:
    """L2-normalise and concatenate colour + LBP."""
    def l2(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
    return np.concatenate([l2(color_feats), l2(lbp_feats)], axis=1).astype(np.float32)


def fuse_color_cnn(color_feats: np.ndarray, cnn_feats: np.ndarray) -> np.ndarray:
    """L2-normalise and concatenate colour + CNN."""
    def l2(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
    return np.concatenate([l2(color_feats), l2(cnn_feats)], axis=1).astype(np.float32)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_color(query_feat: np.ndarray,
                   gallery_feats: np.ndarray,
                   top_k: int = 5) -> tuple:
    dists = cdist(query_feat[None, :], gallery_feats, metric="cosine")[0]
    idx   = np.argsort(dists)[:top_k]
    return idx, dists[idx]
