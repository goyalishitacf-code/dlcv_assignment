"""
task5_hybrid.py
---------------
Task 5 : Hybrid Retrieval Model
fi = [fCNN || fProposed]          (concatenation)
fi = λ·fCNN + (1-λ)·fProposed    (weighted sum)

The idea: CNN captures semantic (high-level) cues; MSFTA captures
texture/frequency (mid/low-level) cues.  Fusion provides complementary
information that neither method alone can achieve.

We evaluate both fusion strategies and also learn an optimal λ via
cross-validated precision maximisation.
"""

import numpy as np
from scipy.spatial.distance import cdist


def fuse_concat(cnn_feats: np.ndarray,
                novel_feats: np.ndarray) -> np.ndarray:
    """
    Concatenation fusion: fi = [fCNN || fProposed]
    Both feature sets are L2-normalised before concatenation so that
    neither dominates purely due to scale.

    Parameters
    ----------
    cnn_feats   : (N, D1) float32
    novel_feats : (N, D2) float32

    Returns
    -------
    fused : (N, D1+D2) float32
    """
    def l2(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10
        return x / norms

    fused = np.concatenate([l2(cnn_feats), l2(novel_feats)], axis=1)
    print(f"[Hybrid-Concat] Fused feature shape: {fused.shape}")
    return fused.astype(np.float32)


def fuse_weighted(cnn_feats: np.ndarray,
                  novel_feats: np.ndarray,
                  lam: float = 0.6) -> np.ndarray:
    """
    Weighted sum fusion: fi = λ·fCNN + (1-λ)·fProposed
    Both inputs are projected to the SAME dimension via PCA if they differ.

    Parameters
    ----------
    cnn_feats   : (N, D1)
    novel_feats : (N, D2)
    lam         : weight for CNN (0.6 → CNN slightly dominates)

    Returns
    -------
    fused : (N, min(D1,D2)) float32
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    d1, d2 = cnn_feats.shape[1], novel_feats.shape[1]
    target_d = min(d1, d2)

    if d1 != target_d:
        cnn_feats = PCA(n_components=target_d).fit_transform(cnn_feats)
    if d2 != target_d:
        novel_feats = PCA(n_components=target_d).fit_transform(novel_feats)

    cnn_n   = normalize(cnn_feats,   norm="l2")
    novel_n = normalize(novel_feats, norm="l2")

    fused = lam * cnn_n + (1 - lam) * novel_n
    print(f"[Hybrid-Weighted λ={lam}] Fused feature shape: {fused.shape}")
    return fused.astype(np.float32)


def retrieve_hybrid(query_feat: np.ndarray,
                    gallery_feats: np.ndarray,
                    top_k: int = 5,
                    metric: str = "cosine") -> tuple:
    """Retrieve top-K from fused gallery features."""
    dists = cdist(query_feat[None, :], gallery_feats, metric=metric)[0]
    idx   = np.argsort(dists)[:top_k]
    return idx, dists[idx]


def find_optimal_lambda(cnn_feats: np.ndarray,
                        novel_feats: np.ndarray,
                        labels: np.ndarray,
                        n_queries: int = 100,
                        top_k: int = 10) -> float:
    """
    Grid-search over λ ∈ {0.1, 0.2, …, 0.9} to maximise mean precision@K.
    Uses a random subset of images as queries against the full gallery.

    Returns
    -------
    best_lam : float
    """
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA

    d1, d2 = cnn_feats.shape[1], novel_feats.shape[1]
    target_d = min(d1, d2)

    cnn_p = PCA(n_components=target_d).fit_transform(cnn_feats) if d1 != target_d else cnn_feats
    nov_p = PCA(n_components=target_d).fit_transform(novel_feats) if d2 != target_d else novel_feats

    cnn_n = normalize(cnn_p, norm="l2")
    nov_n = normalize(nov_p, norm="l2")

    rng = np.random.default_rng(42)
    q_idx = rng.choice(len(labels), size=n_queries, replace=False)

    best_lam, best_prec = 0.5, -1.0
    for lam in np.arange(0.1, 1.0, 0.1):
        fused = lam * cnn_n + (1 - lam) * nov_n
        prec_vals = []
        for qi in q_idx:
            qf    = fused[qi]
            qlab  = labels[qi]
            dists = cdist(qf[None, :], fused, metric="cosine")[0]
            dists[qi] = np.inf     # exclude self
            ranked = np.argsort(dists)[:top_k]
            prec   = np.mean(labels[ranked] == qlab)
            prec_vals.append(prec)
        mean_prec = np.mean(prec_vals)
        print(f"  λ={lam:.1f}  mean_precision@{top_k}={mean_prec:.4f}")
        if mean_prec > best_prec:
            best_prec = mean_prec
            best_lam  = float(lam)

    print(f"[Lambda Search] Best λ={best_lam:.1f}  precision={best_prec:.4f}")
    return best_lam
