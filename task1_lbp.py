"""
task1_lbp.py
------------
Task 1 : Classical Retrieval using Local Binary Patterns (LBP)
fi = LBP(Ii)

LBP encodes local texture by comparing each pixel to its circular
neighbours and building a histogram of binary patterns.  It is
translation-invariant and computationally cheap but ignores global
structure and colour.
"""

import numpy as np
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from skimage.transform import resize
from scipy.spatial.distance import cdist


# ── LBP hyper-parameters ──────────────────────────────────────────────────────
N_POINTS   = 24          # neighbours in circular ring
RADIUS     = 3           # ring radius
METHOD     = "uniform"   # 'uniform' LBP → compact histogram
N_BINS     = N_POINTS + 2  # uniform LBP bins = P+2


def extract_lbp(image: np.ndarray) -> np.ndarray:
    """
    Extract normalised LBP histogram from a single image.

    Parameters
    ----------
    image : np.ndarray  shape (H, W, 3) uint8

    Returns
    -------
    hist : np.ndarray  shape (N_BINS,)  float32, sums to 1
    """
    gray = rgb2gray(image).astype(np.float64)
    lbp  = local_binary_pattern(gray, N_POINTS, RADIUS, method=METHOD)
    hist, _ = np.histogram(lbp.ravel(),
                           bins=N_BINS,
                           range=(0, N_BINS),
                           density=True)
    return hist.astype(np.float32)


def extract_lbp_batch(images: np.ndarray) -> np.ndarray:
    """
    Extract LBP features for a batch of images.

    Parameters
    ----------
    images : np.ndarray  shape (N, H, W, 3)

    Returns
    -------
    features : np.ndarray  shape (N, N_BINS)
    """
    return np.stack([extract_lbp(img) for img in images])


def retrieve_lbp(query_feat: np.ndarray,
                 gallery_feats: np.ndarray,
                 gallery_labels: np.ndarray,
                 top_k: int = 5,
                 metric: str = "chi2") -> tuple:
    """
    Retrieve top-K similar images from gallery using LBP features.

    Parameters
    ----------
    query_feat    : (N_BINS,)
    gallery_feats : (N, N_BINS)
    gallery_labels: (N,)
    top_k         : number of results
    metric        : 'chi2' | 'euclidean' | 'cosine' | 'cityblock'

    Returns
    -------
    indices   : (top_k,)  gallery indices
    distances : (top_k,)  similarity scores (lower = more similar)
    """
    if metric == "chi2":
        # Chi-squared distance: good for histograms
        eps = 1e-10
        diff = query_feat[None, :] - gallery_feats
        summ = query_feat[None, :] + gallery_feats + eps
        dists = 0.5 * np.sum(diff ** 2 / summ, axis=1)
    else:
        dists = cdist(query_feat[None, :], gallery_feats, metric=metric)[0]

    idx = np.argsort(dists)[:top_k]
    return idx, dists[idx]
