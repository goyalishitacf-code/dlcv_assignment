"""
metrics.py
----------
Evaluation metrics for image retrieval:
  - Precision@K
  - Recall@K
  - Average Precision (AP)
  - Mean Average Precision (mAP)
  - Retrieval ranking quality (NDCG@K)
"""

import numpy as np
from scipy.spatial.distance import cdist


def precision_at_k(retrieved_labels: np.ndarray,
                   query_label: int,
                   k: int) -> float:
    """
    Precision@K = (# relevant in top-K) / K
    """
    top_k = retrieved_labels[:k]
    return np.sum(top_k == query_label) / k


def recall_at_k(retrieved_labels: np.ndarray,
                query_label: int,
                total_relevant: int,
                k: int) -> float:
    """
    Recall@K = (# relevant in top-K) / total_relevant_in_gallery
    """
    top_k = retrieved_labels[:k]
    return np.sum(top_k == query_label) / max(total_relevant, 1)


def average_precision(retrieved_labels: np.ndarray,
                      query_label: int) -> float:
    """
    Average Precision (AP) for a single query.
    AP = (1/R) Σ_{k=1}^{N} P(k) · rel(k)
    where R = total relevant docs, rel(k) = 1 if result k is relevant.
    """
    hits   = (retrieved_labels == query_label).astype(float)
    n_rel  = hits.sum()
    if n_rel == 0:
        return 0.0
    cumhits = np.cumsum(hits)
    ranks   = np.arange(1, len(retrieved_labels) + 1)
    prec_at_k = cumhits / ranks
    return float((prec_at_k * hits).sum() / n_rel)


def ndcg_at_k(retrieved_labels: np.ndarray,
              query_label: int,
              k: int) -> float:
    """
    Normalised Discounted Cumulative Gain @ K.
    Binary relevance: rel = 1 if label matches.
    """
    top_k = retrieved_labels[:k]
    rels  = (top_k == query_label).astype(float)
    dcg   = np.sum(rels / np.log2(np.arange(2, k + 2)))
    # Ideal DCG: all top-k slots relevant
    n_ideal = min(k, int(rels.sum()) + int((retrieved_labels == query_label).sum()))
    ideal   = np.sum(np.ones(min(n_ideal, k)) / np.log2(np.arange(2, min(n_ideal, k) + 2)))
    return float(dcg / ideal) if ideal > 0 else 0.0


def evaluate_retrieval(gallery_feats: np.ndarray,
                       gallery_labels: np.ndarray,
                       top_k: int = 10,
                       n_queries: int = 200,
                       metric: str = "cosine",
                       seed: int = 42) -> dict:
    """
    Full evaluation on a random subset of queries.

    Parameters
    ----------
    gallery_feats  : (N, D) feature matrix
    gallery_labels : (N,)   integer class labels
    top_k          : K for precision/recall/NDCG
    n_queries      : how many random queries to evaluate
    metric         : distance metric

    Returns
    -------
    results : dict with keys precision, recall, mAP, ndcg (all floats)
    """
    rng     = np.random.default_rng(seed)
    n       = len(gallery_labels)
    q_idxs  = rng.choice(n, size=min(n_queries, n), replace=False)

    class_counts = {c: int(np.sum(gallery_labels == c))
                    for c in np.unique(gallery_labels)}

    precs, recs, aps, ndcgs = [], [], [], []

    for qi in q_idxs:
        q_feat  = gallery_feats[qi]
        q_label = gallery_labels[qi]

        dists = cdist(q_feat[None, :], gallery_feats, metric=metric)[0]
        dists[qi] = np.inf    # exclude self

        ranked_idx    = np.argsort(dists)
        ranked_labels = gallery_labels[ranked_idx]

        total_rel = class_counts[q_label] - 1   # exclude query itself

        precs.append(precision_at_k(ranked_labels, q_label, top_k))
        recs.append(recall_at_k(ranked_labels, q_label, total_rel, top_k))
        aps.append(average_precision(ranked_labels, q_label))
        ndcgs.append(ndcg_at_k(ranked_labels, q_label, top_k))

    return {
        "precision": float(np.mean(precs)),
        "recall":    float(np.mean(recs)),
        "mAP":       float(np.mean(aps)),
        "ndcg":      float(np.mean(ndcgs)),
    }


def print_results_table(all_results: dict):
    """Pretty-print comparative results table."""
    header = f"{'Method':<20} {'Precision':>10} {'Recall':>10} {'mAP':>10} {'NDCG':>10}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for method, res in all_results.items():
        if res is None:
            print(f"{method:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        else:
            print(f"{method:<20} {res['precision']:>10.4f} {res['recall']:>10.4f}"
                  f" {res['mAP']:>10.4f} {res['ndcg']:>10.4f}")
    print(sep + "\n")
