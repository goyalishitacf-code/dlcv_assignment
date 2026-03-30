"""
visualize.py
------------
Plotting utilities for the retrieval system:
  - show_retrieval_grid()
  - plot_metrics_comparison()
  - plot_feature_tsne()
  - plot_pr_curve()
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch


DARK_BG  = "#0d0d0d"
CARD_BG  = "#1a1a1a"
ACCENT   = "#00b4ff"
GREEN    = "#00ff88"
RED      = "#ff4444"
ORANGE   = "#ff9500"


def show_retrieval_grid(query_img: np.ndarray,
                        retrieved_imgs: np.ndarray,
                        retrieved_labels: np.ndarray,
                        query_label: int,
                        distances: np.ndarray,
                        class_names: list,
                        method_name: str = "",
                        dataset_name: str = ""):
    """
    Display query + top-K results in a dark-themed grid.
    Green border = correct class,  Red = wrong.
    """
    n = len(retrieved_imgs)
    fig = plt.figure(figsize=(2.5 * (n + 1), 3.5), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(1, n + 1, figure=fig,
                            wspace=0.05, left=0.02, right=0.98)

    # Query
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(query_img)
    ax.set_title(f"QUERY\n{class_names[query_label] if class_names else query_label}",
                 color=ACCENT, fontsize=9, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_edgecolor(ACCENT)
        sp.set_linewidth(3)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Retrieved
    for i in range(n):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(retrieved_imgs[i])
        is_correct  = retrieved_labels[i] == query_label
        border_col  = GREEN if is_correct else RED
        label_str   = class_names[retrieved_labels[i]] if class_names else str(retrieved_labels[i])
        ax.set_title(f"#{i+1} {label_str}\n{distances[i]:.3f}",
                     color="white", fontsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(border_col)
            sp.set_linewidth(2.5)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    title = f"Retrieval Results — {method_name}  |  {dataset_name}"
    fig.suptitle(title, color="white", fontsize=11, y=1.01)
    plt.tight_layout()
    return fig


def plot_metrics_comparison(metrics: dict,
                             dataset_name: str = "",
                             metric_keys: list = None):
    """
    Grouped bar chart comparing all methods across chosen metrics.

    Parameters
    ----------
    metrics     : {method_name: {metric_name: float}}
    dataset_name: title suffix
    metric_keys : which metrics to plot (default: precision, recall, mAP, ndcg)
    """
    if metric_keys is None:
        metric_keys = ["precision", "recall", "mAP", "ndcg"]

    methods = list(metrics.keys())
    n_m     = len(methods)
    n_k     = len(metric_keys)
    x       = np.arange(n_m)
    w       = 0.8 / n_k
    colors  = [ACCENT, GREEN, ORANGE, "#c084fc"]

    fig, ax = plt.subplots(figsize=(max(10, n_m * 1.2), 5), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)

    for i, (key, col) in enumerate(zip(metric_keys, colors)):
        vals = [metrics[m].get(key, 0) for m in methods]
        bars = ax.bar(x + i * w - (n_k - 1) * w / 2, vals, w,
                      label=key, color=col, alpha=0.85, edgecolor="none")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    color="white", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", color="white", fontsize=10)
    ax.set_ylabel("Score", color="white")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Method Comparison — {dataset_name}", color="white", fontsize=13)
    ax.legend(facecolor="#2a2a2a", labelcolor="white", framealpha=0.8)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#444")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#555")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_feature_tsne(features: np.ndarray,
                       labels: np.ndarray,
                       class_names: list,
                       title: str = "t-SNE Feature Visualisation",
                       n_samples: int = 1000,
                       seed: int = 42):
    """
    2-D t-SNE scatter plot of feature space (colour-coded by class).
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    rng  = np.random.default_rng(seed)
    idx  = rng.choice(len(features), size=min(n_samples, len(features)), replace=False)
    feat = features[idx]
    lbl  = labels[idx]

    # PCA pre-reduction for speed
    if feat.shape[1] > 50:
        feat = PCA(n_components=50, random_state=seed).fit_transform(feat)

    print(f"[t-SNE] Running on {len(feat)} samples ...")
    embed = TSNE(n_components=2, perplexity=30, random_state=seed).fit_transform(feat)

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)

    cmap   = plt.cm.get_cmap("tab10", len(class_names))
    unique = np.unique(lbl)

    for cls in unique:
        mask = lbl == cls
        name = class_names[cls] if class_names else str(cls)
        ax.scatter(embed[mask, 0], embed[mask, 1],
                   c=[cmap(cls)], label=name,
                   s=18, alpha=0.75, edgecolors="none")

    ax.set_title(title, color="white", fontsize=13)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#444")
    ax.legend(facecolor="#2a2a2a", labelcolor="white",
              fontsize=8, markerscale=2, framealpha=0.8)
    fig.tight_layout()
    return fig


def plot_pr_curve(features: np.ndarray,
                   labels: np.ndarray,
                   n_queries: int = 50,
                   method_name: str = "",
                   seed: int = 42):
    """
    Precision-Recall curve (interpolated) for a set of queries.
    """
    from scipy.spatial.distance import cdist

    rng    = np.random.default_rng(seed)
    q_idxs = rng.choice(len(labels), size=n_queries, replace=False)

    all_prec = np.zeros(len(labels) - 1)
    all_rec  = np.zeros(len(labels) - 1)

    for qi in q_idxs:
        q_feat  = features[qi]
        q_label = labels[qi]
        dists   = cdist(q_feat[None, :], features, metric="cosine")[0]
        dists[qi] = np.inf
        ranked  = np.argsort(dists)
        rels    = (labels[ranked] == q_label).astype(float)
        total_r = rels.sum()
        precs   = np.cumsum(rels) / (np.arange(len(rels)) + 1)
        recs    = np.cumsum(rels) / max(total_r, 1)
        all_prec += precs[:len(all_prec)]
        all_rec  += recs[:len(all_rec)]

    all_prec /= n_queries
    all_rec  /= n_queries

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.plot(all_rec, all_prec, color=ACCENT, lw=2, label=method_name)
    ax.fill_between(all_rec, all_prec, alpha=0.15, color=ACCENT)
    ax.set_xlabel("Recall",    color="white")
    ax.set_ylabel("Precision", color="white")
    ax.set_title(f"Precision-Recall Curve — {method_name}", color="white", fontsize=13)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#444")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.3, color="#555")
    ax.legend(facecolor="#2a2a2a", labelcolor="white")
    fig.tight_layout()
    return fig
