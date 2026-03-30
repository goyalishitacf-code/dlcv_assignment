"""
run_all.py
----------
Master pipeline script — runs all 7 tasks end-to-end.

Usage
-----
    python run_all.py              # full run (5000 images/dataset)
    python run_all.py --quick      # 500 images for fast testing (Colab)
    python run_all.py --dashboard  # launch dashboard after pipeline

Colab example
-------------
    !python run_all.py --quick
    # then
    !python task7_dashboard.py
"""

import argparse
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--quick",     action="store_true",
                    help="Use only 500 images per dataset (fast debug)")
parser.add_argument("--dashboard", action="store_true",
                    help="Launch Gradio dashboard after pipeline completes")
parser.add_argument("--epochs",    type=int, default=10,
                    help="Training epochs for NN/DNN (default 10)")
parser.add_argument("--topk",      type=int, default=10,
                    help="K for evaluation metrics (default 10)")
args = parser.parse_args()

MAX_SAMPLES = 500 if args.quick else 5000
EPOCHS      = args.epochs
TOP_K       = args.topk
DEVICE      = "cuda" if __import__("torch").cuda.is_available() else "cpu"
STORE_PATH  = "./feature_store.pkl"

print(f"\n{'='*60}")
print(f"  Image Retrieval Pipeline")
print(f"  MAX_SAMPLES={MAX_SAMPLES}  EPOCHS={EPOCHS}  DEVICE={DEVICE}")
print(f"{'='*60}\n")

# ── 1. Load datasets ──────────────────────────────────────────────────────────

from dataset_loader import load_all
print("[1/8] Loading datasets ...")
datasets = load_all(root="./data", max_samples=MAX_SAMPLES)
# datasets = {name: (images, labels, class_names)}

IMAGE_STORE   = {k: v[0] for k, v in datasets.items()}
LABEL_STORE   = {k: v[1] for k, v in datasets.items()}
CLASS_STORE   = {k: v[2] for k, v in datasets.items()}
FEATURE_STORE = {k: {}   for k in datasets}
METRICS_STORE = {k: {}   for k in datasets}

# ── 2. Task 1 — LBP ──────────────────────────────────────────────────────────

from task1_lbp import extract_lbp_batch
print("\n[2/8] Task 1 : LBP features ...")
for name, (imgs, labels, _) in datasets.items():
    feats = extract_lbp_batch(imgs)
    FEATURE_STORE[name]["lbp"] = feats

# ── 3. Task 2 — NN & DNN ─────────────────────────────────────────────────────

from task2_nn_dnn import train_and_extract_nn, train_and_extract_dnn
print("\n[3/8] Task 2 : NN & DNN features ...")
for name, (imgs, labels, _) in datasets.items():
    print(f"  Dataset: {name}")
    feats_nn  = train_and_extract_nn(imgs, labels, epochs=EPOCHS, device=DEVICE)
    feats_dnn = train_and_extract_dnn(imgs, labels, epochs=EPOCHS, device=DEVICE)
    FEATURE_STORE[name]["nn"]  = feats_nn
    FEATURE_STORE[name]["dnn"] = feats_dnn

# ── 4. Task 3 — CNN ───────────────────────────────────────────────────────────

from task3_cnn import extract_cnn_features
print("\n[4/8] Task 3 : CNN (ResNet-18) features ...")
for name, (imgs, labels, _) in datasets.items():
    feats = extract_cnn_features(imgs, device=DEVICE)
    FEATURE_STORE[name]["cnn"] = feats

# ── 5. Task 4 — Novel (MSFTA) ─────────────────────────────────────────────────

from task4_novel import extract_msfta_batch
print("\n[5/8] Task 4 : Novel MSFTA features ...")
for name, (imgs, labels, _) in datasets.items():
    feats = extract_msfta_batch(imgs)
    FEATURE_STORE[name]["msfta"] = feats

# ── 6. Task 5 — Hybrid ────────────────────────────────────────────────────────

from task5_hybrid import fuse_concat, fuse_weighted, find_optimal_lambda
print("\n[6/8] Task 5 : Hybrid features ...")
for name in datasets:
    cnn_f    = FEATURE_STORE[name]["cnn"]
    novel_f  = FEATURE_STORE[name]["msfta"]
    labels   = LABEL_STORE[name]
    print(f"  Dataset: {name}")
    FEATURE_STORE[name]["hybrid_concat"]   = fuse_concat(cnn_f, novel_f)
    best_lam = find_optimal_lambda(cnn_f, novel_f, labels, n_queries=50, top_k=TOP_K)
    FEATURE_STORE[name]["hybrid_weighted"] = fuse_weighted(cnn_f, novel_f, lam=best_lam)

# ── 7. Task 6 — Color ─────────────────────────────────────────────────────────

from task6_color import extract_color_batch, fuse_color_lbp, fuse_color_cnn
print("\n[7/8] Task 6 : Colour features ...")
for name, (imgs, labels, _) in datasets.items():
    c_feats = extract_color_batch(imgs)
    FEATURE_STORE[name]["color"]          = c_feats
    FEATURE_STORE[name]["color_lbp"]      = fuse_color_lbp(c_feats, FEATURE_STORE[name]["lbp"])
    FEATURE_STORE[name]["color_cnn"]      = fuse_color_cnn(c_feats, FEATURE_STORE[name]["cnn"])

# ── 8. Evaluation ─────────────────────────────────────────────────────────────

from metrics import evaluate_retrieval, print_results_table
print("\n[8/8] Evaluating all methods ...")

METHOD_METRICS = {
    "lbp":             "cosine",
    "nn":              "cosine",
    "dnn":             "cosine",
    "cnn":             "cosine",
    "msfta":           "cosine",
    "hybrid_concat":   "cosine",
    "hybrid_weighted": "cosine",
    "color":           "cosine",
    "color_lbp":       "cosine",
    "color_cnn":       "cosine",
}

for name in datasets:
    print(f"\n  ─── {name} ───")
    labels = LABEL_STORE[name]
    for method, metric in METHOD_METRICS.items():
        if method not in FEATURE_STORE[name]:
            continue
        feats = FEATURE_STORE[name][method]
        res   = evaluate_retrieval(feats, labels, top_k=TOP_K,
                                   n_queries=200, metric=metric)
        METRICS_STORE[name][method] = res
        print(f"    {method:<22} P={res['precision']:.4f} "
              f"R={res['recall']:.4f} mAP={res['mAP']:.4f} NDCG={res['ndcg']:.4f}")

# ── Print summary tables ───────────────────────────────────────────────────────

for name in datasets:
    print(f"\n{'='*60}")
    print(f"  RESULTS — {name.upper()}")
    print_results_table(METRICS_STORE[name])

# ── Save feature store ─────────────────────────────────────────────────────────

store = {
    "features": FEATURE_STORE,
    "labels":   LABEL_STORE,
    "images":   IMAGE_STORE,
    "classes":  CLASS_STORE,
    "metrics":  METRICS_STORE,
}

with open(STORE_PATH, "wb") as f:
    pickle.dump(store, f, protocol=4)
print(f"\n[✓] Feature store saved to {STORE_PATH}")
print("[✓] Pipeline complete!\n")

# ── Optionally launch dashboard ────────────────────────────────────────────────

if args.dashboard:
    import task7_dashboard as dash
    dash.load_stores(STORE_PATH)
    demo = dash.build_dashboard()
    demo.launch(share=True)
