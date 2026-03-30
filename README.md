# Next-Generation Image Retrieval System

**Deep Learning for Computer Vision Assignment**

A complete multi-paradigm image retrieval system using classical, neural, and novel feature engineering approaches. Supports **CIFAR-10**, **MNIST**, and **Animals-10** datasets.

## Methods Implemented

| Task | Method              | Description                                          | Feature Dim |
| ---- | ------------------- | ---------------------------------------------------- | ----------- |
| 1    | **LBP**             | Local Binary Patterns — texture histogram            | 26-d        |
| 2    | **NN**              | Shallow MLP (1 hidden layer) classifier features     | 512-d       |
| 2    | **DNN**             | Deep MLP (4 hidden layers) classifier features       | 128-d       |
| 3    | **CNN**             | ResNet-18 pretrained on ImageNet                     | 512-d       |
| 4    | **MHSA v2** ⭐      | Multi-scale Hierarchical Structural Analyser (novel) | 89-d        |
| 5    | **Hybrid-Concat**   | CNN ⊕ MHSA concatenated                              | 601-d       |
| 5    | **Hybrid-Weighted** | λ·CNN + (1-λ)·MHSA, λ found by grid search           | 89-d        |
| 6    | **Color**           | RGB/HSV statistics + cross-channel correlations      | 212-d       |
| 6    | **Color+LBP**       | Colour fused with LBP                                | 238-d       |
| 6    | **Color+CNN**       | Colour fused with CNN                                | 724-d       |

---

## Project Structure

```
image_retrieval/
├── dataset_loader.py      # Loads CIFAR-10, MNIST, Animals-10
├── task1_lbp.py           # Task 1  — LBP classical retrieval
├── task2_nn_dnn.py        # Task 2  — Shallow NN & Deep NN features
├── task3_cnn.py           # Task 3  — ResNet-18 CNN features
├── task4_novel.py         # Task 4  — MHSA v2 novel descriptor ⭐
├── task5_hybrid.py        # Task 5  — Hybrid CNN+MHSA fusion
├── task6_color.py         # Task 6  — Intra/inter colour features
├── task7_dashboard.py     # Task 7  — Gradio interactive dashboard
├── metrics.py             # Precision, Recall, mAP, NDCG
├── visualize.py           # t-SNE, bar charts, PR curves
├── run_all.py             # Master pipeline (runs everything)
├── colab_setup.py         # Colab quick-start guide
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Requirements

- Python 3.8 or higher
- pip

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

### CIFAR-10 and MNIST

These download **automatically** on first run. No manual steps needed.

```
data/
├── cifar-10-batches-py/   ← auto-downloaded
└── MNIST/                 ← auto-downloaded
```

### Animals-10 (Manual Download Required)

Animals-10 is not available for automatic download. Follow these steps:

**Step 1:** Go to the Kaggle dataset page:
👉 https://www.kaggle.com/datasets/alessiocorrado99/animals10

**Step 2:** Click **Download** (you need a free Kaggle account)

**Step 3:** You will get a file called `archive.zip` or `animals10.zip`

**Step 4:** Rename it to `animals10.zip` and place it inside the `data/` folder:

```
image_retrieval/
└── data/
    └── animals10.zip      ← place it here
```

**Step 5:** The code will **automatically extract it** when you run the pipeline. You do not need to unzip it manually.

> **Note:** The Animals-10 dataset uses Italian folder names (cane, gatto, etc.). The code automatically translates these to English (dog, cat, etc.) via a mapping dictionary in `dataset_loader.py`.

After extraction, the structure will look like:

```
data/
└── animals10/
    ├── cane/        (dog)
    ├── gatto/       (cat)
    ├── elefante/    (elephant)
    ├── farfalla/    (butterfly)
    ├── gallina/     (chicken)
    ├── mucca/       (cow)
    ├── pecora/      (sheep)
    ├── ragno/       (spider)
    ├── scoiattolo/  (squirrel)
    └── cavallo/     (horse)
```

---

## Running Locally

## Setup local environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Quick Test Run (recommended first)

Runs on 500 images per dataset, 5 epochs — completes in ~5–10 minutes on CPU:

```bash
python run_all.py --quick --epochs 5
```

### Full Run

```bash
python run_all.py --epochs 10
```

### Full Run + Launch Dashboard automatically

```bash
python run_all.py --epochs 10 --dashboard
```

### Available arguments

| Argument      | Default | Description                                     |
| ------------- | ------- | ----------------------------------------------- |
| `--quick`     | off     | Use 500 images per dataset instead of 5000      |
| `--epochs`    | 10      | Training epochs for NN and DNN models           |
| `--topk`      | 10      | K value for Precision@K, Recall@K, NDCG@K       |
| `--dashboard` | off     | Launch Gradio dashboard after pipeline finishes |

### What happens when you run it

The pipeline runs in this order:

```
[1/8] Loading datasets         → CIFAR-10 (auto), MNIST (auto), Animals-10 (from zip)
[2/8] Task 1 - LBP             → extracts texture histograms
[3/8] Task 2 - NN & DNN        → trains models, extracts features
[4/8] Task 3 - CNN             → runs ResNet-18 feature extraction
[5/8] Task 4 - MHSA v2         → extracts novel foreground-aware features
[6/8] Task 5 - Hybrid          → fuses CNN + MHSA features
[7/8] Task 6 - Color           → extracts colour features and fusions
[8/8] Evaluation               → computes Precision, Recall, mAP, NDCG for all methods
      Saves → feature_store.pkl
```

After completion, metrics tables are printed to the terminal and everything is saved to `feature_store.pkl` for the dashboard.

## Dashboard

The dashboard requires the `feature_store.pkl` file to be generated first by `run_all.py`.

### Launch separately (after pipeline is done)

```bash
python task7_dashboard.py
```

### Features

- **Upload any query image** — the system extracts features and searches the gallery in real-time
- **Select dataset** — CIFAR-10, MNIST, or Animals-10
- **Select method** — all 10 methods available
- **Adjust Top-K** — slider from 1 to 20
- **Colour-coded results** — green border = correct class match, red = wrong
- **Similarity scores** — cosine distance shown below each result
- **Metrics panel** — pre-computed Precision@K, Recall@K, mAP, NDCG
- **Comparison charts** — bar charts comparing all methods for the selected dataset

---

## Novel Method: MHSA v2

**Multi-scale Hierarchical Structural Analyser v2** is an original feature descriptor designed specifically for foreground-dominated natural images.

The key innovation is computing ALL features exclusively on foreground pixels after background removal via Otsu thresholding. The descriptor has 6 components:

| Component                           | Dim    | What it measures                                       |
| ----------------------------------- | ------ | ------------------------------------------------------ |
| ZRD — Zonal Radial Descriptor       | 20     | WHERE in the image is mass/energy (5 concentric rings) |
| CSLE — Cross-Scale Laplacian Energy | 3      | WHAT SCALE is the texture (σ=1,2,4)                    |
| SAP — Symmetry Axis Projection      | 18     | WHICH DIRECTION do edges run (horizontal vs vertical)  |
| LBPV — LBP Variance Map             | 16     | HOW IRREGULAR is local texture (4×4 grid)              |
| FSD — Foreground Shape Descriptor   | 8      | WHAT SHAPE is the animal silhouette                    |
| FCH — Foreground Colour Histogram   | 24     | WHAT COLOUR is the animal (not background)             |
| **Total**                           | **89** | L2-normalised                                          |

## Full mathematical formulation is documented in `task4_novel.py`.
