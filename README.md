# Next-Generation Image Retrieval System
**Deep Learning for Computer Vision Assignment**

---

## Project Structure

```
image_retrieval/
├── dataset_loader.py      # Load CIFAR-10, MNIST, Fashion-MNIST
├── task1_lbp.py           # Task 1  — LBP classical retrieval
├── task2_nn_dnn.py        # Task 2  — Shallow NN & Deep NN features
├── task3_cnn.py           # Task 3  — ResNet-18 CNN features
├── task4_novel.py         # Task 4  — MSFTA novel descriptor ⭐
├── task5_hybrid.py        # Task 5  — Hybrid CNN+MSFTA fusion
├── task6_color.py         # Task 6  — Intra/inter colour features
├── task7_dashboard.py     # Task 7  — Gradio interactive dashboard
├── metrics.py             # Precision, Recall, mAP, NDCG
├── visualize.py           # t-SNE, bar charts, PR curves
├── run_all.py             # Master pipeline (runs everything)
├── colab_setup.py         # Colab quick-start guide
└── requirements.txt
```

---

## Quick Start (Google Colab)

```python
# Cell 1: Install
!pip install -q torch torchvision scikit-image scikit-learn gradio tqdm scipy seaborn opencv-python-headless

# Cell 2: Upload zip and extract, then:
%cd image_retrieval

# Cell 3: Quick test (500 images, ~5 min)
!python run_all.py --quick --epochs 5

# Cell 4: Full run (5000 images, ~25 min on GPU)
!python run_all.py --epochs 10

# Cell 5: Launch Dashboard
!python task7_dashboard.py
```

---

## Methods Implemented

| Task | Method | Description |
|------|--------|-------------|
| 1 | **LBP** | Local Binary Patterns — texture histogram |
| 2 | **NN** | Shallow MLP (1 hidden layer) classifier features |
| 2 | **DNN** | Deep MLP (4 hidden layers) classifier features |
| 3 | **CNN** | ResNet-18 pretrained on ImageNet (512-d) |
| 4 | **MSFTA** ⭐ | Multi-Scale Frequency-Texture Attention (novel) |
| 5 | **Hybrid-Concat** | CNN ⊕ MSFTA concatenated |
| 5 | **Hybrid-Weighted** | λ·CNN + (1-λ)·MSFTA, λ found by grid search |
| 6 | **Color** | RGB/HSV statistics + cross-channel correlations |
| 6 | **Color+LBP** | Colour fused with LBP |
| 6 | **Color+CNN** | Colour fused with CNN |

---

## Novel Method: MSFTA

**Multi-Scale Frequency-Texture Attention Descriptor**

The key innovation is combining:
1. **Multi-scale Gaussian pyramid** (3 levels) to capture features at different resolutions
2. **DCT frequency bands** (low/mid/high) per 8×8 block, per channel
3. **Spatial attention** via softmax of local variance — focuses on textured regions
4. **Cross-channel gradient correlation** — captures structural inter-channel relationships
5. **L2 normalisation** for distance-metric compatibility

This is **original work** — not replicated from any paper.

Mathematical formulation is fully documented in `task4_novel.py`.

---

## Datasets

- **CIFAR-10**: 10 natural image classes (airplanes, cars, animals...)
- **MNIST**: Handwritten digits 0-9
- **Fashion-MNIST** *(chosen 3rd dataset)*: 10 clothing categories

**Justification for Fashion-MNIST**: It provides a middle ground between the simplicity of MNIST (grayscale, simple shapes) and the complexity of CIFAR-10 (colour, natural scenes). Clothing items share many textures and shapes, making it a meaningful challenge for texture-based retrieval methods like LBP and MSFTA.

---

## Evaluation

- **Precision@K**: Fraction of top-K results that match the query class
- **Recall@K**: Fraction of all relevant images found in top-K
- **mAP**: Mean Average Precision across all queries
- **NDCG@K**: Normalised Discounted Cumulative Gain (ranking quality)

---

## Dashboard Features

- Upload any query image
- Select dataset and method
- Set Top-K (1–20)
- View retrieved images with colour-coded borders (green=correct, red=wrong)
- See similarity scores per result
- Compare all methods via pre-computed bar charts
