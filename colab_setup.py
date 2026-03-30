"""
colab_setup.py
--------------
Run this ONCE at the top of your Colab notebook to set up the project.

Usage in Colab:
    !pip install -q torch torchvision scikit-image scikit-learn gradio tqdm scipy pandas seaborn opencv-python
    !git clone <your-repo>  OR  upload zip and unzip
    %cd image_retrieval
    !python run_all.py --quick        # fast test (500 imgs/dataset)
    !python run_all.py                # full run  (5000 imgs/dataset)
    !python task7_dashboard.py        # launch dashboard (gets public URL)
"""

COLAB_NOTEBOOK_CELLS = '''
# ============================================================
# Cell 1 : Install dependencies
# ============================================================
!pip install -q torch torchvision scikit-image scikit-learn \\
             gradio tqdm scipy pandas seaborn opencv-python-headless

# ============================================================
# Cell 2 : Upload project zip OR clone repo
# ============================================================
# Option A — upload zip file
from google.colab import files
uploaded = files.upload()   # select image_retrieval.zip
import zipfile
with zipfile.ZipFile("image_retrieval.zip", "r") as z:
    z.extractall(".")
%cd image_retrieval

# Option B — if you put it on GitHub:
# !git clone https://github.com/YOUR_USERNAME/image_retrieval.git
# %cd image_retrieval

# ============================================================
# Cell 3 : Quick test run (500 images, ~5 min on Colab CPU)
# ============================================================
!python run_all.py --quick --epochs 5

# ============================================================
# Cell 4 : Full run (5000 images, ~20-30 min on Colab GPU)
# ============================================================
# !python run_all.py --epochs 10

# ============================================================
# Cell 5 : Launch Dashboard
# ============================================================
!python task7_dashboard.py
# → Gradio will print a public share URL like:
#   Running on public URL: https://XXXXXXXX.gradio.live

# ============================================================
# Cell 6 : Inline visualisations (optional)
# ============================================================
import pickle, numpy as np
import matplotlib.pyplot as plt

with open("feature_store.pkl", "rb") as f:
    store = pickle.load(f)

# Example: t-SNE of CNN features on CIFAR-10
from visualize import plot_feature_tsne
fig = plot_feature_tsne(
    store["features"]["cifar10"]["cnn"],
    store["labels"]["cifar10"],
    store["classes"]["cifar10"],
    title="CNN Features — CIFAR-10",
)
plt.savefig("tsne_cnn_cifar10.png", dpi=150, bbox_inches="tight",
            facecolor="#0d0d0d")
plt.show()

# Example: metrics comparison bar chart
from visualize import plot_metrics_comparison
fig = plot_metrics_comparison(store["metrics"]["cifar10"], dataset_name="CIFAR-10")
plt.savefig("metrics_cifar10.png", dpi=150, bbox_inches="tight",
            facecolor="#0d0d0d")
plt.show()
'''

print("Colab setup guide printed. Copy the cells above into your notebook.")
print("\nQuick-start command:")
print("  python run_all.py --quick --dashboard")
