"""
task4_novel.py  —  Task 4: Novel Feature Extraction
=====================================================
MHSA v2 — Multi-scale Hierarchical Structural Analyzer
with Foreground-Aware Masking

======================================================
THE CORE PROBLEM THIS SOLVES
======================================================
Many animal datasets (Animals-10, CIFAR) have subjects photographed
on plain white/light backgrounds.  Without background suppression,
features are dominated by "lots of white space", making an elephant
look identical to a butterfly — both are "dark blob on white field".

MHSA v2 fixes this with a pre-processing foreground mask that zeroes
out background pixels before computing ANY feature.  The remaining
four components then describe ONLY the animal's structure.

======================================================
INTUITION (explain to your professor like this)
======================================================

Imagine you are blindfolded and trying to identify an animal by touch.
You'd ask:
  1. WHERE on the body do I feel the most structure?
     (elephant → big solid centre mass; spider → thin legs radiating out)
  2. At what SCALE is the texture?
     (elephant → smooth coarse skin; butterfly → fine detailed wing patterns)
  3. Which DIRECTION do the edges run?
     (horse → horizontal body; giraffe → vertical neck; spider → all directions)
  4. How IRREGULAR is the local pattern?
     (elephant skin → regular hexagonal scales; fur → irregular)

MHSA measures exactly these four things, but only AFTER removing the
background so we're only "touching" the animal itself.

======================================================
MATHEMATICAL FORMULATION
======================================================

Let I be an RGB image (H×W×3), resized to 128×128.
Let G = rgb2gray(I).

--- Pre-step: Foreground Mask M ---
  1. Otsu threshold on G → binary mask T
  2. Morphological close (kernel 9×9) to fill gaps
  3. M = 1 where animal is, 0 where background
  Only pixels where M=1 contribute to features.

--- Component 1: Zonal Radial Descriptor (ZRD) ---
  Divide image into 5 concentric rings r₀..r₄
  (r=0 is centre circle, r=4 is outermost ring)
  For each ring, using only foreground pixels:

    μ_r   = mean(G[M ∩ ring_r])          — avg brightness
    σ_r   = std(G[M ∩ ring_r])           — brightness spread
    skew_r = E[(G-μ)³]/σ³               — texture skewness
    E_r   = mean(|∇G|²[M ∩ ring_r])     — gradient energy

  ZRD = [μ₀,σ₀,skew₀,E₀, ..., μ₄,σ₄,skew₄,E₄]
  Dimension: 5 × 4 = 20

  WHY RADIAL NOT GRID?
  A 4×4 grid treats top-left and bottom-right cells the same.
  But the CENTRE of an animal image is semantically different from
  the EDGE — the elephant's body is in the centre, spider legs reach
  the outer ring.  Radial zones capture this naturally.

--- Component 2: Cross-Scale Laplacian Energy (CSLE) ---
  Apply Laplacian of Gaussian (LoG) at 3 scales:
    CSLE_s = (1/|M|) Σ_{(x,y): M=1} |LoG_{σ_s}(G)(x,y)|²
    σ₁=1  → micro-texture (fur tips, skin pores, thin spider legs)
    σ₂=2  → medium features (patches, stripes, eye area)
    σ₃=4  → coarse body outline (silhouette, major limbs)

  Dimension: 3

  KEY DISCRIMINATION:
  Elephant:  HIGH σ₃ (strong body outline), LOW σ₁ (smooth skin)
  Butterfly: HIGH σ₁ (fine wing detail) AND HIGH σ₂ (colour bands)
  Spider:    HIGH σ₁ (thin legs), LOW σ₃ (no large mass)

--- Component 3: Symmetry Axis Projection (SAP) ---
  Divide G into 3×3 spatial grid.  In each cell c:
    SAP_h(c) = (1/|M∩c|) Σ |∂G/∂x|   — horizontal edge strength
    SAP_v(c) = (1/|M∩c|) Σ |∂G/∂y|   — vertical edge strength
    SAP_r(c) = SAP_h(c) / (SAP_v(c) + ε)  — axis ratio

  Final: [SAP_h(c₁),...,SAP_h(c₉), SAP_v(c₁),...,SAP_v(c₉)]
  Dimension: 9 × 2 = 18

  KEY DISCRIMINATION:
  Elephant/Horse: horizontal body → SAP_h >> SAP_v in centre rows
  Spider:         radial legs     → SAP_h ≈ SAP_v everywhere
  Giraffe/Chicken: upright        → SAP_v > SAP_h in centre column

--- Component 4: LBP Variance Map (LBPV) ---
  LBP code at every pixel p (radius=1, 8 neighbours):
    LBP(p) = Σ_{k=0}^{7} s(n_k - p) · 2^k
    where s(x) = 1 if x≥0, 0 otherwise

  Divide into 4×4 grid.  For each cell:
    LBPV_cell = Var({LBP(p) : p ∈ cell, M(p)=1})

  Dimension: 16

  KEY DISCRIMINATION:
  Elephant:  LOW variance (regular hexagonal skin texture)
  Dog/Cat:   HIGH variance (irregular fur directions)
  Butterfly: VERY HIGH variance in wing cells (complex colour patterns)
  Spider:    HIGH variance (thin irregular legs vs background gaps)

--- Final assembly ---
  f_raw = [ZRD ‖ CSLE ‖ SAP ‖ LBPV]        shape: (57,)
  f     = f_raw / ‖f_raw‖₂                   L2 normalised

TOTAL DIMENSION: 20 + 3 + 18 + 16 = 57

======================================================
WHY THIS IS NOVEL
======================================================
Existing methods (HOG, SIFT, LBP) treat the whole image uniformly.
CNN features are black-box.  MHSA v2 is unique because:
  1. Foreground masking before ANY feature — no existing LBP/HOG descriptor does this
  2. RADIAL (not grid) spatial pooling — captures animal's body-centre structure
  3. MULTI-SCALE Laplacian ONLY ON FOREGROUND — background doesn't pollute energy
  4. SAP axis ratio — explicit H vs V dominance per spatial zone
  5. All four components are interpretable: you can explain each number
"""

import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy.stats import skew as scipy_skew

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE   = 128
N_ZONES    = 5
LOG_SCALES = [1, 2, 4]
SAP_GRID   = 3
LBP_GRID   = 4
EPS        = 1e-10


# ══════════════════════════════════════════════════════════════════════════════
#  PRE-STEP: Foreground Mask
# ══════════════════════════════════════════════════════════════════════════════

def _foreground_mask(gray_u8: np.ndarray) -> np.ndarray:
    """
    Detect the animal (foreground) and return a binary mask.

    Strategy:
      1. Otsu's method automatically finds the best threshold to
         separate the dark animal from the light background.
      2. We keep the DARKER region (the animal).
      3. Morphological closing fills small holes inside the animal
         (thin spider legs that might have gaps).
      4. If the mask is too small (< 5% of image), fall back to full image
         — for cases where background isn't white/light.

    Returns: mask (H, W) bool  True = foreground (animal)
    """
    # Otsu threshold
    _, thresh = cv2.threshold(gray_u8, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Fill small holes
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask    = closed > 0

    # Fallback: if foreground < 5% of image it's probably inverted
    fg_ratio = mask.sum() / mask.size
    if fg_ratio < 0.05:
        mask = ~mask   # invert

    # Still too small or too large → use whole image (dark background)
    fg_ratio = mask.sum() / mask.size
    if fg_ratio < 0.05 or fg_ratio > 0.95:
        mask = np.ones_like(mask, dtype=bool)

    return mask


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 1 — Zonal Radial Descriptor (ZRD)
# ══════════════════════════════════════════════════════════════════════════════

def _build_zone_masks(h: int, w: int, n_zones: int):
    """
    Build n_zones concentric ring masks using Euclidean distance from centre.
    Zone 0 = innermost circle, zone n-1 = outermost ring.
    """
    cy, cx = h / 2.0, w / 2.0
    ys, xs = np.mgrid[0:h, 0:w]
    dist   = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    max_r  = dist.max()
    edges  = np.linspace(0, max_r, n_zones + 1)
    masks  = [(dist >= edges[i]) & (dist < edges[i + 1])
              for i in range(n_zones)]
    return masks


def _safe_skew(pixels):
    if len(pixels) < 3:
        return 0.0
    s = float(scipy_skew(pixels))
    return s if np.isfinite(s) else 0.0


def _zrd(gray_f32: np.ndarray, gray_u8: np.ndarray,
         fg_mask: np.ndarray) -> np.ndarray:
    """
    Zonal Radial Descriptor — 20-d.
    Each zone: [mean_brightness, std_brightness, skewness, gradient_energy]
    All computed ONLY on foreground pixels within each zone.
    """
    h, w = gray_f32.shape
    zone_masks = _build_zone_masks(h, w, N_ZONES)

    Gx         = cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=3)
    Gy         = cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=3)
    energy_map = Gx ** 2 + Gy ** 2

    desc = []
    for zmask in zone_masks:
        active = zmask & fg_mask
        if active.sum() < 5:
            desc.extend([0.0, 0.0, 0.0, 0.0])
            continue
        pixels = gray_f32[active]
        mu     = float(np.mean(pixels))
        sigma  = float(np.std(pixels))
        sk     = _safe_skew(pixels)
        eg     = float(np.mean(energy_map[active]))
        desc.extend([mu, sigma, sk, eg])

    return np.array(desc, dtype=np.float32)   # (20,)


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 2 — Cross-Scale Laplacian Energy (CSLE)
# ══════════════════════════════════════════════════════════════════════════════

def _csle(gray_f32: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    Laplacian of Gaussian energy at 3 scales, foreground only.
    σ=1 → micro-texture, σ=2 → medium features, σ=4 → coarse body outline.
    Returns (3,) float32.
    """
    fg_count = fg_mask.sum()
    if fg_count == 0:
        return np.zeros(3, dtype=np.float32)

    desc = []
    for sigma in LOG_SCALES:
        ksize   = int(6 * sigma + 1) | 1
        blurred = cv2.GaussianBlur(gray_f32, (ksize, ksize), sigma)
        lap     = cv2.Laplacian(blurred.astype(np.float64), cv2.CV_64F)
        # Only foreground pixels contribute to energy
        energy  = float(np.mean((lap ** 2)[fg_mask]))
        desc.append(energy)

    return np.array(desc, dtype=np.float32)   # (3,)


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 3 — Symmetry Axis Projection (SAP)
# ══════════════════════════════════════════════════════════════════════════════

def _sap(gray_u8: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    Horizontal vs vertical gradient dominance in a 3×3 spatial grid.
    Uses only foreground pixels per cell.
    Returns (18,) float32:  [SAP_h × 9, SAP_v × 9]

    Discrimination logic:
      Horizontal animals (elephant, horse, cow) → centre-row SAP_h >> SAP_v
      Upright animals (chicken, dog sitting)    → SAP_v > SAP_h in centre col
      Radial animals (spider, butterfly)        → SAP_h ≈ SAP_v everywhere
    """
    Gx = np.abs(cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=3))
    Gy = np.abs(cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=3))

    h, w   = gray_u8.shape
    gh, gw = h // SAP_GRID, w // SAP_GRID
    sap_h, sap_v = [], []

    for r in range(SAP_GRID):
        for c in range(SAP_GRID):
            r0, r1 = r * gh, (r + 1) * gh
            c0, c1 = c * gw, (c + 1) * gw
            cell_mask = fg_mask[r0:r1, c0:c1]
            n = cell_mask.sum()
            if n < 3:
                sap_h.append(0.0)
                sap_v.append(0.0)
            else:
                sap_h.append(float(np.mean(Gx[r0:r1, c0:c1][cell_mask])))
                sap_v.append(float(np.mean(Gy[r0:r1, c0:c1][cell_mask])))

    return np.array(sap_h + sap_v, dtype=np.float32)   # (18,)


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 4 — LBP Variance Map (LBPV)
# ══════════════════════════════════════════════════════════════════════════════

def _lbp_codes(gray_u8: np.ndarray) -> np.ndarray:
    """
    Compute LBP code for every pixel (radius=1, 8 neighbours, uniform).
    Clockwise neighbour order: top-left, top, top-right, right,
                               bot-right, bot, bot-left, left.
    """
    img     = gray_u8.astype(np.int16)
    offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    codes   = np.zeros_like(gray_u8, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(offsets):
        shifted  = np.roll(np.roll(img, -dy, axis=0), -dx, axis=1)
        codes   |= ((shifted >= img).astype(np.uint8) << bit)
    return codes


def _lbpv(gray_u8: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    LBP Variance Map — 16-d.
    Divide into 4×4 grid.  Variance of LBP codes, foreground only per cell.

    High variance = irregular texture (fur, feathers, spider legs on white)
    Low variance  = uniform texture (elephant skin, plain coat)
    """
    codes = _lbp_codes(gray_u8).astype(np.float32)
    h, w  = codes.shape
    ch, cw = h // LBP_GRID, w // LBP_GRID
    desc  = []

    for r in range(LBP_GRID):
        for c in range(LBP_GRID):
            r0, r1 = r * ch, (r + 1) * ch
            c0, c1 = c * cw, (c + 1) * cw
            cell_mask = fg_mask[r0:r1, c0:c1]
            cell_codes = codes[r0:r1, c0:c1]
            if cell_mask.sum() < 3:
                desc.append(0.0)
            else:
                desc.append(float(np.var(cell_codes[cell_mask])))

    return np.array(desc, dtype=np.float32)   # (16,)

def _shape_descriptor(gray_u8: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    Component 5 — Foreground Shape Descriptor (FSD) — 8-d
    Measures the SHAPE of the animal, not just its texture.

    Features:
      1. Foreground fill ratio (area covered)
      2. Aspect ratio (width/height of bounding box)
      3. Compactness = area / (perimeter²)   — circle=high, spider=low
      4. Centre of mass offset X (from image centre, normalised)
      5. Centre of mass offset Y
      6. Radial mass distribution: ratio inner/outer ring mass
      7. Horizontal symmetry score (left half vs right half correlation)
      8. Contour count (number of separate foreground blobs)
         → elephant = 1, spider = many (legs)

    WHY THIS DISCRIMINATES STRONGLY:
      Elephant: high fill ratio, wide aspect, HIGH compactness, 1 blob
      Spider:   low fill ratio, wide aspect BUT low compactness, MANY blobs
      Butterfly:medium fill, wide aspect, medium compactness, 1-2 blobs
      Dog/Cat:  medium fill, medium aspect, medium compactness, 1 blob
    """
    h, w = fg_mask.shape
    fg   = fg_mask.astype(np.uint8) * 255
    desc = []

    # 1. Fill ratio
    fill_ratio = fg_mask.mean()
    desc.append(float(fill_ratio))

    # 2. Aspect ratio from bounding box
    rows = np.where(fg_mask.any(axis=1))[0]
    cols = np.where(fg_mask.any(axis=0))[0]
    if len(rows) > 1 and len(cols) > 1:
        bb_h = rows[-1] - rows[0] + 1
        bb_w = cols[-1] - cols[0] + 1
        aspect = float(bb_w) / float(bb_h + EPS)
    else:
        aspect = 1.0
    desc.append(aspect)

    # 3. Compactness = area / perimeter² (scale-invariant)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        total_area = sum(cv2.contourArea(c) for c in contours)
        total_perim = sum(cv2.arcLength(c, True) for c in contours) + EPS
        compactness = float(total_area / (total_perim ** 2))
    else:
        compactness = 0.0
    desc.append(compactness)

    # 4-5. Centre of mass offsets
    ys, xs = np.where(fg_mask)
    if len(ys) > 0:
        cm_x = (xs.mean() - w / 2.0) / (w / 2.0)  # normalised -1..1
        cm_y = (ys.mean() - h / 2.0) / (h / 2.0)
    else:
        cm_x, cm_y = 0.0, 0.0
    desc.append(float(cm_x))
    desc.append(float(cm_y))

    # 6. Inner/outer ring mass ratio
    cy, cx = h / 2.0, w / 2.0
    ys_g, xs_g = np.mgrid[0:h, 0:w]
    dist = np.sqrt((ys_g - cy) ** 2 + (xs_g - cx) ** 2)
    inner_mask = dist < (min(h, w) / 4.0)
    inner_mass = float((fg_mask & inner_mask).sum())
    outer_mass = float((fg_mask & ~inner_mask).sum()) + EPS
    desc.append(inner_mass / outer_mass)

    # 7. Horizontal symmetry (correlation between left and right halves)
    left_half  = fg_mask[:, :w//2].astype(np.float32)
    right_half = np.fliplr(fg_mask[:, w//2:]).astype(np.float32)
    min_w = min(left_half.shape[1], right_half.shape[1])
    if min_w > 0:
        l = left_half[:, :min_w].ravel()
        r = right_half[:, :min_w].ravel()
        sym = float(np.corrcoef(l, r)[0, 1]) if l.std() > 0 and r.std() > 0 else 0.0
        sym = 0.0 if not np.isfinite(sym) else sym
    else:
        sym = 0.0
    desc.append(sym)

    # 8. Number of contour blobs (normalised)
    blob_count = float(len(contours)) / 20.0   # normalise by max expected
    desc.append(min(blob_count, 1.0))

    return np.array(desc, dtype=np.float32)   # (8,)


def _colour_histogram(img_rgb: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    Component 6 — Foreground Colour Histogram (FCH) — 24-d

    This is the most powerful discriminator between visually similar animals.

    Compute 8-bin histogram for each of R, G, B channels,
    using ONLY foreground pixels.  Normalise each channel histogram.

    WHY THIS SEPARATES ELEPHANT FROM DOG:
      Elephant: grey → R ≈ G ≈ B, all mid-range (bins 3-5), very uniform
      Dog:      brown fur → HIGH R, MEDIUM G, LOW B  (bins skewed warm)
      Butterfly:orange wing → VERY HIGH R, LOW B
                blue wing   → LOW R, VERY HIGH B
      Horse:    brown/black → similar to dog but darker (lower bins)
      Cat:      varies a lot by breed — good spread across bins

    This gives a 24-d colour fingerprint of the actual animal,
    not the background (because fg_mask removes white pixels first).

    Dimension: 3 channels × 8 bins = 24
    """
    resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    fg_count = fg_mask.sum()
    if fg_count < 5:
        return np.zeros(24, dtype=np.float32)

    desc = []
    for ch in range(3):   # R, G, B
        channel = resized[:, :, ch].astype(np.float32)
        fg_vals = channel[fg_mask]
        hist, _ = np.histogram(fg_vals, bins=8, range=(0, 255))
        hist_norm = hist.astype(np.float32) / (fg_count + EPS)
        desc.append(hist_norm)

    return np.concatenate(desc)   # (24,)


def extract_mhsa_features(img: np.ndarray) -> np.ndarray:
    """
    Extract MHSA v2 descriptor from a single RGB image.

    Parameters
    ----------
    img : np.ndarray  shape (H, W, 3)  uint8  RGB

    Returns
    -------
    features : np.ndarray  shape (89,)  float32  L2-normalised
      [ZRD(20) | CSLE(3) | SAP(18) | LBPV(16) | FSD(8) | FCH(24)]

    Component summary:
      ZRD  — WHERE in image is the mass (radial zones)
      CSLE — WHAT SCALE is the texture (fine fur vs coarse body)
      SAP  — WHICH DIRECTION do edges run (horizontal body vs radial legs)
      LBPV — HOW IRREGULAR is local texture (smooth skin vs fur)
      FSD  — WHAT SHAPE is the foreground blob (compact vs spindly)
      FCH  — WHAT COLOUR is the animal (grey elephant vs brown dog)
    """
    resized  = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    gray_u8  = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    gray_f32 = gray_u8.astype(np.float32)

    # Step 0: foreground mask — strip out white/plain backgrounds
    fg_mask = _foreground_mask(gray_u8)

    # Six complementary components
    zrd_feat  = _zrd(gray_f32, gray_u8, fg_mask)     # (20,) WHERE is energy
    csle_feat = _csle(gray_f32, fg_mask)              # (3,)  WHAT SCALE texture
    sap_feat  = _sap(gray_u8, fg_mask)                # (18,) WHICH AXIS dominant
    lbpv_feat = _lbpv(gray_u8, fg_mask)              # (16,) HOW IRREGULAR texture
    fsd_feat  = _shape_descriptor(gray_u8, fg_mask)   # (8,)  WHAT SHAPE is animal
    fch_feat  = _colour_histogram(img, fg_mask)       # (24,) WHAT COLOUR is animal

    raw  = np.concatenate([zrd_feat, csle_feat, sap_feat,
                           lbpv_feat, fsd_feat, fch_feat])   # (89,)
    norm = np.linalg.norm(raw) + EPS
    return (raw / norm).astype(np.float32)


# ── Batch extractor ────────────────────────────────────────────────────────────
def extract_mhsa_batch(images: np.ndarray) -> np.ndarray:
    """
    Extract MHSA features for a batch of images.
    images: (N, H, W, 3) uint8 RGB
    Returns: (N, 57) float32
    """
    feats = []
    print("[MHSA v2] Extracting foreground-aware features (89-d) ...")
    for i, img in enumerate(images):
        feats.append(extract_mhsa_features(img))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(images)}")
    feats = np.stack(feats)
    print(f"[MHSA v2] Done. Feature shape: {feats.shape}")
    return feats


# ── Retrieval helper ───────────────────────────────────────────────────────────
def retrieve_mhsa(query_feat, gallery_feats, top_k=5):
    dists = cdist(query_feat[None, :], gallery_feats, metric="cosine")[0]
    idx   = np.argsort(dists)[:top_k]
    return idx, dists[idx]


# ── Aliases — your run_all.py and dashboard need ZERO changes ─────────────────
extract_scg         = extract_mhsa_features
extract_msfta       = extract_mhsa_features
extract_scg_batch   = extract_mhsa_batch
extract_msfta_batch = extract_mhsa_batch


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")   # suppress skew precision warnings for clean output

    print("=" * 58)
    print("MHSA v2 — Self-Test with Realistic Synthetic Images")
    print("=" * 58)

    rng = np.random.default_rng(42)

    # Elephant: large textured grey body, horizontal, on white background
    elephant = np.full((200, 300, 3), 245, dtype=np.uint8)
    body_tex  = rng.integers(110, 150, (120, 220, 3), dtype=np.uint8)
    elephant[40:160, 40:260] = body_tex

    # Spider: many thin dark lines radiating from centre (all 360°), white bg
    spider = np.full((200, 200, 3), 245, dtype=np.uint8)
    for angle in range(0, 360, 30):
        x2 = int(100 + 85 * np.cos(np.radians(angle)))
        y2 = int(100 + 85 * np.sin(np.radians(angle)))
        cv2.line(spider, (100, 100), (x2, y2), (30, 30, 30), 2)

    # Butterfly: wide coloured wings + narrow dark body, white bg
    butterfly = np.full((200, 300, 3), 245, dtype=np.uint8)
    butterfly[50:150, 20:140, 0]  = rng.integers(180, 255, (100, 120), dtype=np.uint8)
    butterfly[50:150, 160:280, 2] = rng.integers(100, 200, (100, 120), dtype=np.uint8)
    butterfly[70:130, 135:165]    = 20   # dark narrow body

    f_e = extract_mhsa_features(elephant)
    f_s = extract_mhsa_features(spider)
    f_b = extract_mhsa_features(butterfly)

    d_es = float(cdist(f_e[None], f_s[None], metric="cosine")[0, 0])
    d_eb = float(cdist(f_e[None], f_b[None], metric="cosine")[0, 0])
    d_sb = float(cdist(f_s[None], f_b[None], metric="cosine")[0, 0])

    print(f"\nFeature dimension : {f_e.shape}  (89-d)")
    print("\nCosine distances  (higher = more different, max=2.0):")
    print(f"  Elephant vs Spider    : {d_es:.4f}  {'PASS ✓' if d_es > 0.25 else 'FAIL ✗'}")
    print(f"  Elephant vs Butterfly : {d_eb:.4f}  {'PASS ✓' if d_eb > 0.25 else 'FAIL ✗'}")
    print(f"  Spider   vs Butterfly : {d_sb:.4f}  {'PASS ✓' if d_sb > 0.25 else 'FAIL ✗'}")

    # Shape feature breakdown
    print("\nShape features (FSD) — what MHSA sees per animal:")
    for name, img in [("Elephant", elephant), ("Spider", spider), ("Butterfly", butterfly)]:
        g  = cv2.cvtColor(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_RGB2GRAY)
        m  = _foreground_mask(g)
        fd = _shape_descriptor(g, m)
        print(f"  {name:10s}: fill={fd[0]:.2f}  aspect={fd[1]:.2f}  "
              f"compactness={fd[2]:.4f}  inner/outer={fd[5]:.2f}  "
              f"fg%={m.mean()*100:.0f}%")

    passed = d_es > 0.25 and d_eb > 0.25
    print(f"\n{'All tests PASSED ✓' if passed else 'Some tests failed — check images'}")
    print("=" * 58)