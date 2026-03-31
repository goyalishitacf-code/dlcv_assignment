"""
task7_dashboard.py  —  Image Retrieval Dashboard
- Predictions tab shows majority class for every method
- Status banner shows only precision, no majority class text
- All text dark/black throughout
"""
import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os, pickle, cv2
from PIL import Image, ImageDraw
from collections import Counter

C_BG      = "#F5F7FA"
C_SURF    = "#FFFFFF"
C_BORDER  = "#DDE3EE"
C_ACCENT  = "#2563EB"
C_ACCENT2 = "#7C3AED"
C_GREEN   = "#059669"
C_AMBER   = "#D97706"
C_RED     = "#DC2626"
C_TEXT    = "#111827"
C_MUTED   = "#374151"
C_DIM     = "#E5E7EB"

rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.facecolor": "#FFFFFF", "figure.facecolor": "#FFFFFF",
    "text.color": C_TEXT, "axes.labelcolor": C_TEXT, "axes.titlecolor": C_TEXT,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED,
    "axes.edgecolor": C_BORDER, "grid.color": C_DIM, "grid.linewidth": 0.7,
})

FEATURE_STORE = {}
LABEL_STORE   = {}
IMAGE_STORE   = {}
CLASS_STORE   = {}
METRICS_STORE = {}

METHOD_LABELS = {
    "lbp":             "LBP — Task 1",
    "nn":              "NN — Task 2",
    "dnn":             "Deep NN — Task 2",
    "cnn":             "CNN ResNet-18 — Task 3",
    "mhsav2":           "MHSA v2 Novel — Task 4",
    "hybrid_concat":   "Hybrid Concat — Task 5",
    "hybrid_weighted": "Hybrid Weighted — Task 5",
    "color":           "Color Features — Task 6",
    "color_lbp":       "Color + LBP — Task 6",
    "color_cnn":       "Color + CNN — Task 6",
}


def _make_sample_pil(bg_rgb, label_text, size=160):
    img  = Image.new("RGB", (size, size), color=bg_rgb)
    draw = ImageDraw.Draw(img)
    r, g, b = bg_rgb
    tc = (20,20,20) if (r*.299+g*.587+b*.114)>128 else (235,235,235)
    draw.rectangle([4,4,size-5,size-5], outline=tc, width=3)
    bb = draw.textbbox((0,0), label_text)
    draw.text(((size-(bb[2]-bb[0]))//2,(size-(bb[3]-bb[1]))//2), label_text, fill=tc)
    return img


def load_stores(path="./feature_store.pkl"):
    global FEATURE_STORE, LABEL_STORE, IMAGE_STORE, CLASS_STORE, METRICS_STORE
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        d = pickle.load(f)
    FEATURE_STORE = d["features"]
    LABEL_STORE   = d["labels"]
    IMAGE_STORE   = d["images"]
    CLASS_STORE   = d["classes"]
    METRICS_STORE = d.get("metrics", {})
    print("[OK] Feature store loaded.")
    return True

# ── Feature extraction ─────────────────────────────────────────────────────────
def _get_query_feature(query_arr, dataset, method):
    gallery_imgs = IMAGE_STORE[dataset]
    th, tw = gallery_imgs.shape[1], gallery_imgs.shape[2]
    q = cv2.resize(query_arr, (tw, th), interpolation=cv2.INTER_AREA)
    if method == "lbp":
        from task1_lbp import extract_lbp; return extract_lbp(q)
    elif method in ("nn","dnn"):
        from task1_lbp import extract_lbp; return extract_lbp(q)
    elif method == "cnn":
        from task3_cnn import extract_cnn_features; return extract_cnn_features(q[None])[0]
    elif method == "mhsav2":
        from task4_novel import extract_mhsa_features; return extract_mhsa_features(q)
    elif method == "color":
        from task6_color import extract_color_full; return extract_color_full(q)
    elif method == "color_lbp":
        from task6_color import extract_color_full, fuse_color_lbp
        from task1_lbp import extract_lbp
        return fuse_color_lbp(extract_color_full(q)[None], extract_lbp(q)[None])[0]
    elif method == "color_cnn":
        from task6_color import extract_color_full, fuse_color_cnn
        from task3_cnn import extract_cnn_features
        return fuse_color_cnn(extract_color_full(q)[None], extract_cnn_features(q[None]))[0]
    elif method == "hybrid_concat":
        from task3_cnn import extract_cnn_features
        from task4_novel import extract_mhsa_features
        from sklearn.preprocessing import normalize
        return np.concatenate([
            normalize(extract_cnn_features(q[None]), norm="l2")[0],
            normalize(extract_mhsa_features(q)[None], norm="l2")[0],
        ])
    elif method == "hybrid_weighted":
        from task3_cnn import extract_cnn_features
        from task4_novel import extract_mhsa_features
        from sklearn.preprocessing import normalize
        cnn_q = normalize(extract_cnn_features(q[None]), norm="l2")[0]
        nov_q = normalize(extract_mhsa_features(q)[None], norm="l2")[0]
        gf = FEATURE_STORE[dataset][method]
        return np.concatenate([cnn_q,nov_q]) if cnn_q.shape[0]!=gf.shape[1] else 0.6*cnn_q+0.4*nov_q
    return None

# ── Core retrieval helper ──────────────────────────────────────────────────────
def _retrieve_one(query_arr, dataset, method, top_k):
    """Run retrieval for one method. Returns (ret_labels, ret_dists, precision)."""
    gfeats  = FEATURE_STORE[dataset][method]
    glabels = LABEL_STORE[dataset]
    from scipy.spatial.distance import cdist
    qf    = _get_query_feature(query_arr, dataset, method)
    dists = cdist(qf[None,:], gfeats, metric="cosine")[0]
    idx   = np.argsort(dists)[:top_k]
    ret_labels = glabels[idx]
    ret_dists  = dists[idx]
    qlabel     = ret_labels[0]
    precision  = float(np.mean(ret_labels == qlabel))
    return ret_labels, ret_dists, precision

# ── Predictions table HTML ─────────────────────────────────────────────────────
def build_predictions_table(query_arr, dataset, top_k, active_method_label):
    """
    Run every available method on the query image and build an HTML table
    showing: Method | Predicted Class | Confidence | Top-3 Classes | Precision
    The currently selected method is highlighted.
    """
    if dataset not in IMAGE_STORE:
        return ('<div style="padding:20px;color:#111827;font-family:Arial,sans-serif;'
                'font-size:0.85rem;background:#fff;border-radius:8px;border:1px solid #E5E7EB">'
                'Run retrieval first to see predictions.</div>')

    cnames = CLASS_STORE.get(dataset, [str(i) for i in range(10)])
    available = {k:v for k,v in METHOD_LABELS.items()
                 if k in FEATURE_STORE.get(dataset,{})}

    if not available:
        return ('<div style="padding:20px;color:#111827;font-family:Arial,sans-serif;'
                'font-size:0.85rem">No methods loaded yet — run run_all.py first.</div>')

    rows = ""
    for method_key, method_name in available.items():
        try:
            ret_labels, ret_dists, precision = _retrieve_one(
                query_arr, dataset, method_key, top_k)
        except Exception:
            continue

        counts      = Counter(ret_labels)
        top3        = counts.most_common(3)
        maj_idx, maj_count = top3[0]
        maj_name    = cnames[maj_idx].upper()
        confidence  = maj_count / top_k    # fraction of top-k that are majority

        # Top-3 as pill badges
        top3_html = ""
        badge_colors = ["#2563EB","#7C3AED","#374151"]
        for rank,(cls_idx,cnt) in enumerate(top3):
            cls_name = cnames[cls_idx].upper()
            pct      = int(cnt/top_k*100)
            bc       = badge_colors[rank]
            top3_html += (f'<span style="background:{bc}18;color:{bc};'
                          f'border:1px solid {bc}44;border-radius:4px;'
                          f'padding:2px 7px;font-size:0.72rem;font-family:monospace;'
                          f'font-weight:600;margin-right:4px;white-space:nowrap">'
                          f'{cls_name} {pct}%</span>')

        # Confidence bar
        conf_pct   = int(confidence*100)
        conf_color = "#059669" if confidence>=0.7 else "#D97706" if confidence>=0.4 else "#DC2626"
        conf_bar   = (f'<div style="display:flex;align-items:center;gap:6px">'
                      f'<div style="flex:1;background:#F3F4F6;border-radius:4px;height:6px;min-width:60px">'
                      f'<div style="width:{conf_pct}%;background:{conf_color};border-radius:4px;height:6px"></div>'
                      f'</div>'
                      f'<span style="color:{conf_color};font-weight:700;font-size:0.78rem;'
                      f'font-family:monospace;min-width:32px">{conf_pct}%</span></div>')

        # Precision badge
        prec_color = "#059669" if precision>=0.7 else "#D97706" if precision>=0.4 else "#DC2626"
        prec_badge = (f'<span style="color:{prec_color};font-weight:700;'
                      f'font-family:monospace;font-size:0.82rem">'
                      f'{precision*100:.0f}%</span>')

        # Highlight active method
        is_active = (method_name == active_method_label)
        row_bg    = "background:#EFF6FF;" if is_active else (
                    "background:#FAFAFA;" if len(rows)%2==1 else "background:#FFFFFF;")
        active_badge = ('<span style="background:#DBEAFE;color:#1D4ED8;font-size:0.62rem;'
                        'font-weight:700;padding:2px 6px;border-radius:4px;margin-left:6px;'
                        'font-family:monospace">ACTIVE</span>') if is_active else ""

        # Predicted class pill (big, coloured)
        pred_pill = (f'<span style="background:{conf_color}15;color:{conf_color};'
                     f'border:1.5px solid {conf_color}55;border-radius:6px;'
                     f'padding:4px 10px;font-size:0.82rem;font-family:monospace;'
                     f'font-weight:700">{maj_name}</span>')

        rows += (f'<tr style="{row_bg}border-bottom:1px solid #F3F4F6">'
                 f'<td style="padding:11px 16px;color:#111827;font-weight:600;'
                 f'font-size:0.81rem;font-family:Arial,sans-serif;white-space:nowrap">'
                 f'{method_name}{active_badge}</td>'
                 f'<td style="padding:11px 16px">{pred_pill}</td>'
                 f'<td style="padding:11px 16px;min-width:120px">{conf_bar}</td>'
                 f'<td style="padding:11px 16px">{top3_html}</td>'
                 f'<td style="padding:11px 16px;text-align:center">{prec_badge}</td>'
                 f'</tr>')

    th = ('padding:11px 16px;color:#374151;text-align:left;font-size:0.70rem;'
          'letter-spacing:0.1em;font-family:monospace;text-transform:uppercase;font-weight:700')
    th_c = ('padding:11px 16px;color:#374151;text-align:center;font-size:0.70rem;'
            'letter-spacing:0.1em;font-family:monospace;text-transform:uppercase;font-weight:700')

    legend = (
        '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px;'
        'font-size:0.75rem;font-family:Arial,sans-serif;color:#374151">'
        '<span><b style="color:#059669">Green</b> = confident (&gt;70%)</span>'
        '<span><b style="color:#D97706">Amber</b> = uncertain (40–70%)</span>'
        '<span><b style="color:#DC2626">Red</b> = low confidence (&lt;40%)</span>'
        '<span style="margin-left:auto;font-style:italic">Confidence = majority class fraction in Top-K</span>'
        '</div>'
    )

    return (legend +
            f'<div style="border-radius:10px;overflow:hidden;border:1px solid #E5E7EB;'
            f'box-shadow:0 1px 6px rgba(0,0,0,0.05)">'
            f'<table style="width:100%;border-collapse:collapse">'
            f'<thead><tr style="background:#F9FAFB;border-bottom:2px solid #E5E7EB">'
            f'<th style="{th}">Method</th>'
            f'<th style="{th}">Predicted class</th>'
            f'<th style="{th}">Confidence</th>'
            f'<th style="{th}">Top-3 breakdown</th>'
            f'<th style="{th_c}">Precision@{top_k}</th>'
            f'</tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')

# ── Status banner ──────────────────────────────────────────────────────────────
def _status_html(kind, msg):
    styles = {
        "success": ("#ECFDF5","#065F46","#059669","✓"),
        "warning": ("#FFFBEB","#92400E","#D97706","⚠"),
        "error":   ("#FEF2F2","#991B1B","#DC2626","✗"),
        "info":    ("#EFF6FF","#1E40AF","#2563EB","ℹ"),
    }
    bg, text, border, icon = styles.get(kind, styles["info"])
    return (f'<div style="display:flex;align-items:center;gap:10px;padding:12px 18px;'
            f'background:{bg};border-left:4px solid {border};border-radius:8px;'
            f'font-size:0.88rem;color:{text};font-family:Arial,sans-serif;font-weight:500">'
            f'<span style="font-weight:800;font-size:1.1rem;color:{text}">{icon}</span>'
            f'<span style="color:{text}">{msg}</span></div>')

# ── Results figure ─────────────────────────────────────────────────────────────
def _draw_results(query_img, ret_imgs, ret_labels, ret_dists,
                  cnames, qlabel, top_k, method_label, dataset, precision):
    n_cols = min(top_k, 5)
    n_rows = (top_k + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(2.8*(n_cols+1.2), 3.4*(n_rows+0.8)), facecolor=C_BG)

    hax = fig.add_axes([0,0.945,1,0.055]); hax.set_facecolor(C_SURF); hax.axis("off")
    hax.axhline(1, color=C_ACCENT, linewidth=3)
    hax.axhline(0, color=C_BORDER, linewidth=1)
    hax.text(0.014,0.45,"RETRIEVAL RESULTS",color=C_TEXT,fontsize=9,
             fontweight="bold",va="center",transform=hax.transAxes)
    hax.text(0.45,0.45,f"{method_label}  ·  {dataset.upper()}",
             color=C_MUTED,fontsize=7.5,va="center",transform=hax.transAxes)
    pc = C_GREEN if precision>=0.7 else C_AMBER if precision>=0.4 else C_RED
    hax.text(0.84,0.45,f"Precision@{top_k} = {precision*100:.1f}%",
             color=pc,fontsize=9,fontweight="bold",va="center",transform=hax.transAxes)

    gt,gb = 0.935,0.03
    ct = n_cols+1.2; cw = 1.0/ct; ch = (gt-gb)/(n_rows+0.5); pad = 0.012
    NEUTRAL = "#94A3B8"

    def draw_card(x,y,w,h,img,title,sim,is_query=False):
        ax = fig.add_axes([x,y,w-pad,h*0.74])
        ax.imshow(cv2.resize(img,(80,80)),aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(C_ACCENT if is_query else NEUTRAL)
            sp.set_linewidth(3 if is_query else 1.2)
        ax.set_title(title,color=C_ACCENT if is_query else C_TEXT,
                     fontsize=7,pad=3,fontweight="bold" if is_query else "normal")
        if sim is not None:
            bax = fig.add_axes([x,y-0.022,w-pad,0.013])
            bax.set_facecolor(C_DIM); bax.barh(0,sim,color=NEUTRAL,height=1.0)
            bax.set_xlim(0,1); bax.axis("off")
            fig.text(x+(w-pad)/2,y-0.034,f"sim {sim:.3f}",
                     ha="center",color=C_MUTED,fontsize=6)

    qx,qy = 0.008, gb+(n_rows-0.5)*ch
    draw_card(qx,qy,cw*0.92,ch,query_img,"QUERY IMAGE",None,is_query=True)
    bax = fig.add_axes([qx,qy-0.05,cw*0.92-pad,0.034])
    bax.set_facecolor(C_ACCENT); bax.axis("off")
    bax.text(0.5,0.5,(cnames[qlabel] if cnames else str(qlabel)).upper(),
             color="white",fontsize=7,fontweight="bold",
             ha="center",va="center",transform=bax.transAxes)

    for i in range(top_k):
        row=i//n_cols; col=i%n_cols
        sim   = max(0.0,min(1.0,1.0-float(ret_dists[i])))
        lname = (cnames[ret_labels[i]] if cnames else str(ret_labels[i])).upper()
        draw_card((col+1.1)*cw+0.004, gb+(n_rows-row-0.5)*ch,
                  cw*0.9, ch, ret_imgs[i], f"#{i+1}  {lname}", sim)
    return fig

# ── Main retrieval handler ─────────────────────────────────────────────────────
# We store the last query globally so the Predictions tab can use it
_last_query_state = {"arr": None, "dataset": None, "top_k": 10, "method": ""}

def do_retrieve(query_pil, dataset, method_label, top_k):
    global _last_query_state
    if query_pil is None:
        return None, _status_html("warning","Please upload a query image first."), ""
    method = next((k for k,v in METHOD_LABELS.items() if v==method_label), "cnn")
    if dataset not in IMAGE_STORE:
        return None, _status_html("warning",f"Dataset '{dataset}' not loaded — run run_all.py first."), ""
    top_k     = int(top_k)
    query_arr = np.array(query_pil.convert("RGB"))
    gimgs     = IMAGE_STORE[dataset]
    glabels   = LABEL_STORE[dataset]
    cnames    = CLASS_STORE.get(dataset,[str(i) for i in range(10)])

    try:
        qf = _get_query_feature(query_arr, dataset, method)
    except Exception as e:
        return None, _status_html("error",f"Feature extraction error: {e}"), ""

    from scipy.spatial.distance import cdist
    dists      = cdist(qf[None,:], FEATURE_STORE[dataset][method], metric="cosine")[0]
    idx        = np.argsort(dists)[:top_k]
    ret_imgs   = gimgs[idx]
    ret_labels = glabels[idx]
    ret_dists  = dists[idx]
    qlabel     = ret_labels[0]
    precision  = float(np.mean(ret_labels == qlabel))

    # Save state for predictions tab
    _last_query_state = {"arr": query_arr, "dataset": dataset,
                         "top_k": top_k, "method": method_label}

    fig = _draw_results(query_arr, ret_imgs, ret_labels, ret_dists,
                        cnames, qlabel, top_k, method_label, dataset, precision)

    # Status: clean — just method + precision, no majority class
    kind   = "success" if precision>=0.7 else "warning" if precision>=0.4 else "error"
    status = _status_html(kind,
        f"Precision@{top_k}: {precision*100:.1f}%  ·  "
        f"Method: {method_label}  ·  Dataset: {dataset.upper()}  ·  "
        f"Switch to the Predictions tab to compare all methods.")

    # Build predictions table immediately
    pred_html = build_predictions_table(query_arr, dataset, top_k, method_label)

    return fig, status, pred_html

def do_analysis(dataset):
    return _draw_analysis(dataset)

def refresh_predictions(dataset_val, topk_val, method_val):
    """Re-run predictions table when tab is clicked or settings change."""
    s = _last_query_state
    if s["arr"] is None:
        return ('<div style="padding:24px;color:#374151;font-family:Arial,sans-serif;'
                'font-size:0.88rem;text-align:center">'
                'Run a retrieval first — then this tab will compare all methods.</div>')
    return build_predictions_table(s["arr"], dataset_val, int(topk_val), method_val)

# ── Analysis figure ────────────────────────────────────────────────────────────
def _draw_analysis(dataset):
    if dataset not in METRICS_STORE or not METRICS_STORE[dataset]:
        fig, ax = plt.subplots(figsize=(12,5), facecolor=C_BG)
        ax.set_facecolor(C_SURF)
        ax.text(0.5,0.5,"No metrics available — run run_all.py first",
                ha="center",va="center",color=C_TEXT,fontsize=13)
        ax.axis("off"); return fig

    metrics = METRICS_STORE[dataset]
    methods = list(metrics.keys())
    precs   = [metrics[m].get("precision",0) for m in methods]
    maps_   = [metrics[m].get("mAP",0)       for m in methods]
    ndcgs   = [metrics[m].get("ndcg",0)      for m in methods]
    recs    = [metrics[m].get("recall",0)    for m in methods]
    best_i  = int(np.argmax(maps_))

    fig = plt.figure(figsize=(15,9), facecolor=C_BG)
    tax = fig.add_axes([0,0.935,1,0.065]); tax.set_facecolor(C_SURF); tax.axis("off")
    tax.axhline(1,color=C_ACCENT,linewidth=3); tax.axhline(0,color=C_BORDER,linewidth=1)
    tax.text(0.02,0.42,"PERFORMANCE ANALYSIS",color=C_TEXT,fontsize=11,
             fontweight="bold",va="center",transform=tax.transAxes)
    tax.text(0.82,0.42,f"Dataset: {dataset.upper()}",color=C_MUTED,fontsize=9,
             va="center",transform=tax.transAxes)

    short = [m.replace("hybrid_","hyb_").replace("color_","col_") for m in methods]
    x = np.arange(len(methods)); w = 0.19

    ax1 = fig.add_axes([0.05,0.52,0.57,0.37]); ax1.set_facecolor(C_SURF)
    b1=ax1.bar(x-1.5*w,precs,w,label="Precision@10",color=C_ACCENT,alpha=0.85,zorder=3)
    b2=ax1.bar(x-0.5*w,maps_,w,label="mAP",color=C_AMBER,alpha=0.85,zorder=3)
    b3=ax1.bar(x+0.5*w,ndcgs,w,label="NDCG@10",color=C_ACCENT2,alpha=0.85,zorder=3)
    b4=ax1.bar(x+1.5*w,recs,w,label="Recall@10",color=C_GREEN,alpha=0.85,zorder=3)
    for bars in [b1,b2,b3,b4]:
        for bar in bars:
            h=bar.get_height()
            if h>0.025:
                ax1.text(bar.get_x()+bar.get_width()/2,h+0.01,f"{h:.2f}",
                         ha="center",va="bottom",color=C_TEXT,fontsize=5.5)
    ax1.set_xticks(x); ax1.set_xticklabels(short,rotation=30,ha="right",fontsize=8,color=C_TEXT)
    ax1.set_ylim(0,1.18); ax1.set_ylabel("Score",fontsize=9,color=C_TEXT)
    ax1.set_title("All Methods — All Metrics",color=C_TEXT,fontsize=10,fontweight="bold",pad=10)
    ax1.legend(facecolor=C_SURF,labelcolor=C_TEXT,fontsize=7.5,loc="upper right",
               edgecolor=C_BORDER,framealpha=1.0)
    ax1.yaxis.grid(True,linestyle="--",alpha=0.4,zorder=0); ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.get_xticklabels()[best_i].set_color(C_ACCENT)
    ax1.get_xticklabels()[best_i].set_fontweight("bold")

    ax2 = fig.add_axes([0.67,0.52,0.30,0.37]); ax2.set_facecolor(C_SURF)
    si = np.argsort(maps_)
    bars2=ax2.barh(range(len(methods)),[maps_[i] for i in si],
                   color=[C_ACCENT if i==best_i else "#93C5FD" for i in si],
                   alpha=0.9,zorder=3,height=0.6)
    ax2.set_yticks(range(len(methods))); ax2.set_yticklabels([short[i] for i in si],fontsize=7.5,color=C_TEXT)
    ax2.set_xlabel("mAP Score",fontsize=8,color=C_TEXT)
    ax2.set_xlim(0,max(maps_)*1.3 if maps_ else 1)
    ax2.set_title("mAP Ranking",color=C_TEXT,fontsize=10,fontweight="bold",pad=10)
    for bar,val in zip(bars2,[maps_[i] for i in si]):
        ax2.text(val+0.005,bar.get_y()+bar.get_height()/2,f"{val:.3f}",va="center",color=C_TEXT,fontsize=7)
    ax2.xaxis.grid(True,linestyle="--",alpha=0.4,zorder=0); ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    ax3 = fig.add_axes([0.05,0.07,0.38,0.37]); ax3.set_facecolor(C_SURF)
    ax3.scatter(precs,maps_,c=[C_ACCENT if i==best_i else "#6366F1" for i in range(len(methods))],
                s=90,zorder=4,alpha=0.9,edgecolors="white",linewidths=0.8)
    for i,s in enumerate(short):
        ax3.annotate(s,(precs[i],maps_[i]),textcoords="offset points",xytext=(5,4),
                     fontsize=6.5,color=C_TEXT)
    ax3.set_xlabel("Precision@10",fontsize=8,color=C_TEXT)
    ax3.set_ylabel("mAP",fontsize=8,color=C_TEXT)
    ax3.set_title("Precision vs mAP",color=C_TEXT,fontsize=10,fontweight="bold",pad=10)
    ax3.grid(True,linestyle="--",alpha=0.4)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    ax4 = fig.add_axes([0.50,0.07,0.46,0.37],polar=True); ax4.set_facecolor(C_SURF)
    cats=["Precision","Recall×5","mAP","NDCG"]; N=len(cats)
    angles=[n/float(N)*2*np.pi for n in range(N)]+[0]
    top3=np.argsort(maps_)[-3:][::-1]
    for mi,col in zip(top3,[C_ACCENT,C_AMBER,C_GREEN]):
        vals=[precs[mi],recs[mi]*5,maps_[mi],ndcgs[mi],precs[mi]]
        ax4.plot(angles,vals,color=col,linewidth=2,alpha=0.9)
        ax4.fill(angles,vals,color=col,alpha=0.10)
        ax4.scatter(angles[:-1],vals[:-1],color=col,s=35,zorder=5)
    ax4.set_xticks(angles[:-1]); ax4.set_xticklabels(cats,fontsize=8.5,color=C_TEXT)
    ax4.set_ylim(0,1); ax4.set_yticks([0.25,0.5,0.75,1.0])
    ax4.set_yticklabels(["0.25","0.5","0.75","1.0"],fontsize=6,color=C_TEXT)
    ax4.grid(color=C_BORDER,linewidth=0.7)
    ax4.set_title("Top-3 Methods Radar",color=C_TEXT,fontsize=10,fontweight="bold",pad=16)
    ax4.legend([short[i] for i in top3],loc="lower right",facecolor=C_SURF,labelcolor=C_TEXT,
               fontsize=7,edgecolor=C_BORDER,bbox_to_anchor=(1.32,-0.05))
    return fig

# ── Metrics panel ──────────────────────────────────────────────────────────────
def build_summary_cards(dataset):
    if dataset not in METRICS_STORE or not METRICS_STORE[dataset]: return ""
    metrics=METRICS_STORE[dataset]
    best_map=max(v.get("mAP",0) for v in metrics.values())
    best_m=next(k for k,v in metrics.items() if v.get("mAP",0)==best_map)
    avg_prec=np.mean([v.get("precision",0) for v in metrics.values()])
    avg_map=np.mean([v.get("mAP",0) for v in metrics.values()])
    best_prec=max(v.get("precision",0) for v in metrics.values())
    def card(label,value,sub,color):
        return (f'<div style="background:#fff;border:1px solid #E5E7EB;border-top:3px solid {color};'
                f'border-radius:8px;padding:16px 20px;flex:1;min-width:150px;'
                f'box-shadow:0 1px 4px rgba(0,0,0,0.06)">'
                f'<div style="color:#111827;font-size:0.68rem;letter-spacing:0.1em;'
                f'margin-bottom:8px;font-family:monospace;text-transform:uppercase;font-weight:700">{label}</div>'
                f'<div style="color:{color};font-size:1.6rem;font-weight:700;'
                f'font-family:Georgia,serif;margin-bottom:4px">{value}</div>'
                f'<div style="color:#374151;font-size:0.75rem;font-family:Arial,sans-serif">{sub}</div></div>')
    return (f'<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px">'
            +card("Best Method",METHOD_LABELS.get(best_m,best_m).split("—")[0].strip(),f"mAP = {best_map:.4f}","#2563EB")
            +card("Best Precision",f"{best_prec*100:.1f}%","across all methods","#059669")
            +card("Avg Precision",f"{avg_prec*100:.1f}%","mean over all methods","#7C3AED")
            +card("Avg mAP",f"{avg_map:.4f}","mean average precision","#D97706")
            +'</div>')

def build_metrics_table(dataset):
    if dataset not in METRICS_STORE or not METRICS_STORE[dataset]:
        return ('<div style="padding:20px;color:#111827;font-family:Arial,sans-serif;'
                'font-size:0.85rem;background:#fff;border-radius:8px;border:1px solid #E5E7EB">'
                'No metrics available — run run_all.py first.</div>')
    metrics=METRICS_STORE[dataset]; best_map=max(v.get("mAP",0) for v in metrics.values())
    def bar_html(val,color):
        pct=int(val*100)
        return (f'<div style="display:flex;align-items:center;gap:8px">'
                f'<div style="flex:1;background:#F3F4F6;border-radius:4px;height:6px">'
                f'<div style="width:{pct}%;background:{color};border-radius:4px;height:6px"></div></div>'
                f'<span style="color:{color};font-weight:700;min-width:50px;text-align:right;'
                f'font-size:0.8rem;font-family:monospace">{val:.4f}</span></div>')
    rows=""
    for i,(method,m) in enumerate(metrics.items()):
        p=m.get("precision",0); r=m.get("recall",0); mp=m.get("mAP",0); nd=m.get("ndcg",0)
        is_best=mp==best_map
        row_bg="background:#EFF6FF;" if is_best else ("background:#FAFAFA;" if i%2==0 else "background:#FFFFFF;")
        badge=('<span style="background:#DBEAFE;color:#1D4ED8;font-size:0.62rem;font-weight:700;'
               'padding:2px 7px;border-radius:4px;margin-left:8px;font-family:monospace">BEST</span>') if is_best else ""
        p_col="#059669" if p>=0.6 else "#D97706" if p>=0.3 else "#DC2626"
        mp_col="#059669" if mp>=0.5 else "#D97706" if mp>=0.25 else "#DC2626"
        rows+=(f'<tr style="{row_bg}border-bottom:1px solid #F3F4F6">'
               f'<td style="padding:12px 16px;color:#111827;font-weight:600;font-size:0.82rem;'
               f'font-family:Arial,sans-serif">{METHOD_LABELS.get(method,method)}{badge}</td>'
               f'<td style="padding:12px 16px;min-width:150px">{bar_html(p,p_col)}</td>'
               f'<td style="padding:12px 16px;color:#374151;font-size:0.82rem;font-family:monospace">{r:.4f}</td>'
               f'<td style="padding:12px 16px;min-width:150px">{bar_html(mp,mp_col)}</td>'
               f'<td style="padding:12px 16px;color:#374151;font-size:0.82rem;font-family:monospace">{nd:.4f}</td></tr>')
    th='padding:12px 16px;color:#374151;text-align:left;font-size:0.72rem;letter-spacing:0.1em;font-family:monospace;text-transform:uppercase;font-weight:700'
    return (f'<div style="border-radius:10px;overflow:hidden;border:1px solid #E5E7EB;box-shadow:0 1px 6px rgba(0,0,0,0.05)">'
            f'<table style="width:100%;border-collapse:collapse">'
            f'<thead><tr style="background:#F9FAFB;border-bottom:2px solid #E5E7EB">'
            f'<th style="{th}">Method</th><th style="{th}">Precision</th>'
            f'<th style="{th}">Recall</th><th style="{th}">mAP</th><th style="{th}">NDCG</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>')

def refresh_panel(dataset):
    return build_summary_cards(dataset)+build_metrics_table(dataset)

# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
*, *::before, *::after { box-sizing: border-box !important; }
:root {
    --body-text-color: #111827 !important;
    --input-text-fill: #111827 !important;
    --block-label-text-color: #111827 !important;
    --label-text-color: #111827 !important;
    --body-background-fill: #F5F7FA !important;
    --block-background-fill: #FFFFFF !important;
    --input-background-fill: #FFFFFF !important;
    --input-border-color: #D1D5DB !important;
    --border-color-primary: #E5E7EB !important;
    --background-fill-primary: #FFFFFF !important;
    --background-fill-secondary: #F9FAFB !important;
    --neutral-800: #111827 !important;
    --neutral-600: #374151 !important;
    --neutral-400: #9CA3AF !important;
}
body, .gradio-container, .main {
    background-color: #F5F7FA !important;
    color: #111827 !important;
    font-family: Arial, sans-serif !important;
}
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
.block, .gr-box, .gr-panel {
    background: #FFFFFF !important; border: 1px solid #E5E7EB !important;
    border-radius: 10px !important; box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}
label, label span, .block > label > span,
.svelte-1b6s6s, .svelte-1gfkn6j, .label-wrap, .label-wrap span, .gr-label {
    color: #111827 !important; font-weight: 600 !important;
    font-family: Arial, sans-serif !important; font-size: 0.85rem !important;
}
.wrap { background:#FFFFFF !important; border:1.5px solid #D1D5DB !important; border-radius:8px !important; color:#111827 !important; }
.wrap *, .wrap input, .wrap span, .wrap div { color:#111827 !important; background:transparent !important; font-family:Arial,sans-serif !important; font-size:0.85rem !important; }
.wrap input { background:#FFFFFF !important; }
.options, ul.options { background:#FFFFFF !important; border:1px solid #D1D5DB !important; border-radius:8px !important; box-shadow:0 6px 20px rgba(0,0,0,0.1) !important; }
.options li, .item { color:#111827 !important; background:#FFFFFF !important; padding:9px 14px !important; font-size:0.85rem !important; }
.options li:hover, .item:hover { background:#EFF6FF !important; color:#1D4ED8 !important; }
input[type="range"] { accent-color:#2563EB !important; }
input[type="number"] { color:#111827 !important; background:#FFFFFF !important; border:1.5px solid #D1D5DB !important; border-radius:6px !important; font-family:monospace !important; font-size:0.85rem !important; padding:4px 8px !important; }
input[type="text"], input[type="search"], textarea { background:#FFFFFF !important; color:#111827 !important; border:1.5px solid #D1D5DB !important; border-radius:6px !important; font-family:Arial,sans-serif !important; font-size:0.85rem !important; padding:8px 10px !important; }
button.primary, button[variant="primary"] { background:#2563EB !important; color:#FFFFFF !important; border:none !important; border-radius:8px !important; font-weight:600 !important; font-size:0.88rem !important; font-family:Arial,sans-serif !important; padding:11px 20px !important; width:100% !important; box-shadow:0 2px 8px rgba(37,99,235,0.28) !important; cursor:pointer !important; }
button.primary:hover { background:#1D4ED8 !important; }
button.secondary, button[variant="secondary"] { background:#FFFFFF !important; color:#2563EB !important; border:1.5px solid #2563EB !important; border-radius:8px !important; font-weight:600 !important; font-size:0.88rem !important; font-family:Arial,sans-serif !important; padding:11px 20px !important; width:100% !important; cursor:pointer !important; }
button.secondary:hover { background:#EFF6FF !important; }
.tab-nav button, [role="tab"] { color:#6B7280 !important; font-size:0.78rem !important; font-weight:700 !important; letter-spacing:0.08em !important; text-transform:uppercase !important; border-bottom:2px solid transparent !important; padding:10px 18px !important; background:transparent !important; font-family:Arial,sans-serif !important; }
.tab-nav button.selected, [role="tab"][aria-selected="true"] { color:#2563EB !important; border-bottom:2px solid #2563EB !important; }
.tab-nav button:hover { color:#2563EB !important; }
.gr-plot, [class*="plot-container"] { background:#FFFFFF !important; border:1px solid #E5E7EB !important; border-radius:10px !important; padding:4px !important; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#F3F4F6; }
::-webkit-scrollbar-thumb { background:#D1D5DB; border-radius:3px; }
"""

# ── UI ─────────────────────────────────────────────────────────────────────────
def build_dashboard():
    datasets    = list(IMAGE_STORE.keys()) or ["cifar10","mnist","animals10"]
    method_opts = [v for k,v in METHOD_LABELS.items()
                   if k in FEATURE_STORE.get(datasets[0],{})]
    if not method_opts: method_opts = list(METHOD_LABELS.values())
    default_method = method_opts[3] if len(method_opts)>3 else method_opts[0]

    with gr.Blocks(css=CSS, title="Image Retrieval System") as demo:

        gr.HTML("""
        <div style="background:#FFFFFF;border-bottom:1px solid #E5E7EB;padding:24px 36px">
          <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
            <div>
              <div style="font-family:Georgia,serif;font-size:1.5rem;font-weight:700;color:#111827">
                Image Retrieval System
              </div>
              <div style="font-family:monospace;font-size:0.75rem;color:#374151;
                          margin-top:5px;letter-spacing:0.08em;text-transform:uppercase">
                Deep Learning for Computer Vision &nbsp;·&nbsp; Assignment Dashboard
              </div>
            </div>
            <div style="display:flex;gap:10px;flex-wrap:wrap">
              <span style="background:#EFF6FF;color:#1D4ED8;font-size:0.72rem;font-weight:700;padding:5px 14px;border-radius:20px;font-family:monospace;border:1px solid #BFDBFE">CIFAR-10</span>
              <span style="background:#F0FDF4;color:#065F46;font-size:0.72rem;font-weight:700;padding:5px 14px;border-radius:20px;font-family:monospace;border:1px solid #BBF7D0">MNIST</span>
              <span style="background:#FAF5FF;color:#6B21A8;font-size:0.72rem;font-weight:700;padding:5px 14px;border-radius:20px;font-family:monospace;border:1px solid #E9D5FF">ANIMALS-10</span>
            </div>
          </div>
        </div>""")

        with gr.Row(equal_height=False):

            # ── Left panel ─────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=300):
                gr.HTML('<div style="font-family:Arial,sans-serif;font-size:0.78rem;font-weight:700;'
                        'color:#111827;text-transform:uppercase;letter-spacing:0.1em;margin:20px 0 8px 0">Query Image</div>')
                query_img = gr.Image(type="pil", label="", height=220, show_label=False)

                gr.HTML('<div style="font-family:Arial,sans-serif;font-size:0.78rem;font-weight:700;'
                        'color:#111827;text-transform:uppercase;letter-spacing:0.1em;margin:18px 0 8px 0">Configuration</div>')
                dataset_dd = gr.Dropdown(choices=datasets, value=datasets[0],
                                          label="Dataset", interactive=True)
                method_dd  = gr.Dropdown(choices=method_opts, value=default_method,
                                          label="Retrieval Method", interactive=True)
                topk_sl    = gr.Slider(minimum=1, maximum=20, value=10, step=1,
                                       label="Top-K Results")

                gr.HTML('<div style="margin-top:12px"></div>')
                retrieve_btn = gr.Button("Retrieve Similar Images", variant="primary")
                gr.HTML('<div style="margin-top:8px"></div>')
                analysis_btn = gr.Button("Run Full Analysis", variant="secondary")


            # ── Right panel ────────────────────────────────────────────────
            with gr.Column(scale=3):
                status_out = gr.HTML(
                    _status_html("info","Upload an image then click Retrieve — check the Predictions tab to compare all methods.")
                )
                with gr.Tabs():
                    with gr.Tab("RETRIEVED IMAGES"):
                        result_plot = gr.Plot(label="", show_label=False)

                    with gr.Tab("FULL ANALYSIS"):
                        analysis_plot = gr.Plot(label="", show_label=False)

                    with gr.Tab("PREDICTIONS"):
                        gr.HTML('<div style="font-family:Arial,sans-serif;font-size:0.82rem;'
                                'color:#374151;margin-bottom:12px;padding-top:4px">'
                                'After running a retrieval, this table shows what every method predicts '
                                'as the majority class for your query image.</div>')
                        predictions_html = gr.HTML(
                            '<div style="padding:24px;color:#374151;font-family:Arial,sans-serif;'
                            'font-size:0.88rem;text-align:center">'
                            'Run a retrieval first — then this tab will compare all methods.</div>'
                        )
                        refresh_pred_btn = gr.Button("Refresh Predictions", variant="secondary")

                    with gr.Tab("EVALUATION METRICS"):
                        eval_metrics_html = gr.HTML(refresh_panel(datasets[0]))

        gr.HTML("""<div style="text-align:center;padding:18px;border-top:1px solid #E5E7EB;
                    color:#374151;font-size:0.72rem;font-family:monospace;letter-spacing:0.06em">
          IMAGE RETRIEVAL SYSTEM &nbsp;·&nbsp; DEEP LEARNING FOR COMPUTER VISION
        </div>""")


        dataset_dd.change(fn=refresh_panel, inputs=[dataset_dd], outputs=[eval_metrics_html])

        retrieve_btn.click(
            fn=do_retrieve,
            inputs=[query_img, dataset_dd, method_dd, topk_sl],
            outputs=[result_plot, status_out, predictions_html],
        )

        analysis_btn.click(fn=do_analysis, inputs=[dataset_dd], outputs=[analysis_plot])

        refresh_pred_btn.click(
            fn=refresh_predictions,
            inputs=[dataset_dd, topk_sl, method_dd],
            outputs=[predictions_html],
        )

    return demo


if __name__ == "__main__":
    load_stores("./feature_store.pkl")
    build_dashboard().launch(server_name="127.0.0.1", server_port=5000, share=False)