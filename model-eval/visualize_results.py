"""
Professional YOLO training results visualizer.

Reads results.csv from an Ultralytics training run and produces:
  - training_dashboard.png : single-page multi-panel overview
  - losses.png             : train vs val losses (box / cls / dfl)
  - metrics.png            : precision, recall, mAP50, mAP50-95
  - learning_rate.png      : LR schedule per parameter group
  - training_summary.txt   : best epoch, final values, key stats
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

HERE = Path(__file__).resolve().parent
CSV = HERE / "resultsEO.csv"
OUT_DIR = HERE / "viz"
OUT_DIR.mkdir(exist_ok=True)

# ---------- Style ----------
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#222",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "grid.color": "#e3e3e3",
    "grid.linewidth": 0.8,
    "font.family": "DejaVu Sans",
})

COLOR_TRAIN = "#1f77b4"
COLOR_VAL   = "#d62728"
COLOR_MAP50 = "#2ca02c"
COLOR_MAP95 = "#9467bd"
COLOR_PREC  = "#ff7f0e"
COLOR_REC   = "#17becf"

# ---------- Load ----------
df = pd.read_csv(CSV)
df.columns = [c.strip() for c in df.columns]
epochs = df["epoch"].to_numpy()

def smooth(y, k=7):
    """Centered moving-average smoothing, NaN-safe."""
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="same")

def clip_outliers(s, q=0.97):
    """Clip extreme outliers (early-epoch instabilities) to the q-quantile."""
    cap = np.nanquantile(s, q)
    return np.minimum(s, cap), cap

# ---------- Key stats ----------
best_map95_idx = df["metrics/mAP50-95(B)"].idxmax()
best_map50_idx = df["metrics/mAP50(B)"].idxmax()
best_map95_epoch = int(df.loc[best_map95_idx, "epoch"])
best_map50_epoch = int(df.loc[best_map50_idx, "epoch"])
best_map95 = float(df.loc[best_map95_idx, "metrics/mAP50-95(B)"])
best_map50 = float(df.loc[best_map50_idx, "metrics/mAP50(B)"])

final = df.iloc[-1]
total_time_h = df["time"].iloc[-1] / 3600.0

# ---------- 1. DASHBOARD ----------
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.28,
                      left=0.06, right=0.98, top=0.92, bottom=0.07)

fig.suptitle("YOLO Training Results — Dashboard",
             fontsize=20, fontweight="bold", y=0.975)

# --- Row 1: training losses ---
loss_specs = [
    ("train/box_loss", "Box Loss (train)"),
    ("train/cls_loss", "Class Loss (train)"),
    ("train/dfl_loss", "DFL Loss (train)"),
]
for i, (col, title) in enumerate(loss_specs):
    ax = fig.add_subplot(gs[0, i])
    y = df[col].to_numpy()
    ax.plot(epochs, y, color=COLOR_TRAIN, alpha=0.35, linewidth=1, label="raw")
    ax.plot(epochs, smooth(y), color=COLOR_TRAIN, linewidth=2.2, label="smoothed")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")

# --- Row 2 col 0-1: validation losses (clipped to suppress early blow-up) ---
val_cls_clipped, cap_cls = clip_outliers(df["val/cls_loss"], q=0.95)
val_dfl_clipped, cap_dfl = clip_outliers(df["val/dfl_loss"], q=0.95)

ax = fig.add_subplot(gs[1, 0])
y = df["val/box_loss"].to_numpy()
ax.plot(epochs, y, color=COLOR_VAL, alpha=0.35, linewidth=1)
ax.plot(epochs, smooth(y), color=COLOR_VAL, linewidth=2.2, label="smoothed")
ax.set_title("Box Loss (val)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(loc="upper right")

ax = fig.add_subplot(gs[1, 1])
ax.plot(epochs, val_cls_clipped, color=COLOR_VAL, alpha=0.35, linewidth=1)
ax.plot(epochs, smooth(val_cls_clipped), color=COLOR_VAL, linewidth=2.2, label="smoothed")
ax.set_title(f"Class Loss (val) — clipped @ {cap_cls:.1f}")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(loc="upper right")

# --- Row 2 col 2: precision & recall ---
ax = fig.add_subplot(gs[1, 2])
prec = df["metrics/precision(B)"].to_numpy()
rec  = df["metrics/recall(B)"].to_numpy()
ax.plot(epochs, prec, color=COLOR_PREC, alpha=0.35, linewidth=1)
ax.plot(epochs, smooth(prec), color=COLOR_PREC, linewidth=2.2, label="Precision")
ax.plot(epochs, rec,  color=COLOR_REC,  alpha=0.35, linewidth=1)
ax.plot(epochs, smooth(rec),  color=COLOR_REC,  linewidth=2.2, label="Recall")
ax.set_title("Precision & Recall")
ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right")

# --- Row 3 col 0: mAP curves with best-epoch markers ---
ax = fig.add_subplot(gs[2, 0])
m50  = df["metrics/mAP50(B)"].to_numpy()
m95  = df["metrics/mAP50-95(B)"].to_numpy()
ax.plot(epochs, m50, color=COLOR_MAP50, alpha=0.30, linewidth=1)
ax.plot(epochs, smooth(m50), color=COLOR_MAP50, linewidth=2.4, label="mAP@0.5")
ax.plot(epochs, m95, color=COLOR_MAP95, alpha=0.30, linewidth=1)
ax.plot(epochs, smooth(m95), color=COLOR_MAP95, linewidth=2.4, label="mAP@0.5:0.95")
ax.axvline(best_map95_epoch, color=COLOR_MAP95, linestyle="--", alpha=0.6)
ax.scatter([best_map95_epoch], [best_map95], color=COLOR_MAP95, zorder=5,
           s=80, edgecolor="white", linewidth=1.5,
           label=f"best mAP50-95 = {best_map95:.3f} @ {best_map95_epoch}")
ax.set_title("Detection Quality (mAP)")
ax.set_xlabel("Epoch"); ax.set_ylabel("mAP")
ax.set_ylim(0, 1.0)
ax.legend(loc="lower right", fontsize=9)

# --- Row 3 col 1: learning rate ---
ax = fig.add_subplot(gs[2, 1])
ax.plot(epochs, df["lr/pg0"], label="pg0", linewidth=2.0)
ax.plot(epochs, df["lr/pg1"], label="pg1", linewidth=2.0, linestyle="--")
ax.plot(epochs, df["lr/pg2"], label="pg2", linewidth=2.0, linestyle=":")
ax.set_title("Learning Rate Schedule")
ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
ax.set_yscale("log")
ax.legend(loc="upper right")

# --- Row 3 col 2: summary text box ---
ax = fig.add_subplot(gs[2, 2])
ax.axis("off")
summary = (
    f"TRAINING SUMMARY\n"
    f"{'─'*30}\n"
    f"Total epochs       : {int(final['epoch'])}\n"
    f"Total time         : {total_time_h:.2f} h\n"
    f"\n"
    f"BEST CHECKPOINTS\n"
    f"{'─'*30}\n"
    f"mAP@0.5     : {best_map50:.4f}  (epoch {best_map50_epoch})\n"
    f"mAP@0.5:.95 : {best_map95:.4f}  (epoch {best_map95_epoch})\n"
    f"\n"
    f"FINAL EPOCH ({int(final['epoch'])})\n"
    f"{'─'*30}\n"
    f"Precision  : {final['metrics/precision(B)']:.4f}\n"
    f"Recall     : {final['metrics/recall(B)']:.4f}\n"
    f"mAP@0.5    : {final['metrics/mAP50(B)']:.4f}\n"
    f"mAP@0.5:.95: {final['metrics/mAP50-95(B)']:.4f}\n"
    f"train box  : {final['train/box_loss']:.4f}\n"
    f"val box    : {final['val/box_loss']:.4f}\n"
)
ax.text(0.0, 1.0, summary, family="monospace", fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f6f8fa",
                  edgecolor="#cfd6dd"))

dashboard_path = OUT_DIR / "training_dashboard.png"
fig.savefig(dashboard_path, dpi=160, bbox_inches="tight")
plt.close(fig)

# ---------- 2. LOSSES (train vs val side-by-side) ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle("Training vs Validation Losses",
             fontsize=17, fontweight="bold", y=1.02)

loss_pairs = [
    ("box_loss", "Box Loss"),
    ("cls_loss", "Class Loss"),
    ("dfl_loss", "DFL Loss"),
]
for ax, (key, title) in zip(axes, loss_pairs):
    train = df[f"train/{key}"].to_numpy()
    val   = df[f"val/{key}"].to_numpy()
    if key in ("cls_loss", "dfl_loss"):
        val, cap = clip_outliers(val, q=0.95)
        sub = f"  (val clipped @ {cap:.1f})"
    else:
        sub = ""
    ax.plot(epochs, train, color=COLOR_TRAIN, alpha=0.30, linewidth=1)
    ax.plot(epochs, smooth(train), color=COLOR_TRAIN, linewidth=2.2, label="train")
    ax.plot(epochs, val, color=COLOR_VAL, alpha=0.30, linewidth=1)
    ax.plot(epochs, smooth(val), color=COLOR_VAL, linewidth=2.2, label="val")
    ax.set_title(title + sub)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(OUT_DIR / "losses.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ---------- 3. METRICS panel ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Detection Metrics Over Training",
             fontsize=17, fontweight="bold", y=0.995)

ax = axes[0, 0]
ax.plot(epochs, prec, color=COLOR_PREC, alpha=0.3, linewidth=1)
ax.plot(epochs, smooth(prec), color=COLOR_PREC, linewidth=2.4)
ax.set_title("Precision"); ax.set_xlabel("Epoch"); ax.set_ylabel("Precision")
ax.set_ylim(0, 1.05)

ax = axes[0, 1]
ax.plot(epochs, rec, color=COLOR_REC, alpha=0.3, linewidth=1)
ax.plot(epochs, smooth(rec), color=COLOR_REC, linewidth=2.4)
ax.set_title("Recall"); ax.set_xlabel("Epoch"); ax.set_ylabel("Recall")
ax.set_ylim(0, 1.05)

ax = axes[1, 0]
ax.plot(epochs, m50, color=COLOR_MAP50, alpha=0.3, linewidth=1)
ax.plot(epochs, smooth(m50), color=COLOR_MAP50, linewidth=2.4)
ax.axvline(best_map50_epoch, color=COLOR_MAP50, linestyle="--", alpha=0.6)
ax.scatter([best_map50_epoch], [best_map50], color=COLOR_MAP50,
           s=90, edgecolor="white", linewidth=1.5, zorder=5,
           label=f"best = {best_map50:.3f} @ ep{best_map50_epoch}")
ax.set_title("mAP @ 0.5"); ax.set_xlabel("Epoch"); ax.set_ylabel("mAP@0.5")
ax.set_ylim(0, 1.0); ax.legend(loc="lower right")

ax = axes[1, 1]
ax.plot(epochs, m95, color=COLOR_MAP95, alpha=0.3, linewidth=1)
ax.plot(epochs, smooth(m95), color=COLOR_MAP95, linewidth=2.4)
ax.axvline(best_map95_epoch, color=COLOR_MAP95, linestyle="--", alpha=0.6)
ax.scatter([best_map95_epoch], [best_map95], color=COLOR_MAP95,
           s=90, edgecolor="white", linewidth=1.5, zorder=5,
           label=f"best = {best_map95:.3f} @ ep{best_map95_epoch}")
ax.set_title("mAP @ 0.5:0.95"); ax.set_xlabel("Epoch"); ax.set_ylabel("mAP@0.5:0.95")
ax.set_ylim(0, 1.0); ax.legend(loc="lower right")

fig.tight_layout()
fig.savefig(OUT_DIR / "metrics.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ---------- 4. Learning rate ----------
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(epochs, df["lr/pg0"], label="pg0 (bias)", linewidth=2.2)
ax.plot(epochs, df["lr/pg1"], label="pg1 (weights)", linewidth=2.2, linestyle="--")
ax.plot(epochs, df["lr/pg2"], label="pg2 (other)", linewidth=2.2, linestyle=":")
ax.set_title("Learning Rate Schedule (per parameter group)",
             fontsize=15, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
ax.set_yscale("log")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(OUT_DIR / "learning_rate.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ---------- 5. Text summary ----------
summary_path = OUT_DIR / "training_summary.txt"
with open(summary_path, "w") as f:
    f.write("YOLO TRAINING — SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Source CSV         : {CSV}\n")
    f.write(f"Total epochs       : {int(final['epoch'])}\n")
    f.write(f"Total wall-clock   : {total_time_h:.2f} h "
            f"({df['time'].iloc[-1]:.0f} s)\n\n")
    f.write("BEST CHECKPOINTS\n")
    f.write("-" * 50 + "\n")
    f.write(f"  mAP@0.5         : {best_map50:.4f}  at epoch {best_map50_epoch}\n")
    f.write(f"  mAP@0.5:0.95    : {best_map95:.4f}  at epoch {best_map95_epoch}\n\n")
    f.write("FINAL EPOCH METRICS\n")
    f.write("-" * 50 + "\n")
    for col in ["metrics/precision(B)", "metrics/recall(B)",
                "metrics/mAP50(B)", "metrics/mAP50-95(B)",
                "train/box_loss", "train/cls_loss", "train/dfl_loss",
                "val/box_loss", "val/cls_loss", "val/dfl_loss"]:
        f.write(f"  {col:<24}: {final[col]:.4f}\n")

print(f"Saved outputs to: {OUT_DIR}")
for p in sorted(OUT_DIR.iterdir()):
    print(" -", p.name)
