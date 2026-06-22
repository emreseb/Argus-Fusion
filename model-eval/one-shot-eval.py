import csv

from ultralytics import YOLO

# --- config ---------------------------------------------------------------
WEIGHTS = '/home/emre/Desktop/NATO/code/fusion/models/ir/IR-150-epoch.pt'
DATA = '/home/emre/Desktop/NATO/code/yaml-files/ir-model.yaml'
OUT_CSV = '/home/emre/Desktop/NATO/code/model-eval/eval_metrics.csv'


def f1(p, r):
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


model = YOLO(WEIGHTS)

# verbose=True also prints Ultralytics' own per-class table.
metrics = model.val(data=DATA, split='val', verbose=True)
box = metrics.box
names = metrics.names if hasattr(metrics, "names") else model.names


def cname(ci):
    return names.get(ci, str(ci)) if isinstance(names, dict) else names[ci]


# --- overall --------------------------------------------------------------
# NOTE: detection has no single "accuracy" metric (no fixed total-sample
# denominator). The standard reportable set is precision / recall / F1 / mAP.
overall = {
    "precision": float(box.mp),
    "recall": float(box.mr),
    "f1": f1(float(box.mp), float(box.mr)),
    "mAP50": float(box.map50),
    "mAP75": float(box.map75),
    "mAP50-95": float(box.map),
}

print("\n===================== OVERALL (all classes) =====================")
for k, v in overall.items():
    print(f"  {k:<10}: {v:.4f}")

# --- per class ------------------------------------------------------------
# box.p/r/f1/ap50/ap are aligned with box.ap_class_index (classes with GT).
rows = []
print("\n===================== PER CLASS =================================")
print(f"  {'class':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'mAP50':>10}{'mAP50-95':>10}")
for i, ci in enumerate(box.ap_class_index):
    p, r, ap50, ap = float(box.p[i]), float(box.r[i]), float(box.ap50[i]), float(box.ap[i])
    name = cname(ci)
    print(f"  {name:<12}{p:>10.4f}{r:>10.4f}{f1(p, r):>10.4f}{ap50:>10.4f}{ap:>10.4f}")
    rows.append({
        "class": name, "precision": p, "recall": r, "f1": f1(p, r),
        "mAP50": ap50, "mAP50-95": ap,
    })

# --- confusion-matrix counts (TP / FP / FN) -------------------------------
# Gives the raw detection counts behind precision/recall, handy for a report.
try:
    cm = metrics.confusion_matrix.matrix  # shape (nc+1, nc+1); last index = background
    nc = cm.shape[0] - 1
    print("\n===================== COUNTS (TP/FP/FN) ========================")
    print(f"  {'class':<12}{'TP':>8}{'FP':>8}{'FN':>8}")
    for ci in range(nc):
        tp = float(cm[ci, ci])
        fp = float(cm[ci, :].sum() - tp)   # predicted ci, was something else / background
        fn = float(cm[:, ci].sum() - tp)   # was ci, predicted something else / missed
        print(f"  {cname(ci):<12}{tp:>8.0f}{fp:>8.0f}{fn:>8.0f}")
except Exception as e:
    print(f"\n(confusion-matrix counts unavailable: {e})")

# --- write CSV for easy copy-paste into a table ---------------------------
with open(OUT_CSV, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["scope", "precision", "recall", "f1", "mAP50", "mAP75", "mAP50-95"])
    w.writerow(["overall", overall["precision"], overall["recall"], overall["f1"],
                overall["mAP50"], overall["mAP75"], overall["mAP50-95"]])
    for row in rows:
        w.writerow([row["class"], row["precision"], row["recall"], row["f1"],
                    row["mAP50"], "", row["mAP50-95"]])
print(f"\nSaved metrics table -> {OUT_CSV}")
