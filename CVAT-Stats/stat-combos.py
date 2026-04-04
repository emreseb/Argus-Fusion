import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

"""
Dataset Statistics Counter — Drone only (Item=1)
-------------------------------------------------
Filename format:  {I}_{LDB}_{S}_{ExperimentName}_frame{###}.ext

  [0] I   : always 1 (Drone) — Bird skipped
  [1] LDB : 3 digits
              L (digit 0): Light      0=Bright, 1=Low
              D (digit 1): Distance   0=Close,  1=Far
              B (digit 2): Background 0=Clear,  1=Cluttered
  [2] S   : Sensor  0=EO, 1=IR
"""

LIGHT_MAP      = {"0": "Bright",    "1": "Low"}
DISTANCE_MAP   = {"0": "Close",     "1": "Far"}
BACKGROUND_MAP = {"0": "Clear",     "1": "Cluttered"}
SENSOR_MAP     = {"0": "EO",        "1": "IR"}

CATEGORY_CONFIGS = [
    ("Light",      LIGHT_MAP,      ["#EF9F27", "#534AB7"]),
    ("Distance",   DISTANCE_MAP,   ["#D85A30", "#0F6E56"]),
    ("Background", BACKGROUND_MAP, ["#888780", "#D4537E"]),
    ("Sensor",     SENSOR_MAP,     ["#185FA5", "#3B6D11"]),
]

CAT_MAPS = {
    "Light":      LIGHT_MAP,
    "Distance":   DISTANCE_MAP,
    "Background": BACKGROUND_MAP,
    "Sensor":     SENSOR_MAP,
}

# Drone EO vs IR broken down by these conditions
CONDITION_BREAKDOWN = ["Light", "Distance", "Background"]

SENSOR_COLORS = {"0": "#185FA5", "1": "#3B6D11"}  # EO=blue, IR=green

BG_COLOR = "#F8F8F8"


def extract_prefix_tokens(filename):
    stem = os.path.splitext(filename)[0]
    tokens = []
    for token in stem.split("_"):
        if token.isdigit():
            tokens.append(token)
        else:
            break
    return tokens


def extract_experiment_name(filename):
    """Extracts the experiment label (e.g. E1, ERF2, R5) from the filename."""
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    # Skip leading numeric tokens, first non-numeric is the experiment name
    for part in parts:
        if not part.isdigit():
            return part
    return "Unknown"


def parse_file(filename):
    tokens = extract_prefix_tokens(filename)
    if len(tokens) < 3:
        return None
    if tokens[0] != "1":          # skip non-Drone
        return None
    LDB = tokens[1]
    if len(LDB) != 3:
        return None
    frame_match = re.search(r"frame(\d+)", filename)
    frame_id = int(frame_match.group(1)) if frame_match else None
    return {
        "Light":      LDB[0],
        "Distance":   LDB[1],
        "Background": LDB[2],
        "Sensor":     tokens[2],
        "Experiment": extract_experiment_name(filename),
        "FrameID":    frame_id,
    }


def count_files(base_dir):
    counts        = {cat: defaultdict(int) for cat in CAT_MAPS}
    # condition_sensor[(cond_cat, cond_val, sensor_val)] -> count
    condition_sensor = defaultdict(int)
    # experiment_sensor[(exp_name, sensor_val)] -> count
    experiment_sensor = defaultdict(int)
    # experiment_condition[(exp_name, bg_label, light_label)] -> count
    experiment_condition = defaultdict(int)

    warnings = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.endswith((".jpg", ".txt")):
                continue
            parsed = parse_file(file)
            if parsed is None:
                warnings.append(f"Skipped: {file}")
                continue

            s = parsed["Sensor"]
            exp = parsed["Experiment"]

            for cat in CAT_MAPS:
                val = parsed[cat]
                if val in CAT_MAPS[cat]:
                    counts[cat][val] += 1

            for cond in CONDITION_BREAKDOWN:
                cond_val = parsed[cond]
                if cond_val in CAT_MAPS[cond] and s in SENSOR_MAP:
                    condition_sensor[(cond, cond_val, s)] += 1

            if s in SENSOR_MAP:
                experiment_sensor[(exp, s)] += 1

            b_label = BACKGROUND_MAP.get(parsed["Background"], "?")
            l_label = LIGHT_MAP.get(parsed["Light"], "?")
            experiment_condition[(exp, b_label, l_label)] += 1

    return counts, condition_sensor, experiment_sensor, experiment_condition, warnings


def style_ax(ax, title, fontsize=11):
    ax.set_facecolor(BG_COLOR)
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=8)
    ax.yaxis.grid(True, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8, colors="#888888")


def bar_labels(ax, bars, total):
    for bar in bars:
        h = bar.get_height()
        if h == 0:
            continue
        pct = f"{h / total * 100:.1f}%" if total else "0%"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + total * 0.005,
            f"{int(h)}\n({pct})",
            ha="center", va="bottom", fontsize=8, color="#333333"
        )


# ──────────────────────────────────────────────
# Plot 1 — Per-category breakdown (4 charts)
# ──────────────────────────────────────────────
def plot_categories(counts):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Per-Category Breakdown (Drone only)", fontsize=13, fontweight="bold", y=1.02)

    for ax, (cat_name, cat_map, colors) in zip(axes, CATEGORY_CONFIGS):
        labels = list(cat_map.values())
        values = [counts[cat_name][k] for k in cat_map]
        total  = sum(values) or 1
        bars   = ax.bar(labels, values, color=colors, width=0.5, zorder=2)
        style_ax(ax, cat_name)
        bar_labels(ax, bars, total)
        ax.set_ylim(0, max(values, default=1) * 1.3)

    fig.tight_layout()
    fig.savefig("plot1_categories.png", dpi=150, bbox_inches="tight")
    print("Saved: plot1_categories.png")


# ──────────────────────────────────────────────
# Plot 2 — Drone EO vs IR per condition
# ──────────────────────────────────────────────
def plot_condition_sensor(condition_sensor):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Drone EO vs IR — broken down by condition", fontsize=13, fontweight="bold", y=1.02)

    width = 0.35
    sensor_keys   = ["0", "1"]
    sensor_labels = ["EO", "IR"]
    sensor_colors = [SENSOR_COLORS["0"], SENSOR_COLORS["1"]]

    for ax, cond in zip(axes, CONDITION_BREAKDOWN):
        cond_keys   = list(CAT_MAPS[cond].keys())
        cond_labels = [CAT_MAPS[cond][k] for k in cond_keys]
        x     = np.arange(len(cond_keys))
        total = sum(condition_sensor[(cond, ck, sk)]
                    for ck in cond_keys for sk in sensor_keys) or 1

        for j, (sk, slabel, color) in enumerate(zip(sensor_keys, sensor_labels, sensor_colors)):
            values = [condition_sensor[(cond, ck, sk)] for ck in cond_keys]
            offset = (j - 0.5) * width
            bars = ax.bar(x + offset, values, width=width,
                          label=slabel, color=color, zorder=2)
            bar_labels(ax, bars, total)

        style_ax(ax, cond)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels)
        ax.set_ylim(0, max(
            condition_sensor[(cond, ck, sk)]
            for ck in cond_keys for sk in sensor_keys
        ) * 1.35 or 1)
        ax.legend(fontsize=9, framealpha=0)

    fig.tight_layout()
    fig.savefig("plot2_eo_ir_by_condition.png", dpi=150, bbox_inches="tight")
    print("Saved: plot2_eo_ir_by_condition.png")


# ──────────────────────────────────────────────
# Plot 3 — Frame count per experiment × sensor
# ──────────────────────────────────────────────
def plot_experiment_sensor(experiment_sensor):
    experiments = sorted({exp for exp, _ in experiment_sensor})
    eo_counts = [experiment_sensor[(exp, "0")] for exp in experiments]
    ir_counts = [experiment_sensor[(exp, "1")] for exp in experiments]

    x     = np.arange(len(experiments))
    width = 0.35
    total = sum(experiment_sensor.values()) or 1

    fig_h = max(4.5, len(experiments) * 0.45)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    fig.patch.set_facecolor(BG_COLOR)

    bars_eo = ax.barh(x + width / 2, eo_counts, width, label="EO",
                      color=SENSOR_COLORS["0"], zorder=2)
    bars_ir = ax.barh(x - width / 2, ir_counts, width, label="IR",
                      color=SENSOR_COLORS["1"], zorder=2)

    for bar in list(bars_eo) + list(bars_ir):
        w = bar.get_width()
        if w == 0:
            continue
        ax.text(w + total * 0.002, bar.get_y() + bar.get_height() / 2,
                str(int(w)), va="center", fontsize=8, color="#333333")

    ax.set_facecolor(BG_COLOR)
    ax.set_title("Frame count per experiment (EO vs IR)", fontsize=12,
                 fontweight="bold", pad=8)
    ax.xaxis.grid(True, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.set_yticks(x)
    ax.set_yticklabels(experiments, fontsize=9)
    ax.tick_params(axis="x", labelsize=8, colors="#888888")
    ax.legend(fontsize=9, framealpha=0)

    fig.tight_layout()
    fig.savefig("plot3_frames_per_experiment.png", dpi=150, bbox_inches="tight")
    print("Saved: plot3_frames_per_experiment.png")


# ──────────────────────────────────────────────
# Plot 4 — Frame count per experiment × condition
# ──────────────────────────────────────────────
def plot_experiment_condition(experiment_condition):
    """Stacked horizontal bar: each experiment bar split by
       (Background × Light) combination."""
    experiments = sorted({exp for exp, _, _ in experiment_condition})

    combos = [
        ("Clear",     "Bright", "#9FE1CB"),
        ("Clear",     "Low",    "#5DCAA5"),
        ("Cluttered", "Bright", "#F5C4B3"),
        ("Cluttered", "Low",    "#D85A30"),
    ]
    combo_labels = [f"{b} + {l}" for b, l, _ in combos]

    data = np.array([
        [experiment_condition[(exp, b, l)] for b, l, _ in combos]
        for exp in experiments
    ])

    fig_h = max(4.5, len(experiments) * 0.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    fig.patch.set_facecolor(BG_COLOR)

    lefts = np.zeros(len(experiments))
    for col_idx, (_, _, color) in enumerate(combos):
        values = data[:, col_idx]
        bars = ax.barh(experiments, values, left=lefts,
                       label=combo_labels[col_idx], color=color, zorder=2)
        for bar, val in zip(bars, values):
            if val < 1:
                continue
            cx = bar.get_x() + bar.get_width() / 2
            cy = bar.get_y() + bar.get_height() / 2
            ax.text(cx, cy, str(int(val)), ha="center", va="center",
                    fontsize=7.5, color="#222222")
        lefts += values

    ax.set_facecolor(BG_COLOR)
    ax.set_title("Frames per experiment — Background × Light condition",
                 fontsize=12, fontweight="bold", pad=8)
    ax.xaxis.grid(True, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, colors="#888888")
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=9, framealpha=0, loc="lower right")

    fig.tight_layout()
    fig.savefig("plot4_experiment_conditions.png", dpi=150, bbox_inches="tight")
    print("Saved: plot4_experiment_conditions.png")


def main():
    base_dir = "/home/emre/Desktop/NATO/DATASETv3/images(all)"
    counts, condition_sensor, experiment_sensor, experiment_condition, warnings = count_files(base_dir)

    # Console output
    print("Dataset Statistics (Drone only):")
    print("-" * 35)
    for cat_name, cat_map, _ in CATEGORY_CONFIGS:
        print(f"\n  {cat_name}:")
        for val, label in cat_map.items():
            print(f"    {label:<12}: {counts[cat_name][val]}")

    print("\n  Drone EO vs IR by condition:")
    for cond in CONDITION_BREAKDOWN:
        print(f"\n    {cond}:")
        for ck, cl in CAT_MAPS[cond].items():
            for sk, sl in SENSOR_MAP.items():
                n = condition_sensor[(cond, ck, sk)]
                print(f"      {cl} + {sl:<4}: {n}")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings[:20]:
            print(f"  ⚠  {w}")

    plot_categories(counts)
    plot_condition_sensor(condition_sensor)
    plot_experiment_sensor(experiment_sensor)
    plot_experiment_condition(experiment_condition)

    plt.show()


if __name__ == "__main__":
    main()