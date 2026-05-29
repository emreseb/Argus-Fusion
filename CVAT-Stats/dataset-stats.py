import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

"""
Dataset Statistics Counter — Correct Naming Convention
-------------------------------------------------------
Filename format:  {I}_{LDB}_{S}_{ExperimentName}_frame{###}.ext

  [0] I   : 1 digit  — Item       0=Bird, 1=Drone
  [1] LDB : 3 digits — combined field:
              L (digit 0): Light      0=Bright, 1=Low (rain/fog/snow)
              D (digit 1): Distance   0=Close (x<50), 1=Far (x>50)
              B (digit 2): Background 0=Clear, 1=Cluttered
  [2] S   : 1 digit  — Sensor     0=EO, 1=IR
  [3+]    : Experiment name + frame ID — ignored

Example: 1_001_0_ERF2_frame000032
  I=1 (Drone), L=0 (Bright), D=0 (Close), B=1 (Cluttered), S=0 (EO)
"""

CATEGORY_CONFIGS = [
    ("Item",       {"0": "Bird",      "1": "Drone", "2": "No Drone"    }, ["#378ADD", "#1D9E75"]),
    ("Light",      {"0": "Bright",    "1": "Low"      }, ["#EF9F27", "#534AB7"]),
    ("Distance",   {"0": "Close",     "1": "Far"      }, ["#D85A30", "#0F6E56"]),
    ("Background", {"0": "Clear",     "1": "Cluttered"}, ["#888780", "#D4537E"]),
    ("Sensor",     {"0": "EO",        "1": "IR"       }, ["#185FA5", "#3B6D11"]),
]


def extract_prefix_tokens(filename):
    stem = os.path.splitext(filename)[0]
    tokens = []
    for token in stem.split("_"):
        if token.isdigit():
            tokens.append(token)
        else:
            break
    return tokens


def count_files_by_category(base_dir):
    counts = {cat: {} for cat, _, _ in CATEGORY_CONFIGS}
    warnings = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.endswith((".jpg", ".txt")):
                continue

            tokens = extract_prefix_tokens(file)

            if len(tokens) < 3:
                warnings.append(f"Skipped (only {len(tokens)} tokens): {file}")
                continue

            I_tok   = tokens[0]
            LDB_tok = tokens[1]
            S_tok   = tokens[2]

            if len(LDB_tok) != 3:
                warnings.append(f"Unexpected LDB token '{LDB_tok}' in: {file}")
                continue

            L_tok = LDB_tok[0]
            D_tok = LDB_tok[1]
            B_tok = LDB_tok[2]

            token_map = {
                "Item":       I_tok,
                "Light":      L_tok,
                "Distance":   D_tok,
                "Background": B_tok,
                "Sensor":     S_tok,
            }

            for cat_name, cat_map, _ in CATEGORY_CONFIGS:
                val = token_map[cat_name]
                if val not in cat_map:
                    warnings.append(f"Unknown value '{val}' for {cat_name} in: {file}")
                    continue
                counts[cat_name][val] = counts[cat_name].get(val, 0) + 1

    return counts, warnings


def plot_stats(counts, output_path="dataset_stats.png"):
    n = len(CATEGORY_CONFIGS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    fig.patch.set_facecolor("#F8F8F8")

    for ax, (cat_name, cat_map, colors) in zip(axes, CATEGORY_CONFIGS):
        labels = list(cat_map.values())
        values = [counts[cat_name].get(k, 0) for k in cat_map]
        total  = sum(values) or 1

        bars = ax.bar(labels, values, color=colors, width=0.5, zorder=2)

        ax.set_facecolor("#F8F8F8")
        ax.set_title(cat_name, fontsize=13, fontweight="bold", pad=10)
        ax.yaxis.grid(True, color="#DDDDDD", zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=9, colors="#888888")

        for bar, val in zip(bars, values):
            pct = f"{val / total * 100:.1f}%"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.01,
                f"{val}\n({pct})",
                ha="center", va="bottom", fontsize=10, color="#333333"
            )

        ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)

    plt.suptitle("Dataset Statistics", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    plt.show()


def main():
    base_dir = "/home/emre/Desktop/NATO/DATASETv3/images(all)"
    counts, warnings = count_files_by_category(base_dir)

    print("Dataset Statistics:")
    print("-" * 35)
    for cat_name, cat_map, _ in CATEGORY_CONFIGS:
        print(f"\n  {cat_name}:")
        for val, label in cat_map.items():
            n = counts[cat_name].get(val, 0)
            print(f"    {label:<12}: {n}")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠  {w}")

    plot_stats(counts, output_path="dataset_stats.png")


if __name__ == "__main__":
    main()