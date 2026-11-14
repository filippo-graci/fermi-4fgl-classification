"""Draws the two deck-specific figures from the committed result CSVs.

Run from the repository root:  python docs/presentation/make_figures.py
The other images in figures/ are copied unchanged from the run's own plots.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CSV = Path("outputs_4FGL_DR4_macro_f1_final_single_split/validation/csv_reports")
OUT = Path(__file__).parent / "figures"

# Reference palette, categorical slots 1-3 (validated: CVD dE >= 9, normal-vision dE >= 27).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})


def family_macro_f1():
    d = pd.read_csv(CSV / "train_validation_test_metric_comparison_with_ci.csv").set_index("family")
    fams = [("fine", "Flat (fine)", BLUE), ("merged", "2-level (merged)", ORANGE),
            ("3level", "3-level", AQUA)]
    fig, ax = plt.subplots(figsize=(5.6, 2.1))
    for y, (key, label, color) in enumerate(reversed(fams)):
        r = d.loc[key]
        mid, lo, hi = r["test_macro_f1_mean"], r["test_macro_f1_ci_low"], r["test_macro_f1_ci_high"]
        ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round")
        ax.plot(mid, y, "o", ms=8, color=color, mec="white", mew=2)
        ax.text(hi + 0.004, y, f"{mid:.3f}  [{lo:.3f}, {hi:.3f}]", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(3), [f[1] for f in reversed(fams)], color=INK)
    ax.set_xlim(0.25, 0.44)
    ax.set_xlabel("Test macro-F1, 95% bootstrap interval")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.savefig(OUT / "family_macro_f1.pdf", metadata={"CreationDate": None})


def reliability_fine():
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, label="perfect calibration")
    for name, label, color in [("raw", "raw (ECE 0.128)", BLUE),
                               ("calibrated", "calibrated (ECE 0.043)", ORANGE)]:
        r = pd.read_csv(CSV / f"calibration_reliability_fine_{name}.csv").dropna()
        r = r[r["n"] >= 5]  # ponytail: bins with < 5 test rows are noise; the CSV keeps them all
        ax.plot(r["mean_predicted_confidence"], r["empirical_accuracy"], "-o", lw=2, ms=6,
                color=color, mec="white", mew=1.5, label=label)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    fig.savefig(OUT / "reliability_fine.pdf", metadata={"CreationDate": None})


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    family_macro_f1()
    reliability_fine()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
