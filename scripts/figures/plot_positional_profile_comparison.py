#!/usr/bin/env python3
"""
Grouped bars of Spearman rho against the observed held-out positional profile,
with bootstrap 95% CIs and the split-half noise ceiling.

Reads the metrics tables written by scripts/controls/compare_positional_profiles.py.

Three series are drawn: the two models and one site-frequency baseline. The
second baseline variant (all rows vs unique pairs) differs by at most 0.035 and
by 0.000 on spike, so it stays in the metrics CSV rather than adding a fourth
bar; a fourth categorical slot would also put yellow beside orange, which fails
the normal-vision separation floor.

The noise ceiling is a reference level, not a predictor, so it is drawn as a
neutral rule across each group rather than as another bar. Bars that reach it
cannot be improved on at this sample size.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# Categorical slots 1-3 of the reference palette, in their documented order.
SERIES_COLORS: Dict[str, str] = {
    "evoPLM-Tree": "#2a78d6",
    "sequence-only": "#eb6834",
    "site-freq (unique pairs)": "#1baf7a",
}
SERIES_LABELS: Dict[str, str] = {
    "evoPLM-Tree": "evoPLM-Tree",
    "sequence-only": "Sequence-only",
    "site-freq (unique pairs)": "Site-frequency baseline",
}
CEILING_ROW = "noise ceiling (Spearman-Brown)"

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#d8d7d2"
CEILING_COLOR = "#52514e"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument(
        "--metrics_dir",
        default="results/controls",
        help="Directory holding positional_{task}_{conv}_metrics.csv",
    )
    ap.add_argument(
        "--out_png",
        default="results/controls/plots/positional_profile_comparison.png",
    )
    return ap.parse_args()


def load_groups(metrics_dir: Path) -> List[dict]:
    """One entry per task x counting convention, in display order."""
    groups = []
    for task, task_label in (("rbd", "RBD"), ("spike", "Full spike")):
        for conv, conv_label in (("raw", "all positions"), ("noamb", "X / * excluded")):
            path = metrics_dir / f"positional_{task}_{conv}_metrics.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Run cluster/slurm/controls/compare_positional_profiles.sh first."
                )
            df = pd.read_csv(path).set_index("predictor")
            groups.append(
                {
                    "label": f"{task_label}\n({conv_label})",
                    "df": df,
                    "ceiling": float(df.loc[CEILING_ROW, "spearman_rho"]),
                }
            )
    return groups


def main() -> None:
    args = parse_args()
    groups = load_groups(Path(args.metrics_dir))

    n_groups = len(groups)
    series = list(SERIES_COLORS)
    n_series = len(series)
    bar_w = 0.24
    gap = 0.02  # 2px-equivalent surface gap between adjacent bars

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for g_i, group in enumerate(groups):
        df = group["df"]
        # Centre the series block on the group position.
        offsets = [(s_i - (n_series - 1) / 2) * (bar_w + gap) for s_i in range(n_series)]

        for s_i, name in enumerate(series):
            if name not in df.index:
                continue
            row = df.loc[name]
            x = g_i + offsets[s_i]
            rho = float(row["spearman_rho"])

            ax.bar(
                x,
                rho,
                width=bar_w,
                color=SERIES_COLORS[name],
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )

            # CI drawn as an absolute interval, not as a +/- around the bar top:
            # resampling perturbs both correlated vectors, so the point estimate
            # can sit near or above the upper percentile.
            lo, hi = float(row["ci95_low"]), float(row["ci95_high"])
            if pd.notna(lo) and pd.notna(hi):
                ax.vlines(x, lo, hi, color=TEXT_PRIMARY, linewidth=1.4, zorder=5)
                for y in (lo, hi):
                    ax.hlines(y, x - bar_w * 0.22, x + bar_w * 0.22,
                              color=TEXT_PRIMARY, linewidth=1.4, zorder=5)

            # Keep the value clear of both the whisker and the ceiling rule.
            label_y = max(rho, hi if pd.notna(hi) else rho) + 0.022
            if abs(label_y - group["ceiling"]) < 0.035:
                label_y = group["ceiling"] + 0.028
            ax.text(
                x,
                label_y,
                f"{rho:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                color=TEXT_PRIMARY,
                zorder=6,
            )

        half = (n_series * bar_w + (n_series - 1) * gap) / 2 + 0.05
        ax.hlines(
            group["ceiling"],
            g_i - half,
            g_i + half,
            color=CEILING_COLOR,
            linestyle=(0, (5, 3)),
            linewidth=1.6,
            zorder=4,
        )

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([g["label"] for g in groups], fontsize=11, color=TEXT_PRIMARY)
    ax.set_ylabel("Spearman ρ vs observed profile", fontsize=12, color=TEXT_PRIMARY)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="y", labelsize=10, colors=TEXT_SECONDARY)
    ax.tick_params(axis="x", length=0)

    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=SERIES_COLORS[s], edgecolor="white")
        for s in series
    ]
    labels = [SERIES_LABELS[s] for s in series]
    handles.append(Line2D([0], [0], color=CEILING_COLOR, linestyle=(0, (5, 3)), linewidth=1.6))
    labels.append("Noise ceiling (split-half, Spearman–Brown)")

    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        labelcolor=TEXT_PRIMARY,
    )

    fig.tight_layout()
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[INFO] Saved -> {out_png}")


if __name__ == "__main__":
    main()
