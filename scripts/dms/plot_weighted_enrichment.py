#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.figures import _plos_export as plos

BAR_COLOR = "#0072B2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--metrics_csv",
        default="results/dms/metrics_summary_all_datasets.csv",
        help="Input metrics CSV with weighted_enrichment_vs_background.",
    )
    parser.add_argument(
        "--out_dir",
        default="results/dms/plots",
        help="Directory for output figure.",
    )
    parser.add_argument(
        "--out_name",
        default="dms_weighted_enrichment_treeaware",
        help="Output figure stem (no extension); png/pdf/tiff/eps are all written.",
    )
    parser.add_argument(
        "--model",
        default="seqonly",
        choices=["seqonly", "treeaware"],
        help="Which model to plot.",
    )
    parser.add_argument("--dpi", type=int, default=plos.DPI)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_labels = {
        "urn_mavedb_00000670-a-1": "ACE2 binding",
        "urn_mavedb_00000671-a-1": "RBD expression",
    }

    df = pd.read_csv(args.metrics_csv)
    df = df[df["dms_dataset"].isin(dataset_labels.keys())]
    df = df[df["model"] == args.model]
    df["dataset_label"] = df["dms_dataset"].map(dataset_labels)
    df = df.sort_values("dataset_label")

    if df.empty:
        raise ValueError("No rows found after filtering dataset IDs/model.")

    fold = df["weighted_enrichment_vs_background"].to_numpy()
    pct_excess = (fold - 1.0) * 100.0
    labels = df["dataset_label"].to_numpy()

    plt.style.use("seaborn-v0_8-whitegrid")
    plos.apply_rcparams()  

    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ax.grid(axis="y", visible=False)
    bars = ax.barh(labels, pct_excess, color=BAR_COLOR, edgecolor="#3a3a3a", linewidth=0.5, height=0.55)
    ax.invert_yaxis() 

    for bar, f in zip(bars, fold):
        ax.text(
            bar.get_width() + max(pct_excess) * 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{f:.2f}-fold",
            ha="left", va="center", fontsize=8.5,
        )

    ax.set_xlim(left=0, right=max(pct_excess) * 1.75)
    ax.set_xlabel("Weighted enrichment above\nrandom allocation (%)", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_stem = out_dir / args.out_name
    plos.save_all_formats(fig, out_stem, args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
