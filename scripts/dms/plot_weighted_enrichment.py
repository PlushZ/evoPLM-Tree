#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


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
        default="dms_weighted_enrichment.png",
        help="Output figure filename.",
    )
    parser.add_argument(
        "--model",
        default="seqonly",
        choices=["seqonly", "treeaware"],
        help="Which model to plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_labels = {
        "urn_mavedb_00000670-a-1": "RBD-ACE2 binding\n(urn:mavedb:00000670-a-1)",
        "urn_mavedb_00000671-a-1": "RBD expression\n(urn:mavedb:00000671-a-1)",
    }

    df = pd.read_csv(args.metrics_csv)
    df = df[df["dms_dataset"].isin(dataset_labels.keys())]
    df = df[df["model"] == args.model]
    df["dataset_label"] = df["dms_dataset"].map(dataset_labels)
    df = df.sort_values("dataset_label")

    if df.empty:
        raise ValueError("No rows found after filtering dataset IDs/model.")

    values = df["weighted_enrichment_vs_background"].values
    labels = df["dataset_label"].values
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    bars = ax.bar(
        labels,
        values,
        color=["#4c78a8", "#72b7b2"],
        edgecolor="#3a3a3a",
        linewidth=0.5,
    )

    ax.axhline(
        1.0,
        linestyle="--",
        color="#444444",
        linewidth=1.2,
        label="Random expectation (baseline = 1.0)",
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}x",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Weighted enrichment", fontsize=12)
    ax.set_xlabel("")
    #ax.set_title(
    #    "EvoPLM enrichment for tolerated mutations\nBA.2 RBD deep mutational scanning",
    #    fontsize=13,
    #    pad=10,
    #)
    ax.set_ylim(bottom=0.0, top=max(values) + 0.12)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
        fancybox=True,
        framealpha=0.9,
    )
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    fig.tight_layout()

    out_path = out_dir / args.out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[ok] Saved figure: {out_path}")


if __name__ == "__main__":
    main()
