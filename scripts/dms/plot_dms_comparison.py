#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--metrics_dir", default="results/dms")
    ap.add_argument("--out_dir", default="results/dms/plots")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_m = pd.read_csv(metrics_dir / "seqonly_merged_probs_dms.csv")
    tr_m = pd.read_csv(metrics_dir / "treeaware_merged_probs_dms.csv")
    seq_t = pd.read_csv(metrics_dir / f"seqonly_top{args.k}_mutations.csv")
    tr_t = pd.read_csv(metrics_dir / f"treeaware_top{args.k}_mutations.csv")
    summary = pd.read_csv(metrics_dir / "metrics_summary.csv")

    seq_m["model"] = "seqonly"
    tr_m["model"] = "treeaware"
    merged = pd.concat([seq_m, tr_m], ignore_index=True)

    seq_t["model"] = "seqonly"
    tr_t["model"] = "treeaware"
    topk = pd.concat([seq_t, tr_t], ignore_index=True)

    # Plot 1: probability vs DMS score, split by model
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, model_name in zip(axes, ["seqonly", "treeaware"]):
        d = merged.loc[merged["model"] == model_name]
        sns.scatterplot(data=d, x="prob", y="dms_score", alpha=0.35, s=12, ax=ax)
        ax.set_title(f"{model_name}: probability vs DMS")
        ax.set_xlabel("Mutation probability")
        ax.set_ylabel("DMS score")
    fig.tight_layout()
    p1 = out_dir / "probability_vs_dms_both_models.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)

    # Plot 2: top-k DMS distribution by model
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=topk, x="model", y="dms_score", cut=0, inner="quartile", ax=ax)
    sns.stripplot(data=topk, x="model", y="dms_score", color="black", alpha=0.2, size=2, ax=ax)
    ax.set_title(f"Top-{args.k} per-site DMS score distribution")
    ax.set_xlabel("Model")
    ax.set_ylabel("DMS score")
    fig.tight_layout()
    p2 = out_dir / "topk_dms_distribution_both_models.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)

    # Plot 3: summary comparison bars
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=summary, x="model", y="spearman_prob_vs_dms", ax=axes[0])
    axes[0].set_title("Spearman(prob, DMS)")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Correlation")

    sns.barplot(data=summary, x="model", y="topk_enrichment_vs_background", ax=axes[1])
    axes[1].set_title("Top-k enrichment")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Enrichment")
    fig.tight_layout()
    p3 = out_dir / "metrics_summary_bars.png"
    fig.savefig(p3, dpi=300)
    plt.close(fig)

    print(f"[INFO] Saved: {p1}")
    print(f"[INFO] Saved: {p2}")
    print(f"[INFO] Saved: {p3}")


if __name__ == "__main__":
    main()
