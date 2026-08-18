#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


MUTATION_BINS = [0, 1, 2, 3, 5, 10, 20]
MUTATION_BIN_LABELS = ["1", "2", "3", "4–5", "6–10", "11–20"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_glob",
        default="results/generation/*sample2000.csv",
        help="Glob pattern for input CSV files.",
    )
    parser.add_argument(
        "--out_dir",
        default="results/gen/plots",
        help="Directory where PNG plots will be written.",
    )
    return parser.parse_args()


def plot_csv(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)

    required_cols = {"recovered_mut_count", "true_mut_count"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing_cols)}")

    df["recovery_rate"] = df["recovered_mut_count"] / df["true_mut_count"].replace(0, pd.NA)
    df["mut_bin"] = pd.cut(
        df["true_mut_count"], bins=MUTATION_BINS, labels=MUTATION_BIN_LABELS, include_lowest=True
    )
    agg = (
        df.groupby("mut_bin", observed=False)["recovery_rate"]
        .agg(mean_recovery="mean", n="count")
        .reset_index()
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(
        agg["mut_bin"].astype(str),
        agg["mean_recovery"],
        color="#4c78a8",
        linewidth=2.2,
        marker="o",
        markersize=5.5,
    )
    for bin_label, mean_recovery, n in agg.itertuples(index=False, name=None):
        if pd.notna(mean_recovery):
            ax.annotate(
                f"n={int(n):,}",
                xy=(str(bin_label), mean_recovery),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#333333",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.8,
                },
            )

    ax.set_xlabel("Observed amino-acid mutation count (evolutionary distance)")
    ax.set_ylabel("Mean exact mutation recovery rate")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()

    out_path = out_dir / f"{csv_path.stem}_recovery_vs_distance.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[ok] {csv_path} -> {out_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(Path().glob(args.csv_glob))
    if not csv_files:
        raise FileNotFoundError(f"No files matched: {args.csv_glob}")

    for csv_path in csv_files:
        plot_csv(csv_path, out_dir)


if __name__ == "__main__":
    main()
