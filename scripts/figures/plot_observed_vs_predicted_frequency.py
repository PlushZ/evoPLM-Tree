#!/usr/bin/env python3
"""Rank-rank and optional log-scale plots for generated mutation frequencies."""
import argparse
from pathlib import Path
from typing import NamedTuple
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

class DatasetConfig(NamedTuple):
    name: str
    csv_path: Path
    expected_rho: float

def parse_args():
    p = argparse.ArgumentParser(description="Generate rank-rank hexbin and optional log-scale frequency plots.")
    p.add_argument("--rbd_csv", type=Path, default=Path("results/generation/split1/rbd_treeaware_sample2000.csv"))
    p.add_argument("--spike_csv", type=Path, default=Path("results/generation/split1/spike_treeaware_sample2000.csv"))
    p.add_argument("--out_dir", type=Path, default=Path("results/generation/split1/plots"))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--hexbin_gridsize", type=int, default=35)
    p.add_argument("--rho_tolerance", type=float, default=0.005)
    p.add_argument("--include_log_plots", action="store_true", help="Also write log-scale plots for frequency magnitudes.")
    return p.parse_args()

def validate_dataframe(df, csv_path):
    required = {"input_seq", "true_seq", "pred_seq"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{csv_path} contains no rows.")
    for col in required:
        if df[col].isna().any():
            raise ValueError(f"{csv_path}: column '{col}' contains missing values.")

def calculate_mutation_frequencies(csv_path):
    """Use the same positional count/normalization calculation as the hotspot plots."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {csv_path}")
    df = pd.read_csv(csv_path)
    validate_dataframe(df, csv_path)
    seq = df[["input_seq", "true_seq", "pred_seq"]].astype(str)
    n_ref = len(seq["input_seq"].mode().iloc[0])
    true_counts = np.zeros(n_ref, dtype=np.int64)
    pred_counts = np.zeros(n_ref, dtype=np.int64)
    for row in seq.itertuples(index=False):
        max_len = min(len(row.input_seq), len(row.true_seq), len(row.pred_seq), n_ref)
        for i in range(max_len):
            true_counts[i] += row.input_seq[i] != row.true_seq[i]
            pred_counts[i] += row.input_seq[i] != row.pred_seq[i]
    nonzero_positions = np.flatnonzero((true_counts > 0) | (pred_counts > 0))
    if nonzero_positions.size == 0:
        raise ValueError(f"{csv_path} contains no observed or predicted mutations.")

    # Match the normN correlation in 76_3 exactly: RBD uses the full
    # reference sequence, while spike uses positions through the last
    # observed or predicted mutation. This retains zero-frequency positions.
    if "rbd" in csv_path.stem.lower():
        plot_positions = np.arange(n_ref)
    else:
        plot_positions = np.arange(nonzero_positions[-1] + 1)

    denominator = float(len(seq))
    return plot_positions, true_counts[plot_positions] / denominator, pred_counts[plot_positions] / denominator

def add_annotation(ax, rho, p_value):
    text = f"Spearman $\\rho$ = {rho:.3f}\nTwo-sided $p$ = {p_value:.2e}"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, ha="left", va="top", fontsize=14,
            linespacing=1.25,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9})


def add_regression_line(ax, x, y, line_x, log_scale=False):
    if log_scale:
        slope, intercept = np.polyfit(np.log10(x), np.log10(y), 1)
        line_y = 10 ** (slope * np.log10(line_x) + intercept)
    else:
        slope, intercept = np.polyfit(x, y, 1)
        line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, color="#1565c0", linewidth=1.8, label="Linear regression")
    ax.legend(handles=[
        Line2D([0], [0], color="#1565c0", linewidth=1.8, label="Linear regression"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="1:1 reference"),
    ], loc="lower right", fontsize=11, frameon=True, framealpha=0.9)

def save_rank_rank_hexbin(observed, predicted, rho, p_value, out_path, dpi, gridsize):
    n = float(len(observed))
    x = rankdata(observed, method="average") / n
    y = rankdata(predicted, method="average") / n
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    hb = ax.hexbin(x, y, gridsize=gridsize, mincnt=1, bins="log", extent=(0, 1, 0, 1), linewidths=0.25)
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1.2, alpha=0.65, label="1:1 reference")
    add_regression_line(ax, x, y, np.array([0.0, 1.0]))
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Observed frequency rank percentile", ylabel="Predicted frequency rank percentile")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=12)
    add_annotation(ax, rho, p_value)
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("Number of positions", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] Wrote rank-rank hexbin: {out_path}")

def save_log_frequency_plot(observed, predicted, rho, p_value, out_path, dpi):
    positive = np.concatenate((observed[observed > 0], predicted[predicted > 0]))
    if positive.size == 0:
        raise ValueError("Cannot create a log-scale plot: all frequencies are zero.")
    replacement = float(positive.min()) / 2.0
    x = np.maximum(observed, replacement)
    y = np.maximum(predicted, replacement)
    lo, hi = min(float(x.min()), float(y.min())), max(float(x.max()), float(y.max()))
    padding = 10 ** (0.04 * (np.log10(hi) - np.log10(lo)))
    lo, hi = lo / padding, hi * padding
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=42, alpha=0.72, edgecolors="white", linewidths=0.45)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Observed mutation frequency", ylabel="Predicted mutation frequency")
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.2, alpha=0.65, label="1:1 reference")
    add_regression_line(ax, x, y, np.array([lo, hi]), log_scale=True)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=12)
    add_annotation(ax, rho, p_value)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] Wrote log-scale frequency plot: {out_path}")

def process_dataset(dataset, args):
    _, observed, predicted = calculate_mutation_frequencies(dataset.csv_path)
    rho, p_value = spearmanr(observed, predicted)
    if np.isnan(rho):
        raise ValueError("Spearman correlation is undefined for these frequencies.")
    print(f"[info] {dataset.name}: CSV={dataset.csv_path}, positions={len(observed)}, rho={rho:.6f}, p={p_value:.6e}, expected={dataset.expected_rho:.6f}")
    if abs(rho - dataset.expected_rho) > args.rho_tolerance:
        print(f"[warning] {dataset.name}: rho differs from expected by {abs(rho - dataset.expected_rho):.6f}")
    save_rank_rank_hexbin(observed, predicted, rho, p_value, args.out_dir / f"{dataset.name}_rank_rank_frequency_hexbin.png", args.dpi, args.hexbin_gridsize)
    if args.include_log_plots:
        save_log_frequency_plot(observed, predicted, rho, p_value, args.out_dir / f"{dataset.name}_observed_vs_predicted_frequency_log.png", args.dpi)

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in (DatasetConfig("rbd", args.rbd_csv, 0.823), DatasetConfig("spike", args.spike_csv, 0.736)):
        process_dataset(dataset, args)

if __name__ == "__main__":
    main()
