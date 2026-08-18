#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import pandas as pd
from scipy.stats import spearmanr


TRUE_COLOR = "#d1007a"
PRED_COLOR = "#1a9d49"
OUTPUT_SUFFIX = "_mss_talk_adapted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_glob",
        default="results/generation/split1/*sample2000.csv",
        help="Glob pattern for input CSV files.",
    )
    parser.add_argument(
        "--out_dir",
        default="results/generation/split1/plots",
        help="Directory where PNG plots will be written.",
    )
    return parser.parse_args()


def get_position_series(
    csv_path: Path,
    true_counts: list[int],
    pred_counts: list[int],
) -> tuple[list[int], list[int]]:
    stem_lower = csv_path.stem.lower()
    is_rbd = "rbd" in stem_lower

    nonzero_positions = [i for i, (t, p) in enumerate(zip(true_counts, pred_counts)) if t > 0 or p > 0]
    if not nonzero_positions:
        return [], []

    if is_rbd:
        plot_positions = list(range(len(true_counts)))
        axis_positions = [p + 319 for p in plot_positions]
    else:
        last_position = nonzero_positions[-1]
        plot_positions = list(range(last_position + 1))
        axis_positions = plot_positions

    return plot_positions, axis_positions


def style_axes(ax: plt.Axes, csv_path: Path, axis_positions: list[int], n_ref: int) -> None:
    stem_lower = csv_path.stem.lower()
    is_spike = "spike" in stem_lower
    is_rbd = "rbd" in stem_lower

    ax.set_xlabel("Mutation frequency", fontsize=18)
    ax.set_ylabel("Sequence position", fontsize=18)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)

    if is_rbd:
        ax.set_ylim(319 + n_ref - 1, 319)
        ax.yaxis.set_major_locator(MultipleLocator(25))
    elif is_spike and axis_positions:
        ax.set_ylim(axis_positions[-1], 0)
        ax.yaxis.set_major_locator(MultipleLocator(50))
    elif axis_positions:
        ax.set_ylim(axis_positions[-1], 0)

    ax.margins(x=0, y=0)


def save_plot(
    csv_path: Path,
    out_dir: Path,
    axis_positions: list[int],
    true_values: list[float],
    pred_values: list[float],
    corr_text: str,
    out_name: str,
) -> None:
    stem_lower = csv_path.stem.lower()
    is_spike = "spike" in stem_lower
    is_rbd = "rbd" in stem_lower
    figsize = (7.0, 11.0) if (is_spike or is_rbd) else (6.5, 7.5)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.plot(
        true_values,
        axis_positions,
        label="True",
        color=TRUE_COLOR,
        linewidth=2.2,
        alpha=0.9,
    )
    ax.plot(
        pred_values,
        axis_positions,
        label="Predicted",
        color=PRED_COLOR,
        linewidth=2.2,
        alpha=0.6,
    )

    style_axes(ax, csv_path, axis_positions, n_ref=len(axis_positions) if is_rbd else 0)
    
    legend_handles = [
        Line2D([0], [0], color=TRUE_COLOR, linewidth=2.2, alpha=0.9, label="Observed"),
        Line2D([0], [0], color=PRED_COLOR, linewidth=2.2, alpha=0.6, label="Predicted"),
        Line2D([0], [0], linestyle="None", color="none", label=f"Spearman ρ={corr_text}"),
    ]
    ax.legend(
        fontsize=16,
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
    )
    fig.tight_layout()

    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[ok] {csv_path} -> {out_path}")


def plot_csv(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    required_cols = {"input_seq", "true_seq", "pred_seq"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing_cols)}")

    if df.empty:
        raise ValueError(f"{csv_path} has 0 rows.")

    reference_seq = df["input_seq"].astype(str).mode().iloc[0]
    n_ref = len(reference_seq)
    true_counts = [0] * n_ref
    pred_counts = [0] * n_ref

    for _, row in df.iterrows():
        inp = row["input_seq"]
        true = row["true_seq"]
        pred = row["pred_seq"]

        max_len = min(len(inp), len(true), len(pred), n_ref)
        for i in range(max_len):
            if inp[i] != true[i]:
                true_counts[i] += 1
            if inp[i] != pred[i]:
                pred_counts[i] += 1

    count_positions = [i for i in range(n_ref) if true_counts[i] > 0 or pred_counts[i] > 0]
    true_freq = [true_counts[p] for p in count_positions]
    pred_freq = [pred_counts[p] for p in count_positions]
    corr, _ = spearmanr(true_freq, pred_freq) if count_positions else (float("nan"), None)
    corr_text = "nan" if pd.isna(corr) else f"{corr:.3f}"

    plot_positions, axis_positions = get_position_series(csv_path, true_counts, pred_counts)
    if not plot_positions:
        raise ValueError(f"{csv_path} has no mutated positions in true or predicted sequences.")

    count_plot_positions = [p + 319 for p in count_positions] if "rbd" in csv_path.stem.lower() else count_positions
    save_plot(
        csv_path=csv_path,
        out_dir=out_dir,
        axis_positions=count_plot_positions,
        true_values=true_freq,
        pred_values=pred_freq,
        corr_text=corr_text,
        out_name=f"{csv_path.stem}_mutation_hotspots{OUTPUT_SUFFIX}.png",
    )

    denom = float(len(df))
    true_norm = [true_counts[p] / denom for p in plot_positions]
    pred_norm = [pred_counts[p] / denom for p in plot_positions]
    corr_norm, _ = spearmanr(true_norm, pred_norm)
    corr_norm_text = "nan" if pd.isna(corr_norm) else f"{corr_norm:.3f}"

    save_plot(
        csv_path=csv_path,
        out_dir=out_dir,
        axis_positions=axis_positions,
        true_values=true_norm,
        pred_values=pred_norm,
        corr_text=corr_norm_text,
        out_name=f"{csv_path.stem}_mutation_hotspots_normN{OUTPUT_SUFFIX}.png",
    )


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
