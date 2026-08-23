#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

import _plos_export as plos

Y_MAX = 2.0

SEQONLY_COLOR = "#0072B2"
TREEAWARE_COLOR = "#E69F00"
EDGE_COLOR = "#3a3a3a"
PARITY_COLOR = "#444444"

RATIO_RE = re.compile(r"ratio masked/normal\s*=\s*([0-9.]+)")


def read_ratio(log_dir: Path, domain: str, model: str) -> float:
    log_path = log_dir / f"{domain}_{model}.txt"
    text = log_path.read_text()
    m = RATIO_RE.search(text)
    if not m:
        raise ValueError(f"Could not find 'ratio masked/normal' in {log_path}")
    return float(m.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        default="results/context_dependence",
        help="Directory containing the diagnose_conditioning.py stdout logs "
             "({domain}_{model}.txt, e.g. rbd_seqonly.txt), one per "
             "cluster/slurm/diagnose/diagnose_conditioning_*.sh run.",
    )
    parser.add_argument(
        "--out_dir",
        default="results/context_dependence/plots",
        help="Directory where the panel figures will be written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=plos.DPI,
        help="Raster resolution for the PNG/TIFF outputs (PLOS requires 300-600 dpi).",
    )
    return parser.parse_args()


def make_plot(domain: str, show_treeaware: bool, log_dir: Path, out_dir: Path, dpi: int) -> None:
    if show_treeaware:
        keys = ["seqonly", "treeaware"]
        labels = ["seq-only", "tree-aware\n(evoPLM-Tree)"]
        colors = [SEQONLY_COLOR, TREEAWARE_COLOR]
        positions = [0, 1]
        xlim = (-0.6, 1.6)
    else:
        keys = ["seqonly"]
        labels = ["seq-only"]
        colors = [SEQONLY_COLOR]
        positions = [0]
        xlim = (-0.6, 0.6)

    values = [read_ratio(log_dir, domain, k) for k in keys]

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    ax.axhline(1.0, linestyle="--", color=PARITY_COLOR, linewidth=1.0, zorder=1)

    ax.vlines(positions, ymin=1.0, ymax=values, color=EDGE_COLOR, linewidth=1.4, zorder=2)
    ax.scatter(positions, values, s=70, color=colors, edgecolor=EDGE_COLOR, linewidth=0.8, zorder=3)

    for x, v in zip(positions, values):
        ax.text(x, v + 0.035, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlim(*xlim)
    ax.set_ylim(1.0 - 0.06, Y_MAX)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
    ax.set_ylabel("Loss ratio (masked / visible)", fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    yticks = [t for t in ax.get_yticks() if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    ax.set_yticks(yticks)
    ax.set_yticklabels(
        ["parity" if abs(t - 1.0) < 1e-9 else f"{t:.1f}" for t in yticks],
        fontsize=10,
    )

    variant = "both" if show_treeaware else "seqonly_only"
    out_stem = out_dir / f"conditioning_context_dependence_{domain}_{variant}"
    fig.tight_layout()
    plos.save_all_formats(fig, out_stem, dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    make_plot("rbd", show_treeaware=False, log_dir=log_dir, out_dir=out_dir, dpi=args.dpi)
    make_plot("rbd", show_treeaware=True, log_dir=log_dir, out_dir=out_dir, dpi=args.dpi)
    make_plot("spike", show_treeaware=False, log_dir=log_dir, out_dir=out_dir, dpi=args.dpi)
    make_plot("spike", show_treeaware=True, log_dir=log_dir, out_dir=out_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()
