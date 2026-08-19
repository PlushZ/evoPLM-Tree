#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


SEQONLY_COLOR = "#d1007a"
TREEAWARE_COLOR = "#1a9d49"
EDGE_COLOR = "#3a3a3a"
PARITY_COLOR = "#444444"

MODEL_RATIOS = {
    "rbd": {
        "seqonly": 1.048,
        "treeaware": 1.242,
    },
    "spike": {
        "seqonly": 1.275,
        "treeaware": 1.736,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="results/context_dependence/plots",
        help="Directory where the presentation plots will be written.",
    )
    return parser.parse_args()


def make_plot(domain: str, show_treeaware: bool, out_dir: Path) -> None:
    ratios = MODEL_RATIOS[domain]
    labels = ["seq-only", "tree-aware"]
    positions = [0, 1]
    width = 0.62

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=150)

    seq_bar = ax.bar(
        [positions[0]],
        [ratios["seqonly"]],
        width=width,
        color=SEQONLY_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
    )
    if show_treeaware:
        tree_bar = ax.bar(
            [positions[1]],
            [ratios["treeaware"]],
            width=width,
            color=TREEAWARE_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=0.6,
        )
    else:
        tree_bar = []

    ax.axhline(1.0, linestyle="--", color=PARITY_COLOR, linewidth=1.2)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.0, 1.95)
    if show_treeaware:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=18, ha="right")
    else:
        ax.set_xticks([positions[0]])
        ax.set_xticklabels([labels[0]], rotation=18, ha="right")
    ax.set_ylabel("Loss ratio (masked / visible)")

    for bar in list(seq_bar) + list(tree_bar):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.03,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    variant = "both" if show_treeaware else "seqonly_only"
    out_path = out_dir / f"conditioning_context_dependence_{domain}_{variant}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[ok] saved plot to {out_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    make_plot("rbd", show_treeaware=False, out_dir=out_dir)
    make_plot("rbd", show_treeaware=True, out_dir=out_dir)
    make_plot("spike", show_treeaware=False, out_dir=out_dir)
    make_plot("spike", show_treeaware=True, out_dir=out_dir)


if __name__ == "__main__":
    main()
