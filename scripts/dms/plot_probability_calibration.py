#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def dataset_dirs(metrics_root: Path):
    return sorted([p for p in metrics_root.iterdir() if p.is_dir() and p.name.startswith("urn_")])


def load_merged_csv(dataset_dir: Path, model_name: str, file_suffix: str, input_subdir: str):
    base = dataset_dir / input_subdir if input_subdir else dataset_dir
    return base / f"{model_name}_merged_probs_dms{file_suffix}.csv"


def plot_calibration(df: pd.DataFrame, out_png: Path, title: str, n_bins: int):
    d = df.copy()
    d["prob"] = pd.to_numeric(d["prob"], errors="coerce")
    d["dms_score"] = pd.to_numeric(d["dms_score"], errors="coerce")
    d = d.dropna(subset=["prob", "dms_score"])
    if d.empty:
        raise ValueError(f"No numeric rows available for calibration plot: {out_png}")

    d["prob_bin"] = pd.qcut(d["prob"], q=n_bins, labels=False, duplicates="drop")
    b = d.groupby("prob_bin", as_index=False).agg(
        mean_prob=("prob", "mean"),
        mean_dms=("dms_score", "mean"),
        n=("dms_score", "size"),
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(b["mean_prob"], b["mean_dms"], marker="o")
    ax.set_xlabel("Mean model probability (bin)")
    ax.set_ylabel("Mean DMS score")
    ax.set_title(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--metrics_root", default="results/dms_validation/metrics")
    ap.add_argument("--out_root", default="results/dms_validation/calibration")
    ap.add_argument("--file_suffix", default="", help="Optional suffix used in merged filenames (e.g. __corrDmsGt_m3p0__corrTopQ_0p8).")
    ap.add_argument("--input_subdir", default="", help="Optional subdirectory under each dataset in metrics_root (e.g. filtered).")
    ap.add_argument("--n_bins", type=int, default=10)
    args = ap.parse_args()

    if args.n_bins < 2:
        raise ValueError("--n_bins must be >= 2")

    metrics_root = Path(args.metrics_root)
    out_root = Path(args.out_root)
    dirs = dataset_dirs(metrics_root)
    if not dirs:
        raise FileNotFoundError(f"No dataset dirs found in {metrics_root}")

    for ds in dirs:
        for model in ["seqonly", "treeaware"]:
            in_csv = load_merged_csv(ds, model, args.file_suffix, args.input_subdir)
            if not in_csv.exists():
                print(f"[WARN] skip missing: {in_csv}")
                continue
            df = pd.read_csv(in_csv)
            out_png = out_root / ds.name / f"{model}_probability_calibration{args.file_suffix}.png"
            title = f"{ds.name} | {model} | probability calibration"
            plot_calibration(df, out_png, title, n_bins=args.n_bins)
            print(f"[INFO] saved: {out_png}")


if __name__ == "__main__":
    main()
