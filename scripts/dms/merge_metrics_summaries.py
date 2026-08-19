#!/usr/bin/env python3
"""Merge per-dataset DMS metrics_summary.csv files into one combined table.

evaluate_dms.py already writes metrics_summary_all_datasets.csv as part of its own run
(concatenating every dataset it just processed), but this script lets that combined file be
regenerated on its own from the individual results/dms/<dataset_id>/metrics_summary.csv
files, without re-running inference/merging.
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--metrics_root", default="results/dms")
    ap.add_argument(
        "--dataset_ids",
        nargs="+",
        default=["urn_mavedb_00000670-a-1", "urn_mavedb_00000671-a-1"],
    )
    ap.add_argument("--out_csv", default="results/dms/metrics_summary_all_datasets.csv")
    args = ap.parse_args()

    root = Path(args.metrics_root)
    frames = []
    for dataset_id in args.dataset_ids:
        csv_path = root / dataset_id / "metrics_summary.csv"
        df = pd.read_csv(csv_path)
        frames.append(df)
        print(f"[INFO] loaded {len(df)} rows from {csv_path}")

    combined = pd.concat(frames, ignore_index=True)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"[INFO] wrote combined summary: {out_path}")


if __name__ == "__main__":
    main()
