#!/usr/bin/env python3
"""Filter trivial (non-mutating) ancestor->descendant trajectory pairs per task.

Reads the combined pair files produced by 11_gen_trajectories_tree_aware.py and,
for each of {rbd, spike}, keeps only rows where the task's start/end sequences
differ and at least one nucleotide substitution separates them
(delta_nt_mutcount >= 1). Writes one parquet per (split, train/test, task) into
data/trajectories/, matching the filenames scripts/train_prefix_lm.py expects.
"""
import argparse
from pathlib import Path

import pandas as pd

SPLIT_STEMS = {"split1": "first3Omi2laterOmi"}
TASKS = ["rbd", "spike"]
SUBSETS = ["train", "test"]


def filter_one(in_path: Path, task: str, out_path: Path) -> None:
    df = pd.read_parquet(in_path)
    seq_t, seq_t1 = f"{task}_t", f"{task}_t1"
    keep = (df[seq_t] != df[seq_t1]) & (df["delta_nt_mutcount"] >= 1)
    out = df[keep].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[INFO] {in_path.name} task={task}: {len(df):,} -> {len(out):,} rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/processed/spike_aa/trajectories")
    ap.add_argument("--out_dir", default="data/trajectories")
    ap.add_argument("--split", default="split1", choices=list(SPLIT_STEMS))
    args = ap.parse_args()

    stem = SPLIT_STEMS[args.split]
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    for subset in SUBSETS:
        in_path = in_dir / f"{args.split}_{subset}_{stem}_pairs.parquet"
        for task in TASKS:
            out_name = f"{args.split}_{subset}_{stem}_pairs__{task}_t_ne_{task}_t1__delta_nt_mutcount_ge_1.parquet"
            filter_one(in_path, task, out_dir / out_name)


if __name__ == "__main__":
    main()
