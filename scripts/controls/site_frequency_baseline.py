#!/usr/bin/env python3
"""
Site-frequency baseline: predict each position's mutation frequency as the
fraction of TRAINING pairs in which that position changed.

This is a tree-free, model-free, context-free predictor built only from the
training split. If it reproduces the observed held-out positional profile about
as well as evoPLM-Tree does, then the model's positional agreement reflects
memorised marginal site frequencies rather than learned structure. The
comparison itself is done by compare_positional_profiles.py.

Because the pair tables are highly redundant (RBD: 87,370 rows but 9,809 unique
sequence pairs over 1,052 unique source sequences), --dedup reports the
baseline over distinct pairs as well; the two answer different questions and
both are worth having.
"""

import argparse
from pathlib import Path

import pandas as pd

from evoplm.profiles import modal_length, profile_from_pairs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--train_pairs", required=True, help="Training pair parquet.")
    ap.add_argument("--task", required=True, choices=["rbd", "spike"])
    ap.add_argument(
        "--n_ref",
        type=int,
        default=None,
        help="Reference length. Defaults to the modal source-sequence length "
             "(223 for RBD, 1274 for spike).",
    )
    ap.add_argument(
        "--dedup",
        action="store_true",
        help="Count each distinct (source, target) sequence pair once.",
    )
    ap.add_argument(
        "--exclude_ambiguous",
        action="store_true",
        help="Drop positions where either sequence carries X, * or - from both "
             "numerator and denominator.",
    )
    ap.add_argument("--out_csv", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    src_col = f"{args.task}_t"
    dst_col = f"{args.task}_t1"

    df = pd.read_parquet(args.train_pairs, columns=[src_col, dst_col])
    n_rows = len(df)
    if n_rows == 0:
        raise ValueError(f"No rows in {args.train_pairs}")

    n_unique_pairs = len(df.drop_duplicates([src_col, dst_col]))
    n_unique_src = df[src_col].nunique()
    if args.dedup:
        df = df.drop_duplicates([src_col, dst_col])

    sources = df[src_col].astype(str).tolist()
    targets = df[dst_col].astype(str).tolist()

    n_ref = args.n_ref if args.n_ref is not None else modal_length(sources)

    n_mut, n_cmp, freq = profile_from_pairs(
        sources, targets, n_ref, exclude_ambiguous=args.exclude_ambiguous
    )

    out = pd.DataFrame(
        {
            "position": range(n_ref),
            "n_mutated": n_mut,
            "n_comparable": n_cmp,
            "freq": freq,
        }
    )
    # One-based model coordinates, plus spike numbering for the RBD window.
    out["model_site"] = out["position"] + 1
    if args.task == "rbd":
        out["spike_site"] = out["position"] + 319

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[INFO] task={args.task} n_ref={n_ref}")
    print(f"[INFO] training rows: {n_rows:,}")
    print(f"[INFO]   unique source sequences: {n_unique_src:,}")
    print(f"[INFO]   unique (source,target) pairs: {n_unique_pairs:,} "
          f"(redundancy {n_rows / max(n_unique_pairs, 1):.1f}x)")
    print(f"[INFO] pairs counted: {len(sources):,} ({'deduplicated' if args.dedup else 'all rows'})")
    print(f"[INFO] exclude_ambiguous: {bool(args.exclude_ambiguous)}")
    print(f"[INFO] positions with no comparable pair: {int((n_cmp == 0).sum())}")
    print(f"[INFO] Saved baseline profile -> {out_path}")


if __name__ == "__main__":
    main()
