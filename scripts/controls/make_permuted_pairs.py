#!/usr/bin/env python3
"""
Build permuted copies of a held-out pair table for the conditioning controls.

Two families of perturbation, both leaving the target sequence untouched:

  source            replace the ancestral sequence with one drawn from another
                    pair, so the model is asked for the observed descendant
                    from the wrong background (R1.3)
  tree-joint        replace all four tree-feature values with those of another
                    pair, keeping them mutually consistent
  tree-independent  permute each tree feature separately, destroying their
                    correlation as well as their truth

Masking removes information; permutation supplies wrong information, which is
the stronger test of whether the conditioning is actually used.

Why the source control is stratified
------------------------------------
Every held-out pair is early-to-mid Omicron, so a randomly chosen replacement
ancestor is typically >99% identical to the true one. An unstratified shuffle
would therefore barely perturb the input, and a small loss increase would mean
"backgrounds here are interchangeable", not "the model ignores the background".
Variants are emitted within-lineage (near-zero perturbation, a floor),
across-lineage (realistic), and global, and every row records the Hamming
distance actually achieved so loss can be read against perturbation size
rather than against the label of the variant.

All variants are built from one subsample of rows, so losses are comparable
across variants without re-matching rows afterwards.
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

TREE_COLS = ["delta_edges", "delta_nt_mutcount", "depth_t", "pt_t"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--pairs_parquet", required=True)
    ap.add_argument("--task", required=True, choices=["rbd", "spike"])
    ap.add_argument("--n_rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()


def hamming(a: str, b: str) -> int:
    m = min(len(a), len(b))
    return sum(1 for i in range(m) if a[i] != b[i]) + abs(len(a) - len(b))


def donor_indices(
    df: pd.DataFrame, rng: np.random.Generator, mode: str
) -> np.ndarray:
    """For each row, pick the row whose source sequence will be substituted in."""
    n = len(df)
    if mode == "global":
        # Sample a donor != self.
        out = rng.integers(0, n, size=n)
        clash = out == np.arange(n)
        while clash.any():
            out[clash] = rng.integers(0, n, size=int(clash.sum()))
            clash = out == np.arange(n)
        return out

    lineages = df["lineage_t"].to_numpy()
    by_lineage = {lin: np.flatnonzero(lineages == lin) for lin in np.unique(lineages)}
    out = np.empty(n, dtype=int)

    for i in range(n):
        if mode == "within-lineage":
            pool = by_lineage[lineages[i]]
            pool = pool[pool != i]
        elif mode == "across-lineage":
            pool = np.flatnonzero(lineages != lineages[i])
        else:
            raise ValueError(f"unknown mode {mode}")
        # No eligible donor (singleton lineage, or every row shares a lineage):
        # leave the row unperturbed and let perturb_distance record the zero.
        out[i] = i if pool.size == 0 else int(rng.choice(pool))

    return out


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    src_col = f"{args.task}_t"
    dst_col = f"{args.task}_t1"

    df = pd.read_parquet(args.pairs_parquet)
    for col in [src_col, dst_col, "lineage_t"] + TREE_COLS:
        if col not in df.columns:
            raise KeyError(f"{args.pairs_parquet} missing column '{col}'")

    if args.n_rows > 0 and args.n_rows < len(df):
        df = df.sample(n=args.n_rows, random_state=args.seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}

    # Reference: unpermuted, same rows.
    ref = df.copy()
    ref["perturb_distance"] = 0
    ref_path = out_dir / f"{args.task}_reference.parquet"
    ref.to_parquet(ref_path, index=False)
    manifest["reference"] = {"path": str(ref_path), "mean_perturb_distance": 0.0}
    print(f"[INFO] reference                 rows={len(ref)}  mean dist=0.0")

    # Source-substitution variants.
    for mode in ("within-lineage", "across-lineage", "global"):
        idx = donor_indices(df, np.random.default_rng(args.seed), mode)
        var = df.copy()
        new_src = df[src_col].to_numpy()[idx]
        dist = np.array([hamming(a, b) for a, b in zip(df[src_col], new_src)])
        var[src_col] = new_src
        var["perturb_distance"] = dist
        var["donor_lineage_t"] = df["lineage_t"].to_numpy()[idx]

        name = f"source_{mode.replace('-', '_')}"
        path = out_dir / f"{args.task}_{name}.parquet"
        var.to_parquet(path, index=False)
        manifest[name] = {
            "path": str(path),
            "mean_perturb_distance": float(dist.mean()),
            "median_perturb_distance": float(np.median(dist)),
            "frac_unchanged": float((dist == 0).mean()),
        }
        print(f"[INFO] {name:24s} rows={len(var)}  mean dist={dist.mean():6.2f}  "
              f"median={np.median(dist):5.1f}  unchanged={100*(dist==0).mean():.1f}%")

    # Tree-feature variants.
    for name, independent in (("tree_joint", False), ("tree_independent", True)):
        var = df.copy()
        r = np.random.default_rng(args.seed + 1)
        if independent:
            for col in TREE_COLS:
                var[col] = df[col].to_numpy()[r.permutation(len(df))]
        else:
            perm = r.permutation(len(df))
            for col in TREE_COLS:
                var[col] = df[col].to_numpy()[perm]

        changed = np.zeros(len(df), dtype=int)
        for col in TREE_COLS:
            changed += (var[col].to_numpy() != df[col].to_numpy()).astype(int)
        var["perturb_distance"] = changed

        path = out_dir / f"{args.task}_{name}.parquet"
        var.to_parquet(path, index=False)
        manifest[name] = {
            "path": str(path),
            "mean_features_changed": float(changed.mean()),
            "frac_unchanged": float((changed == 0).mean()),
        }
        print(f"[INFO] {name:24s} rows={len(var)}  mean features changed="
              f"{changed.mean():.2f}/4  unchanged={100*(changed==0).mean():.1f}%")

    manifest_path = out_dir / f"{args.task}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[INFO] Saved manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
