#!/usr/bin/env python3
"""
How much does the phylogeny actually constrain which pairs get built?

The shuffle controls test whether tree-derived *inputs* are used at inference.
This tests the other half of the claim: whether the tree-derived *training
signal* - which (ancestor, descendant) pairs exist at all - carries information
that metadata alone would not give. It needs no retraining.

For each tree-constructed pair (u -> v) it builds the candidate pool a tree-free
rule would have drawn from (same clade or lineage as v, collected earlier than
v) and asks where the actual tree-chosen ancestor u sits in that pool, ranked by
similarity to v:

  percentile 0    u is the single most similar admissible sequence; the tree is
                  selecting something a metadata rule would have to be lucky to find
  percentile 50   u is a typical member of the pool; the tree adds nothing
                  beyond the metadata constraints

Also reports admissibility: the fraction of tree pairs a tree-free rule could
have produced at all, which is capped by the pairs whose descendant predates
their ancestor.
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--train_pairs", required=True)
    ap.add_argument("--task", required=True, choices=["rbd", "spike"])
    ap.add_argument("--n_sample", type=int, default=4000, help="Pairs to analyse.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", required=True)
    return ap.parse_args()


def encode(seqs, width: int) -> np.ndarray:
    """Sequences -> uint8 matrix, right-padded so rows are comparable."""
    out = np.full((len(seqs), width), 0, dtype=np.uint8)
    for i, s in enumerate(seqs):
        b = np.frombuffer(s[:width].encode("ascii"), dtype=np.uint8)
        out[i, : len(b)] = b
    return out


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    src_col, dst_col = f"{args.task}_t", f"{args.task}_t1"

    df = pd.read_parquet(args.train_pairs)
    df["t"] = pd.to_datetime(df["t"])
    df["t1"] = pd.to_datetime(df["t1"])

    # Candidate pool: every distinct sequence the pair table knows about,
    # carrying the metadata a tree-free rule is allowed to use.
    left = df[["mat_strain_t", src_col, "t", "lineage_t", "clade_t"]].rename(
        columns={"mat_strain_t": "strain", src_col: "seq", "t": "date",
                 "lineage_t": "lineage", "clade_t": "clade"}
    )
    right = df[["mat_strain_t1", dst_col, "t1", "lineage_t1", "clade_t1"]].rename(
        columns={"mat_strain_t1": "strain", dst_col: "seq", "t1": "date",
                 "lineage_t1": "lineage", "clade_t1": "clade"}
    )
    nodes = (
        pd.concat([left, right], ignore_index=True)
        .drop_duplicates("strain")
        .reset_index(drop=True)
    )
    width = int(nodes["seq"].str.len().mode().iloc[0])
    node_mat = encode(nodes["seq"].tolist(), width)
    node_date = nodes["date"].to_numpy()
    node_lineage = nodes["lineage"].to_numpy()
    node_clade = nodes["clade"].to_numpy()
    strain_to_idx: Dict[str, int] = {s: i for i, s in enumerate(nodes["strain"])}

    sample = df.sample(n=min(args.n_sample, len(df)), random_state=args.seed)

    rows = []
    for _, r in sample.iterrows():
        u_idx = strain_to_idx.get(str(r["mat_strain_t"]))
        if u_idx is None:
            continue
        v_vec = encode([str(r[dst_col])], width)[0]
        true_dist = int((encode([str(r[src_col])], width)[0] != v_vec).sum())

        date_ok = node_date < np.datetime64(r["t1"])
        rec = {
            "task": args.task,
            "true_distance": true_dist,
            "delta_edges": int(r["delta_edges"]),
            "same_lineage": bool(r["lineage_t"] == r["lineage_t1"]),
            "date_ordered": bool(r["t"] < r["t1"]),
        }

        for rule, mask in (
            ("clade", (node_clade == r["clade_t1"]) & date_ok),
            ("lineage", (node_lineage == r["lineage_t1"]) & date_ok),
        ):
            cand = np.flatnonzero(mask)
            # The chosen ancestor must itself be admissible for a rank to exist.
            admissible = u_idx in set(cand.tolist())
            if cand.size < 2 or not admissible:
                rec[f"{rule}_n_candidates"] = int(cand.size)
                rec[f"{rule}_admissible"] = admissible
                rec[f"{rule}_percentile"] = np.nan
                rec[f"{rule}_median_cand_distance"] = np.nan
                continue

            d = (node_mat[cand] != v_vec[None, :]).sum(axis=1)
            # Percentile of the true ancestor among candidates, by similarity.
            pct = float((d < true_dist).sum() / d.size * 100.0)
            rec[f"{rule}_n_candidates"] = int(cand.size)
            rec[f"{rule}_admissible"] = True
            rec[f"{rule}_percentile"] = pct
            rec[f"{rule}_median_cand_distance"] = float(np.median(d))

        rows.append(rec)

    out = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    n = len(out)
    print(f"\n=== {args.task}: pairing overlap diagnostic (n={n}, pool={len(nodes):,} sequences) ===")
    print(f"descendant collected after ancestor : {100*out.date_ordered.mean():5.1f}%")
    print(f"same Pango lineage                  : {100*out.same_lineage.mean():5.1f}%")
    print(f"tree-chosen ancestor distance to v  : median {out.true_distance.median():.0f} aa")
    print()
    for rule in ("clade", "lineage"):
        adm = out[f"{rule}_admissible"]
        sub = out[out[f"{rule}_percentile"].notna()]
        print(f"-- tree-free rule: same {rule} + earlier date")
        print(f"   admissible (rule could produce this pair): {100*adm.mean():5.1f}%")
        print(f"   candidate pool size                      : median {out[f'{rule}_n_candidates'].median():.0f}")
        if len(sub):
            p = sub[f"{rule}_percentile"]
            print(f"   percentile of tree ancestor by similarity: median {p.median():5.1f}  "
                  f"mean {p.mean():5.1f}   (50 = no better than metadata)")
            print(f"   distance to v: tree {sub.true_distance.median():.0f} aa vs "
                  f"typical candidate {sub[f'{rule}_median_cand_distance'].median():.0f} aa")
            print(f"   tree ancestor is the single closest candidate: {100*(p==0).mean():5.1f}%")
        print()
    print(f"[INFO] Saved -> {out_path}")


if __name__ == "__main__":
    main()
