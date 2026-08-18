#!/usr/bin/env python3
"""
Tree-aware trajectory generation using per-split MATs.

Key idea:
- Internal nodes have no sequences in your data.
- We create a mapping internal_node -> representative leaf in its subtree.
- For each descendant leaf v, choose internal ancestor a = ancestor_at_k(v,k),
  then ancestor leaf u = rep_leaf[a], emit pair u -> v with delta_edges = dist(u,a)+k.

This avoids the incorrect assumption that a leaf's parent chain contains other leaves.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import pandas as pd
from tqdm import tqdm
from bte import MATree  # type: ignore


@dataclass
class LeafRec:
    strain: str
    spike: str
    rbd: str
    date: pd.Timestamp
    lineage: str
    clade: str
    dup: int


REQ = ["mat_is_present","mat_strain","spike_aa","rbd_aa","ncbi.collectionDate",
       "Nextclade_pango","clade_nextstrain","dup_count"]


def get_node(tree: MATree, name: str):
    try:
        return tree.get_node(name)
    except Exception:
        try:
            return tree[name]
        except Exception:
            return None


def node_parent(node):
    return getattr(node, "parent", None)


def node_id(node) -> str:
    return str(getattr(node, "id", getattr(node, "node_id", "UNKNOWN")))


def ancestor_at_k(node, k: int):
    cur = node
    for _ in range(k):
        p = node_parent(cur)
        if p is None:
            return None
        cur = p
    return cur


def load_leaf_recs(path: str, representative: str) -> Dict[str, LeafRec]:
    df = pd.read_parquet(path)
    miss = [c for c in REQ if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in {path}: {miss}")

    df = df[df["mat_is_present"] == True].copy()
    df = df.dropna(subset=["mat_strain","spike_aa","rbd_aa"])
    df["ncbi.collectionDate"] = pd.to_datetime(df["ncbi.collectionDate"], errors="coerce")
    df = df.dropna(subset=["ncbi.collectionDate"])
    df["dup_count"] = pd.to_numeric(df["dup_count"], errors="coerce").fillna(1).astype(int)

    if representative == "max_dup":
        df = df.sort_values(["mat_strain","dup_count","ncbi.collectionDate"], ascending=[True, False, False])
    elif representative == "latest_date":
        df = df.sort_values(["mat_strain","ncbi.collectionDate","dup_count"], ascending=[True, False, False])
    else:
        df = df.sort_values(["mat_strain","ncbi.collectionDate","dup_count"], ascending=[True, True, False])

    df = df.drop_duplicates("mat_strain", keep="first")

    recs = {}
    for _, r in df.iterrows():
        s = str(r["mat_strain"])
        recs[s] = LeafRec(
            strain=s,
            spike=str(r["spike_aa"]),
            rbd=str(r["rbd_aa"]),
            date=r["ncbi.collectionDate"],
            lineage=str(r["Nextclade_pango"]),
            clade=str(r["clade_nextstrain"]),
            dup=int(r["dup_count"]),
        )
    return recs


def build_rep_map(tree: MATree, train_recs: Dict[str, LeafRec], max_up: int = 512) -> Dict[str, Tuple[str,int]]:
    """
    Build mapping internal_node_id -> (representative_train_leaf_strain, dist_leaf_to_node)

    For each training leaf, walk up to root. For every visited internal node,
    store the closest (min dist) training leaf encountered so far.

    This gives a subtree representative for many internal nodes.
    """
    rep: Dict[str, Tuple[str,int]] = {}

    for leaf in tqdm(train_recs.keys(), desc="Building rep_map (train leaves)"):
        n = get_node(tree, leaf)
        if n is None:
            continue
        cur = n
        dist = 0
        steps = 0
        while steps <= max_up:
            nid = node_id(cur)
            if nid not in rep or dist < rep[nid][1]:
                rep[nid] = (leaf, dist)
            p = node_parent(cur)
            if p is None:
                break
            cur = p
            dist += 1
            steps += 1

    return rep


def compute_depth_pt(tree: MATree, node) -> Tuple[int,int]:
    # depth and pseudotime from root by walking up and counting edge-mutations
    depth = 0
    pt = 0
    cur = node
    while True:
        p = node_parent(cur)
        if p is None:
            break
        depth += 1
        muts = getattr(cur, "mutations", getattr(cur, "muts", [])) or []
        pt += len(muts)
        cur = p
    return depth, pt


def make_pair(u: LeafRec, v: LeafRec, k: int, delta_edges: int, depth_u: int, depth_v: int, pt_u: int, pt_v: int) -> dict:
    return {
        "spike_t": u.spike, "rbd_t": u.rbd,
        "spike_t1": v.spike, "rbd_t1": v.rbd,
        "k": int(k),
        "delta_edges": int(delta_edges),
        "delta_nt_mutcount": int(pt_v - pt_u),
        "depth_t": int(depth_u), "depth_t1": int(depth_v),
        "pt_t": int(pt_u), "pt_t1": int(pt_v),
        "dup_count_t": int(u.dup), "dup_count_t1": int(v.dup),
        "clade_t": u.clade, "clade_t1": v.clade,
        "lineage_t": u.lineage, "lineage_t1": v.lineage,
        "t": u.date, "t1": v.date,
        "mat_strain_t": u.strain, "mat_strain_t1": v.strain,
    }


def gen_pairs_single(tree: MATree,
                     recs: Dict[str, LeafRec],
                     k_list: List[int],
                     max_pairs_per_desc: int,
                     max_delta_edges: int,
                     out_path: str,
                     label: str):

    rep_map = build_rep_map(tree, recs, max_up=max_delta_edges)
    rows = []

    for v_strain, v in tqdm(recs.items(), desc=f"{label} pairs"):
        v_node = get_node(tree, v_strain)
        if v_node is None:
            continue
        depth_v, pt_v = compute_depth_pt(tree, v_node)

        emitted = 0
        for k in k_list:
            if emitted >= max_pairs_per_desc:
                break
            a = ancestor_at_k(v_node, k)
            if a is None:
                continue
            aid = node_id(a)
            if aid not in rep_map:
                continue
            u_strain, dist_u_to_a = rep_map[aid]
            if u_strain == v_strain:
                continue
            u = recs.get(u_strain)
            if u is None:
                continue
            u_node = get_node(tree, u_strain)
            if u_node is None:
                continue
            depth_u, pt_u = compute_depth_pt(tree, u_node)

            delta_edges = dist_u_to_a + k
            if delta_edges <= 0 or delta_edges > max_delta_edges:
                continue
            if pt_u > pt_v:
                continue

            rows.append(make_pair(u, v, k, delta_edges, depth_u, depth_v, pt_u, pt_v))
            emitted += 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"[INFO] {label} pairs: {len(df):,} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_csv", required=True)
    ap.add_argument("--out_dir", default="data/processed/spike_aa/trajectories")
    ap.add_argument("--k_list", default="1,2,4,8")
    ap.add_argument("--max_train_pairs_per_desc", type=int, default=2)
    ap.add_argument("--max_test_pairs_per_desc", type=int, default=2)
    ap.add_argument("--max_delta_edges", type=int, default=512)
    ap.add_argument("--representative", default="max_dup", choices=["max_dup","latest_date","earliest_date"])
    args = ap.parse_args()

    cfg = pd.read_csv(args.config_csv)

    # We assume naming in your config matches these outputs:
    basename_to_out = {
        "split1_train_first3Omi2laterOmi.with_mat.parquet": "split1_train_first3Omi2laterOmi_pairs.parquet",
        "split1_test_first3Omi2laterOmi.with_mat.parquet":  "split1_test_first3Omi2laterOmi_pairs.parquet",
        "split2_train_preOmi2Omi.with_mat.parquet":         "split2_train_preOmi2Omi_pairs.parquet",
        "split2_test_preOmi2Omi.with_mat.parquet":          "split2_test_preOmi2Omi_pairs.parquet",
    }

    def find_row(basename: str):
        for _, r in cfg.iterrows():
            if os.path.basename(str(r["out_parquet"])) == basename:
                return r
        raise KeyError(f"Not found in config out_parquet: {basename}")

    k_list = sorted(set(int(x) for x in args.k_list.split(",") if x.strip()))
    os.makedirs(args.out_dir, exist_ok=True)

    for basename, out_name in basename_to_out.items():
        r = find_row(basename)
        mat = str(r["mat_pb_gz"])
        out_path = os.path.join(args.out_dir, out_name)

        label = os.path.basename(out_name).replace("_pairs.parquet", "")
        max_pairs = args.max_train_pairs_per_desc if "train" in basename else args.max_test_pairs_per_desc

        print(f"\n=== {label} (MAT: {os.path.basename(mat)}) ===")
        tree = MATree(mat)
        recs = load_leaf_recs(str(r["out_parquet"]), args.representative)
        gen_pairs_single(
            tree=tree,
            recs=recs,
            k_list=k_list,
            max_pairs_per_desc=max_pairs,
            max_delta_edges=args.max_delta_edges,
            out_path=out_path,
            label=label,
        )


if __name__ == "__main__":
    main()
