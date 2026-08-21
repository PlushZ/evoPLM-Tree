#!/usr/bin/env python3
"""
Augment split parquet files with UShER MAT placement features using accession matching.

Requires:
  pip install pandas pyarrow tqdm
  pip install bte  (Big Tree Explorer)  https://jmcbroome.github.io/BTE/  :contentReference[oaicite:1]{index=1}

Inputs per split:
  - split parquet (contains ncbi.accession)
  - MAT protobuf .pb.gz
  - corresponding UShER metadata.tsv.gz (contains genbank_accession and strain)

Outputs:
  - augmented parquet with:
      mat_is_present, mat_strain, mat_node_id, parent_mat_node_id,
      tree_depth, branch_nt_mutcount, root_to_tip_nt_mutcount
"""

import os
import re
import argparse
import pandas as pd
from tqdm import tqdm

# BTE (Big Tree Explorer) provides MATree/MATNode for MAT protobuf traversal :contentReference[oaicite:2]{index=2}
from bte import MATree  # type: ignore


def normalize_accession(acc: str) -> str:
    """Normalize GenBank accessions (strip whitespace; optionally strip version)."""
    if acc is None:
        return ""
    acc = str(acc).strip()
    if acc == "" or acc.lower() == "nan":
        return ""
    return acc


def strip_version(acc: str) -> str:
    """MT396241.1 -> MT396241"""
    return re.sub(r"\.\d+$", "", acc)


def load_usher_metadata_map(metadata_tsv_gz: str) -> pd.DataFrame:
    """
    Load UShER metadata TSV and return a mapping DataFrame with:
      genbank_accession, genbank_accession_noversion, strain
    """
    print(f"[INFO] Loading UShER metadata: {metadata_tsv_gz}")
    md = pd.read_csv(metadata_tsv_gz, sep="\t", compression="gzip", low_memory=False)

    # Common column names: 'genbank_accession', 'strain' (UCSC/UShER distribution)
    required = ["genbank_accession", "strain"]
    for c in required:
        if c not in md.columns:
            raise ValueError(f"Metadata missing required column '{c}'. Columns: {list(md.columns)[:50]} ...")

    md["genbank_accession"] = md["genbank_accession"].astype(str).map(normalize_accession)
    md["genbank_accession_noversion"] = md["genbank_accession"].map(strip_version)

    # drop empties
    md = md[(md["genbank_accession"] != "") & (md["genbank_accession"] != "nan")].copy()

    # Keep only what needed
    md = md[["genbank_accession", "genbank_accession_noversion", "strain"]].drop_duplicates()

    return md


def build_accession_to_strain(md_map: pd.DataFrame) -> dict:
    """
    Create accession -> strain dict supporting both versioned and unversioned accessions.
    """
    acc2strain = {}

    # versioned
    for acc, strain in zip(md_map["genbank_accession"], md_map["strain"]):
        if acc and acc not in acc2strain:
            acc2strain[acc] = strain

    # unversioned fallback
    for acc_nv, strain in zip(md_map["genbank_accession_noversion"], md_map["strain"]):
        if acc_nv and acc_nv not in acc2strain:
            acc2strain[acc_nv] = strain

    return acc2strain


def compute_node_features(tree: MATree, strain_names: list[str]) -> pd.DataFrame:
    """
    Given a MATree and a list of strain names (leaf labels), compute per-strain:
      mat_node_id, parent_mat_node_id, tree_depth, branch_nt_mutcount, root_to_tip_nt_mutcount, mat_is_present

    Notes:
      - For BTE, node objects expose .id, .parent, and .mutations (list of mutations on incoming edge).
      - Exact attribute names can vary slightly by BTE version; we defensively handle common variants.
    """
    records = []

    # Build a fast lookup of leaf nodes by name
    # BTE typically supports tree.get_node(name) or tree[name]; we try a few patterns.
    def get_node_by_name(name: str):
        for getter in (
            lambda: tree.get_node(name),      # common
            lambda: tree[name],               # __getitem__
        ):
            try:
                return getter()
            except Exception:
                pass
        return None

    for name in tqdm(strain_names, desc="Computing MAT features"):
        node = get_node_by_name(name)
        if node is None:
            records.append({
                "mat_is_present": False,
                "mat_strain": name,
                "mat_node_id": None,
                "parent_mat_node_id": None,
                "tree_depth": None,
                "branch_nt_mutcount": None,
                "root_to_tip_nt_mutcount": None,
            })
            continue

        # Node id
        node_id = getattr(node, "id", None)
        if node_id is None:
            node_id = getattr(node, "node_id", None)

        # Parent
        parent = getattr(node, "parent", None)
        parent_id = None
        if parent is not None:
            parent_id = getattr(parent, "id", None)
            if parent_id is None:
                parent_id = getattr(parent, "node_id", None)

        # Branch mutations on incoming edge to node
        muts = getattr(node, "mutations", None)
        if muts is None:
            muts = getattr(node, "muts", None)
        branch_mutcount = len(muts) if muts is not None else None

        # Depth + root-to-tip mutation count by walking to root
        depth = 0
        root_to_tip = 0
        cur = node
        while True:
            par = getattr(cur, "parent", None)
            if par is None:
                break
            depth += 1
            cur_muts = getattr(cur, "mutations", None)
            if cur_muts is None:
                cur_muts = getattr(cur, "muts", None)
            if cur_muts is not None:
                root_to_tip += len(cur_muts)
            cur = par

        records.append({
            "mat_is_present": True,
            "mat_strain": name,
            "mat_node_id": node_id,
            "parent_mat_node_id": parent_id,
            "tree_depth": depth,
            "branch_nt_mutcount": branch_mutcount,
            "root_to_tip_nt_mutcount": root_to_tip,
        })

    return pd.DataFrame.from_records(records)


def augment_split(
    split_parquet: str,
    mat_pb_gz: str,
    metadata_tsv_gz: str,
    out_parquet: str,
):
    print(f"\n[INFO] Split: {split_parquet}")
    df = pd.read_parquet(split_parquet)
    if "ncbi.accession" not in df.columns:
        raise ValueError("Split parquet missing required column 'ncbi.accession'")

    # Load metadata and map accession -> strain
    md_map = load_usher_metadata_map(metadata_tsv_gz)
    acc2strain = build_accession_to_strain(md_map)

    # Map split accessions to strains
    df["ncbi.accession"] = df["ncbi.accession"].astype(str).map(normalize_accession)
    df["acc_noversion"] = df["ncbi.accession"].map(strip_version)

    # Prefer versioned, fallback to no-version
    df["mat_strain"] = df["ncbi.accession"].map(acc2strain)
    missing = df["mat_strain"].isna()
    df.loc[missing, "mat_strain"] = df.loc[missing, "acc_noversion"].map(acc2strain)

    # Load MAT tree
    print(f"[INFO] Loading MAT: {mat_pb_gz}")
    tree = MATree(mat_pb_gz)  # BTE can load MAT protobuf directly :contentReference[oaicite:3]{index=3}

    # Compute node features for unique strains that mapped
    unique_strains = sorted([s for s in df["mat_strain"].dropna().unique().tolist() if s != ""])
    print(f"[INFO] Mapped to {len(unique_strains):,} unique MAT strain names (before node lookup).")

    feat = compute_node_features(tree, unique_strains)

    # Merge back
    df = df.merge(feat, on="mat_strain", how="left", suffixes=("", "_mat"))

    # For rows without a mapped strain at all
    df["mat_is_present"] = df["mat_is_present"].fillna(False)

    # Cleanup
    df.drop(columns=["acc_noversion"], inplace=True)

    print(f"[INFO] Match rate (mat_is_present): {df['mat_is_present'].mean():.3f}")

    # Write
    os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"[INFO] Wrote: {out_parquet}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_csv", required=True,
                    help="CSV with columns: split_parquet, mat_pb_gz, metadata_tsv_gz, out_parquet")
    args = ap.parse_args()

    cfg = pd.read_csv(args.config_csv)
    required = ["split_parquet", "mat_pb_gz", "metadata_tsv_gz", "out_parquet"]
    for c in required:
        if c not in cfg.columns:
            raise ValueError(f"Config missing column '{c}'. Need: {required}")

    for _, row in cfg.iterrows():
        augment_split(
            split_parquet=row["split_parquet"],
            mat_pb_gz=row["mat_pb_gz"],
            metadata_tsv_gz=row["metadata_tsv_gz"],
            out_parquet=row["out_parquet"],
        )


if __name__ == "__main__":
    main()
