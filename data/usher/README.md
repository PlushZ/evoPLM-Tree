# UShER mutation-annotated tree (external download)

This is **not** repository-authored data — it is a public UShER/UCSC global phylogeny
snapshot, not hosted on Zenodo with the rest of this project's artifacts. Not included here
because of its size and because it is versioned/updated externally.

| File (expected here) | Read by |
|---|---|
| `public-2022-06-01.all.masked.pb.gz` | `cluster/slurm/preprocessing/10_add_usher_tree_to_splits.sh` (step 10, train split) |
| `public-2026-01-21.all.masked.pb.gz` | `10_add_usher_tree_to_splits.sh` (step 10, test split) |
| `public-2022-06-01.metadata.tsv.gz` | `10_add_usher_tree_to_splits.sh` (step 10, both train and test) |

These exact filenames are what
`data/processed/spike_aa/splits/config_tree2splits_split1.csv` (tracked in git) points at —
that manifest is also read by step 11 (`11_gen_trajectories_tree_aware.sh`), which consumes
step 10's `*.with_mat.parquet` output rather than these files directly.

**TODO (fill in before publishing):** the exact download source/URL for each dated snapshot
above, e.g. the UShER public MAT distribution. The metadata TSV must contain
`genbank_accession` and `strain` columns — see
`scripts/preprocessing/10_add_usher_tree_to_splits.py`.

If you use a different snapshot, place the files here and update the paths in
`config_tree2splits_split1.csv` to match.
