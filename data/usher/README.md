# UShER mutation-annotated tree (external download)

This is **not** repository-authored data — it is a public UShER/UCSC global phylogeny
snapshot.

| File (expected here) | Read by | Download URL |
|---|---|---|
| `public-2022-06-01.all.masked.pb.gz` | `cluster/slurm/preprocessing/10_add_usher_tree_to_splits.sh` (step 10, train split) | https://hgdownload.soe.ucsc.edu/goldenPath/wuhCor1/UShER_SARS-CoV-2/2022/06/01/public-2022-06-01.all.masked.pb.gz |
| `public-2022-06-01.metadata.tsv.gz` | `10_add_usher_tree_to_splits.sh` (step 10, train split) | https://hgdownload.soe.ucsc.edu/goldenPath/wuhCor1/UShER_SARS-CoV-2/2022/06/01/public-2022-06-01.metadata.tsv.gz |
| `public-2026-01-21.all.masked.pb.gz` | `10_add_usher_tree_to_splits.sh` (step 10, test split) | https://hgdownload.soe.ucsc.edu/goldenPath/wuhCor1/UShER_SARS-CoV-2/2026/01/21/public-2026-01-21.all.masked.pb.gz |
| `public-2026-01-21.metadata.tsv.gz` | `10_add_usher_tree_to_splits.sh` (step 10, test split) | https://hgdownload.soe.ucsc.edu/goldenPath/wuhCor1/UShER_SARS-CoV-2/2026/01/21/public-2026-01-21.metadata.tsv.gz |

These exact filenames are what
`data/processed/spike_aa/splits/config_tree2splits_split1.csv` points at.
