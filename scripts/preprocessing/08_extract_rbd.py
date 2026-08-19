#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# === Input / Output ===
in_path = Path("data/processed/spike_aa/spike_nodup_location_coldate.parquet")
out_path = Path("data/processed/spike_aa/spike_with_rbd.parquet")

# === Load ===
print(f"Loading {in_path} ...")
df = pd.read_parquet(in_path)
print("Rows:", len(df))

# === Add RBD column ===
# Spike AA length ~1273 aa
# RBD is aa 319–541 (inclusive, 1-based numbering)
# -> Python slice [318:541] (0-based, end exclusive)
df["rbd_aa"] = df["spike_aa"].str.slice(318, 541)

# Count unknown residues ("X") in RBD
df["rbd_num_X"] = df["rbd_aa"].str.count("X")

# === Save ===
df.to_parquet(out_path, index=False)

# Write only the first 1000 rows to CSV
df.head(1000).to_csv("data/processed/spike_aa/spike_with_rbd_head1000.csv", index=False)


print(f"Saved parquet with RBD column -> {out_path}")