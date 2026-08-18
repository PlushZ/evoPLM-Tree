#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# === Input / Output ===
in_path = Path("data/processed/spike_aa/spike.parquet")
out_dir = Path("data/processed/spike_aa/splits")
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading {in_path} ...")
df = pd.read_parquet(in_path)
print("Total rows:", len(df))

# --- Clade sets ---
omicron_train_3 = ["21M", "21L", "21K"]
omicron_test_descendants = [
    "22A","22B","22C","22D","22E",
    "23C","23I",
    "24A","24B","24C","24E","24G","24H","24I",
    "25A"
]

date_series = df["ncbi.collectionDate"]
if not pd.api.types.is_datetime64_any_dtype(date_series):
    date_series = pd.to_datetime(date_series, errors="coerce", cache=True)

non_recombinant_mask = ~df["Nextclade_pango"].astype(str).str.startswith("X")

# --- Split 1: first 3 Omicrons vs later Omicrons (exclude recombinants) ---
omicron_test_all = omicron_train_3 + omicron_test_descendants
split1_train_mask = (
    non_recombinant_mask
    & df["clade"].isin(omicron_train_3)
    & (date_series < "2022-06-01")
)
split1_test_mask = (
    non_recombinant_mask
    & df["clade"].isin(omicron_test_all)
    & (date_series >= "2022-06-01")
)

train1 = df[split1_train_mask].copy()
test1 = df[split1_test_mask].copy()

train1.to_parquet(out_dir / "split1_train_first3Omi2laterOmi.parquet", index=False)
test1.to_parquet(out_dir / "split1_test_first3Omi2laterOmi.parquet", index=False)
print("Split 1 ->", len(train1), "train,", len(test1), "test")

# --- Split 2: pre-Omicron vs Omicron (exclude recombinants) ---
split2_train_mask = non_recombinant_mask & (date_series < "2021-11-01")
split2_test_mask = non_recombinant_mask & (date_series >= "2021-11-01")

train2 = df[split2_train_mask].copy()
test2 = df[split2_test_mask].copy()

train2.to_parquet(out_dir / "split2_train_preOmi2Omi.parquet", index=False)
test2.to_parquet(out_dir / "split2_test_preOmi2Omi.parquet", index=False)
print("Split 2 ->", len(train2), "train,", len(test2), "test")

print(f"\nSaved splits into {out_dir}")
