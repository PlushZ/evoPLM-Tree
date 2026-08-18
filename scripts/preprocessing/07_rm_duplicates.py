import pandas as pd

df = pd.read_parquet("data/processed/spike_aa/spike.parquet")

print("Before:", len(df))
"""
df_unique_accession = df.drop_duplicates(subset=["ncbi.accession", "spike_aa"])
print("After (rm seq+ncbi.accession):", len(df_unique_accession))
df_unique_accession.to_parquet("data/processed/spike_aa/spike_nodup_accession.parquet", index=False)


df_unique_coldate = df.drop_duplicates(subset=["ncbi.collectionDate", "spike_aa"])
print("After (rm seq+ncbi.collectionDate):", len(df_unique_coldate))
df_unique_coldate.to_parquet("data/processed/spike_aa/spike_nodup_coldate.parquet", index=False)


df_unique_region = df.drop_duplicates(subset=["ncbi.geographicRegion", "spike_aa"])
print("After (rm seq+ncbi.geographicRegion):", len(df_unique_region))
df_unique_region.to_parquet("data/processed/spike_aa/spike_nodup_region.parquet", index=False)


df_unique_location = df.drop_duplicates(subset=["ncbi.geographicLocation", "spike_aa"])
print("After (rm seq+ncbi.geographicLocation):", len(df_unique_location))
df_unique_location.to_parquet("data/processed/spike_aa/spike_nodup_location.parquet", index=False)
"""

#df_unique_location_coldate = df.drop_duplicates(subset=["ncbi.geographicLocation", "spike_aa", "ncbi.collectionDate"])
#print("After (rm seq+ncbi.geographicLocation+ncbi.collectionDate):", len(df_unique_location_coldate))


# === Define grouping keys ===
group_cols = ["spike_aa", "ncbi.geographicLocation", "ncbi.collectionDate"]

# === Deduplicate: take first row in each group ===
df_unique = (
    df.sort_values("ncbi.collectionDate")        # optional: ensures consistent order
      .groupby(group_cols, as_index=False)
      .first()                                   # keep first record per group
)

# === Add dup_count column ===
dup_counts = df.groupby(group_cols).size().reset_index(name="dup_count")
df_unique = df_unique.merge(dup_counts, on=group_cols, how="left")

print("Rows after deduplication:", len(df_unique))
print("dup_count stats:")
print(df_unique["dup_count"].describe())

df_unique.to_parquet("data/processed/spike_aa/spike_nodup_location_coldate.parquet", index=False)
df_unique.to_csv("data/processed/spike_aa/spike_nodup_location_coldate.csv", index=False)