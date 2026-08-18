import pandas as pd
import json
from pathlib import Path

nextclade_path = Path("data/raw/nextclade/all/nextclade.tsv")
ncbi_jsonl_path = Path("data/raw/ncbi_virus/all/ncbi_dataset/data/data_report.jsonl")
out_path = nextclade_path.with_name("nextclade_augmented.tsv")

df = pd.read_csv(nextclade_path, sep="\t", low_memory=False)
df["ncbi.accession"] = df["seqName"].astype(str).str.split(" ").str[0].str.strip()

records = []
with open(ncbi_jsonl_path) as fh:
    for line in fh:
        try:
            obj = json.loads(line)
            acc = obj.get("accession")
            if not acc:
                continue
            r = {
                "ncbi.accession": acc,
                "ncbi.collectionDate": obj.get("isolate", {}).get("collectionDate"),
                "ncbi.geographicLocation": obj.get("location", {}).get("geographicLocation"),
                "ncbi.geographicRegion": obj.get("location", {}).get("geographicRegion"),
                "ncbi.bioprojects": ";".join(obj.get("bioprojects", [])),
                "ncbi.organismName": obj.get("host", {}).get("organismName"),
                "ncbi.pangolinClassification": obj.get("virus", {}).get("pangolinClassification"),
            }
            records.append(r)
        except json.JSONDecodeError:
            continue

meta = pd.DataFrame(records)

meta["ncbi.collectionDate"] = pd.to_datetime(meta["ncbi.collectionDate"], errors="coerce")

df_aug = df.merge(meta, on="ncbi.accession", how="left")

df_aug.to_csv(out_path, sep="\t", index=False)
print(f" Saved augmented file: {out_path}")
