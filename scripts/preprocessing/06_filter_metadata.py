import pandas as pd
from pathlib import Path

meta_path = Path("data/raw/nextclade/all/nextclade_augmented.tsv")
fasta_path = Path("data/raw/nextclade/all/gene_spike.fasta")

out_dir = Path("data/processed/spike_aa")
out_dir.mkdir(parents=True, exist_ok=True)

tsv_out = out_dir / "spike.tsv"
parquet_out = out_dir / "spike.parquet"
fasta_out = out_dir / "spike.fasta"

df = pd.read_csv(meta_path, sep="\t", low_memory=False)
print(f"Initial rows: {len(df)}")

# load spike AA fasta
def load_fasta_to_dict(fasta_file):
    seqs = {}
    with open(fasta_file) as f:
        sid, buf = None, []
        for line in f:
            if line.startswith(">"):
                if sid and buf:
                    seqs[sid] = "".join(buf)
                sid = line[1:].strip()
                buf = []
            else:
                buf.append(line.strip())
        if sid and buf:
            seqs[sid] = "".join(buf)
    return seqs

spike_map = load_fasta_to_dict(fasta_path)
print(f"FASTA sequences loaded: {len(spike_map)}")

# merge sequences into metadata
df["spike_aa"] = df["seqName"].map(spike_map)
before = len(df)
df = df.dropna(subset=["spike_aa"])
print(f"Missing spike_aa: {before - len(df)} removed; {len(df)} remain")

# add number of X residues in spike
df["num_X"] = df["spike_aa"].str.count("X")

# ensure collectionDate is datetime
df["ncbi.collectionDate"] = pd.to_datetime(df["ncbi.collectionDate"], errors="coerce")

# apply filters
filters = []

filters.append(("valid collectionDate (2019-2025)",
                df["ncbi.collectionDate"].between("2019-12-01", "2025-12-31")))

filters.append(("qc.overallStatus not 'bad'",
                df["qc.overallStatus"] != "bad"))

filters.append(("errors empty",
                df["errors"].isna() | (df["errors"] == "")))

filters.append(("failedCdses empty",
                df["failedCdses"].isna() | (df["failedCdses"] == "")))

filters.append(("qc.privateMutations.status not 'bad'",
                df["qc.privateMutations.status"] != "bad"))

filters.append(("qc.snpClusters.status not 'bad'",
                df["qc.snpClusters.status"] != "bad"))

filters.append(("spike_aa length 1260-1285",
                df["spike_aa"].str.len().between(1260, 1285)))

filters.append(("spike_aa ≤ 3 unknown residues (X)", df["num_X"] <= 3))


# apply filters and count drops after each filter
current = df
for name, mask in filters:
    before = len(current)
    current = current[mask].copy()
    after = len(current)
    print(f"{name}: {before - after} removed; {after} remain")

# save tsv and parquet
current.to_csv(tsv_out, sep="\t", index=False)
current.to_parquet(parquet_out, index=False)
print(f"\n Saved metadata:\n - {tsv_out}\n - {parquet_out}")

# save fasta
keep_seqs = set(current["seqName"])

with open(fasta_out, "w") as out_fa:
    for name in keep_seqs:
        seq = spike_map.get(name)
        if seq:
            out_fa.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                out_fa.write(seq[i:i+80] + "\n")

print(f" Saved filtered FASTA: {fasta_out}")
