#!/usr/bin/env bash
#SBATCH -J nc_filter_split
#SBATCH -p cpu-single
#SBATCH -N 1 -c 2 --mem=8G
#SBATCH -t 02:00:00
#SBATCH --export=NONE
#SBATCH -o logs/02_prefilter_split.out
#SBATCH -e logs/02_prefilter_split.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate evoplm-paper

IN=data/raw/ncbi_virus/all_genomes.fasta
FIL=data/raw/ncbi_virus/all_genomes_len28500_31000.fasta
SHDIR=data/raw/ncbi_virus/shards

mkdir -p "$(dirname "$FIL")" "$SHDIR" logs

# Sanity: input must exist
if [ ! -s "$IN" ]; then
  echo "ERROR: Input FASTA not found or empty: $IN" >&2
  exit 1
fi

echo "=== Input stats ==="
# quick counts
grep -c '^>' "$IN" | awk '{print "Sequences:", $1}'
# length range histogram (first 10 lines)
seqkit fx2tab -l "$IN" | awk '{print $2}' | awk '
  {b=int($1/1000)} {cnt[b]++}
  END{for(i in cnt) printf("%s-%sk: %d\n", i, i+1, cnt[i]) | "sort -n"}' | head -n 20

# keep near-complete genomes
seqkit seq -m 28500 -M 31000 "$IN" > "$FIL" || true

N_KEEP=$(grep -c '^>' "$FIL" 2>/dev/null || echo 0)
echo "Kept (28.5-31 kb): $N_KEEP sequences"

if [ "$N_KEEP" -eq 0 ]; then
  echo "No sequences in 28.5-31 kb. Relaxing to 25-32 kb..."
  FIL=data/raw/ncbi_virus/all_genomes_len25000_32000.fasta
  seqkit seq -m 25000 -M 32000 "$IN" > "$FIL"
  N_KEEP=$(grep -c '^>' "$FIL" || echo 0)
  echo "Kept (25-32 kb): $N_KEEP sequences"
  if [ "$N_KEEP" -eq 0 ]; then
    echo "ERROR: Still zero sequences after relaxed bounds. Check input." >&2
    exit 1
  fi
fi

# split into ~5k sequences per shard
seqkit split -s 5000 -O "$SHDIR" "$FIL"

# record shard list for the array job
ls -1 "$SHDIR"/*.fasta > "$SHDIR/shard_list.txt"

N_SHARDS=$(wc -l < "$SHDIR/shard_list.txt")
echo "Shards: $N_SHARDS written to $SHDIR"
