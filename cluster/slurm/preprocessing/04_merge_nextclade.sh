#!/usr/bin/env bash
#SBATCH -J nextclade_merge
#SBATCH -p cpu-single
#SBATCH -N 1 -c 1 --mem=4G
#SBATCH -t 09:00:00
#SBATCH --export=NONE
#SBATCH -o logs/04_merge_nextclade.out
#SBATCH -e logs/04_merge_nextclade.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"

PARTS=data/raw/nextclade/all_parts
OUTDIR=data/raw/nextclade/all
mkdir -p "$OUTDIR"

# merge TSVs
first=1
for tsv in "$PARTS"/*/nextclade.tsv; do
  if [ $first -eq 1 ]; then
    cat "$tsv" > "$OUTDIR/nextclade.tsv"
    first=0
  else
    tail -n +2 "$tsv" >> "$OUTDIR/nextclade.tsv"
  fi
done

# merge Spike AA FASTAs
cat "$PARTS"/*/nextclade.cds_translation.S.fasta > "$OUTDIR/gene_spike.fasta"

#echo "Merged TSV -> $OUTDIR/nextclade.tsv"
echo "Merged Spike -> $OUTDIR/gene_spike.fasta"
