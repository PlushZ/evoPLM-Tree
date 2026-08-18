#!/usr/bin/env bash
#SBATCH -J nextclade_arr
#SBATCH -p cpu-single
#SBATCH -N 1 -c 4 --mem=16G
#SBATCH -t 24:00:00
#SBATCH --export=NONE
#SBATCH -o logs/03_nextclade_all_%A_%a.out
#SBATCH -e logs/03_nextclade_all_%A_%a.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate evoplm-paper

DATASET_DIR=assets/nextclade/sars-cov-2
DEFAULT_SHARD_LIST="data/raw/ncbi_virus/shards/shard_list.txt"
SHARD_LIST="${SHARD_LIST:-$DEFAULT_SHARD_LIST}"   # may be injected by submitter
OUTROOT=data/raw/nextclade/all_parts

mkdir -p "$(dirname "$DATASET_DIR")" "$OUTROOT"

# fetch ref dataset
#nextclade dataset get --name sars-cov-2 --tag latest --output-dir "$DATASET_DIR"

# compute global shard index using OFFSET
OFFSET=${OFFSET:-0}
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "SLURM_ARRAY_TASK_ID is not set" >&2
  exit 1
fi
TASK_ID=$(( SLURM_ARRAY_TASK_ID + OFFSET ))   # 1-based

# guardrails: ensure TASK_ID is valid
N=$(wc -l < "$SHARD_LIST")
if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt "$N" ]; then
  echo "No shard for TASK_ID=$TASK_ID (OFFSET=$OFFSET, SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID; N=$N)" >&2
  exit 0
fi

# get the TASK_ID-th line (1-based)
FA=$(sed -n "${TASK_ID}p" "$SHARD_LIST")
if [ -z "$FA" ]; then
  echo "Empty shard path at line $TASK_ID" >&2
  exit 1
fi

BASENAME=$(basename "$FA" .fasta)
OUTDIR="$OUTROOT/$BASENAME"
mkdir -p "$OUTDIR"

# run nextclade with 4 threads
nextclade run -D "$DATASET_DIR" -j 4 -O "$OUTDIR" "$FA"
