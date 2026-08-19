#!/usr/bin/env bash
#SBATCH -J dms_prepare_table
#SBATCH -p cpu-single
#SBATCH -N 1 -c 1 --mem=4G
#SBATCH -t 00:15:00
#SBATCH --export=NONE
#SBATCH -o logs/prepare_dms_table.out
#SBATCH -e logs/prepare_dms_table.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate evoplm
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p results/dms

python scripts/dms/prepare_dms_table.py \
  --in_csv data/dms/urn_mavedb_00000670-a-1_scores.csv \
  --out_csv results/dms/dms_scores/urn_mavedb_00000670-a-1_table.csv

python scripts/dms/prepare_dms_table.py \
  --in_csv data/dms/urn_mavedb_00000671-a-1_scores.csv \
  --out_csv results/dms/dms_scores/urn_mavedb_00000671-a-1_table.csv
