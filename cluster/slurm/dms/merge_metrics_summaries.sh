#!/usr/bin/env bash
#SBATCH -J dms_merge_metrics
#SBATCH -p cpu-single
#SBATCH -o logs/merge_metrics_summaries.out
#SBATCH -e logs/merge_metrics_summaries.err
#SBATCH -N 1 -c 1 --mem=4G
#SBATCH -t 00:10:00
#SBATCH --export=NONE

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate evoplm
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/dms/merge_metrics_summaries.py \
  --metrics_root results/dms \
  --dataset_ids urn_mavedb_00000670-a-1 urn_mavedb_00000671-a-1 \
  --out_csv results/dms/metrics_summary_all_datasets.csv
