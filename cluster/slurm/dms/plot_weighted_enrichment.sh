#!/usr/bin/env bash
#SBATCH -J dms_weighted_enrich
#SBATCH -p cpu-single
#SBATCH -N 1 -c 2 --mem=16G
#SBATCH -t 00:30:00
#SBATCH --export=NONE
#SBATCH -o logs/plot_weighted_enrichment.out
#SBATCH -e logs/plot_weighted_enrichment.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate evoplm
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs

python scripts/dms/plot_weighted_enrichment.py \
  --metrics_csv results/dms/metrics_summary_all_datasets.csv \
  --out_dir results/dms/plots \
  --out_name dms_weighted_enrichment_treeaware.png \
  --model treeaware

# Uncomment to also plot the seq-only model:
# python scripts/dms/plot_weighted_enrichment.py \
#   --metrics_csv results/dms/metrics_summary_all_datasets.csv \
#   --out_dir results/dms/plots \
#   --out_name dms_weighted_enrichment_seqonly.png \
#   --model seqonly
