#!/usr/bin/env bash
#SBATCH -J dms_plots
#SBATCH -p cpu-single
#SBATCH -N 1 -c 2 --mem=32G
#SBATCH -t 01:00:00
#SBATCH --export=NONE
#SBATCH -o logs/plot_dms_comparison.out
#SBATCH -e logs/plot_dms_comparison.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate evoplm-paper
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# plot_dms_comparison.py expects a flat --metrics_dir (one dataset's files directly inside),
# but evaluate_dms.py's --out_root results/dms creates one subdirectory per dataset -
# so this runs once per dataset rather than pointing at the shared root.
python scripts/dms/plot_dms_comparison.py \
  --metrics_dir results/dms/urn_mavedb_00000670-a-1 \
  --out_dir results/dms/plots/urn_mavedb_00000670-a-1 \
  --k 3

python scripts/dms/plot_dms_comparison.py \
  --metrics_dir results/dms/urn_mavedb_00000671-a-1 \
  --out_dir results/dms/plots/urn_mavedb_00000671-a-1 \
  --k 3
