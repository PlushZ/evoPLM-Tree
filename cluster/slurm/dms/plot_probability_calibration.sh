#!/usr/bin/env bash
#SBATCH -J dms_calib
#SBATCH -p cpu-single
#SBATCH -N 1 -c 2 --mem=32G
#SBATCH -t 01:00:00
#SBATCH --export=NONE
#SBATCH -o logs/plot_probability_calibration.out
#SBATCH -e logs/plot_probability_calibration.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate evoplm-paper
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/dms/plot_probability_calibration.py \
  --metrics_root results/dms \
  --out_root results/dms/calibration \
  --n_bins 10
