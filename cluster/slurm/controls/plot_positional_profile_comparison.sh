#!/usr/bin/env bash
#SBATCH -J ctl_plot_profiles
#SBATCH -p cpu-single
#SBATCH -N 1 -c 2 --mem=16G
#SBATCH -t 00:30:00
#SBATCH --export=NONE
#SBATCH -o logs/plot_positional_profile_comparison.out
#SBATCH -e logs/plot_positional_profile_comparison.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
[[ -f cluster/env.local.sh ]] && source cluster/env.local.sh
conda activate "${CONDA_ENV:-evoplm}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs results/controls/plots

python scripts/figures/plot_positional_profile_comparison.py \
  --metrics_dir results/controls \
  --out_png results/controls/plots/positional_profile_comparison.png

echo "[DONE] Figure written to results/controls/plots/"
