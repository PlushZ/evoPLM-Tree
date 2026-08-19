#!/usr/bin/env bash
#SBATCH -J fig_conditioning_context
#SBATCH -p cpu-single
#SBATCH -N 1 -c 2 --mem=32G
#SBATCH -t 01:00:00
#SBATCH --export=NONE
#SBATCH -o logs/plot_conditioning_context.out
#SBATCH -e logs/plot_conditioning_context.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate evoplm
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs

python scripts/figures/plot_conditioning_context.py \
  --out_dir results/context_dependence/plots
