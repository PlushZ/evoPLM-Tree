#!/usr/bin/env bash
#SBATCH -J fig_mutation_hotspots
#SBATCH -p cpu-single
#SBATCH -N 1 -c 2 --mem=32G
#SBATCH -t 01:00:00
#SBATCH --export=NONE
#SBATCH -o logs/plot_mutation_hotspots.out
#SBATCH -e logs/plot_mutation_hotspots.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate evoplm
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs

python scripts/figures/plot_mutation_hotspots.py \
  --csv_glob "results/generation/split1/*sample2000.csv" \
  --out_dir results/generation/split1/plots
