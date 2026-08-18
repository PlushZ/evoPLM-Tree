#!/usr/bin/env bash
#SBATCH -J fig_obs_pred_freq
#SBATCH -p cpu-single
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH --export=NONE
#SBATCH -o logs/plot_observed_vs_predicted_frequency.out
#SBATCH -e logs/plot_observed_vs_predicted_frequency.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate evoplm-paper
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs results/generation/split1/plots

python scripts/figures/plot_observed_vs_predicted_frequency.py \
  --rbd_csv results/generation/split1/rbd_treeaware_sample2000.csv \
  --spike_csv results/generation/split1/spike_treeaware_sample2000.csv \
  --out_dir results/generation/split1/plots \
  --dpi 300 \
  --hexbin_gridsize 35 \
  --rho_tolerance 0.005 \
  --include_log_plots
