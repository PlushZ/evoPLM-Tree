#!/usr/bin/env bash
#SBATCH -J add_tree_to_splits
#SBATCH -p cpu-single
#SBATCH -N 1 -c 2 --mem=96G
#SBATCH -t 99:00:00
#SBATCH --export=NONE
#SBATCH -o logs/10_add_usher_tree_to_splits.out
#SBATCH -e logs/10_add_usher_tree_to_splits.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate evoplm-paper-bte
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python scripts/preprocessing/10_add_usher_tree_to_splits.py --config_csv data/processed/spike_aa/splits/config_tree2splits_all_splits.csv
