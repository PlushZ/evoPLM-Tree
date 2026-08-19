#!/usr/bin/env bash
#SBATCH -J splits
#SBATCH -p cpu-single
#SBATCH -N 1 -c 2 --mem=32G
#SBATCH -t 01:00:00
#SBATCH --export=NONE
#SBATCH -o logs/09_splits_with_cutoff.out
#SBATCH -e logs/09_splits_with_cutoff.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate evoplm
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python scripts/preprocessing/09_splits_with_cutoff.py
