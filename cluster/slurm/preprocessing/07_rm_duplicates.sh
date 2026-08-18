#!/usr/bin/env bash
#SBATCH -J rm_dup
#SBATCH -p cpu-single
#SBATCH -N 1 -c 4 --mem=128G
#SBATCH -t 24:00:00
#SBATCH --export=NONE
#SBATCH -o logs/07_rm_duplicates.out
#SBATCH -e logs/07_rm_duplicates.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate evoplm-paper

python scripts/preprocessing/07_rm_duplicates.py
