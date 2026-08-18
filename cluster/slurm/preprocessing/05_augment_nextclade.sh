#!/usr/bin/env bash
#SBATCH -J augment_nextclade
#SBATCH -p cpu-single
#SBATCH -N 1 -c 4 --mem=64G
#SBATCH -t 09:00:00
#SBATCH --export=NONE
#SBATCH -o logs/05_augment_nextclade.out
#SBATCH -e logs/05_augment_nextclade.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate evoplm-paper

python scripts/preprocessing/05_augment_nextclade.py
