#!/usr/bin/env bash
#SBATCH -J filter_metadata
#SBATCH -p cpu-single
#SBATCH -N 1 -c 4 --mem=128G
#SBATCH -t 24:00:00
#SBATCH --export=NONE
#SBATCH -o logs/06_filter_metadata.out
#SBATCH -e logs/06_filter_metadata.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
[[ -f cluster/env.local.sh ]] && source cluster/env.local.sh
conda activate "${CONDA_ENV:-evoplm}"

python scripts/preprocessing/06_filter_metadata.py
