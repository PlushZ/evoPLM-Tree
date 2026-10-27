#!/usr/bin/env bash
#SBATCH -J rbd_extract
#SBATCH -p cpu-single
#SBATCH -N 1 -c 1 --mem=24G
#SBATCH -t 24:00:00
#SBATCH --export=NONE
#SBATCH -o logs/08_extract_rbd.out
#SBATCH -e logs/08_extract_rbd.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
[[ -f cluster/env.local.sh ]] && source cluster/env.local.sh
conda activate "${CONDA_ENV:-evoplm}"

python scripts/preprocessing/08_extract_rbd.py
