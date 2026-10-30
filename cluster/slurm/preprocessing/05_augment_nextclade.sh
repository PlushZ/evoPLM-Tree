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
[[ -f cluster/env.local.sh ]] && source cluster/env.local.sh
conda activate "${CONDA_ENV:-evoplm}"
# pandas/scipy load C++ extensions built against the environment's libstdc++;
# compute nodes can ship an older /lib64 one (GLIBCXX_3.4.29 not found).
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python scripts/preprocessing/05_augment_nextclade.py
