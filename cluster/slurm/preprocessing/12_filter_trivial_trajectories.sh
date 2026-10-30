#!/usr/bin/env bash
#SBATCH -J filter_trivial_traj
#SBATCH -p cpu-single
#SBATCH -N 1 -c 2 --mem=32G
#SBATCH -t 01:00:00
#SBATCH --export=NONE
#SBATCH -o logs/12_filter_trivial_trajectories.out
#SBATCH -e logs/12_filter_trivial_trajectories.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
[[ -f cluster/env.local.sh ]] && source cluster/env.local.sh
conda activate "${CONDA_ENV:-evoplm}"
# pandas/scipy load C++ extensions built against the environment's libstdc++;
# compute nodes can ship an older /lib64 one (GLIBCXX_3.4.29 not found).
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python scripts/preprocessing/12_filter_trivial_trajectories.py \
  --in_dir data/processed/spike_aa/trajectories \
  --out_dir data/trajectories \
  --split split1
