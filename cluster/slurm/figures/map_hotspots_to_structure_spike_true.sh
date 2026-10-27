#!/usr/bin/env bash
#SBATCH -J fig_hotspots_spike_true
#SBATCH -p cpu-single
#SBATCH -N 1 -c 2 --mem=32G
#SBATCH -t 01:00:00
#SBATCH --export=NONE
#SBATCH -o logs/map_hotspots_to_structure_spike_true.out
#SBATCH -e logs/map_hotspots_to_structure_spike_true.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
[[ -f cluster/env.local.sh ]] && source cluster/env.local.sh
conda activate "${CONDA_ENV:-evoplm}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs

python scripts/figures/map_hotspots_to_structure.py \
  --csv_path results/generation/split1/spike_treeaware_sample2000.csv \
  --pdb_path data/pdb/6VSB.pdb --chain A --score true
