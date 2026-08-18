#!/usr/bin/env bash
#SBATCH -J gen_traj_12
#SBATCH -p cpu-single
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH --export=NONE
#SBATCH -o logs/11_gen_trajectories_tree_aware.out
#SBATCH -e logs/11_gen_trajectories_tree_aware.err

set -euo pipefail

export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate evoplm-paper-bte
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python scripts/preprocessing/11_gen_trajectories_tree_aware.py \
  --config_csv data/processed/spike_aa/splits/config_tree2splits_all_splits.csv \
  --out_dir data/processed/spike_aa/trajectories \
  --k_list 1,2,4,8 \
  --max_train_pairs_per_desc 2 \
  --max_test_pairs_per_desc 2 \
  --max_delta_edges 200 \
  --representative max_dup
