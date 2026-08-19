#!/usr/bin/env bash
#SBATCH -J dms_metrics
#SBATCH -p cpu-single
#SBATCH -N 1 -c 2 --mem=32G
#SBATCH -t 01:00:00
#SBATCH --export=NONE
#SBATCH -o logs/evaluate_dms.out
#SBATCH -e logs/evaluate_dms.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate evoplm
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/dms/evaluate_dms.py \
  --dms_scores_dir results/dms/dms_scores \
  --seqonly_probs_csv results/dms/rbd_split1_seqonly_mutation_probs.csv \
  --treeaware_probs_csv results/dms/rbd_split1_treeaware_mutation_probs.csv \
  --k 3 \
  --threshold 0.0 \
  --out_root results/dms
