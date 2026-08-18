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
conda activate evoplm-paper
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# evaluate_dms.py globs *.csv in --dms_scores_dir and treats each as a DMS table, so it
# needs a directory containing only the *_table.csv files (not the mixed results/dms/ root).
mkdir -p results/dms/dms_scores
cp -n results/dms/urn_mavedb_00000670-a-1_table.csv results/dms/dms_scores/ 2>/dev/null || true
cp -n results/dms/urn_mavedb_00000671-a-1_table.csv results/dms/dms_scores/ 2>/dev/null || true

python scripts/dms/evaluate_dms.py \
  --dms_scores_dir results/dms/dms_scores \
  --seqonly_probs_csv results/dms/rbd_split1_seqonly_mutation_probs.csv \
  --treeaware_probs_csv results/dms/rbd_split1_treeaware_mutation_probs.csv \
  --k 3 \
  --threshold 0.0 \
  --out_root results/dms
