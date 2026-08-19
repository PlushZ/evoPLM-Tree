#!/usr/bin/env bash
#SBATCH --job-name=evoplm-teacher-forcing-rbd-treeaware
#SBATCH -o logs/evaluate_rbd_treeaware.out
#SBATCH -e logs/evaluate_rbd_treeaware.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --export=ALL
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate evoplm
mkdir -p logs
python scripts/evaluate_teacher_forcing.py \
  --model_dir models/treeaware/rbd/split1/maxT1024_tree1_maskp0.1_span3/final \
  --tokenizer_json configs/protein_tree_tokenizer.json \
  --test_pairs data/trajectories/split1_test_first3Omi2laterOmi_pairs__rbd_t_ne_rbd_t1__delta_nt_mutcount_ge_1.parquet \
  --train_pairs_for_unigram data/trajectories/split1_train_first3Omi2laterOmi_pairs__rbd_t_ne_rbd_t1__delta_nt_mutcount_ge_1.parquet \
  --seq_in_col rbd_t --seq_out_col rbd_t1 \
  --use_tree_tokens_in_eval \
  --out_json results/teacher_forcing/split1/rbd_treeaware.json \
  --metrics_csv results/teacher_forcing/split1/rbd_treeaware_metrics.csv \
  --out_csv results/teacher_forcing/split1/rbd_treeaware_examples.csv \
  --batch_size 64
