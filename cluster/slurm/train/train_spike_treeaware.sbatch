#!/usr/bin/env bash
#SBATCH --job-name=evoplm-train-spike-treeaware
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --export=ALL
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs
python scripts/train_prefix_lm.py \
  --task spike --split split1 \
  --traj_dir data/trajectories \
  --tokenizer_json configs/protein_tree_tokenizer.json \
  --max_tokens 4096 \
  --mask_prob 0.10 --span_len 3 \
  --train_bs 32 --eval_bs 32 \
  --out_root models/treeaware \
  --use_tree_tokens
