#!/usr/bin/env bash
# GPU job (requests 1 GPU via --gres=gpu:1; CPU-only launchers elsewhere in cluster/slurm/ use -p cpu-single).
#SBATCH --job-name=evoplm-train-rbd-seqonly
#SBATCH -o logs/train_rbd_seqonly.out
#SBATCH -e logs/train_rbd_seqonly.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --export=ALL
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source "$HOME/miniconda/etc/profile.d/conda.sh"
[[ -f cluster/env.local.sh ]] && source cluster/env.local.sh
conda activate "${CONDA_ENV:-evoplm}"
mkdir -p logs
python scripts/train_prefix_lm.py \
  --task rbd --split split1 \
  --traj_dir data/trajectories \
  --tokenizer_json configs/protein_tree_tokenizer.json \
  --mask_prob 0.10 --span_len 3 \
  --train_bs 32 --eval_bs 32 \
  --out_root models/seqonly
