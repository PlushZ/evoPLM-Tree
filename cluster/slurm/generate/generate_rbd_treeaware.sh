#!/usr/bin/env bash
# GPU job (requests 1 GPU via --gres=gpu:1; CPU-only launchers elsewhere in cluster/slurm/ use -p cpu-single).
#SBATCH --job-name=evoplm-generate-rbd-treeaware
#SBATCH -o logs/generate_rbd_treeaware.out
#SBATCH -e logs/generate_rbd_treeaware.err
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
python scripts/generate_sequences.py \
  --model_dir models/treeaware/rbd/split1/maxT1024_tree1_maskp0.1_span3/final \
  --tokenizer_json configs/protein_tree_tokenizer.json \
  --test_pairs data/trajectories/split1_test_first3Omi2laterOmi_pairs__rbd_t_ne_rbd_t1__delta_nt_mutcount_ge_1.parquet \
  --seq_in_col rbd_t --seq_out_col rbd_t1 \
  --use_tree_tokens_in_eval \
  --gen_n_samples 2000 --gen_seed 0 \
  --out_json results/generation/split1/rbd_treeaware_sample2000.json \
  --out_csv results/generation/split1/rbd_treeaware_sample2000.csv
