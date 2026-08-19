#!/usr/bin/env bash
#SBATCH --job-name=evoplm-generate-spike-treeaware
#SBATCH -o logs/generate_spike_treeaware.out
#SBATCH -e logs/generate_spike_treeaware.err
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
  --model_dir models/treeaware/spike/split1/maxT4096_tree1_maskp0.1_span3/final \
  --tokenizer_json configs/protein_tree_tokenizer.json \
  --test_pairs data/trajectories/split1_test_first3Omi2laterOmi_pairs__spike_t_ne_spike_t1__delta_nt_mutcount_ge_1.parquet \
  --seq_in_col spike_t --seq_out_col spike_t1 \
  --use_tree_tokens_in_eval \
  --gen_n_samples 2000 --gen_seed 0 \
  --max_new_tokens_cap 1400 \
  --out_json results/generation/split1/spike_treeaware_sample2000.json \
  --out_csv results/generation/split1/spike_treeaware_sample2000.csv
