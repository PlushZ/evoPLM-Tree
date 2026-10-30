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
# Under sbatch the script runs from SLURM's spool copy, so a path derived from
# ${BASH_SOURCE[0]} points outside the repository. SLURM_SUBMIT_DIR is the
# reliable anchor; the BASH_SOURCE form is the fallback for direct shell runs.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_ROOT"
[[ -f pyproject.toml && -d cluster/slurm ]] || {
  echo "[FATAL] not the repository root: $REPO_ROOT" >&2
  echo "        submit with: sbatch cluster/slurm/<dir>/<script>.sh from the repo root" >&2
  exit 1
}
source "$HOME/miniconda/etc/profile.d/conda.sh"
[[ -f cluster/env.local.sh ]] && source cluster/env.local.sh
conda activate "${CONDA_ENV:-evoplm}"
# pandas/scipy load C++ extensions built against the environment's libstdc++;
# compute nodes can ship an older /lib64 one (GLIBCXX_3.4.29 not found).
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
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
