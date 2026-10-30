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
python scripts/train_prefix_lm.py \
  --task rbd --split split1 \
  --traj_dir data/trajectories \
  --tokenizer_json configs/protein_tree_tokenizer.json \
  --mask_prob 0.10 --span_len 3 \
  --train_bs 32 --eval_bs 32 \
  --out_root models/seqonly
