#!/usr/bin/env bash
# GPU job (requests 1 GPU via --gres=gpu:1; CPU-only launchers elsewhere in cluster/slurm/ use -p cpu-single).
#SBATCH --job-name=evoplm-shuffle-controls
#SBATCH -o logs/shuffle_controls.out
#SBATCH -e logs/shuffle_controls.err
#SBATCH -p gpu-single
#SBATCH --gres=gpu:1
#SBATCH --account=bw24h008
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
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
mkdir -p logs results/controls/shuffle results/controls/permuted

TRAJ=data/trajectories
STEM=split1_test_first3Omi2laterOmi_pairs
N_ROWS=512
SEED=0

for TASK in rbd spike; do
  PAIRS="$TRAJ/${STEM}__${TASK}_t_ne_${TASK}_t1__delta_nt_mutcount_ge_1.parquet"

  echo "######## building permuted variants: $TASK ########"
  python scripts/controls/make_permuted_pairs.py \
    --pairs_parquet "$PAIRS" \
    --task "$TASK" \
    --n_rows "$N_ROWS" \
    --seed "$SEED" \
    --out_dir results/controls/permuted

  if [[ "$TASK" == "rbd" ]]; then
    TREE_MODEL=models/treeaware/rbd/split1/maxT1024_tree1_maskp0.1_span3/final
    SEQ_MODEL=models/seqonly/rbd/split1/maxT1024_tree0_maskp0.1_span3/final
    BS=32
  else
    TREE_MODEL=models/treeaware/spike/split1/maxT4096_tree1_maskp0.1_span3/final
    SEQ_MODEL=models/seqonly/spike/split1/maxT4096_tree0_maskp0.1_span3/final
    BS=8
  fi

  # Tree-aware model: source substitutions and tree-feature permutations.
  for VAR in reference source_within_lineage source_across_lineage source_global \
             tree_joint tree_independent; do
    echo "=== $TASK / treeaware / $VAR ==="
    python scripts/controls/teacher_forced_loss.py \
      --model_dir "$TREE_MODEL" \
      --pairs_parquet "results/controls/permuted/${TASK}_${VAR}.parquet" \
      --task "$TASK" \
      --use_tree_tokens \
      --label "$VAR" \
      --batch_size "$BS" \
      --out_csv "results/controls/shuffle/${TASK}_treeaware_${VAR}.csv"
  done

  # Sequence-only model: tree variants are meaningless, it reads no tree tokens.
  for VAR in reference source_within_lineage source_across_lineage source_global; do
    echo "=== $TASK / seqonly / $VAR ==="
    python scripts/controls/teacher_forced_loss.py \
      --model_dir "$SEQ_MODEL" \
      --pairs_parquet "results/controls/permuted/${TASK}_${VAR}.parquet" \
      --task "$TASK" \
      --label "$VAR" \
      --batch_size "$BS" \
      --out_csv "results/controls/shuffle/${TASK}_seqonly_${VAR}.csv"
  done
done

echo "######## summary ########"
python scripts/controls/summarize_shuffle_controls.py \
  --loss_dir results/controls/shuffle \
  --out_prefix results/controls/shuffle_controls

echo "[DONE] Shuffle controls written to results/controls/"
