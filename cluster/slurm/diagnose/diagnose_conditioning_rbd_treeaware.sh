#!/usr/bin/env bash
#SBATCH -J diag_cond_rbd_treeaware
#SBATCH -p cpu-single
#SBATCH -o logs/diagnose_conditioning_rbd_treeaware.out
#SBATCH -e logs/diagnose_conditioning_rbd_treeaware.err
#SBATCH -N 1 -c 1 --mem=24G
#SBATCH -t 10:00:00
#SBATCH --export=NONE

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
[[ -f cluster/env.local.sh ]] && source cluster/env.local.sh
conda activate "${CONDA_ENV:-evoplm}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p results/context_dependence

# Produces the masked/visible teacher-forced loss ratio (RBD, tree-aware).
# plot_conditioning_context.py reads the "ratio masked/normal" line from this log.
python scripts/diagnose_conditioning.py \
  --model_dir models/treeaware/rbd/split1/maxT1024_tree1_maskp0.1_span3/final \
  --tokenizer_json configs/protein_tree_tokenizer.json \
  --pairs_parquet data/trajectories/split1_test_first3Omi2laterOmi_pairs__rbd_t_ne_rbd_t1__delta_nt_mutcount_ge_1.parquet \
  --seq_in_col rbd_t \
  --seq_out_col rbd_t1 \
  --device cpu \
  --batch_size 16 \
  --n_rows 256 \
  --max_new_tokens 260 \
  --compute_teacher_forced_loss \
  --do_sample --top_p 0.9 --temperature 1.0 \
  --use_tree_tokens \
  2>&1 | tee results/context_dependence/rbd_treeaware.txt
