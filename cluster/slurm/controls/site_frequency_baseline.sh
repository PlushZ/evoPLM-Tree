#!/usr/bin/env bash
#SBATCH -J ctl_site_freq
#SBATCH -p cpu-single
#SBATCH -N 1 -c 2 --mem=32G
#SBATCH -t 01:00:00
#SBATCH --export=NONE
#SBATCH -o logs/site_frequency_baseline.out
#SBATCH -e logs/site_frequency_baseline.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
[[ -f cluster/env.local.sh ]] && source cluster/env.local.sh
conda activate "${CONDA_ENV:-evoplm}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs results/controls

TRAJ=data/trajectories
STEM=split1_train_first3Omi2laterOmi_pairs

# Four variants per task:
#   all / dedup       counting every row vs each distinct sequence pair, since
#                     the tables are ~9x redundant
#   raw / noamb       keeping vs dropping positions that carry X, * or -
for TASK in rbd spike; do
  PAIRS="$TRAJ/${STEM}__${TASK}_t_ne_${TASK}_t1__delta_nt_mutcount_ge_1.parquet"

  for MODE in all dedup; do
    DEDUP_FLAG=""
    [[ "$MODE" == "dedup" ]] && DEDUP_FLAG="--dedup"

    for AMB in raw noamb; do
      AMB_FLAG=""
      [[ "$AMB" == "noamb" ]] && AMB_FLAG="--exclude_ambiguous"

      echo "=== ${TASK} / ${MODE} / ${AMB} ==="
      python scripts/controls/site_frequency_baseline.py \
        --train_pairs "$PAIRS" \
        --task "$TASK" \
        $DEDUP_FLAG $AMB_FLAG \
        --out_csv "results/controls/site_frequency_${TASK}_${MODE}_${AMB}.csv"
      echo
    done
  done
done

echo "[DONE] Baseline profiles written. Next: cluster/slurm/controls/compare_positional_profiles.sh"
