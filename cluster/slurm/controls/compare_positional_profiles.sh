#!/usr/bin/env bash
#SBATCH -J ctl_compare_profiles
#SBATCH -p cpu-single
#SBATCH -N 1 -c 4 --mem=32G
#SBATCH -t 02:00:00
#SBATCH --export=NONE
#SBATCH -o logs/compare_positional_profiles.out
#SBATCH -e logs/compare_positional_profiles.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
[[ -f cluster/env.local.sh ]] && source cluster/env.local.sh
conda activate "${CONDA_ENV:-evoplm}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs results/controls

GEN=results/generation/split1
CTL=results/controls

# Run twice per task. The "raw" pass reproduces the manuscript's counting
# convention; the "noamb" pass drops X / * / - positions, which for full spike
# removes the terminal-stop artefact that registers a predicted mutation in
# 1998 of 2000 pairs. Baselines must match the pass they are compared in.
for TASK in rbd spike; do
  for AMB in raw noamb; do
    AMB_FLAG=""
    [[ "$AMB" == "noamb" ]] && AMB_FLAG="--exclude_ambiguous"

    echo "=== ${TASK} / ${AMB} ==="
    python scripts/controls/compare_positional_profiles.py \
      --treeaware_csv "$GEN/${TASK}_treeaware_sample2000.csv" \
      --seqonly_csv   "$GEN/${TASK}_seqonly_sample2000.csv" \
      --baseline "site-freq (all rows)=$CTL/site_frequency_${TASK}_all_${AMB}.csv" \
      --baseline "site-freq (unique pairs)=$CTL/site_frequency_${TASK}_dedup_${AMB}.csv" \
      --task "$TASK" \
      $AMB_FLAG \
      --n_bootstrap 200 \
      --n_ceiling 100 \
      --seed 0 \
      --out_prefix "$CTL/positional_${TASK}_${AMB}"
    echo
  done
done

echo "[DONE] Comparison tables written to results/controls/"
