#!/usr/bin/env bash
# Link the large, un-versioned inputs (pair datasets, trained checkpoints, UShER
# snapshots) into the layout the launchers under cluster/slurm/ expect.
#
# Nothing is copied: every target created here is a symlink, so the repository
# stays small enough to live on GitHub while the scripts keep using plain
# repo-relative paths such as `data/trajectories/...` and `models/treeaware/...`.
#
# Source root defaults to the sibling evoPLM working tree; override with
#   EVOPLM_DATA_ROOT=/path/to/evoPLM bash cluster/setup_data_links.sh
#
# Safe to re-run: existing symlinks are refreshed, real files are never replaced.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${EVOPLM_DATA_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/evoPLM}"

# Some linked paths are pipeline OUTPUTS, not inputs: steps 10, 11 and 12 write
# to them. Writing through a symlink would overwrite the originals in the
# working tree, so clear those links before re-running preprocessing.
#   bash cluster/setup_data_links.sh --unlink-outputs
OUTPUT_LINKS=(
  "$REPO_ROOT/data/processed/spike_aa/splits/split1_train_first3Omi2laterOmi.with_mat.parquet"
  "$REPO_ROOT/data/processed/spike_aa/splits/split1_test_first3Omi2laterOmi.with_mat.parquet"
  "$REPO_ROOT/data/trajectories/split1_train_first3Omi2laterOmi_pairs__rbd_t_ne_rbd_t1__delta_nt_mutcount_ge_1.parquet"
  "$REPO_ROOT/data/trajectories/split1_test_first3Omi2laterOmi_pairs__rbd_t_ne_rbd_t1__delta_nt_mutcount_ge_1.parquet"
  "$REPO_ROOT/data/trajectories/split1_train_first3Omi2laterOmi_pairs__spike_t_ne_spike_t1__delta_nt_mutcount_ge_1.parquet"
  "$REPO_ROOT/data/trajectories/split1_test_first3Omi2laterOmi_pairs__spike_t_ne_spike_t1__delta_nt_mutcount_ge_1.parquet"
)

if [[ "${1:-}" == "--unlink-outputs" ]]; then
  echo "[INFO] clearing pipeline-output links so steps 10-12 write real files"
  n=0
  for p in "${OUTPUT_LINKS[@]}"; do
    if [[ -L "$p" ]]; then
      rm "$p"; echo "[UNLINK] ${p#$REPO_ROOT/}"; n=$((n + 1))
    elif [[ -e "$p" ]]; then
      echo "[KEEP  ] ${p#$REPO_ROOT/} (a real file, left alone)"
    fi
  done
  echo "[DONE ] $n link(s) removed. Originals remain under $DATA_ROOT."
  exit 0
fi

echo "[INFO] repo root : $REPO_ROOT"
echo "[INFO] data root : $DATA_ROOT"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] data root not found: $DATA_ROOT" >&2
  echo "        Set EVOPLM_DATA_ROOT to the directory holding data/ and models_fulldata/." >&2
  exit 1
fi

n_linked=0
n_missing=0

# link <source> <destination>
# Creates destination as a symlink to source. Refuses to clobber a real file or
# directory, so committed READMEs and configs are never shadowed.
link() {
  local src="$1" dst="$2"

  if [[ ! -e "$src" ]]; then
    echo "[WARN ] missing source, skipped: $src"
    n_missing=$((n_missing + 1))
    return 0
  fi

  if [[ -e "$dst" && ! -L "$dst" ]]; then
    echo "[ERROR] refusing to replace real path: $dst" >&2
    exit 1
  fi

  mkdir -p "$(dirname "$dst")"
  ln -sfn "$src" "$dst"
  echo "[LINK ] ${dst#$REPO_ROOT/}  ->  $src"
  n_linked=$((n_linked + 1))
}

# ---------------------------------------------------------------------------
# 1. Filtered train/test pair tables  (read by train/, generate/, diagnose/)
#    Flat files inside data/trajectories/, alongside the committed README.
# ---------------------------------------------------------------------------
TRAJ_SRC="$DATA_ROOT/data/processed/spike_aa/trajectories/filtered_trivial"
for task in rbd spike; do
  for part in train test; do
    f="split1_${part}_first3Omi2laterOmi_pairs__${task}_t_ne_${task}_t1__delta_nt_mutcount_ge_1.parquet"
    link "$TRAJ_SRC/$f" "$REPO_ROOT/data/trajectories/$f"
  done
done

# ---------------------------------------------------------------------------
# 2. Trained checkpoints  (models_fulldata is the set used in the manuscript:
#    RBD checkpoint-8193 = 87,370 pairs / batch 32 x 3 epochs)
# ---------------------------------------------------------------------------
link "$DATA_ROOT/models_fulldata/prefixlm_treeaware_filtered" "$REPO_ROOT/models/treeaware"
link "$DATA_ROOT/models_fulldata/prefixlm_seqonly_filtered"   "$REPO_ROOT/models/seqonly"

# ---------------------------------------------------------------------------
# 3. Upstream inputs needed for the v2 analyses only (not on Zenodo).
#    - unfiltered pair tables   -> identical-pair variant, site-frequency baseline
#    - *.with_mat.parquet       -> re-running pair construction with a date filter
#    - spike.parquet            -> tree-free pairing control
#    - UShER .pb.gz snapshots   -> re-running pair construction
# ---------------------------------------------------------------------------
link "$DATA_ROOT/data/processed/spike_aa/trajectories" "$REPO_ROOT/data/processed/spike_aa/trajectories_unfiltered"
link "$DATA_ROOT/data/processed/spike_aa/spike.parquet" "$REPO_ROOT/data/processed/spike_aa/spike.parquet"

# Step 08 reads spike_nodup_location_coldate.parquet. Only spike.parquet
# survives in the working tree, and it is that same table (1,886,819 rows,
# already carrying dup_count and rbd_aa), so point step 08's input at it.
# Re-running 08 over it is idempotent apart from stripping the stop codon:
# the RBD slice [318:541] does not reach index 1273 where the '*' sits.
link "$DATA_ROOT/data/processed/spike_aa/spike.parquet" \
     "$REPO_ROOT/data/processed/spike_aa/spike_nodup_location_coldate.parquet"

SPLIT_SRC="$DATA_ROOT/data/processed/spike_aa/splits"
for f in split1_train_first3Omi2laterOmi.with_mat.parquet \
         split1_test_first3Omi2laterOmi.with_mat.parquet; do
  link "$SPLIT_SRC/$f" "$REPO_ROOT/data/processed/spike_aa/splits/$f"
done

link "$DATA_ROOT/assets/usher" "$REPO_ROOT/data/usher/snapshots"

echo
echo "[DONE ] $n_linked link(s) created, $n_missing source(s) missing."
echo "        Re-running is safe and refreshes every link."
