#!/usr/bin/env bash
#SBATCH -J dms_infer_probs
#SBATCH -p cpu-single
#SBATCH -N 1 -c 2 --mem=32G
#SBATCH -t 01:00:00
#SBATCH --export=NONE
#SBATCH -o logs/dms_infer_mutation_probabilities.out
#SBATCH -e logs/dms_infer_mutation_probabilities.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
[[ -f cluster/env.local.sh ]] && source cluster/env.local.sh
conda activate "${CONDA_ENV:-evoplm}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p results/dms results/dms/sensitivity

# ---------------------------------------------------------------------------
# Ancestor background.
#
# The RBD models are trained on spike residues 319-541 (223 aa) with learned
# absolute positional embeddings, so inference must use that same window: a
# shorter input shifts every residue out of its trained frame.
#
# The Starr et al. BA.2 assay construct covers spike 331-531 (201 aa), which is
# the window the DMS tables are numbered in. The sequence below is the modal
# BA.2 RBD among held-out BA.2 source sequences
# (results/generation/split1/rbd_treeaware_sample2000.csv, 35/85 rows); its
# residues 13-213 reproduce all 201 assay wild-type residues exactly.
#
# Mapping used below:  assay_site = model_position - 12  (--site_offset -12)
# ---------------------------------------------------------------------------
ANCESTOR_AA="RVQPTESIVRFPNITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"

# Tree-feature values: medians over the held-out BA.2 source rows
# (delta_edges 10 [IQR 6-10], delta_nt_mutcount 6 [5-8], depth_t 28 [27-30],
#  pt_t 83 [82-84]).
DELTA_EDGES=10
DELTA_NT=6
DEPTH_T=28
PT_T=83

TREEAWARE_MODEL=models/treeaware/rbd/split1/maxT1024_tree1_maskp0.1_span3/final
SEQONLY_MODEL=models/seqonly/rbd/split1/maxT1024_tree0_maskp0.1_span3/final

COMMON=(
  --tokenizer_json configs/protein_tree_tokenizer.json
  --ancestor_seq "$ANCESTOR_AA"
  --seq_type aa
  --expected_len 223
  --site_offset -12
)

echo "=== tree-aware (tree tokens populated) ==="
python scripts/dms/infer_mutation_probabilities.py \
  --model_dir "$TREEAWARE_MODEL" \
  "${COMMON[@]}" \
  --use_tree_tokens \
  --delta_edges "$DELTA_EDGES" \
  --delta_nt_mutcount "$DELTA_NT" \
  --depth_t "$DEPTH_T" \
  --pt_t "$PT_T" \
  --out_csv results/dms/rbd_split1_treeaware_mutation_probs.csv

echo "=== sequence-only baseline ==="
python scripts/dms/infer_mutation_probabilities.py \
  --model_dir "$SEQONLY_MODEL" \
  "${COMMON[@]}" \
  --out_csv results/dms/rbd_split1_seqonly_mutation_probs.csv

# ---------------------------------------------------------------------------
# Sensitivity to the conditioned evolutionary step size.
#
# If the tree tokens are genuinely used at inference, varying delta_nt_mutcount
# across the observed BA.2 range should change the probability table. Identical
# outputs would indicate the conditioning is being ignored.
# ---------------------------------------------------------------------------
echo "=== tree-aware: delta_nt_mutcount sweep ==="
for DNT in 1 6 20; do
  python scripts/dms/infer_mutation_probabilities.py \
    --model_dir "$TREEAWARE_MODEL" \
    "${COMMON[@]}" \
    --use_tree_tokens \
    --delta_edges "$DELTA_EDGES" \
    --delta_nt_mutcount "$DNT" \
    --depth_t "$DEPTH_T" \
    --pt_t "$PT_T" \
    --out_csv "results/dms/sensitivity/rbd_split1_treeaware_dnt${DNT}_mutation_probs.csv"
done

echo
echo "[DONE] Probability tables written. Next: cluster/slurm/dms/evaluate_dms.sh"
