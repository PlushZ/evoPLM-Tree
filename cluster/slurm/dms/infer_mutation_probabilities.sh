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
conda activate evoplm-paper
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p results/dms

# BA.2 RBD ancestor nucleotide sequence used as the DMS-assay background for both models.
ANCESTOR_NT="AACATCACCAACTTGTGTCCATTCGACGAAGTTTTCAATGCTACTAGATTCGCTTCTGTTTACGCTTGGAATAGAAAGAGAATCTCTAACTGCGTTGCTGACTATTCTGTCTTGTACAATTTTGCTCCATTCTTCGCTTTCAAGTGCTATGGTGTTTCTCCAACTAAGTTGAACGATTTGTGTTTCACCAACGTTTACGCCGATTCCTTTGTTATTAGAGGTAACGAAGTCTCCCAAATTGCTCCAGGTCAAACTGGTAATATTGCCGATTACAATTACAAGTTGCCAGATGATTTCACCGGTTGTGTTATTGCTTGGAACTCTAACAAGTTGGATTCTAAGGTTGGTGGCAACTACAATTACTTGTACAGGTTGTTCCGTAAGTCCAACTTGAAGCCATTCGAAAGAGATATTTCCACCGAAATCTATCAAGCTGGTAACAAGCCATGTAATGGTGTTGCTGGTTTTAACTGTTACTTCCCATTGAGATCTTACGGTTTCAGACCAACTTATGGTGTTGGTCATCAACCATACAGAGTTGTTGTTTTGTCTTTCGAGTTGTTGCATGCTCCAGCTACTGTTTGTGGTCCAAAGAAATCTACT"

python scripts/dms/infer_mutation_probabilities.py \
  --model_dir models/treeaware/rbd/split1/maxT1024_tree1_maskp0.1_span3/final \
  --tokenizer_json configs/protein_tree_tokenizer.json \
  --ancestor_seq "$ANCESTOR_NT" \
  --seq_type nt \
  --use_tree_tokens \
  --out_csv results/dms/rbd_split1_treeaware_mutation_probs.csv

python scripts/dms/infer_mutation_probabilities.py \
  --model_dir models/seqonly/rbd/split1/maxT1024_tree0_maskp0.1_span3/final \
  --tokenizer_json configs/protein_tree_tokenizer.json \
  --ancestor_seq "$ANCESTOR_NT" \
  --seq_type nt \
  --out_csv results/dms/rbd_split1_seqonly_mutation_probs.csv
