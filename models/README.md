# Trained model checkpoints

**Not included** in this repository — checkpoints are large binary files.

**TODO:** add the Zenodo record URL/DOI for the trained checkpoints once published.

Expected layout (split1 only):

| Path | Written by | Read by |
|---|---|---|
| `models/seqonly/rbd/split1/maxT1024_tree0_maskp0.1_span3/final/` | `cluster/slurm/train/train_rbd_seqonly.sbatch` | `evaluate/generate_rbd_seqonly.sbatch`, `evaluate/evaluate_rbd_seqonly.sbatch`, `cluster/slurm/dms/infer_mutation_probabilities.sh` |
| `models/seqonly/spike/split1/maxT4096_tree0_maskp0.1_span3/final/` | `train/train_spike_seqonly.sbatch` | `evaluate/generate_spike_seqonly.sbatch`, `evaluate/evaluate_spike_seqonly.sbatch` |
| `models/treeaware/rbd/split1/maxT1024_tree1_maskp0.1_span3/final/` | `train/train_rbd_treeaware.sbatch` | `evaluate/generate_rbd_treeaware.sbatch`, `evaluate/evaluate_rbd_treeaware.sbatch`, `cluster/slurm/dms/infer_mutation_probabilities.sh` |
| `models/treeaware/spike/split1/maxT4096_tree1_maskp0.1_span3/final/` | `train/train_spike_treeaware.sbatch` | `evaluate/generate_spike_treeaware.sbatch`, `evaluate/evaluate_spike_treeaware.sbatch` |

Each `final/` directory is a Hugging Face `transformers` checkpoint (`BertLMHeadModel`)
saved via `trainer.save_model(...)` in `scripts/train_prefix_lm.py`
(see the top-level [README.md](../README.md), "Training"). The training input,
`data/trajectories/*.parquet`, is also not included — see `data/trajectories/README.md`.

Regenerate by running the training pipeline end to end, or download the checkpoints from
Zenodo once the link above is filled in.
