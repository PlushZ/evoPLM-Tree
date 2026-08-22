# Trained model checkpoints
Can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.22045909).

**Not included** in this repository.

Expected layout:

| Path | Written by | Read by |
|---|---|---|
| `models/seqonly/rbd/split1/maxT1024_tree0_maskp0.1_span3/final/` | `cluster/slurm/train/train_rbd_seqonly.sh` | `generate/generate_rbd_seqonly.sh`, `diagnose/diagnose_conditioning_rbd_seqonly.sh`, `cluster/slurm/dms/infer_mutation_probabilities.sh` |
| `models/seqonly/spike/split1/maxT4096_tree0_maskp0.1_span3/final/` | `train/train_spike_seqonly.sh` | `generate/generate_spike_seqonly.sh`, `diagnose/diagnose_conditioning_spike_seqonly.sh` |
| `models/treeaware/rbd/split1/maxT1024_tree1_maskp0.1_span3/final/` | `train/train_rbd_treeaware.sh` | `generate/generate_rbd_treeaware.sh`, `diagnose/diagnose_conditioning_rbd_treeaware.sh`, `cluster/slurm/dms/infer_mutation_probabilities.sh` |
| `models/treeaware/spike/split1/maxT4096_tree1_maskp0.1_span3/final/` | `train/train_spike_treeaware.sh` | `generate/generate_spike_treeaware.sh`, `diagnose/diagnose_conditioning_spike_treeaware.sh` |

Each `final/` directory is a Hugging Face `transformers` checkpoint (`BertLMHeadModel`)
saved via `trainer.save_model(...)` in `scripts/train_prefix_lm.py`
(see the top-level [README.md](../README.md), "Training"). The training input,
`data/trajectories/*.parquet` — see `data/trajectories/README.md`.

Regenerate by running the training pipeline end to end, or download from Zenodo (vial link above).
