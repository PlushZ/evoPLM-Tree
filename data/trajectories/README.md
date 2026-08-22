# Ancestor-descendant sequence pairs (train/test datasets)
Can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.22045909).

**Not included** in this repository — these parquet files are the final training/test
datasets for `scripts/train_prefix_lm.py`.

Expected contents:

| File | Written by | Read by |
|---|---|---|
| `split1_train_first3Omi2laterOmi_pairs__rbd_t_ne_rbd_t1__delta_nt_mutcount_ge_1.parquet` | `cluster/slurm/preprocessing/12_filter_trivial_trajectories.sh` | `train/train_rbd_{seqonly,treeaware}.sh`, `generate/generate_rbd_{seqonly,treeaware}.sh`, `diagnose/diagnose_conditioning_rbd_{seqonly,treeaware}.sh` |
| `split1_test_first3Omi2laterOmi_pairs__rbd_t_ne_rbd_t1__delta_nt_mutcount_ge_1.parquet` | same | same |
| `split1_train_first3Omi2laterOmi_pairs__spike_t_ne_spike_t1__delta_nt_mutcount_ge_1.parquet` | same | `train/train_spike_{seqonly,treeaware}.sh`, `generate/generate_spike_{seqonly,treeaware}.sh`, `diagnose/diagnose_conditioning_spike_{seqonly,treeaware}.sh` |
| `split1_test_first3Omi2laterOmi_pairs__spike_t_ne_spike_t1__delta_nt_mutcount_ge_1.parquet` | same | same |

`12_filter_trivial_trajectories.sh` is the last preprocessing step: it filters trivial pairs
(no sequence change / no nt substitutions) out of step 11's output
(`data/processed/spike_aa/trajectories/`, see `data/processed/README.md`), split per task.

Regenerate via the full preprocessing pipeline (`cluster/slurm/preprocessing/01`
through `12`, see the top-level [README.md](../../README.md)), or download them from
Zenodo (via link above).