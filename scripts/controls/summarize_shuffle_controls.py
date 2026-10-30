#!/usr/bin/env python3
"""
Collect the per-example losses from the shuffle controls into one table.

For each (task, model, variant) it reports the mean descendant-token loss, the
ratio to that model's unperturbed reference, and a paired test against the
reference over the same rows. The pairing matters: every variant is built from
one row subsample, so the same descendant appears in all of them and a paired
comparison removes between-pair difficulty from the contrast.

Also bins the source-substitution variants by the Hamming distance actually
achieved between the true and substituted ancestor, which is what distinguishes
"the model ignores its background" from "these backgrounds are interchangeable".
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REFERENCE = "reference"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--loss_dir", default="results/controls/shuffle")
    ap.add_argument("--out_prefix", default="results/controls/shuffle_controls")
    return ap.parse_args()


def load_all(loss_dir: Path) -> pd.DataFrame:
    files = sorted(loss_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No per-example loss CSVs in {loss_dir}")
    frames = []
    for path in files:
        # <task>_<model>_<label>.csv
        stem = path.stem
        task, model, label = stem.split("_", 2)
        df = pd.read_csv(path)
        df["task"] = task
        df["model"] = model
        df["label"] = label
        df["row_id"] = np.arange(len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    data = load_all(Path(args.loss_dir))

    rows: List[dict] = []
    for (task, model), grp in data.groupby(["task", "model"]):
        ref = grp[grp.label == REFERENCE]
        if ref.empty:
            raise ValueError(f"No reference run for {task}/{model}")
        ref_idx = ref.set_index("row_id")["nll_per_token"]
        ref_mean = float(ref_idx.mean())

        for label, var in grp.groupby("label"):
            v = var.set_index("row_id")["nll_per_token"]
            common = ref_idx.index.intersection(v.index)
            delta = (v.loc[common] - ref_idx.loc[common]).to_numpy()

            if label == REFERENCE or np.allclose(delta, 0):
                pval = float("nan")
            else:
                pval = float(wilcoxon(delta).pvalue)

            rows.append(
                {
                    "task": task,
                    "model": model,
                    "variant": label,
                    "n": int(len(common)),
                    "mean_nll": float(v.mean()),
                    "ref_nll": ref_mean,
                    "delta_nll": float(delta.mean()),
                    "loss_ratio": float(v.mean() / ref_mean) if ref_mean else float("nan"),
                    "frac_rows_worse": float((delta > 0).mean()),
                    "wilcoxon_p": pval,
                    "mean_perturb_distance": (
                        float(var["perturb_distance"].mean())
                        if "perturb_distance" in var.columns else float("nan")
                    ),
                }
            )

    summary = pd.DataFrame(rows).sort_values(["task", "model", "loss_ratio"])

    # Loss increase as a function of how far the substituted ancestor actually sits.
    bins = [-0.5, 0.5, 1.5, 3.5, 7.5, 15.5, np.inf]
    names = ["0", "1", "2-3", "4-7", "8-15", "16+"]
    curve_rows: List[dict] = []
    for (task, model), grp in data.groupby(["task", "model"]):
        ref_idx = grp[grp.label == REFERENCE].set_index("row_id")["nll_per_token"]
        for label, var in grp.groupby("label"):
            if not label.startswith("source_") or "perturb_distance" not in var.columns:
                continue
            v = var.set_index("row_id")
            common = ref_idx.index.intersection(v.index)
            d = v.loc[common]
            delta = d["nll_per_token"] - ref_idx.loc[common]
            band = pd.cut(d["perturb_distance"], bins=bins, labels=names)
            for b, sub in delta.groupby(band, observed=True):
                curve_rows.append(
                    {
                        "task": task,
                        "model": model,
                        "variant": label,
                        "distance_band": str(b),
                        "n": int(len(sub)),
                        "mean_delta_nll": float(sub.mean()),
                    }
                )
    curve = pd.DataFrame(curve_rows)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_prefix.with_name(out_prefix.name + "_summary.csv"), index=False)
    if not curve.empty:
        curve.to_csv(out_prefix.with_name(out_prefix.name + "_distance_curve.csv"), index=False)

    for (task, model), grp in summary.groupby(["task", "model"]):
        print(f"\n=== {task} / {model} ===")
        print(f"{'variant':26s} {'mean NLL':>9s} {'ratio':>7s} {'ΔNLL':>8s} "
              f"{'% worse':>8s} {'dist':>7s} {'p':>10s}")
        print("-" * 82)
        for _, r in grp.iterrows():
            p = "" if np.isnan(r.wilcoxon_p) else f"{r.wilcoxon_p:.2e}"
            dist = "" if np.isnan(r.mean_perturb_distance) else f"{r.mean_perturb_distance:.1f}"
            print(f"{r.variant:26s} {r.mean_nll:9.4f} {r.loss_ratio:7.3f} {r.delta_nll:8.4f} "
                  f"{100*r.frac_rows_worse:7.1f}% {dist:>7s} {p:>10s}")

    if not curve.empty:
        print("\n=== loss increase vs achieved ancestor substitution distance ===")
        for (task, model), grp in curve.groupby(["task", "model"]):
            print(f"\n{task} / {model}")
            piv = grp.pivot_table(index="distance_band", columns="variant",
                                  values="mean_delta_nll", observed=True)
            cnt = grp.pivot_table(index="distance_band", columns="variant",
                                  values="n", observed=True)
            print(piv.round(4).to_string())
            print("n per cell:")
            print(cnt.to_string())

    print(f"\n[INFO] Saved -> {out_prefix.with_name(out_prefix.name + '_summary.csv')}")


if __name__ == "__main__":
    main()
