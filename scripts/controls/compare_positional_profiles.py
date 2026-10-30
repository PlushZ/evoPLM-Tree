#!/usr/bin/env python3
"""
Score every positional predictor against the same observed held-out profile.

Predictors compared:
  - evoPLM-Tree generated descendants      (tree-aware sample CSV, pred_seq)
  - sequence-only generated descendants    (sequence-only sample CSV, pred_seq)
  - one or more site-frequency baselines   (--baseline LABEL=PATH)

Both sample CSVs must cover the same held-out pairs, which is asserted, so the
observed profile is computed once and shared.

Two things the manuscript currently lacks are produced here:

  error margins  Spearman rho is bootstrapped by resampling held-out pairs.
                 Model profiles are recomputed inside each draw, since they are
                 derived from the same pairs as the observed profile; baseline
                 profiles come from the training split and stay fixed.

  noise ceiling  Repeated random half-splits of the held-out pairs are
                 correlated against each other, giving the highest rho any
                 predictor could reach at this sample size. The Spearman-Brown
                 corrected value rescales that reliability to the full sample.
                 Without it a rho of 0.82 cannot be read as good or poor.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evoplm.profiles import mutation_matrices, profile_from_matrices


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--treeaware_csv", required=True, help="*_treeaware_sample2000.csv")
    ap.add_argument("--seqonly_csv", required=True, help="*_seqonly_sample2000.csv")
    ap.add_argument(
        "--baseline",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Site-frequency baseline CSV (position,freq). Repeatable.",
    )
    ap.add_argument("--task", required=True, choices=["rbd", "spike"])
    ap.add_argument(
        "--exclude_ambiguous",
        action="store_true",
        help="Drop positions carrying X, * or -. Must match how the baselines were built.",
    )
    ap.add_argument("--n_bootstrap", type=int, default=200)
    ap.add_argument("--n_ceiling", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_prefix", required=True, help="e.g. results/controls/positional_rbd")
    return ap.parse_args()


def load_sample(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"pair_index", "input_seq", "true_seq", "pred_seq"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{path} has 0 rows")
    return df


def load_baselines(specs: List[str], n_ref: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--baseline must be LABEL=PATH, got: {spec}")
        label, path = spec.split("=", 1)
        df = pd.read_csv(path)
        for col in ("position", "freq"):
            if col not in df.columns:
                raise KeyError(f"{path} missing column '{col}'")
        arr = np.full(n_ref, np.nan)
        pos = df["position"].to_numpy(dtype=int)
        if pos.max(initial=-1) >= n_ref:
            raise ValueError(
                f"{path} covers positions up to {pos.max()} but n_ref is {n_ref}; "
                "the baseline and the held-out sample use different reference lengths."
            )
        arr[pos] = df["freq"].to_numpy(dtype=float)
        out[label] = arr
    return out


def bootstrap_rho(
    obs_mut: np.ndarray,
    obs_cmp: np.ndarray,
    pred_mats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    baselines: Dict[str, np.ndarray],
    valid: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> Dict[str, Tuple[float, float]]:
    """Percentile CIs for each predictor, resampling held-out pairs."""
    n_pairs = obs_mut.shape[0]
    draws: Dict[str, List[float]] = {k: [] for k in list(pred_mats) + list(baselines)}

    for _ in range(n_boot):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        obs_b = profile_from_matrices(obs_mut[idx], obs_cmp[idx])

        for label, (mut, cmp_) in pred_mats.items():
            pred_b = profile_from_matrices(mut[idx], cmp_[idx])
            ok = valid & np.isfinite(obs_b) & np.isfinite(pred_b)
            if ok.sum() >= 3:
                draws[label].append(spearmanr(pred_b[ok], obs_b[ok]).statistic)

        for label, base in baselines.items():
            ok = valid & np.isfinite(obs_b) & np.isfinite(base)
            if ok.sum() >= 3:
                draws[label].append(spearmanr(base[ok], obs_b[ok]).statistic)

    return {
        k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        if len(v) >= 10 else (float("nan"), float("nan"))
        for k, v in draws.items()
    }


def noise_ceiling(
    obs_mut: np.ndarray,
    obs_cmp: np.ndarray,
    valid: np.ndarray,
    n_iter: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Split-half reliability, and its Spearman-Brown value at full sample size."""
    n_pairs = obs_mut.shape[0]
    half = n_pairs // 2
    rhos: List[float] = []

    for _ in range(n_iter):
        perm = rng.permutation(n_pairs)
        a, b = perm[:half], perm[half: 2 * half]
        fa = profile_from_matrices(obs_mut[a], obs_cmp[a])
        fb = profile_from_matrices(obs_mut[b], obs_cmp[b])
        ok = valid & np.isfinite(fa) & np.isfinite(fb)
        if ok.sum() >= 3:
            rhos.append(spearmanr(fa[ok], fb[ok]).statistic)

    if not rhos:
        return float("nan"), float("nan")
    r_half = float(np.mean(rhos))
    r_full = 2.0 * r_half / (1.0 + r_half) if r_half > -1 else float("nan")
    return r_half, r_full


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    tree_df = load_sample(args.treeaware_csv)
    seq_df = load_sample(args.seqonly_csv)

    if not tree_df["pair_index"].equals(seq_df["pair_index"]):
        raise ValueError("The two sample CSVs cover different held-out pairs.")
    if not tree_df["true_seq"].equals(seq_df["true_seq"]):
        raise ValueError("The two sample CSVs disagree on the observed descendants.")

    inputs = tree_df["input_seq"].astype(str).tolist()
    truths = tree_df["true_seq"].astype(str).tolist()
    n_ref = len(pd.Series(inputs).mode().iloc[0])

    obs_mut, obs_cmp = mutation_matrices(inputs, truths, n_ref, args.exclude_ambiguous)
    obs_freq = profile_from_matrices(obs_mut, obs_cmp)

    pred_mats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for label, df in (("evoPLM-Tree", tree_df), ("sequence-only", seq_df)):
        pred_mats[label] = mutation_matrices(
            inputs, df["pred_seq"].astype(str).tolist(), n_ref, args.exclude_ambiguous
        )

    baselines = load_baselines(args.baseline, n_ref)

    # Positions usable by every predictor, so all rho values are over one set.
    valid = np.isfinite(obs_freq)
    for mut, cmp_ in pred_mats.values():
        valid &= np.isfinite(profile_from_matrices(mut, cmp_))
    for base in baselines.values():
        valid &= np.isfinite(base)

    n_valid = int(valid.sum())
    if n_valid < 3:
        raise ValueError(f"Only {n_valid} positions usable by all predictors.")

    point: Dict[str, Tuple[float, float]] = {}
    profiles: Dict[str, np.ndarray] = {}
    for label, (mut, cmp_) in pred_mats.items():
        freq = profile_from_matrices(mut, cmp_)
        profiles[label] = freq
        res = spearmanr(freq[valid], obs_freq[valid])
        point[label] = (float(res.statistic), float(res.pvalue))
    for label, base in baselines.items():
        profiles[label] = base
        res = spearmanr(base[valid], obs_freq[valid])
        point[label] = (float(res.statistic), float(res.pvalue))

    ci = bootstrap_rho(obs_mut, obs_cmp, pred_mats, baselines, valid, args.n_bootstrap, rng)
    r_half, r_full = noise_ceiling(obs_mut, obs_cmp, valid, args.n_ceiling, rng)

    rows = []
    for label, (rho, pval) in point.items():
        lo, hi = ci[label]
        rows.append(
            {
                "task": args.task,
                "predictor": label,
                "spearman_rho": rho,
                "ci95_low": lo,
                "ci95_high": hi,
                "pvalue": pval,
                "n_positions": n_valid,
                "n_pairs": len(inputs),
                "mean_predicted_freq": float(np.nanmean(profiles[label][valid])),
                "mean_observed_freq": float(np.nanmean(obs_freq[valid])),
                "exclude_ambiguous": int(args.exclude_ambiguous),
            }
        )
    rows.append(
        {
            "task": args.task,
            "predictor": "noise ceiling (split-half)",
            "spearman_rho": r_half,
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "pvalue": float("nan"),
            "n_positions": n_valid,
            "n_pairs": len(inputs),
            "mean_predicted_freq": float("nan"),
            "mean_observed_freq": float(np.nanmean(obs_freq[valid])),
            "exclude_ambiguous": int(args.exclude_ambiguous),
        }
    )
    rows.append({**rows[-1], "predictor": "noise ceiling (Spearman-Brown)", "spearman_rho": r_full})

    metrics = pd.DataFrame(rows).sort_values("spearman_rho", ascending=False)

    tidy = pd.DataFrame({"position": np.arange(n_ref), "observed": obs_freq, "usable": valid})
    for label, freq in profiles.items():
        tidy[label] = freq
    if args.task == "rbd":
        tidy["spike_site"] = tidy["position"] + 319

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = out_prefix.with_name(out_prefix.name + "_metrics.csv")
    profiles_path = out_prefix.with_name(out_prefix.name + "_profiles.csv")
    metrics.to_csv(metrics_path, index=False)
    tidy.to_csv(profiles_path, index=False)

    print(f"[INFO] task={args.task} n_ref={n_ref} usable positions={n_valid} pairs={len(inputs)}")
    print(f"[INFO] exclude_ambiguous={bool(args.exclude_ambiguous)} "
          f"bootstrap={args.n_bootstrap} ceiling_iters={args.n_ceiling}")
    print()
    print(f"{'predictor':34s} {'rho':>7s} {'95% CI':>17s}  {'mean pred freq':>14s}")
    print("-" * 78)
    for _, r in metrics.iterrows():
        ci_txt = "" if not np.isfinite(r.ci95_low) else f"[{r.ci95_low:.3f}, {r.ci95_high:.3f}]"
        mpf = "" if not np.isfinite(r.mean_predicted_freq) else f"{r.mean_predicted_freq:.5f}"
        print(f"{r.predictor:34s} {r.spearman_rho:7.3f} {ci_txt:>17s}  {mpf:>14s}")
    print(f"\n[INFO] observed mean frequency: {float(np.nanmean(obs_freq[valid])):.5f}")
    print(f"[INFO] Saved -> {metrics_path}")
    print(f"[INFO] Saved -> {profiles_path}")


if __name__ == "__main__":
    main()
