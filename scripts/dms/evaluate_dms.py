#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List

import pandas as pd
from scipy.stats import spearmanr


def analyze_one_model(
    model_name: str,
    probs_csv: str,
    dms_df: pd.DataFrame,
    out_dir: Path,
    k: int,
    threshold: float,
    corr_min_dms: float | None,
    corr_top_prob_quantile: float | None,
    file_suffix: str,
):
    probs_df = pd.read_csv(probs_csv)

    req_probs = {"site", "wt", "mut", "prob"}
    req_dms = {"site", "wt", "mut", "dms_score"}
    if not req_probs.issubset(probs_df.columns):
        missing = sorted(req_probs - set(probs_df.columns))
        raise KeyError(f"[{model_name}] Missing columns in probs CSV: {missing}")
    if not req_dms.issubset(dms_df.columns):
        missing = sorted(req_dms - set(dms_df.columns))
        raise KeyError(f"Missing columns in DMS CSV: {missing}")

    merged = probs_df.merge(dms_df, on=["site", "wt", "mut"], how="inner")

    # Safety: exclude WT-equivalent rows from all metrics.
    merged = merged.loc[merged["mut"] != merged["wt"]].copy()
    merged["prob"] = pd.to_numeric(merged["prob"], errors="coerce")
    merged["dms_score"] = pd.to_numeric(merged["dms_score"], errors="coerce")
    merged = merged.dropna(subset=["prob", "dms_score"]).copy()
    merged["tolerated"] = (merged["dms_score"] >= threshold).astype(int)

    if merged.empty:
        raise ValueError(f"[{model_name}] No overlapping non-WT mutations after merge.")

    corr_df = merged.copy()
    if corr_min_dms is not None:
        corr_df = corr_df.loc[corr_df["dms_score"] > float(corr_min_dms)].copy()
    if corr_top_prob_quantile is not None:
        q = float(corr_top_prob_quantile)
        cutoff = corr_df["prob"].quantile(q)
        corr_df = corr_df.loc[corr_df["prob"] >= cutoff].copy()

    if len(corr_df) >= 2:
        corr, pval = spearmanr(corr_df["prob"], corr_df["dms_score"])
    else:
        corr, pval = float("nan"), float("nan")

    topk = (
        merged.sort_values(["site", "prob"], ascending=[True, False])
        .groupby("site", as_index=False)
        .head(k)
        .copy()
    )

    frac_tol_topk = float((topk["dms_score"] >= threshold).mean())
    frac_tol_bg = float(merged["tolerated"].mean())
    enrich = float(frac_tol_topk / frac_tol_bg) if frac_tol_bg > 0 else float("nan")
    total_prob = merged["prob"].sum()
    weighted_tol = float((merged["prob"] * merged["tolerated"]).sum() / total_prob)
    weighted_enrich = float(weighted_tol / frac_tol_bg) if frac_tol_bg > 0 else float("nan")

    merged_out = out_dir / f"{model_name}_merged_probs_dms{file_suffix}.csv"
    topk_out = out_dir / f"{model_name}_top{k}_mutations{file_suffix}.csv"
    merged.to_csv(merged_out, index=False)
    topk.to_csv(topk_out, index=False)

    result = {
        "model": model_name,
        "n_merged_nonwt": int(len(merged)),
        "n_corr_nonwt": int(len(corr_df)),
        "spearman_prob_vs_dms": float(corr),
        "spearman_pvalue": float(pval),
        "corr_min_dms": (float(corr_min_dms) if corr_min_dms is not None else ""),
        "corr_top_prob_quantile": (float(corr_top_prob_quantile) if corr_top_prob_quantile is not None else ""),
        "k": int(k),
        "topk_n": int(len(topk)),
        "topk_mean_dms": float(topk["dms_score"].mean()),
        "topk_frac_tolerated": frac_tol_topk,
        "background_frac_tolerated": frac_tol_bg,
        "topk_enrichment_vs_background": enrich,
        "weighted_tolerance": weighted_tol,
        "weighted_enrichment_vs_background": weighted_enrich,
        "tolerance_threshold": float(threshold),
        "merged_csv": str(merged_out),
        "topk_csv": str(topk_out),
    }
    return result


def dataset_id_from_path(p: Path) -> str:
    stem = p.stem
    if stem.endswith("_table"):
        return stem[:-6]
    return stem


def collect_dms_csvs(dms_scores_dir: Path) -> List[Path]:
    files = sorted([p for p in dms_scores_dir.glob("*.csv") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No CSV files found in {dms_scores_dir}")
    return files


def fmt_for_filename(x: float) -> str:
    s = str(x).replace("-", "m").replace(".", "p")
    return s


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--dms_scores_dir", default="results/dms/dms_scores")
    ap.add_argument("--seqonly_probs_csv", default="results/dms/rbd_split1_seqonly_mutation_probs.csv")
    ap.add_argument("--treeaware_probs_csv", default="results/dms/rbd_split1_treeaware_mutation_probs.csv")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Inclusive tolerance threshold: tolerated when dms_score >= threshold.",
    )
    ap.add_argument("--corr_min_dms", type=float, default=None, help="Optional correlation-only filter: keep rows with dms_score > this.")
    ap.add_argument("--corr_top_prob_quantile", type=float, default=None, help="Optional correlation-only filter: keep rows with prob >= quantile cutoff (e.g. 0.8 = top 20%).")
    ap.add_argument("--out_root", default="results/dms")
    args = ap.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be >= 1")
    if args.corr_top_prob_quantile is not None and not (0.0 < args.corr_top_prob_quantile < 1.0):
        raise ValueError("--corr_top_prob_quantile must be between 0 and 1 (exclusive).")

    suffix_parts = []
    if args.corr_min_dms is not None:
        suffix_parts.append(f"corrDmsGt_{fmt_for_filename(args.corr_min_dms)}")
    if args.corr_top_prob_quantile is not None:
        suffix_parts.append(f"corrTopQ_{fmt_for_filename(args.corr_top_prob_quantile)}")
    file_suffix = "__" + "__".join(suffix_parts) if suffix_parts else ""

    dms_scores_dir = Path(args.dms_scores_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    dms_csvs = collect_dms_csvs(dms_scores_dir)
    all_rows = []

    for dms_csv in dms_csvs:
        dataset_id = dataset_id_from_path(dms_csv)
        out_dir = out_root / dataset_id
        out_dir.mkdir(parents=True, exist_ok=True)

        dms_df = pd.read_csv(dms_csv)

        results = []
        results.append(
            analyze_one_model(
                model_name="seqonly",
                probs_csv=args.seqonly_probs_csv,
                dms_df=dms_df,
                out_dir=out_dir,
                k=args.k,
                threshold=args.threshold,
                corr_min_dms=args.corr_min_dms,
                corr_top_prob_quantile=args.corr_top_prob_quantile,
                file_suffix=file_suffix,
            )
        )
        results.append(
            analyze_one_model(
                model_name="treeaware",
                probs_csv=args.treeaware_probs_csv,
                dms_df=dms_df,
                out_dir=out_dir,
                k=args.k,
                threshold=args.threshold,
                corr_min_dms=args.corr_min_dms,
                corr_top_prob_quantile=args.corr_top_prob_quantile,
                file_suffix=file_suffix,
            )
        )
        summary_df = pd.DataFrame(results)
        summary_df.insert(0, "dms_dataset", dataset_id)
        summary_df.insert(1, "dms_csv", str(dms_csv))

        summary_csv = out_dir / f"metrics_summary{file_suffix}.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"[INFO] wrote summary: {summary_csv}")
        print(summary_df.to_string(index=False))

        all_rows.append(summary_df)

    combined = pd.concat(all_rows, ignore_index=True)
    combined_csv = out_root / f"metrics_summary_all_datasets{file_suffix}.csv"
    combined.to_csv(combined_csv, index=False)
    print(f"[INFO] wrote combined summary: {combined_csv}")


if __name__ == "__main__":
    main()
