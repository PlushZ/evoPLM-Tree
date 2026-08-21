#!/usr/bin/env python3
# scripts/generate_sequences.py

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import BertLMHeadModel, AutoConfig

from evoplm.tokenizer import ProteinTreeTokenizer

TREE_FEATURE_ORDER_DEFAULT = ["delta_edges", "delta_nt_mutcount", "depth_t", "pt_t"]

def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        m = min(len(a), len(b))
        return sum(a[i] != b[i] for i in range(m)) + abs(len(a) - len(b))
    return sum(x != y for x, y in zip(a, b))

def mutation_sets(src: str, dst: str) -> set:
    m = min(len(src), len(dst))
    muts = set()
    for i in range(m):
        if src[i] != dst[i]:
            muts.add((i, src[i], dst[i]))
    for i in range(m, len(dst)):
        muts.add((i, "-", dst[i]))
    for i in range(m, len(src)):
        muts.add((i, src[i], "-"))
    return muts

def recovered_mutation_metrics(inp: str, true: str, pred: str) -> Dict[str, float]:
    true_muts = mutation_sets(inp, true)
    pred_muts = mutation_sets(inp, pred)

    tp = len(true_muts & pred_muts)
    fp = len(pred_muts - true_muts)
    fn = len(true_muts - pred_muts)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

    return {
        "mut_tp": float(tp),
        "mut_fp": float(fp),
        "mut_fn": float(fn),
        "mut_precision": float(prec),
        "mut_recall": float(rec),
        "mut_f1": float(f1),
        "true_mut_count": float(len(true_muts)),
        "pred_mut_count": float(len(pred_muts)),
        "recovered_mut_count": float(tp),
    }

def build_prompt_ids(
    row: Dict[str, Any],
    tokenizer: ProteinTreeTokenizer,
    seq_in_col: str,
    use_tree_tokens: bool,
    tree_feature_order: List[str],
) -> List[int]:
    ids: List[int] = [int(tokenizer.cls_token_id)]

    if use_tree_tokens:
        feats = {
            "delta_edges": int(row["delta_edges"]) if "delta_edges" in row and pd.notna(row["delta_edges"]) else None,
            "delta_nt_mutcount": int(row["delta_nt_mutcount"]) if "delta_nt_mutcount" in row and pd.notna(row["delta_nt_mutcount"]) else None,
            "depth_t": int(row["depth_t"]) if "depth_t" in row and pd.notna(row["depth_t"]) else None,
            "pt_t": int(row["pt_t"]) if "pt_t" in row and pd.notna(row["pt_t"]) else None,
        }
        for k in tree_feature_order:
            tok = tokenizer.tree_token(k, feats.get(k))
            ids.append(int(tokenizer.token_to_id[tok]))
        ids.append(int(tokenizer.sep_token_id))

    s_in = str(row[seq_in_col])
    for ch in s_in:
        ids.append(int(tokenizer.token_to_id.get(ch, tokenizer.token_to_id["X"])))

    ids.append(int(tokenizer.sep_token_id))
    return ids

def decode_continuation_only(tokenizer: ProteinTreeTokenizer, full_generated_ids: List[int], prompt_len: int) -> str:
    cont_ids = full_generated_ids[prompt_len:]
    return tokenizer.decode(cont_ids, remove_special_tokens=True)

def stratified_sample(
    df: pd.DataFrame,
    n: int,
    seed: int,
    n_bins_edges: int = 4,
    n_bins_muts: int = 4,
) -> pd.DataFrame:
    """
    Quantile-bin stratification over (delta_edges, delta_nt_mutcount).
    Returns up to n rows, balanced across non-empty strata.
    """
    if n <= 0:
        raise ValueError("n must be > 0")

    # Ensure columns exist; if missing, fall back to random
    if "delta_edges" not in df.columns or "delta_nt_mutcount" not in df.columns:
        return df.sample(n=min(n, len(df)), random_state=seed).copy()

    d = df.copy()
    d["delta_edges"] = pd.to_numeric(d["delta_edges"], errors="coerce")
    d["delta_nt_mutcount"] = pd.to_numeric(d["delta_nt_mutcount"], errors="coerce")
    d = d.dropna(subset=["delta_edges", "delta_nt_mutcount"])
    if len(d) == 0:
        return df.sample(n=min(n, len(df)), random_state=seed).copy()

    # quantile bins (duplicates='drop' handles low-variance cols)
    d["bin_e"] = pd.qcut(d["delta_edges"], q=n_bins_edges, duplicates="drop")
    d["bin_m"] = pd.qcut(d["delta_nt_mutcount"], q=n_bins_muts, duplicates="drop")
    d["stratum"] = d["bin_e"].astype(str) + " | " + d["bin_m"].astype(str)

    strata = d["stratum"].value_counts().index.tolist()
    rng = np.random.default_rng(seed)

    # target per stratum
    nonempty = len(strata)
    per = max(1, n // max(nonempty, 1))

    sampled_idx = []
    for s in strata:
        idx = d.index[d["stratum"] == s].to_numpy()
        take = min(per, idx.size)
        if take > 0:
            sampled_idx.extend(rng.choice(idx, size=take, replace=False).tolist())

    # top up if needed
    if len(sampled_idx) < min(n, len(d)):
        remaining = np.setdiff1d(d.index.to_numpy(), np.array(sampled_idx, dtype=int), assume_unique=False)
        need = min(n, len(d)) - len(sampled_idx)
        if need > 0 and remaining.size > 0:
            extra = rng.choice(remaining, size=min(need, remaining.size), replace=False)
            sampled_idx.extend(extra.tolist())

    sampled = d.loc[sampled_idx].copy()
    # If we dropped rows due to NaN, we might be below n. That's fine.
    return sampled.sample(frac=1.0, random_state=seed).copy()

@torch.no_grad()
def generate_and_score(
    model: BertLMHeadModel,
    tokenizer: ProteinTreeTokenizer,
    df: pd.DataFrame,
    device: str,
    seq_in_col: str,
    seq_out_col: str,
    use_tree_tokens: bool,
    tree_feature_order: List[str],
    max_new_tokens_cap: int,
    num_beams: int,
    out_csv: Path,
    out_json: Path,
):
    model.eval()
    model.to(device)

    pad_id = int(tokenizer.pad_token_id)

    rows_out: List[Dict[str, Any]] = []

    n = 0
    exact = 0
    ham_sum = 0.0
    mut_tp = mut_fp = mut_fn = 0.0
    recovered_sum = 0.0
    true_mut_sum = 0.0

    for idx, r in df.iterrows():
        prompt_ids = build_prompt_ids(r, tokenizer, seq_in_col, use_tree_tokens, tree_feature_order)
        true_seq = str(r[seq_out_col])
        inp_seq = str(r[seq_in_col])

        max_new = min(len(true_seq) + 1, max_new_tokens_cap)
        px = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        pattn = (px != pad_id).long()

        gen = model.generate(
            input_ids=px,
            attention_mask=pattn,
            max_new_tokens=int(max_new),
            num_beams=int(num_beams),
            do_sample=False,
            eos_token_id=int(tokenizer.eos_token_id),
            pad_token_id=int(tokenizer.pad_token_id),
        )
        gen_ids = gen[0].detach().cpu().tolist()
        pred_seq = decode_continuation_only(tokenizer, gen_ids, prompt_len=len(prompt_ids))

        n += 1
        if pred_seq == true_seq:
            exact += 1

        hd = hamming(pred_seq, true_seq)
        ham_sum += hd

        m = recovered_mutation_metrics(inp_seq, true_seq, pred_seq)
        mut_tp += m["mut_tp"]
        mut_fp += m["mut_fp"]
        mut_fn += m["mut_fn"]
        recovered_sum += m["recovered_mut_count"]
        true_mut_sum += m["true_mut_count"]

        rows_out.append({
            "pair_index": int(idx),
            "input_seq": inp_seq,
            "true_seq": true_seq,
            "pred_seq": pred_seq,
            "hamming_pred_true": int(hd),
            "true_mut_count": m["true_mut_count"],
            "pred_mut_count": m["pred_mut_count"],
            "recovered_mut_count": m["recovered_mut_count"],
            "mut_precision": m["mut_precision"],
            "mut_recall": m["mut_recall"],
            "mut_f1": m["mut_f1"],
            "delta_edges": float(r.get("delta_edges", np.nan)),
            "delta_nt_mutcount": float(r.get("delta_nt_mutcount", np.nan)),
            "pt_t": float(r.get("pt_t", np.nan)),
            "depth_t": float(r.get("depth_t", np.nan)),
            "t": str(r.get("t", "")),
            "lineage_t": str(r.get("lineage_t", "")),
            "clade_t": str(r.get("clade_t", "")),
            "mat_strain_t": str(r.get("mat_strain_t", "")),
            "mat_strain_t1": str(r.get("mat_strain_t1", "")),
        })

    precision = mut_tp / (mut_tp + mut_fp) if (mut_tp + mut_fp) else 0.0
    recall = mut_tp / (mut_tp + mut_fn) if (mut_tp + mut_fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    recovered_rate = recovered_sum / max(true_mut_sum, 1.0)

    report = {
        "n_examples": int(n),
        "seq_exact_match_rate": float(exact / max(n, 1)),
        "mean_hamming_pred_true": float(ham_sum / max(n, 1)),
        "mutation_precision": float(precision),
        "mutation_recall": float(recall),
        "mutation_f1": float(f1),
        "recovered_mutations_rate": float(recovered_rate),
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"[INFO] Wrote sample generation CSV -> {out_csv}")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    print(f"[INFO] Wrote sample generation JSON -> {out_json}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tokenizer_json", required=True)
    ap.add_argument("--test_pairs", required=True)

    ap.add_argument("--seq_in_col", default="rbd_t")
    ap.add_argument("--seq_out_col", default="rbd_t1")

    ap.add_argument("--use_tree_tokens_in_eval", action="store_true")
    ap.add_argument("--tree_feature_order", default="delta_edges,delta_nt_mutcount,depth_t,pt_t")

    ap.add_argument("--gen_n_samples", type=int, default=2000)
    ap.add_argument("--gen_seed", type=int, default=0)

    ap.add_argument("--max_new_tokens_cap", type=int, default=1600)
    ap.add_argument("--num_beams", type=int, default=1)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)

    ap.add_argument("--n_bins_edges", type=int, default=4)
    ap.add_argument("--n_bins_muts", type=int, default=4)

    args = ap.parse_args()
    set_seed(args.gen_seed)

    tokenizer = ProteinTreeTokenizer.load(args.tokenizer_json)
    config = AutoConfig.from_pretrained(args.model_dir)
    model = BertLMHeadModel.from_pretrained(args.model_dir, config=config)

    model.config.bos_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    df = pd.read_parquet(args.test_pairs)

    sampled = stratified_sample(
        df=df,
        n=int(args.gen_n_samples),
        seed=int(args.gen_seed),
        n_bins_edges=int(args.n_bins_edges),
        n_bins_muts=int(args.n_bins_muts),
    )
    print(f"[INFO] Sampled {len(sampled)} rows (requested {args.gen_n_samples}) from {len(df)} total.")

    tree_order = [s.strip() for s in args.tree_feature_order.split(",") if s.strip()]
    if not tree_order:
        tree_order = TREE_FEATURE_ORDER_DEFAULT

    generate_and_score(
        model=model,
        tokenizer=tokenizer,
        df=sampled,
        device=str(args.device),
        seq_in_col=str(args.seq_in_col),
        seq_out_col=str(args.seq_out_col),
        use_tree_tokens=bool(args.use_tree_tokens_in_eval),
        tree_feature_order=tree_order,
        max_new_tokens_cap=int(args.max_new_tokens_cap),
        num_beams=int(args.num_beams),
        out_csv=Path(args.out_csv),
        out_json=Path(args.out_json),
    )

if __name__ == "__main__":
    main()