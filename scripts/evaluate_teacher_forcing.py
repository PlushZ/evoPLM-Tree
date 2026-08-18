#!/usr/bin/env python3
# scripts/evaluate_teacher_forcing.py

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

def pad_batch(seqs: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in seqs)
    x = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    attn = (x != pad_id).long()
    return x, attn

def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=-1)
    ent = -(p * (p.clamp_min(1e-12).log())).sum(dim=-1)
    return ent

def build_unigram_logp_from_targets(target_seqs: pd.Series) -> Dict[str, float]:
    counts = {aa: 0 for aa in list("ACDEFGHIKLMNPQRSTVWY") + ["X", "-"]}
    total = 0
    for s in target_seqs.astype(str).tolist():
        for ch in s:
            counts[ch if ch in counts else "X"] += 1
            total += 1
    probs = np.array([counts[k] for k in counts.keys()], dtype=np.float64)
    probs = probs / max(probs.sum(), 1.0)
    probs = np.clip(probs, 1e-12, 1.0)
    return {aa: float(np.log(p)) for aa, p in zip(counts.keys(), probs)}

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

def build_full_ids_and_labels(
    row: Dict[str, Any],
    tokenizer: ProteinTreeTokenizer,
    seq_in_col: str,
    seq_out_col: str,
    use_tree_tokens: bool,
    tree_feature_order: List[str],
) -> Tuple[List[int], List[int], int]:
    prompt = build_prompt_ids(row, tokenizer, seq_in_col, use_tree_tokens, tree_feature_order)
    t0 = len(prompt)

    ids = list(prompt)
    s_out = str(row[seq_out_col])
    for ch in s_out:
        ids.append(int(tokenizer.token_to_id.get(ch, tokenizer.token_to_id["X"])))
    ids.append(int(tokenizer.eos_token_id))

    labels = [-100] * t0 + ids[t0:]
    return ids, labels, t0

@torch.no_grad()
def eval_tf_only(
    model: BertLMHeadModel,
    tokenizer: ProteinTreeTokenizer,
    df: pd.DataFrame,
    batch_size: int,
    device: str,
    seq_in_col: str,
    seq_out_col: str,
    use_tree_tokens: bool,
    tree_feature_order: List[str],
    unigram_logp: Optional[Dict[str, float]],
    out_json: Path,
    out_csv: Optional[Path],
    metrics_csv: Optional[Path],
    max_rows: int,
):
    model.eval()
    model.to(device)

    if max_rows > 0 and len(df) > max_rows:
        df = df.iloc[:max_rows].copy()

    token_logprob_sum = 0.0
    token_count = 0
    entropy_sum = 0.0
    entropy_count = 0

    unigram_nll_sum = 0.0
    unigram_token_count = 0

    per_ex_rows: List[Dict[str, Any]] = []

    pad_id = int(tokenizer.pad_token_id)

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start: start + batch_size]

        full_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        target_starts: List[int] = []

        for _, r in batch.iterrows():
            full_ids, labels, t0 = build_full_ids_and_labels(
                row=r,
                tokenizer=tokenizer,
                seq_in_col=seq_in_col,
                seq_out_col=seq_out_col,
                use_tree_tokens=use_tree_tokens,
                tree_feature_order=tree_feature_order,
            )
            full_ids_list.append(full_ids)
            labels_list.append(labels)
            target_starts.append(t0)

        x, attn = pad_batch(full_ids_list, pad_id)
        L = x.shape[1]
        y = torch.full((len(labels_list), L), -100, dtype=torch.long)
        for i, lab in enumerate(labels_list):
            y[i, :len(lab)] = torch.tensor(lab, dtype=torch.long)

        x = x.to(device)
        attn = attn.to(device)
        y = y.to(device)

        out = model(input_ids=x, attention_mask=attn, labels=y)
        logits = out.logits

        logp = torch.log_softmax(logits, dim=-1)
        mask = (y != -100)
        y_safe = y.masked_fill(~mask, 0)
        gathered = logp.gather(dim=-1, index=y_safe.unsqueeze(-1)).squeeze(-1)

        token_logprob_sum += float(gathered[mask].sum().cpu())
        token_count += int(mask.sum().cpu())

        ent = softmax_entropy(logits)
        entropy_sum += float(ent[mask].sum().cpu())
        entropy_count += int(mask.sum().cpu())

        # per-example TF metrics
        mask_f = mask.float()
        tok_counts_per_ex = mask_f.sum(dim=1).clamp(min=1.0)
        nll_per_ex = (-(gathered) * mask_f).sum(dim=1) / tok_counts_per_ex
        ent_per_ex = (ent * mask_f).sum(dim=1) / tok_counts_per_ex

        nll_per_ex = nll_per_ex.detach().cpu().numpy()
        ent_per_ex = ent_per_ex.detach().cpu().numpy()

        # unigram baseline
        if unigram_logp is not None:
            for _, r in batch.iterrows():
                true_seq = str(r[seq_out_col])
                for ch in true_seq:
                    unigram_nll_sum += -(unigram_logp.get(ch, unigram_logp["X"]))
                    unigram_token_count += 1

        # optional per-example CSV rows
        if out_csv is not None:
            for i in range(len(batch)):
                rmeta = batch.iloc[i]
                per_ex_rows.append({
                    "pair_index": int(rmeta.name),
                    "input_seq": str(rmeta.get(seq_in_col, "")),
                    "true_seq": str(rmeta.get(seq_out_col, "")),
                    "delta_edges": float(rmeta.get("delta_edges", np.nan)),
                    "delta_nt_mutcount": float(rmeta.get("delta_nt_mutcount", np.nan)),
                    "pt_t": float(rmeta.get("pt_t", np.nan)),
                    "depth_t": float(rmeta.get("depth_t", np.nan)),
                    "t": str(rmeta.get("t", "")),
                    "lineage_t": str(rmeta.get("lineage_t", "")),
                    "clade_t": str(rmeta.get("clade_t", "")),
                    "mat_strain_t": str(rmeta.get("mat_strain_t", "")),
                    "mat_strain_t1": str(rmeta.get("mat_strain_t1", "")),
                    "mean_token_nll_example": float(nll_per_ex[i]),
                    "mean_entropy_example": float(ent_per_ex[i]),
                })

    mean_token_logprob = token_logprob_sum / max(token_count, 1)
    mean_token_nll = -mean_token_logprob
    mean_entropy = entropy_sum / max(entropy_count, 1)

    unigram_mean_nll = (unigram_nll_sum / unigram_token_count) if unigram_token_count else None
    ll_improvement_over_unigram = None
    if unigram_mean_nll is not None:
        ll_improvement_over_unigram = float(unigram_mean_nll - mean_token_nll)

    report = {
        "n_examples": int(len(df)),
        "mean_token_logprob_true": float(mean_token_logprob),
        "mean_token_nll_true": float(mean_token_nll),
        "mean_per_position_entropy": float(mean_entropy),
        "unigram_baseline_mean_nll": float(unigram_mean_nll) if unigram_mean_nll is not None else None,
        "ll_improvement_over_unigram": float(ll_improvement_over_unigram) if ll_improvement_over_unigram is not None else None,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    print(f"[INFO] Wrote JSON -> {out_json}")

    if metrics_csv is not None:
        metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([report]).to_csv(metrics_csv, index=False)
        print(f"[INFO] Wrote metrics CSV -> {metrics_csv}")

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_ex_rows).to_csv(out_csv, index=False)
        print(f"[INFO] Wrote per-example TF CSV -> {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tokenizer_json", required=True)
    ap.add_argument("--test_pairs", required=True)
    ap.add_argument("--train_pairs_for_unigram", default=None)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--metrics_csv", default=None)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seq_in_col", default="rbd_t")
    ap.add_argument("--seq_out_col", default="rbd_t1")

    ap.add_argument("--use_tree_tokens_in_eval", action="store_true")
    ap.add_argument("--tree_feature_order", default="delta_edges,delta_nt_mutcount,depth_t,pt_t")
    ap.add_argument("--max_rows", type=int, default=0, help="0 = all rows, else truncate for quick tests")

    args = ap.parse_args()
    set_seed(0)

    tokenizer = ProteinTreeTokenizer.load(args.tokenizer_json)
    config = AutoConfig.from_pretrained(args.model_dir)
    model = BertLMHeadModel.from_pretrained(args.model_dir, config=config)

    model.config.bos_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    df = pd.read_parquet(args.test_pairs)

    unigram_logp = None
    if args.train_pairs_for_unigram is not None:
        train_df = pd.read_parquet(args.train_pairs_for_unigram)
        unigram_logp = build_unigram_logp_from_targets(train_df[args.seq_out_col])

    tree_order = [s.strip() for s in args.tree_feature_order.split(",") if s.strip()]
    if not tree_order:
        tree_order = TREE_FEATURE_ORDER_DEFAULT

    eval_tf_only(
        model=model,
        tokenizer=tokenizer,
        df=df,
        batch_size=int(args.batch_size),
        device=str(args.device),
        seq_in_col=str(args.seq_in_col),
        seq_out_col=str(args.seq_out_col),
        use_tree_tokens=bool(args.use_tree_tokens_in_eval),
        tree_feature_order=tree_order,
        unigram_logp=unigram_logp,
        out_json=Path(args.out_json),
        out_csv=Path(args.out_csv) if args.out_csv else None,
        metrics_csv=Path(args.metrics_csv) if args.metrics_csv else None,
        max_rows=int(args.max_rows),
    )

if __name__ == "__main__":
    main()