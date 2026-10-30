#!/usr/bin/env python3
"""
Per-example teacher-forced cross-entropy over the descendant sequence.

Serialises each pair exactly as training did,

    [CLS] (tree tokens [SEP])? seq_t [SEP] seq_t1 [EOS]

supervises only seq_t1 and the terminal [EOS], and reports the mean negative
log-likelihood per descendant token for every row.

Per-example output is the point: the shuffle controls need loss binned by how
far the substituted ancestor actually sits from the true one, which a single
corpus-level number cannot give. It is also token-weighted, where averaging
per-batch losses would weight short and long batches equally.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import BertLMHeadModel

from evoplm.tokenizer import ProteinTreeTokenizer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tokenizer_json", default="configs/protein_tree_tokenizer.json")
    ap.add_argument("--pairs_parquet", required=True)
    ap.add_argument("--task", required=True, choices=["rbd", "spike"])
    ap.add_argument("--use_tree_tokens", action="store_true")
    ap.add_argument(
        "--tree_feature_order",
        default="delta_edges,delta_nt_mutcount,depth_t,pt_t",
    )
    ap.add_argument("--label", required=True, help="Variant name recorded in the output.")
    ap.add_argument("--max_tokens", type=int, default=0, help="0 = model's max_position_embeddings.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_csv", required=True)
    return ap.parse_args()


def aa_ids(tok: ProteinTreeTokenizer, seq: str) -> List[int]:
    x = int(tok.token_to_id["X"])
    return [int(tok.token_to_id.get(c, x)) for c in seq]


def build_stream(
    row: pd.Series,
    tok: ProteinTreeTokenizer,
    src_col: str,
    dst_col: str,
    use_tree: bool,
    tree_order: List[str],
) -> Tuple[List[int], int]:
    ids: List[int] = [int(tok.cls_token_id)]
    if use_tree:
        for key in tree_order:
            val = row.get(key)
            val = None if pd.isna(val) else int(val)
            ids.append(int(tok.token_to_id[tok.tree_token(key, val)]))
        ids.append(int(tok.sep_token_id))
    ids.extend(aa_ids(tok, str(row[src_col])))
    ids.append(int(tok.sep_token_id))
    target_start = len(ids)
    ids.extend(aa_ids(tok, str(row[dst_col])))
    ids.append(int(tok.eos_token_id))
    return ids, target_start


@torch.no_grad()
def main() -> None:
    args = parse_args()
    tree_order = [s for s in args.tree_feature_order.split(",") if s]
    src_col, dst_col = f"{args.task}_t", f"{args.task}_t1"

    tok = ProteinTreeTokenizer.load(args.tokenizer_json)
    model = BertLMHeadModel.from_pretrained(args.model_dir).to(args.device)
    model.eval()

    max_tokens = args.max_tokens or int(model.config.max_position_embeddings)
    df = pd.read_parquet(args.pairs_parquet)

    streams, starts, keep = [], [], []
    for i, row in df.iterrows():
        ids, t0 = build_stream(row, tok, src_col, dst_col, args.use_tree_tokens, tree_order)
        if len(ids) > max_tokens:
            continue
        streams.append(ids)
        starts.append(t0)
        keep.append(i)

    n_skipped = len(df) - len(streams)
    if not streams:
        raise ValueError("Every row exceeded the model context length.")

    pad = int(tok.pad_token_id)
    per_example_nll: List[float] = []
    per_example_ntok: List[int] = []

    for b0 in range(0, len(streams), args.batch_size):
        chunk = streams[b0: b0 + args.batch_size]
        chunk_starts = starts[b0: b0 + args.batch_size]
        width = max(len(s) for s in chunk)

        x = torch.full((len(chunk), width), pad, dtype=torch.long)
        for j, s in enumerate(chunk):
            x[j, : len(s)] = torch.tensor(s, dtype=torch.long)
        x = x.to(args.device)
        attn = (x != pad).long()

        logits = model(input_ids=x, attention_mask=attn).logits
        logprobs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = x[:, 1:]
        gathered = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        # Supervise descendant tokens and the terminal [EOS] only. Position p of
        # `targets` predicts stream index p+1, so the target region starts at
        # target_start - 1.
        idx = torch.arange(width - 1, device=args.device).unsqueeze(0)
        start_t = torch.tensor(chunk_starts, device=args.device).unsqueeze(1) - 1
        mask = (idx >= start_t) & (targets != pad)

        nll = -(gathered * mask).sum(dim=1)
        ntok = mask.sum(dim=1).clamp(min=1)
        per_example_nll.extend((nll / ntok).cpu().tolist())
        per_example_ntok.extend(mask.sum(dim=1).cpu().tolist())

    out = df.loc[keep, :].copy()
    out["nll_per_token"] = per_example_nll
    out["n_target_tokens"] = per_example_ntok
    out["label"] = args.label
    out["model_dir"] = args.model_dir
    out["use_tree_tokens"] = int(args.use_tree_tokens)

    cols = [c for c in ("perturb_distance", "lineage_t", "donor_lineage_t") if c in out.columns]
    out = out[cols + ["nll_per_token", "n_target_tokens", "label", "model_dir", "use_tree_tokens"]]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # Token-weighted corpus loss, alongside the unweighted mean of per-example means.
    w = np.asarray(per_example_ntok, dtype=float)
    v = np.asarray(per_example_nll, dtype=float)
    weighted = float((v * w).sum() / w.sum())

    print(json.dumps(
        {
            "label": args.label,
            "n_rows": len(out),
            "n_skipped_too_long": n_skipped,
            "mean_nll_per_token": float(v.mean()),
            "token_weighted_nll": weighted,
            "out_csv": str(out_path),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
