#!/usr/bin/env python3
# scripts/train_prefix_lm.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    BertConfig,
    BertLMHeadModel,
)

from evoplm.tokenizer import ProteinTreeTokenizer


TREE_FEATURE_ORDER = ["delta_edges", "delta_nt_mutcount", "depth_t", "pt_t"]


def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_paths(traj_dir: Path, split: str, task: str) -> Tuple[Path, Path]:
    if split != "split1":
        raise ValueError("split must be split1")
    stem = "first3Omi2laterOmi"

    if task not in {"rbd", "spike"}:
        raise ValueError("task must be rbd or spike")

    fname_train = f"{split}_train_{stem}_pairs__{task}_t_ne_{task}_t1__delta_nt_mutcount_ge_1.parquet"
    fname_test  = f"{split}_test_{stem}_pairs__{task}_t_ne_{task}_t1__delta_nt_mutcount_ge_1.parquet"

    train_path = traj_dir / fname_train
    test_path = traj_dir / fname_test
    if not train_path.exists():
        raise FileNotFoundError(f"Train parquet not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test parquet not found: {test_path}")
    return train_path, test_path


def pad_to_max(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def apply_span_masking_prefix_region(
    x: torch.Tensor,
    tokenizer: ProteinTreeTokenizer,
    mask_prob: float,
    span_len: int,
    prefix_end_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Optional: mask random spans in the *prefix* seq_t region to force robustness.
    We do NOT mask the target seq_t1 region (the thing we supervise).
    prefix_end_positions: (B,) index where target region begins (inclusive).
    """
    if mask_prob <= 0.0 or span_len <= 0:
        return x

    B, L = x.shape
    x = x.clone()
    pad = int(tokenizer.pad_token_id)
    mask_id = int(tokenizer.mask_token_id)

    for i in range(B):
        row = x[i]
        tgt_start = int(prefix_end_positions[i].item())
        # Only mask before target start, excluding [CLS] at position 0
        start = 1
        end = max(start, tgt_start)
        if end <= start:
            continue

        # find effective non-pad region within prefix
        pad_pos = (row == pad).nonzero(as_tuple=False)
        if pad_pos.numel() > 0:
            end = min(end, int(pad_pos[0].item()))
        if end <= start:
            continue

        pref_len = end - start
        n_mask_tokens = int(round(mask_prob * pref_len))
        n_spans = max(0, n_mask_tokens // span_len)
        if n_spans == 0:
            continue

        max_start = end - span_len
        if max_start < start:
            continue

        starts = torch.randint(low=start, high=max_start + 1, size=(n_spans,))
        for s in starts.tolist():
            e = min(s + span_len, end)
            row[s:e] = mask_id
        x[i] = row

    return x


class PrefixLMCollator:
    """
    Builds:
      input_ids: [CLS] tree_tokens [SEP] seq_t [SEP] seq_t1 [EOS]
      attention_mask: 1 where not PAD
      labels: same length, but -100 for prefix region and real token ids for target region
    """
    def __init__(self, tokenizer: ProteinTreeTokenizer, max_tokens: int, mask_prob: float = 0.0, span_len: int = 0):
        self.tok = tokenizer
        self.pad_id = int(tokenizer.pad_token_id)
        self.sep_id = int(tokenizer.sep_token_id)
        self.eos_id = int(tokenizer.eos_token_id)
        self.max_tokens = int(max_tokens)
        self.mask_prob = float(mask_prob)
        self.span_len = int(span_len)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        seqs = [f["input_ids"] for f in features]
        prefix_end = torch.tensor([f["target_start"] for f in features], dtype=torch.long)

        max_len = max(len(s) for s in seqs)
        if max_len > self.max_tokens:
            raise ValueError(f"Batch has len={max_len} > max_tokens={self.max_tokens}. Increase --max_tokens.")

        x = pad_to_max(seqs, self.pad_id)                  # (B,L)
        attn = (x != self.pad_id).long()                   # (B,L)

        # Optional: mask spans in prefix only
        if self.mask_prob > 0.0 and self.span_len > 0:
            x = apply_span_masking_prefix_region(x, self.tok, self.mask_prob, self.span_len, prefix_end)

        # labels: -100 everywhere except target region (seq_t1 + EOS)
        labels = x.clone()
        labels[labels == self.pad_id] = -100

        B, L = labels.shape
        for i in range(B):
            t0 = int(prefix_end[i].item())
            # mask prefix
            labels[i, :t0] = -100

        return {"input_ids": x, "attention_mask": attn, "labels": labels}


def build_example(
    row: Dict[str, Any],
    tokenizer: ProteinTreeTokenizer,
    seq_in_col: str,
    seq_out_col: str,
    use_tree: bool,
) -> Dict[str, Any]:
    ids: List[int] = [int(tokenizer.cls_token_id)]

    # tree tokens
    if use_tree:
        feats = {
            "delta_edges": int(row["delta_edges"]) if "delta_edges" in row and pd.notna(row["delta_edges"]) else None,
            "delta_nt_mutcount": int(row["delta_nt_mutcount"]) if "delta_nt_mutcount" in row and pd.notna(row["delta_nt_mutcount"]) else None,
            "depth_t": int(row["depth_t"]) if "depth_t" in row and pd.notna(row["depth_t"]) else None,
            "pt_t": int(row["pt_t"]) if "pt_t" in row and pd.notna(row["pt_t"]) else None,
        }
        for k in TREE_FEATURE_ORDER:
            tok = tokenizer.tree_token(k, feats.get(k))
            ids.append(int(tokenizer.token_to_id[tok]))
        ids.append(int(tokenizer.sep_token_id))

    # seq_t
    s_in = str(row[seq_in_col])
    for ch in s_in:
        ids.append(int(tokenizer.token_to_id.get(ch, tokenizer.token_to_id["X"])))

    # separator between conditioning and target
    ids.append(int(tokenizer.sep_token_id))

    # target start index (this position is the first token of seq_t1)
    target_start = len(ids)

    # seq_t1
    s_out = str(row[seq_out_col])
    for ch in s_out:
        ids.append(int(tokenizer.token_to_id.get(ch, tokenizer.token_to_id["X"])))
    ids.append(int(tokenizer.eos_token_id))

    return {"input_ids": ids, "target_start": target_start}


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--task", choices=["rbd", "spike"], required=True)
    ap.add_argument("--split", choices=["split1"], required=True)
    ap.add_argument("--traj_dir", default="data/trajectories")
    ap.add_argument("--tokenizer_json", default="configs/protein_tree_tokenizer.json")
    ap.add_argument("--out_root", default="models/prefixlm_treeaware_filtered")
    ap.add_argument("--use_tree_tokens", action="store_true", help="Include tree tokens in prefix.")
    ap.add_argument("--max_tokens", type=int, default=0, help="0=auto (RBD 1024, Spike 2048).")

    # optional prefix masking
    ap.add_argument("--mask_prob", type=float, default=0.0)
    ap.add_argument("--span_len", type=int, default=0)

    # model size
    ap.add_argument("--hidden_size", type=int, default=384)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--intermediate_mult", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)

    # train
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--train_bs", type=int, default=16)
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--logging_steps", type=int, default=100)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    set_seed(args.seed)

    tokenizer = ProteinTreeTokenizer.load(args.tokenizer_json)
    vocab_size = tokenizer.vocab_size()

    if args.task == "rbd":
        seq_in_col, seq_out_col = "rbd_t", "rbd_t1"
        default_max_tokens = 1024
    else:
        seq_in_col, seq_out_col = "spike_t", "spike_t1"
        # spike can exceed 1536 with tree tokens + separators; 2048 is safer
        default_max_tokens = 2048

    max_tokens = int(args.max_tokens) if args.max_tokens > 0 else default_max_tokens

    traj_dir = Path(args.traj_dir)
    train_path, test_path = pick_paths(traj_dir, args.split, args.task)

    print("[INFO] task:", args.task, "split:", args.split)
    print("[INFO] train:", train_path)
    print("[INFO] test :", test_path)
    print("[INFO] use_tree_tokens:", bool(args.use_tree_tokens))
    print("[INFO] max_tokens:", max_tokens)
    print("[INFO] mask_prob/span_len:", args.mask_prob, args.span_len)
    print("[INFO] vocab_size:", vocab_size)

    train_df = pd.read_parquet(train_path)
    test_df  = pd.read_parquet(test_path)
    print("[INFO] rows train/test:", len(train_df), len(test_df))

    def map_fn(row: Dict[str, Any]) -> Dict[str, Any]:
        return build_example(row, tokenizer, seq_in_col, seq_out_col, use_tree=bool(args.use_tree_tokens))

    train_ds = Dataset.from_pandas(train_df).map(map_fn, remove_columns=train_df.columns.tolist())
    test_ds  = Dataset.from_pandas(test_df).map(map_fn, remove_columns=test_df.columns.tolist())

    collator = PrefixLMCollator(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        mask_prob=args.mask_prob,
        span_len=args.span_len,
    )

    intermediate_size = int(args.hidden_size * args.intermediate_mult)

    # BERT LM head model can be used as a decoder-only causal LM if we enable is_decoder
    cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_tokens,
        pad_token_id=tokenizer.pad_token_id,
        is_decoder=True,             # IMPORTANT: causal mask
        add_cross_attention=False,   # decoder-only
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
    )

    model = BertLMHeadModel(cfg)

    # Generation tokens
    model.config.bos_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    out_dir = (
        Path(args.out_root)
        / args.task
        / args.split
        / f"maxT{max_tokens}_tree{int(args.use_tree_tokens)}_maskp{args.mask_prob}_span{args.span_len}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(args.lr),
        per_device_train_batch_size=int(args.train_bs),
        per_device_eval_batch_size=int(args.eval_bs),
        num_train_epochs=int(args.epochs),
        weight_decay=float(args.wd),
        logging_steps=int(args.logging_steps),
        save_total_limit=int(args.save_total_limit),
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    print(f"[INFO] Saved -> {out_dir / 'final'}")


if __name__ == "__main__":
    main()
