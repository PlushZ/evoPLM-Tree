#!/usr/bin/env python3
# scripts/diagnose_conditioning.py
"""
Prefix-LM (decoder-only) conditioning debug script.

What it tests (ONLY, for a prefix-LM checkpoint):
1) Builds the *correct* prefix prompt:
     [CLS] tree_tokens [SEP] seq_t [SEP]
   (NO [EOS] at end of prefix)

2) Generation dependence on prompt:
   - normal attention_mask
   - masked-prefix attention_mask (prefix tokens masked out, only last [SEP] kept visible)
   It reports uniqueness of the *continuation only* (seq_t1 prediction).

3) Teacher-forced loss dependence on prompt:
   Builds full stream:
     [CLS] tree [SEP] seq_t [SEP] seq_t1 [EOS]
   Labels are -100 on the prefix region, supervised only on seq_t1+[EOS].
   Compares loss with:
   - normal attention mask
   - masked-prefix attention mask (prefix tokens masked out)

Notes:
- Works for both rbd/spike (set seq cols + max_new_tokens accordingly).
- Prints *continuation-only* stats (not prompt+continuation).
"""

from __future__ import annotations
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, BertLMHeadModel

from evoplm.tokenizer import ProteinTreeTokenizer


TREE_FEATURE_ORDER_DEFAULT = ["delta_edges", "delta_nt_mutcount", "depth_t", "pt_t"]


def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_to_max(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    x = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return x


def make_tree_token_ids(
    r: pd.Series,
    tokenizer: ProteinTreeTokenizer,
    tree_feature_order: List[str],
) -> List[int]:
    feats = {
        "delta_edges": int(r["delta_edges"]) if "delta_edges" in r and pd.notna(r["delta_edges"]) else None,
        "delta_nt_mutcount": int(r["delta_nt_mutcount"]) if "delta_nt_mutcount" in r and pd.notna(r["delta_nt_mutcount"]) else None,
        "depth_t": int(r["depth_t"]) if "depth_t" in r and pd.notna(r["depth_t"]) else None,
        "pt_t": int(r["pt_t"]) if "pt_t" in r and pd.notna(r["pt_t"]) else None,
    }
    ids: List[int] = []
    for k in tree_feature_order:
        tok = tokenizer.tree_token(k, feats.get(k))
        ids.append(int(tokenizer.token_to_id[tok]))
    return ids


def aa_ids(tokenizer: ProteinTreeTokenizer, seq: str) -> List[int]:
    x_id = int(tokenizer.token_to_id["X"])
    return [int(tokenizer.token_to_id.get(ch, x_id)) for ch in seq]


def build_prefix_prompt(
    r: pd.Series,
    tokenizer: ProteinTreeTokenizer,
    seq_in_col: str,
    use_tree_tokens: bool,
    tree_feature_order: List[str],
) -> Tuple[List[int], int]:
    """
    Returns:
      prefix_ids = [CLS] tree [SEP] seq_t [SEP]
      sep2_index = index of the SECOND [SEP] within prefix_ids (0-based)
    """
    ids: List[int] = [int(tokenizer.cls_token_id)]

    if use_tree_tokens:
        ids.extend(make_tree_token_ids(r, tokenizer, tree_feature_order))
        ids.append(int(tokenizer.sep_token_id))

    ids.extend(aa_ids(tokenizer, str(r[seq_in_col])))
    ids.append(int(tokenizer.sep_token_id))  # the crucial boundary token

    sep2_index = len(ids) - 1
    return ids, sep2_index


def build_full_stream_for_tf_loss(
    r: pd.Series,
    tokenizer: ProteinTreeTokenizer,
    seq_in_col: str,
    seq_out_col: str,
    use_tree_tokens: bool,
    tree_feature_order: List[str],
) -> Tuple[List[int], int]:
    """
    Full stream:
      [CLS] tree [SEP] seq_t [SEP] seq_t1 [EOS]
    Returns:
      stream_ids
      target_start = first index of seq_t1 (i.e., right after second [SEP])
    """
    prefix, sep2_index = build_prefix_prompt(r, tokenizer, seq_in_col, use_tree_tokens, tree_feature_order)
    target_start = sep2_index + 1

    ids = list(prefix)
    ids.extend(aa_ids(tokenizer, str(r[seq_out_col])))
    ids.append(int(tokenizer.eos_token_id))
    return ids, target_start


def build_attention_masks_for_prompt_test(
    x: torch.Tensor,
    pad_id: int,
    sep2_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      attn_normal: 1 where not PAD
      attn_masked_prefix: masks out EVERYTHING before the second [SEP], but keeps:
        - the second [SEP] token itself visible (so generation can start right after)
        - any non-pad tokens after it (during TF loss full stream)
    """
    attn_normal = (x != pad_id).long()
    attn_masked_prefix = attn_normal.clone()

    B, L = x.shape
    for i in range(B):
        s2 = int(sep2_indices[i].item())
        # mask [0..s2-1]
        if s2 > 0:
            attn_masked_prefix[i, :s2] = 0
        # keep s2 itself and after as-is
    return attn_normal, attn_masked_prefix


def decode_continuation_only(
    tokenizer: ProteinTreeTokenizer,
    gen_ids: List[int],
    prompt_len: int,
) -> str:
    """
    gen_ids includes prompt + continuation. We slice continuation part only.
    """
    cont = gen_ids[prompt_len:]
    return tokenizer.decode(cont, remove_special_tokens=True)


def summarize_uniques(name: str, preds: List[str], show_k: int = 5):
    uniq = {}
    for p in preds:
        uniq[p] = uniq.get(p, 0) + 1
    items = sorted(uniq.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\n[{name}] n={len(preds)} unique={len(items)}")
    for s, c in items[:show_k]:
        preview = s[:80].replace("\n", " ")
        print(f"  count={c:4d}  preview='{preview}'  len={len(s)}")


@torch.no_grad()
def teacher_forced_loss_test(
    model: BertLMHeadModel,
    tokenizer: ProteinTreeTokenizer,
    df: pd.DataFrame,
    seq_in_col: str,
    seq_out_col: str,
    use_tree_tokens: bool,
    tree_feature_order: List[str],
    batch_size: int,
    device: str,
):
    streams: List[List[int]] = []
    target_starts: List[int] = []
    sep2_indices: List[int] = []

    for _, r in df.iterrows():
        ids, t0 = build_full_stream_for_tf_loss(
            r, tokenizer, seq_in_col, seq_out_col, use_tree_tokens, tree_feature_order
        )
        # sep2 index is t0 - 1
        streams.append(ids)
        target_starts.append(t0)
        sep2_indices.append(t0 - 1)

    x = pad_to_max(streams, int(tokenizer.pad_token_id)).to(device)

    # labels: -100 on prefix, supervise only target region (seq_t1 + EOS)
    labels = x.clone()
    labels[labels == int(tokenizer.pad_token_id)] = -100
    B, L = labels.shape
    for i in range(B):
        labels[i, :target_starts[i]] = -100
    labels = labels.to(device)

    sep2_t = torch.tensor(sep2_indices, dtype=torch.long, device=device)
    attn_normal, attn_masked_prefix = build_attention_masks_for_prompt_test(
        x, int(tokenizer.pad_token_id), sep2_t
    )
    attn_normal = attn_normal.to(device)
    attn_masked_prefix = attn_masked_prefix.to(device)

    # Compute mean losses over batches
    losses_n: List[float] = []
    losses_m: List[float] = []

    for start in range(0, x.shape[0], batch_size):
        xb = x[start:start+batch_size]
        yb = labels[start:start+batch_size]
        an = attn_normal[start:start+batch_size]
        am = attn_masked_prefix[start:start+batch_size]

        out_n = model(input_ids=xb, attention_mask=an, labels=yb)
        out_m = model(input_ids=xb, attention_mask=am, labels=yb)

        losses_n.append(float(out_n.loss.detach().cpu()))
        losses_m.append(float(out_m.loss.detach().cpu()))

    loss_normal = float(np.mean(losses_n))
    loss_masked_prefix = float(np.mean(losses_m))
    ratio = loss_masked_prefix / max(loss_normal, 1e-12)

    print("\n[TEACHER-FORCED PROMPT DEPENDENCE TEST]")
    print(f"  loss_normal(prompt visible)      = {loss_normal:.6f}")
    print(f"  loss_masked_prefix(prompt masked)= {loss_masked_prefix:.6f}")
    print(f"  ratio masked/normal              = {ratio:.3f}")
    if ratio > 1.05:
        print("  -> Model uses the prompt (masking prompt worsens loss). ✅")
    else:
        print("  -> Weak prompt dependence under TF loss (may still be early training / small data). ⚠️")


@torch.no_grad()
def generation_prompt_dependence_test(
    model: BertLMHeadModel,
    tokenizer: ProteinTreeTokenizer,
    df: pd.DataFrame,
    seq_in_col: str,
    seq_out_col: str,
    use_tree_tokens: bool,
    tree_feature_order: List[str],
    batch_size: int,
    device: str,
    max_new_tokens: int,
    num_beams: int,
    do_sample: bool,
    top_p: float,
    temperature: float,
):
    # Build prompts
    prompts: List[List[int]] = []
    sep2_indices: List[int] = []
    prompt_lens: List[int] = []
    true_t1: List[str] = []

    for _, r in df.iterrows():
        p, s2 = build_prefix_prompt(r, tokenizer, seq_in_col, use_tree_tokens, tree_feature_order)
        prompts.append(p)
        sep2_indices.append(s2)
        prompt_lens.append(len(p))
        true_t1.append(str(r[seq_out_col]))

    x = pad_to_max(prompts, int(tokenizer.pad_token_id)).to(device)
    sep2_t = torch.tensor(sep2_indices, dtype=torch.long, device=device)
    attn_normal, attn_masked_prefix = build_attention_masks_for_prompt_test(
        x, int(tokenizer.pad_token_id), sep2_t
    )
    attn_normal = attn_normal.to(device)
    attn_masked_prefix = attn_masked_prefix.to(device)

    preds_normal: List[str] = []
    preds_masked: List[str] = []

    # Generate
    for start in range(0, x.shape[0], batch_size):
        xb = x[start:start+batch_size]
        an = attn_normal[start:start+batch_size]
        am = attn_masked_prefix[start:start+batch_size]

        gen_n = model.generate(
            input_ids=xb,
            attention_mask=an,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            top_p=top_p if do_sample else None,
            temperature=temperature if do_sample else None,
        )
        gen_m = model.generate(
            input_ids=xb,
            attention_mask=am,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            top_p=top_p if do_sample else None,
            temperature=temperature if do_sample else None,
        )

        gen_n = gen_n.detach().cpu().tolist()
        gen_m = gen_m.detach().cpu().tolist()

        # decode continuation only (slice off prompt)
        for j in range(len(gen_n)):
            idx = start + j
            if idx >= len(prompt_lens):
                break
            pl = prompt_lens[idx]
            preds_normal.append(decode_continuation_only(tokenizer, gen_n[j], pl))
            preds_masked.append(decode_continuation_only(tokenizer, gen_m[j], pl))

    summarize_uniques("GEN continuation (prompt visible)", preds_normal, show_k=5)
    summarize_uniques("GEN continuation (prompt masked)", preds_masked, show_k=5)

    same = sum(a == b for a, b in zip(preds_normal, preds_masked))
    print("\n[GEN PROMPT DEPENDENCE TEST]")
    print(f"  same(visible, masked) = {same}/{len(preds_normal)} ({same/len(preds_normal):.3f})")
    if same == len(preds_normal):
        print("  -> Continuation does NOT change when prompt is masked. Conditioning not used. ❌")
    else:
        print("  -> Continuation changes when prompt is masked. Conditioning is being used. ✅")

    # show a few examples
    print("\n[EXAMPLES: seq_t prefix -> pred_t1 (visible) | pred_t1 (masked) | true_t1]")
    for i in range(min(5, len(df))):
        inp = str(df.loc[i, seq_in_col])
        print(f"\n  ex{i}")
        print("    seq_t :", inp[:120])
        print("    predV :", preds_normal[i][:120])
        print("    predM :", preds_masked[i][:120])
        print("    true  :", true_t1[i][:120])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tokenizer_json", required=True)
    ap.add_argument("--pairs_parquet", required=True)

    ap.add_argument("--seq_in_col", default="rbd_t")
    ap.add_argument("--seq_out_col", default="rbd_t1")

    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_rows", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--use_tree_tokens", action="store_true")
    ap.add_argument("--tree_feature_order", default="delta_edges,delta_nt_mutcount,depth_t,pt_t")

    ap.add_argument("--max_new_tokens", type=int, default=260)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=1.0)

    ap.add_argument("--compute_teacher_forced_loss", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    tokenizer = ProteinTreeTokenizer.load(args.tokenizer_json)

    # Load decoder-only model
    config = AutoConfig.from_pretrained(args.model_dir)
    # Some checkpoints carry loss_type=None in config.json, which triggers
    # a warning at runtime. Set the explicit default used by Transformers.
    if getattr(config, "loss_type", None) in (None, "", "None", "none", "null"):
        config.loss_type = "ForCausalLMLoss"
    model = BertLMHeadModel.from_pretrained(args.model_dir, config=config)
    model.eval()
    model.to(args.device)

    # token ids
    model.config.bos_token_id = int(tokenizer.cls_token_id)
    model.config.eos_token_id = int(tokenizer.eos_token_id)
    model.config.pad_token_id = int(tokenizer.pad_token_id)

    print("\n[CONFIG CHECK]")
    print("  is_encoder_decoder:", getattr(model.config, "is_encoder_decoder", False))
    print("  bos_token_id:", getattr(model.config, "bos_token_id", None))
    print("  eos_token_id:", getattr(model.config, "eos_token_id", None))
    print("  pad_token_id:", getattr(model.config, "pad_token_id", None))
    print("  is_decoder:", getattr(model.config, "is_decoder", None))
    print("  add_cross_attention:", getattr(model.config, "add_cross_attention", None))

    df = pd.read_parquet(args.pairs_parquet)
    if len(df) == 0:
        raise ValueError("Empty parquet.")
    n_take = min(args.n_rows, len(df))
    df = df.sample(n=n_take, random_state=args.seed).reset_index(drop=True)

    tree_order = [s.strip() for s in args.tree_feature_order.split(",") if s.strip()]
    if len(tree_order) == 0:
        tree_order = list(TREE_FEATURE_ORDER_DEFAULT)

    # quick check inputs differ
    print("\n[INPUT CHECK]")
    if len(df) >= 2:
        s0 = str(df.loc[0, args.seq_in_col])[:80]
        s1 = str(df.loc[1, args.seq_in_col])[:80]
        print("  seq_in[0] preview:", s0)
        print("  seq_in[1] preview:", s1)

    if args.compute_teacher_forced_loss:
        teacher_forced_loss_test(
            model=model,
            tokenizer=tokenizer,
            df=df,
            seq_in_col=args.seq_in_col,
            seq_out_col=args.seq_out_col,
            use_tree_tokens=bool(args.use_tree_tokens),
            tree_feature_order=tree_order,
            batch_size=args.batch_size,
            device=args.device,
        )

    generation_prompt_dependence_test(
        model=model,
        tokenizer=tokenizer,
        df=df,
        seq_in_col=args.seq_in_col,
        seq_out_col=args.seq_out_col,
        use_tree_tokens=bool(args.use_tree_tokens),
        tree_feature_order=tree_order,
        batch_size=args.batch_size,
        device=args.device,
        max_new_tokens=int(args.max_new_tokens),
        num_beams=int(args.num_beams),
        do_sample=bool(args.do_sample),
        top_p=float(args.top_p),
        temperature=float(args.temperature),
    )


if __name__ == "__main__":
    main()
