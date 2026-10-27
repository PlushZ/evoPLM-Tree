#!/usr/bin/env python3

import argparse
import csv
from typing import Dict, List, Optional

import torch
from transformers import BertLMHeadModel

from evoplm.tokenizer import ProteinTreeTokenizer

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
TREE_FEATURE_ORDER = ["delta_edges", "delta_nt_mutcount", "depth_t", "pt_t"]

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def translate_nt(seq_nt: str) -> str:
    seq = seq_nt.upper().replace("U", "T").replace(" ", "").replace("\n", "")
    if len(seq) % 3 != 0:
        raise ValueError(f"Nucleotide length must be divisible by 3, got {len(seq)}.")
    aa = "".join(CODON_TABLE.get(seq[i:i + 3], "X") for i in range(0, len(seq), 3))
    if "*" in aa:
        raise ValueError("Stop codon detected after translation. Check reading frame/input sequence.")
    return aa


def normalize_input_sequence(raw_seq: str, seq_type: str) -> str:
    s = raw_seq.strip().upper().replace(" ", "").replace("\n", "")
    if seq_type == "nt":
        return translate_nt(s)
    if seq_type == "aa":
        return s

    nt_alphabet = set("ACGTUN")
    aa_alphabet = set("ACDEFGHIKLMNPQRSTVWYX-")
    if set(s).issubset(nt_alphabet) and len(s) % 3 == 0:
        return translate_nt(s)
    if set(s).issubset(aa_alphabet):
        return s
    raise ValueError("Could not auto-detect sequence type. Use --seq_type aa or --seq_type nt.")


def tree_feature_dict(args: argparse.Namespace) -> Dict[str, Optional[int]]:
    return {
        "delta_edges": args.delta_edges,
        "delta_nt_mutcount": args.delta_nt_mutcount,
        "depth_t": args.depth_t,
        "pt_t": args.pt_t,
    }


def aa_ids(tokenizer: ProteinTreeTokenizer, seq_aa: str) -> List[int]:
    x_id = int(tokenizer.token_to_id["X"])
    return [int(tokenizer.token_to_id.get(ch, x_id)) for ch in seq_aa]


def build_prefix_ids(
    tokenizer: ProteinTreeTokenizer,
    seq_in_aa: str,
    use_tree_tokens: bool,
    tree_features: Dict[str, Optional[int]],
) -> List[int]:
    ids: List[int] = [int(tokenizer.cls_token_id)]
    if use_tree_tokens:
        for k in TREE_FEATURE_ORDER:
            tok = tokenizer.tree_token(k, tree_features.get(k))
            ids.append(int(tokenizer.token_to_id[tok]))
        ids.append(int(tokenizer.sep_token_id))
    ids.extend(aa_ids(tokenizer, seq_in_aa))
    ids.append(int(tokenizer.sep_token_id))
    return ids


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tokenizer_json", default="configs/protein_tree_tokenizer.json")
    ap.add_argument("--ancestor_seq", required=True, help="AA or NT sequence. NT is translated if --seq_type nt/auto.")
    ap.add_argument("--seq_type", choices=["auto", "aa", "nt"], default="auto")
    ap.add_argument("--use_tree_tokens", action="store_true")
    ap.add_argument("--delta_edges", type=int, default=None)
    ap.add_argument("--delta_nt_mutcount", type=int, default=None)
    ap.add_argument("--depth_t", type=int, default=None)
    ap.add_argument("--pt_t", type=int, default=None)
    ap.add_argument(
        "--allow_masked_tree_features",
        action="store_true",
        help="Permit --use_tree_tokens with unset feature values (emits [MASK_TREE]). "
             "Off by default: an unset value silently removes the tree conditioning.",
    )
    ap.add_argument(
        "--expected_len",
        type=int,
        default=None,
        help="Assert the ancestor translates to exactly this many residues. Use the "
             "window the model was trained on (223 for RBD, spike 319-541).",
    )
    ap.add_argument(
        "--site_offset",
        type=int,
        default=0,
        help="Added to the 1-based model position to produce the emitted 'site'. "
             "Use to express sites in assay coordinates, e.g. -12 maps model "
             "position 13 (spike 331) to assay site 1.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_csv", default="mutation_probs.csv")
    args = ap.parse_args()

    seq_aa = normalize_input_sequence(args.ancestor_seq, args.seq_type)

    if args.expected_len is not None and len(seq_aa) != args.expected_len:
        raise ValueError(
            f"Ancestor is {len(seq_aa)} aa but --expected_len is {args.expected_len}. "
            "This model uses learned absolute positional embeddings, so a sequence of "
            "the wrong length shifts every residue out of its trained frame."
        )

    if args.use_tree_tokens:
        unset = [k for k, v in tree_feature_dict(args).items() if v is None]
        if unset and not args.allow_masked_tree_features:
            raise ValueError(
                f"--use_tree_tokens was given but these features are unset: {unset}. "
                "They would be encoded as [MASK_TREE], leaving the run with no tree "
                "conditioning at all. Supply values, or pass "
                "--allow_masked_tree_features if that is genuinely intended."
            )

    tokenizer = ProteinTreeTokenizer.load(args.tokenizer_json)
    model = BertLMHeadModel.from_pretrained(args.model_dir).to(args.device)
    model.eval()

    prefix_ids = build_prefix_ids(
        tokenizer=tokenizer,
        seq_in_aa=seq_aa,
        use_tree_tokens=bool(args.use_tree_tokens),
        tree_features=tree_feature_dict(args),
    )
    target_start = len(prefix_ids)

    # Teacher-forcing target context: same sequence as ancestor background.
    full_ids = prefix_ids + aa_ids(tokenizer, seq_aa) + [int(tokenizer.eos_token_id)]
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=args.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
        probs = torch.softmax(logits, dim=-1)

    records = []
    for i, wt in enumerate(seq_aa):
        pred_pos = target_start + i - 1
        for aa in AA20:
            if aa == wt:
                continue
            token_id = int(tokenizer.token_to_id[aa])
            records.append(
                {
                    "site": i + 1 + args.site_offset,
                    "model_site": i + 1,
                    "wt": wt,
                    "mut": aa,
                    "prob": float(probs[pred_pos, token_id].item()),
                    "model_dir": args.model_dir,
                    "use_tree_tokens": int(args.use_tree_tokens),
                }
            )

    fieldnames = ["site", "model_site", "wt", "mut", "prob", "model_dir", "use_tree_tokens"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[INFO] Input interpreted as AA length: {len(seq_aa)}")
    if args.use_tree_tokens:
        feats = tree_feature_dict(args)
        shown = ", ".join(f"{k}={tokenizer.tree_token(k, feats[k])}" for k in TREE_FEATURE_ORDER)
        print(f"[INFO] Tree tokens: {shown}")
    else:
        print("[INFO] Tree tokens: none (sequence-only input)")
    sites = [r["site"] for r in records]
    print(f"[INFO] Emitted site range: {min(sites)}..{max(sites)} (offset {args.site_offset:+d})")
    print(f"[INFO] Saved mutation probabilities -> {args.out_csv}")


if __name__ == "__main__":
    main()
