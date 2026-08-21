#!/usr/bin/env python3
"""
Protein tokenizer (1 AA = 1 token) with optional discretized MAT/tree feature tokens.

- AA tokens: 20 canonical + X + '-'
- Special tokens: [PAD], [CLS], [EOS], [MASK], [SEP]
- Tree tokens: bucketed tokens for:
    delta_edges, delta_nt_mutcount, depth_t, depth_t1, pt_t, pt_t1

Tree tokens are designed to be OPTIONAL:
- Include them during TRAIN (phylogeny-regularized training).
- Mask or omit them during TEST (sequence-only inference).

"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# -------------------------
# Bucketing config
# -------------------------
@dataclass(frozen=True)
class BucketSpec:
    name: str
    prefix: str
    max_value: int          # inclusive max representable value
    pad_width: int          # formatting, e.g. 3 -> 007
    clamp: bool = True      # clamp values above max to max token


DEFAULT_BUCKETS: Dict[str, BucketSpec] = {
    # can tune these maxima to match the observed ranges.
    "delta_edges":        BucketSpec(name="delta_edges",        prefix="DE",     max_value=512, pad_width=3),
    "delta_nt_mutcount":  BucketSpec(name="delta_nt_mutcount",  prefix="DNT",    max_value=2048, pad_width=4),
    "depth_t":            BucketSpec(name="depth_t",            prefix="DEP_T",  max_value=512, pad_width=3),
    "depth_t1":           BucketSpec(name="depth_t1",           prefix="DEP_T1", max_value=512, pad_width=3),
    "pt_t":               BucketSpec(name="pt_t",               prefix="PT_T",   max_value=4096, pad_width=4),
    "pt_t1":              BucketSpec(name="pt_t1",              prefix="PT_T1",  max_value=4096, pad_width=4),
}


class ProteinTreeTokenizer:
    """
    Simple tokenizer that:
      - encodes AA sequences (1 char -> 1 token)
      - can prepend discretized tree tokens as special tokens
      - supports save/load of vocab + bucket config
    """

    def __init__(
        self,
        vocab: Optional[List[str]] = None,
        buckets: Optional[Dict[str, BucketSpec]] = None,
    ):
        # Base AA vocab
        if vocab is None:
            amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
            extra = ["X", "-"]
            specials = ["[PAD]", "[CLS]", "[EOS]", "[MASK]", "[SEP]", "[MASK_TREE]"]
            vocab = amino_acids + extra + specials

        self.buckets: Dict[str, BucketSpec] = buckets if buckets is not None else DEFAULT_BUCKETS

        # Expand vocab with all bucket tokens
        tree_tokens = self._build_all_tree_tokens(self.buckets)
        # ensure deterministic order
        vocab = list(vocab) + [t for t in tree_tokens if t not in vocab]

        self.vocab = vocab
        self.token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token: Dict[int, str] = {i: tok for i, tok in enumerate(vocab)}

        # Convenience ids
        self.pad_token_id = self.token_to_id["[PAD]"]
        self.cls_token_id = self.token_to_id["[CLS]"]
        self.eos_token_id = self.token_to_id["[EOS]"]
        self.mask_token_id = self.token_to_id["[MASK]"]
        self.sep_token_id = self.token_to_id["[SEP]"]
        self.mask_tree_token_id = self.token_to_id["[MASK_TREE]"]

        # sanity
        for aa in list("ACDEFGHIKLMNPQRSTVWY") + ["X", "-"]:
            if aa not in self.token_to_id:
                raise ValueError(f"AA token missing from vocab: {aa}")

    # -------------------------
    # Tree token helpers
    # -------------------------
    @staticmethod
    def _format_bucket_token(spec: BucketSpec, value: int) -> str:
        s = str(value).zfill(spec.pad_width)
        return f"[{spec.prefix}={s}]"

    @classmethod
    def _build_all_tree_tokens(cls, buckets: Dict[str, BucketSpec]) -> List[str]:
        tokens: List[str] = []
        for _, spec in buckets.items():
            for v in range(spec.max_value + 1):
                tokens.append(cls._format_bucket_token(spec, v))
        return tokens

    def tree_token(self, feature_name: str, value: Optional[int]) -> str:
        """
        Convert a numeric feature value into a bucket token.
        If value is None or NaN-like, returns [MASK_TREE].
        """
        if value is None:
            return "[MASK_TREE]"
        try:
            iv = int(value)
        except Exception:
            return "[MASK_TREE]"

        if feature_name not in self.buckets:
            raise KeyError(f"Unknown feature_name '{feature_name}'. Known: {list(self.buckets.keys())}")

        spec = self.buckets[feature_name]
        if iv < 0:
            return "[MASK_TREE]"
        if iv > spec.max_value:
            iv = spec.max_value if spec.clamp else iv
        return self._format_bucket_token(spec, iv)

    # -------------------------
    # Core encode/decode
    # -------------------------
    def encode_sequence(self, seq: str, add_special_tokens: bool = True) -> List[int]:
        ids = [self.token_to_id.get(ch, self.token_to_id["X"]) for ch in seq]
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.eos_token_id]
        return ids

    def encode(
        self,
        seq: str,
        tree_features: Optional[Dict[str, Optional[int]]] = None,
        add_special_tokens: bool = True,
        add_sep: bool = True,
        tree_feature_order: Optional[List[str]] = None,
    ) -> List[int]:
        """
        Encode AA sequence with optional prepended tree tokens.

        Format:
          [CLS] <tree tokens...> [SEP] A A ... [EOS]

        tree_features: dict with keys among DEFAULT_BUCKETS:
          delta_edges, delta_nt_mutcount, depth_t, depth_t1, pt_t, pt_t1
        """
        ids: List[int] = []

        if add_special_tokens:
            ids.append(self.cls_token_id)

        # tree tokens (optional)
        if tree_features is not None:
            order = tree_feature_order or list(self.buckets.keys())
            for key in order:
                tok = self.tree_token(key, tree_features.get(key))
                ids.append(self.token_to_id[tok])
            if add_sep:
                ids.append(self.sep_token_id)

        # AA tokens
        ids.extend([self.token_to_id.get(ch, self.token_to_id["X"]) for ch in seq])

        if add_special_tokens:
            ids.append(self.eos_token_id)

        return ids

    def decode(self, ids: Sequence[int], remove_special_tokens: bool = True) -> str:
        toks = [self.id_to_token[int(i)] for i in ids]
        if remove_special_tokens:
            drop = {"[PAD]", "[CLS]", "[EOS]", "[MASK]", "[SEP]", "[MASK_TREE]"}
            # also drop all tree bucket tokens like [DE=007], ...
            # anything that starts with '[' and contains '=' is treated as a feature token
            filtered = []
            for t in toks:
                if t in drop:
                    continue
                if t.startswith("[") and "=" in t and t.endswith("]"):
                    continue
                filtered.append(t)
            toks = filtered
        # Remaining tokens should be AA characters
        return "".join([t for t in toks if len(t) == 1])

    # -------------------------
    # Utilities
    # -------------------------
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: str):
        payload = {
            "vocab": self.vocab,
            "buckets": {
                k: {
                    "name": v.name,
                    "prefix": v.prefix,
                    "max_value": v.max_value,
                    "pad_width": v.pad_width,
                    "clamp": v.clamp,
                }
                for k, v in self.buckets.items()
            },
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[INFO] Tokenizer saved to {path}")

    def save_pretrained(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str]:
        """
        Hugging Face compatibility: allow Trainer to persist custom tokenizer objects.
        """
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = "protein_tree_tokenizer.json"
        if filename_prefix:
            stem = f"{filename_prefix}-{stem}"
        out_path = out_dir / stem
        self.save(str(out_path))
        return (str(out_path),)

    @classmethod
    def load(cls, path: str) -> "ProteinTreeTokenizer":
        with open(path) as f:
            payload = json.load(f)
        buckets = {
            k: BucketSpec(
                name=v["name"],
                prefix=v["prefix"],
                max_value=int(v["max_value"]),
                pad_width=int(v["pad_width"]),
                clamp=bool(v.get("clamp", True)),
            )
            for k, v in payload["buckets"].items()
        }
        return cls(vocab=payload["vocab"], buckets=buckets)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> "ProteinTreeTokenizer":
        """
        Hugging Face compatibility counterpart to save_pretrained.
        """
        p = Path(pretrained_model_name_or_path)
        if p.is_file():
            return cls.load(str(p))
        candidate = p / "protein_tree_tokenizer.json"
        if candidate.exists():
            return cls.load(str(candidate))
        raise FileNotFoundError(
            f"No tokenizer file found at {p}. Expected a file path or {candidate}."
        )


# -------------------------
# Example usage
# -------------------------

if __name__ == "__main__":
    tok = ProteinTreeTokenizer()
    print("Vocab size:", tok.vocab_size())

    seq = "MFVFLVLLPL-X"
    feats = {
        "delta_edges": 23,
        "delta_nt_mutcount": 55,
        "depth_t": 40,
        "depth_t1": 52,
        "pt_t": 100,
        "pt_t1": 155,
    }
    ids = tok.encode(seq, tree_features=feats)
    print("Encoded ids:", ids[:20], "...")

    decoded = tok.decode(ids)
    print("Decoded seq:", decoded)

    tok.save("protein_tree_tokenizer.json")
    tok2 = ProteinTreeTokenizer.load("protein_tree_tokenizer.json")
    print("Reload vocab size:", tok2.vocab_size())
