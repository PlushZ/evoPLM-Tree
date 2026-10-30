#!/usr/bin/env python3
"""
Positional mutation-frequency profiles shared by the control analyses.

The counting convention follows scripts/figures/plot_mutation_hotspots.py, so
baseline profiles and model profiles are directly comparable: a fixed reference
length n_ref, position-wise comparison truncated to the shortest sequence
involved, and a frequency of mutated pairs over comparable pairs.

With exclude_ambiguous=False every position of every pair is comparable, so the
denominator is the pair count and the result reproduces the existing figure
code exactly. With exclude_ambiguous=True, positions where either sequence
carries X, * or - are dropped from both numerator and denominator, which
removes two artefacts of the current pipeline:

  - the tokenizer maps the terminal '*' of a translated spike ORF to 'X'
    (src/evoplm/tokenizer.py), so the observed sequence ends in '*' while the
    generated one ends in 'X' and the final position registers as a mutation in
    almost every full-spike pair;
  - QC admits up to three X per spike, and 42% of training spike sequences
    contain at least one.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

# Residues that carry no amino-acid identity and so cannot support a
# meaningful "did this position change" comparison.
AMBIGUOUS = frozenset("X*-")


def mutation_matrices(
    sources: Sequence[str],
    targets: Sequence[str],
    n_ref: int,
    exclude_ambiguous: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-pair, per-position boolean matrices.

    Returns (mutated, comparable), both shape (n_pairs, n_ref):
      comparable[p, i]  position i of pair p contributes to the denominator
      mutated[p, i]     ... and the residue differs between source and target
    """
    if len(sources) != len(targets):
        raise ValueError(f"sources/targets length mismatch: {len(sources)} vs {len(targets)}")

    n_pairs = len(sources)
    mutated = np.zeros((n_pairs, n_ref), dtype=bool)
    comparable = np.zeros((n_pairs, n_ref), dtype=bool)

    for p, (src, dst) in enumerate(zip(sources, targets)):
        m = min(len(src), len(dst), n_ref)
        if m == 0:
            continue
        a = np.frombuffer(src[:m].encode("ascii"), dtype=np.uint8)
        b = np.frombuffer(dst[:m].encode("ascii"), dtype=np.uint8)

        ok = np.ones(m, dtype=bool)
        if exclude_ambiguous:
            for ch in AMBIGUOUS:
                code = ord(ch)
                ok &= (a != code) & (b != code)

        comparable[p, :m] = ok
        mutated[p, :m] = ok & (a != b)

    return mutated, comparable


def profile_from_matrices(mutated: np.ndarray, comparable: np.ndarray) -> np.ndarray:
    """Per-position mutation frequency; NaN where no pair was comparable."""
    num = mutated.sum(axis=0).astype(float)
    den = comparable.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        freq = np.where(den > 0, num / den, np.nan)
    return freq


def profile_from_pairs(
    sources: Sequence[str],
    targets: Sequence[str],
    n_ref: int,
    exclude_ambiguous: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper: returns (n_mutated, n_comparable, freq) per position."""
    mutated, comparable = mutation_matrices(sources, targets, n_ref, exclude_ambiguous)
    return (
        mutated.sum(axis=0),
        comparable.sum(axis=0),
        profile_from_matrices(mutated, comparable),
    )


def modal_length(seqs: Iterable[str]) -> int:
    """Reference length: the most common sequence length, as the figure code uses."""
    lengths: List[int] = [len(s) for s in seqs]
    if not lengths:
        raise ValueError("no sequences given")
    values, counts = np.unique(np.asarray(lengths), return_counts=True)
    return int(values[int(np.argmax(counts))])
