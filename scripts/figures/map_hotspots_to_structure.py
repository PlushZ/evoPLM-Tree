#!/usr/bin/env python3

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "ASX": "B",
    "GLX": "Z",
    "SEC": "U",
    "PYL": "O",
    "UNK": "X",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gap-safe, region-aware hotspot mapping to a structure (writes _remapped outputs)."
    )
    parser.add_argument(
        "--csv_path",
        required=True,
        help="Input generation CSV (e.g., results/generation/split1/rbd_treeaware_sample2000.csv).",
    )
    parser.add_argument(
        "--pdb_path",
        required=True,
        help="Input PDB file used for structure visualization.",
    )
    parser.add_argument(
        "--chain",
        default="A",
        help="PDB chain ID to use for mapping.",
    )
    parser.add_argument(
        "--score",
        choices=["pred", "true", "pred_minus_true", "abs_pred_minus_true"],
        default="pred",
        help="Which hotspot score to project onto structure B-factors.",
    )
    parser.add_argument(
        "--value_mode",
        choices=["freq", "count"],
        default="freq",
        help="Use normalized frequencies (count/N) or raw absolute counts for hotspot values.",
    )
    parser.add_argument(
        "--reference_col",
        default="input_seq",
        help="CSV column used as the reference sequence for sequence->structure alignment.",
    )
    parser.add_argument("--model_type", choices=["auto", "rbd", "spike"], default="auto",
                        help="RBD maps to PDB residues 319-541; spike maps to the full chain.")
    parser.add_argument("--region_start", type=int, default=319)
    parser.add_argument("--region_end", type=int, default=541)
    parser.add_argument(
        "--bfactor_scale",
        type=float,
        default=100.0,
        help="Scale factor applied to hotspot scores before writing to B-factor.",
    )
    parser.add_argument(
        "--out_prefix",
        default=None,
        help="Output prefix. Default includes the required _remapped suffix.",
    )
    parser.add_argument(
        "--out_dir",
        default="results/generation/split1/hotspots_to_structure",
        help="Output directory for mapped files.",
    )
    return parser.parse_args()


def compute_hotspot_profiles(
    rows: List[Dict[str, str]], reference_col: str
) -> Tuple[str, List[int], List[int], List[float], List[float]]:
    if not rows:
        raise ValueError("Input CSV has 0 rows.")

    required_cols = {"input_seq", "true_seq", "pred_seq", reference_col}
    missing_cols = required_cols - set(rows[0].keys())
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {sorted(missing_cols)}")

    ref_values = [str(r[reference_col]) for r in rows if r.get(reference_col)]
    if not ref_values:
        raise ValueError(f"No non-null sequence values found in reference column '{reference_col}'.")
    reference_seq = Counter(ref_values).most_common(1)[0][0]
    # Alignment gaps are not residues; remove their columns before scoring.
    keep = [i for i, aa in enumerate(reference_seq) if aa != "-"]
    reference_seq = "".join(reference_seq[i] for i in keep)

    n_ref = len(reference_seq)
    true_counts = [0] * n_ref
    pred_counts = [0] * n_ref

    for row in rows:
        inp = str(row["input_seq"])
        true = str(row["true_seq"])
        pred = str(row["pred_seq"])
        max_len = min(len(inp), len(true), len(pred))
        for out_i, src_i in enumerate(keep):
            if src_i >= max_len:
                continue
            if inp[src_i] != true[src_i]:
                true_counts[out_i] += 1
            if inp[src_i] != pred[src_i]:
                pred_counts[out_i] += 1

    denom = float(len(rows))

    true_freq = [c / denom for c in true_counts]
    pred_freq = [c / denom for c in pred_counts]
    return reference_seq, true_counts, pred_counts, true_freq, pred_freq


def extract_chain_residues_from_pdb(pdb_path: Path, chain: str) -> List[Dict[str, str]]:
    residues: List[Dict[str, str]] = []
    seen = set()
    chain = chain.strip()

    with pdb_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if len(line) < 54:
                continue

            atom_name = line[12:16].strip()
            alt_loc = line[16]
            chain_id = line[21].strip()
            resname = line[17:20].strip().upper()
            resseq = line[22:26].strip()
            icode = line[26].strip()

            if atom_name != "CA":
                continue
            if alt_loc not in (" ", "A"):
                continue
            if chain and chain_id != chain:
                continue

            key = (chain_id, resseq, icode)
            if key in seen:
                continue
            seen.add(key)

            residues.append(
                {
                    "chain": chain_id,
                    "resseq": resseq,
                    "icode": icode,
                    "resname": resname,
                    "aa": AA3_TO_1.get(resname, "X"),
                    "key": key,
                }
            )

    if not residues:
        raise ValueError(f"No residues extracted from chain '{chain}' in {pdb_path}.")
    return residues


def select_region(residues: List[Dict[str, str]], start: int, end: int) -> List[Dict[str, str]]:
    selected = []
    for r in residues:
        try:
            number = int(r["resseq"])
        except ValueError:
            continue
        if start <= number <= end:
            selected.append(r)
    if not selected:
        raise ValueError(f"No chain residues found in requested region {start}-{end}.")
    return selected


def global_align(seq_a: str, seq_b: str, match: int = 2, mismatch: int = -1, gap: int = -2) -> Tuple[str, str]:
    n = len(seq_a)
    m = len(seq_b)
    ncols = m + 1

    trace = bytearray((n + 1) * (m + 1))

    prev = [0] * (m + 1)
    for j in range(1, m + 1):
        prev[j] = prev[j - 1] + gap
        trace[j] = 2

    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        curr[0] = prev[0] + gap
        trace[i * ncols] = 1

        a = seq_a[i - 1]
        base = i * ncols
        for j in range(1, m + 1):
            b = seq_b[j - 1]
            score_diag = prev[j - 1] + (match if a == b else mismatch)
            score_up = prev[j] + gap
            score_left = curr[j - 1] + gap

            if score_diag >= score_up and score_diag >= score_left:
                curr[j] = score_diag
                trace[base + j] = 0
            elif score_up >= score_left:
                curr[j] = score_up
                trace[base + j] = 1
            else:
                curr[j] = score_left
                trace[base + j] = 2
        prev = curr

    aligned_a: List[str] = []
    aligned_b: List[str] = []
    i = n
    j = m
    while i > 0 or j > 0:
        t = trace[i * ncols + j]
        if i > 0 and j > 0 and t == 0:
            aligned_a.append(seq_a[i - 1])
            aligned_b.append(seq_b[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or t == 1):
            aligned_a.append(seq_a[i - 1])
            aligned_b.append("-")
            i -= 1
        else:
            aligned_a.append("-")
            aligned_b.append(seq_b[j - 1])
            j -= 1

    aligned_a.reverse()
    aligned_b.reverse()
    return "".join(aligned_a), "".join(aligned_b)


def build_model_to_residue_mapping(model_seq: str, pdb_seq: str) -> Dict[int, int]:
    aligned_model, aligned_pdb = global_align(model_seq, pdb_seq)
    mapping: Dict[int, int] = {}
    i_model = 0
    i_pdb = 0

    for a, b in zip(aligned_model, aligned_pdb):
        model_idx = None
        pdb_idx = None
        if a != "-":
            model_idx = i_model
            i_model += 1
        if b != "-":
            pdb_idx = i_pdb
            i_pdb += 1
        if model_idx is not None and pdb_idx is not None:
            mapping[model_idx] = pdb_idx
    return mapping


def score_vector_from_choice(true_vals: List[float], pred_vals: List[float], score_name: str) -> List[float]:
    if score_name == "pred":
        return pred_vals
    if score_name == "true":
        return true_vals
    if score_name == "pred_minus_true":
        return [p - t for p, t in zip(pred_vals, true_vals)]
    if score_name == "abs_pred_minus_true":
        return [abs(p - t) for p, t in zip(pred_vals, true_vals)]
    raise ValueError(f"Unsupported score name: {score_name}")


def write_mapped_csv(
    out_csv: Path,
    reference_seq: str,
    true_counts: List[int],
    pred_counts: List[int],
    true_freq: List[float],
    pred_freq: List[float],
    residues: List[Dict[str, str]],
    model_to_pdb: Dict[int, int],
) -> None:
    out_rows = []
    for pos0 in range(len(reference_seq)):
        row = {
            "model_pos_0": pos0,
            "model_pos_1": pos0 + 1,
            "model_aa": reference_seq[pos0],
            "true_hotspot_count": true_counts[pos0],
            "pred_hotspot_count": pred_counts[pos0],
            "true_hotspot_freq": true_freq[pos0],
            "pred_hotspot_freq": pred_freq[pos0],
            "mapped_to_pdb": False,
            "pdb_chain": "",
            "pdb_resseq": "",
            "pdb_icode": "",
            "pdb_resname": "",
            "pdb_aa": "",
        }
        if pos0 in model_to_pdb:
            r = residues[model_to_pdb[pos0]]
            row.update(
                {
                    "mapped_to_pdb": True,
                    "pdb_chain": r["chain"],
                    "pdb_resseq": r["resseq"],
                    "pdb_icode": r["icode"],
                    "pdb_resname": r["resname"],
                    "pdb_aa": r["aa"],
                }
            )
        out_rows.append(row)

    fieldnames = [
        "model_pos_0",
        "model_pos_1",
        "model_aa",
        "true_hotspot_count",
        "pred_hotspot_count",
        "true_hotspot_freq",
        "pred_hotspot_freq",
        "mapped_to_pdb",
        "pdb_chain",
        "pdb_resseq",
        "pdb_icode",
        "pdb_resname",
        "pdb_aa",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_bfactor_pdb(
    in_pdb: Path,
    out_pdb: Path,
    chain: str,
    score_by_reskey: Dict[Tuple[str, str, str], float],
    bfactor_scale: float,
) -> None:
    with in_pdb.open("r", encoding="utf-8", errors="replace") as fin, out_pdb.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if (line.startswith("ATOM") or line.startswith("HETATM")) and len(line) >= 66:
                chain_id = line[21].strip()
                resseq = line[22:26].strip()
                icode = line[26].strip()
                key = (chain_id, resseq, icode)
                if not chain or chain_id == chain:
                    # Reset selected chain to 0 first so unmapped residues do not keep native PDB B-factors.
                    b = 0.0
                    if key in score_by_reskey:
                        b = score_by_reskey[key] * bfactor_scale
                    line = f"{line[:60]}{b:6.2f}{line[66:]}"
            fout.write(line)


def write_pymol_script(
    out_pml: Path,
    out_pdb: Path,
    chain: str,
    score_name: str,
    bfactor_min: float,
    bfactor_max: float,
    png_path: Path,
) -> None:
    obj = out_pdb.stem
    lines = [
        "reinitialize",
        f"load {out_pdb}, {obj}",
        "hide everything, all",
        f"show cartoon, {obj} and chain {chain}",
        f"spectrum b, blue_white_red, {obj} and chain {chain}, minimum={bfactor_min:.3f}, maximum={bfactor_max:.3f}",
        f"set cartoon_transparency, 0.10, {obj}",
        "bg_color white",
        f"set_name {obj}, {obj}_{score_name}",
        "orient",
        "zoom",
        f"png {png_path}, 1800, 1400, dpi=300, ray=1",
    ]
    out_pml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    pdb_path = Path(args.pdb_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.out_prefix:
        out_prefix = Path(args.out_prefix)
    else:
        out_prefix = out_dir / f"{csv_path.stem}_to_{pdb_path.stem}_chain{args.chain}_remapped"

    rows = read_csv_rows(csv_path)
    reference_seq, true_counts, pred_counts, true_freq, pred_freq = compute_hotspot_profiles(
        rows, reference_col=args.reference_col
    )

    residues = extract_chain_residues_from_pdb(pdb_path, chain=args.chain)
    model_type = args.model_type if args.model_type != "auto" else ("rbd" if len(reference_seq) <= 300 else "spike")
    mapping_residues = select_region(residues, args.region_start, args.region_end) if model_type == "rbd" else residues
    pdb_seq = "".join(r["aa"] for r in mapping_residues)
    local_mapping = build_model_to_residue_mapping(reference_seq, pdb_seq)
    full_index = {id(r): i for i, r in enumerate(residues)}
    model_to_pdb = {m: full_index[id(mapping_residues[p])] for m, p in local_mapping.items()}

    if args.value_mode == "count":
        true_vals = [float(v) for v in true_counts]
        pred_vals = [float(v) for v in pred_counts]
    else:
        true_vals = true_freq
        pred_vals = pred_freq
    chosen_scores = score_vector_from_choice(true_vals, pred_vals, args.score)

    score_by_reskey: Dict[Tuple[str, str, str], float] = {}
    for model_pos, pdb_pos in model_to_pdb.items():
        r = residues[pdb_pos]
        score_by_reskey[r["key"]] = chosen_scores[model_pos]

    mapped_csv = Path(f"{out_prefix}_mapping.csv")
    out_pdb = Path(f"{out_prefix}_{args.score}_bfactor.pdb")
    out_pml = Path(f"{out_prefix}_{args.score}.pml")
    png_path = Path(f"{out_prefix}_{args.score}.png")

    write_mapped_csv(
        out_csv=mapped_csv,
        reference_seq=reference_seq,
        true_counts=true_counts,
        pred_counts=pred_counts,
        true_freq=true_freq,
        pred_freq=pred_freq,
        residues=residues,
        model_to_pdb=model_to_pdb,
    )
    write_bfactor_pdb(
        in_pdb=pdb_path,
        out_pdb=out_pdb,
        chain=args.chain,
        score_by_reskey=score_by_reskey,
        bfactor_scale=float(args.bfactor_scale),
    )

    mapped_scores = [chosen_scores[i] for i in model_to_pdb.keys()]
    if mapped_scores:
        bmin = min(mapped_scores) * float(args.bfactor_scale)
        bmax = max(mapped_scores) * float(args.bfactor_scale)
    else:
        bmin = 0.0
        bmax = 1.0
    write_pymol_script(
        out_pml=out_pml,
        out_pdb=out_pdb,
        chain=args.chain,
        score_name=args.score,
        bfactor_min=bmin,
        bfactor_max=bmax,
        png_path=png_path,
    )

    coverage = len(model_to_pdb) / max(len(reference_seq), 1)
    print(f"[ok] CSV hotspot source: {csv_path}")
    print(f"[ok] Structure source: {pdb_path} | chain={args.chain}")
    print(f"[ok] Mapped positions: {len(model_to_pdb)}/{len(reference_seq)} (coverage={coverage:.3f})")
    print(f"[ok] Mapping mode: {model_type}; " + (f"region={args.region_start}-{args.region_end}" if model_type == "rbd" else "region=full-chain"))
    print(f"[ok] Wrote mapping table: {mapped_csv}")
    print(f"[ok] Wrote B-factor PDB: {out_pdb}")
    print(f"[ok] Wrote PyMOL script: {out_pml}")
    print(f"[ok] Value mode: {args.value_mode}")
    print("[hint] In PyMOL: run the .pml file, then ray/save figure from desired view.")


if __name__ == "__main__":
    main()
