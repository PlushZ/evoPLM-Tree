#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Stop": "*",
}


def aa3_to_aa1(token: str) -> str:
    return AA3_TO_1.get(token, token)


def parse_hgvs_pro(value: str):
    s = str(value).strip()
    if s.startswith("p."):
        s = s[2:]

    # One-letter notation, e.g. N501Y
    m1 = re.fullmatch(r"([A-Z\*])(\d+)([A-Z\*])", s)
    if m1:
        return int(m1.group(2)), m1.group(1), m1.group(3)

    # Three-letter notation, e.g. Asn501Tyr
    m3 = re.fullmatch(r"([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter|Stop)", s)
    if m3:
        wt = aa3_to_aa1(m3.group(1))
        mut = aa3_to_aa1(m3.group(3))
        return int(m3.group(2)), wt, mut

    return None


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--in_csv", default="data/dms/urn_mavedb_00000670-a-1_scores.csv")
    ap.add_argument("--out_csv", default="results/dms_validation/dms_scores/urn_mavedb_00000670-a-1_table.csv")
    ap.add_argument("--hgvs_col", default="hgvs_pro")
    ap.add_argument("--score_col", default="score")
    args = ap.parse_args()

    records = []
    n_rows = 0
    skipped = 0

    with open(args.in_csv, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or args.hgvs_col not in reader.fieldnames:
            raise KeyError(f"Missing HGVS column: {args.hgvs_col}")
        if args.score_col not in reader.fieldnames:
            raise KeyError(f"Missing score column: {args.score_col}")

        for row in reader:
            n_rows += 1
            parsed = parse_hgvs_pro(row[args.hgvs_col])
            if parsed is None:
                skipped += 1
                continue
            site, wt, mut = parsed
            records.append(
                {
                    "site": int(site),
                    "wt": wt,
                    "mut": mut,
                    "dms_score": row[args.score_col],
                }
            )

    records.sort(key=lambda r: (r["site"], r["wt"], r["mut"]))
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["site", "wt", "mut", "dms_score"])
        writer.writeheader()
        writer.writerows(records)

    print(f"[INFO] input rows: {n_rows}")
    print(f"[INFO] parsed rows: {len(records)}")
    print(f"[INFO] skipped rows: {skipped}")
    print(f"[INFO] wrote: {out_path}")


if __name__ == "__main__":
    main()
