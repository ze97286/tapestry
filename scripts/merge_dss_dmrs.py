#!/usr/bin/env python3
"""Merge chromosome-level DSS DMR calls and select the top regions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def score(row: dict[str, str]) -> float:
    raw = row.get("abs_areaStat") or row.get("areaStat") or row.get("stat") or "0"
    try:
        return abs(float(raw))
    except ValueError:
        return 0.0


def read_dmrs(input_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    seen_fields: set[str] = set()

    for path in sorted(input_dir.glob("*/dss_dmrs.tsv")):
        with path.open() as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames:
                continue
            for field in reader.fieldnames:
                if field not in seen_fields:
                    seen_fields.add(field)
                    fieldnames.append(field)
            for row in reader:
                if not row:
                    continue
                row["_source_file"] = str(path)
                rows.append(row)

    return rows, fieldnames


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing per-chromosome dss_dmrs.tsv files")
    parser.add_argument("--output-all", required=True, help="Merged DSS DMR TSV")
    parser.add_argument("--output-top", required=True, help="Top-N DSS DMR TSV")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--target-group", default="T")
    args = parser.parse_args()

    rows, fieldnames = read_dmrs(Path(args.input_dir))
    if not rows:
        raise SystemExit(f"no per-chromosome DMRs found under {args.input_dir}")

    for required in ("abs_areaStat", "ctype", "dmr_id"):
        if required not in fieldnames:
            fieldnames.append(required)

    rows.sort(key=score, reverse=True)
    for idx, row in enumerate(rows):
        row["abs_areaStat"] = str(score(row))
        row["ctype"] = row.get("ctype") or args.target_group
        row["dmr_id"] = str(idx)

    output_all = Path(args.output_all)
    output_top = Path(args.output_top)
    write_tsv(output_all, rows, fieldnames)
    write_tsv(output_top, rows[: args.top_n], fieldnames)

    print(f"wrote {len(rows)} merged DMRs to {output_all}")
    print(f"wrote top {min(args.top_n, len(rows))} DMRs to {output_top}")


if __name__ == "__main__":
    main()
