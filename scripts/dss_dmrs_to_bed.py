#!/usr/bin/env python3
"""Convert a DSS DMR TSV to BED intervals for region filtering."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="DSS DMR TSV with chr/start/end columns")
    parser.add_argument("--output", required=True, help="Output BED")
    parser.add_argument(
        "--start-base",
        type=int,
        choices=[0, 1],
        default=1,
        help="Coordinate base of the DSS start column. Default: 1, matching the hg38 CpG index used here.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    with input_path.open() as inp, output_path.open("w") as out:
        reader = csv.DictReader(inp, delimiter="\t")
        missing = {"chr", "start", "end"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{input_path} is missing required columns: {sorted(missing)}")
        for row in reader:
            chrom = row["chr"]
            start = int(float(row["start"]))
            end = int(float(row["end"]))
            bed_start = start - 1 if args.start_base == 1 else start
            if bed_start < 0:
                bed_start = 0
            if end <= bed_start:
                continue
            name = row.get("dmr_id") or f"dmr_{n_rows}"
            out.write(f"{chrom}\t{bed_start}\t{end}\t{name}\n")
            n_rows += 1

    print(f"wrote {n_rows} BED intervals to {output_path}")


if __name__ == "__main__":
    main()
