#!/usr/bin/env python3
"""Generate random genomic regions matched to a DMR set (count, width, chromosome).

DMR-shuffle control for MethylBERT. Feeding these in place of the real DSS DMRs tests
whether the read classifier relies on the specific tumour-specific methylation regions
or on any region carrying a tissue-vs-plasma coverage/length difference. If accuracy on
random regions matches accuracy on the real DMRs, the signal is generic source/coverage,
not DMR methylation.

Output matches the DMR TSV schema the preprocessor expects (chr/start/end/ctype/dmr_id).
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def read_fai(path: Path) -> dict[str, int]:
    lengths: dict[str, int] = {}
    with path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                lengths[parts[0]] = int(parts[1])
    return lengths


def read_template(path: Path) -> list[tuple[str, int]]:
    """Return (chrom, width) per region from a DMR TSV with chr/start/end columns."""
    regions: list[tuple[str, int]] = []
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = {"chr", "start", "end"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            width = int(float(row["end"])) - int(float(row["start"]))
            if width > 0:
                regions.append((row["chr"], width))
    if not regions:
        raise SystemExit(f"no usable regions in {path}")
    return regions


def resolve_length(lengths: dict[str, int], chrom: str) -> int | None:
    if chrom in lengths:
        return lengths[chrom]
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    return lengths.get(alt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--like", required=True, help="Real DMR TSV to match (chr/start/end)")
    parser.add_argument("--fai", required=True, help="Reference .fai for chromosome lengths")
    parser.add_argument("--output", required=True, help="Output random-region TSV")
    parser.add_argument("--ctype", default="T", help="Positive-class label for the regions")
    parser.add_argument("--seed", type=int, default=950410)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    lengths = read_fai(Path(args.fai))
    template = read_template(Path(args.like))

    rows = []
    skipped = 0
    for idx, (chrom, width) in enumerate(template):
        chrom_len = resolve_length(lengths, chrom)
        if chrom_len is None or chrom_len <= width:
            skipped += 1
            continue
        start = rng.randint(1, chrom_len - width)
        rows.append(
            {"chr": chrom, "start": start, "end": start + width, "ctype": args.ctype, "dmr_id": idx}
        )

    if not rows:
        raise SystemExit("no random regions generated")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["chr", "start", "end", "ctype", "dmr_id"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)
    suffix = f" (skipped {skipped} regions with no fitting chromosome)" if skipped else ""
    print(f"wrote {len(rows)} random regions to {output}{suffix}")


if __name__ == "__main__":
    main()
