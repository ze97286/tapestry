#!/usr/bin/env python3
"""Write a minimal samtools-compatible FASTA .fai index."""

from __future__ import annotations

import argparse
from pathlib import Path


def write_record(handle, name: str | None, length: int, offset: int | None, line_bases: int | None, line_width: int | None) -> None:
    if name is None:
        return
    if offset is None or line_bases is None or line_width is None:
        raise SystemExit(f"sequence {name!r} has no sequence lines")
    handle.write(f"{name}\t{length}\t{offset}\t{line_bases}\t{line_width}\n")


def index_fasta(fasta: Path, output: Path) -> None:
    name: str | None = None
    length = 0
    offset: int | None = None
    line_bases: int | None = None
    line_width: int | None = None

    with fasta.open("rb") as src, output.open("w") as dst:
        while True:
            line_start = src.tell()
            line = src.readline()
            if not line:
                break

            if line.startswith(b">"):
                write_record(dst, name, length, offset, line_bases, line_width)
                name = line[1:].strip().split()[0].decode("ascii")
                length = 0
                offset = src.tell()
                line_bases = None
                line_width = None
                continue

            seq = line.rstrip(b"\r\n")
            if not seq:
                continue
            if name is None:
                raise SystemExit(f"sequence data before first FASTA header at byte {line_start}")
            if line_bases is None:
                line_bases = len(seq)
                line_width = len(line)
            length += len(seq)

        write_record(dst, name, length, offset, line_bases, line_width)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    fasta = Path(args.fasta)
    output = Path(args.output) if args.output else Path(f"{fasta}.fai")
    index_fasta(fasta, output)


if __name__ == "__main__":
    main()
