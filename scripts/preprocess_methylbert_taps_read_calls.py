#!/usr/bin/env python3
"""Create MethylBERT fine-tuning tables from TAPS per-read call BEDs.

TAPS convention is explicit here: ``mod_cps`` are methylated CpGs and
``unmod_cpgs`` are unmethylated CpGs.  Read-call coordinates are BED-style hg38
0-based half-open intervals.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import pandas as pd


@dataclass
class Dmr:
    chrom: str
    start0: int
    end0: int
    dmr_id: str
    ctype: str


class IndexedFasta:
    def __init__(self, fasta_path: Path):
        self.fasta_path = fasta_path
        self.index_path = Path(f"{fasta_path}.fai")
        self.index: dict[str, tuple[int, int, int, int]] = {}
        with self.index_path.open() as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 5:
                    self.index[parts[0]] = tuple(int(value) for value in parts[1:5])

    def resolve_chrom(self, chrom: str) -> str:
        if chrom in self.index:
            return chrom
        alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
        if alt in self.index:
            return alt
        return chrom

    def fetch0(self, chrom: str, start0: int, end0: int) -> str:
        chrom = self.resolve_chrom(chrom)
        if chrom not in self.index:
            raise KeyError(f"{chrom} not found in FASTA index {self.index_path}")
        length, offset, line_bases, line_width = self.index[chrom]
        start0 = max(start0, 0)
        end0 = min(end0, length)
        if end0 <= start0:
            return ""
        first_byte = offset + (start0 // line_bases) * line_width + (start0 % line_bases)
        last_base = end0 - 1
        last_byte = offset + (last_base // line_bases) * line_width + (last_base % line_bases)
        with self.fasta_path.open("rb") as handle:
            handle.seek(first_byte)
            raw = handle.read(last_byte - first_byte + 1 + line_width)
        return raw.decode("ascii").replace("\n", "").replace("\r", "")[: end0 - start0].upper()


def open_text(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def parse_offsets(value: object) -> list[int]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    offsets = []
    for item in text.split(","):
        item = item.strip()
        if item:
            offsets.append(int(item))
    return offsets


def read_list(path: Path, label: str) -> list[tuple[Path, str]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append((Path(line.split()[0]), label))
    return rows


def read_sample_sheet(path: Path) -> list[tuple[Path, str]]:
    rows = []
    with path.open() as handle:
        reader = csv.reader(handle, delimiter="\t")
        for parts in reader:
            if not parts or not parts[0] or parts[0].startswith("#"):
                continue
            if len(parts) < 2:
                raise SystemExit(f"sample sheet row needs call path and label: {parts!r}")
            rows.append((Path(parts[0]), parts[1]))
    return rows


def read_dmrs(path: Path, top_n: int, start_base: int) -> list[Dmr]:
    dmrs = []
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise SystemExit(f"{path} has no header")
        missing = {"chr", "start", "end"} - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"{path} missing required columns: {sorted(missing)}")
        for idx, row in enumerate(reader):
            if top_n > 0 and len(dmrs) >= top_n:
                break
            start = int(float(row["start"]))
            end = int(float(row["end"]))
            start0 = start - 1 if start_base == 1 else start
            if start0 < 0:
                start0 = 0
            if end <= start0:
                continue
            dmrs.append(Dmr(row["chr"], start0, end, row.get("dmr_id") or str(idx), row.get("ctype") or "T"))
    if not dmrs:
        raise SystemExit(f"no DMRs loaded from {path}")
    return dmrs


def build_dmr_index(dmrs: list[Dmr]) -> dict[str, list[Dmr]]:
    by_chrom: dict[str, list[Dmr]] = defaultdict(list)
    for dmr in dmrs:
        by_chrom[dmr.chrom].append(dmr)
    for chrom in by_chrom:
        by_chrom[chrom].sort(key=lambda item: (item.start0, item.end0))
    return by_chrom


def matching_dmrs(dmrs: list[Dmr], start0: int, end0: int, mode: str) -> list[Dmr]:
    if mode == "contained":
        return [dmr for dmr in dmrs if dmr.start0 <= start0 and end0 <= dmr.end0]
    return [dmr for dmr in dmrs if start0 < dmr.end0 and end0 > dmr.start0]


def make_tokens(seq: str, methyl: list[int], k: int) -> tuple[str, str] | None:
    if len(seq) != len(methyl) or len(seq) <= k:
        return None
    mid = k // 2
    dna_tokens = []
    methyl_tokens = []
    for idx in range(len(seq) - k):
        token = seq[idx : idx + k].upper()
        token = "".join(base if base in "ACGT" else "N" for base in token)
        dna_tokens.append(token)
        methyl_tokens.append(str(methyl[idx + mid]))
    if not dna_tokens:
        return None
    return " ".join(dna_tokens), "".join(methyl_tokens)


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["name", "filename", "dna_seq", "methyl_seq", "ctype", "dmr_ctype", "dmr_label", "non_null_col"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def cap_rows_per_label(rows: list[dict[str, str]], max_reads: int, rng: random.Random) -> list[dict[str, str]]:
    if max_reads <= 0:
        return rows
    capped = []
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["ctype"]].append(row)
    for label, label_rows in by_label.items():
        if len(label_rows) > max_reads:
            print(f"downsampling label {label} from {len(label_rows)} to {max_reads} reads")
            capped.extend(rng.sample(label_rows, max_reads))
        else:
            capped.extend(label_rows)
    rng.shuffle(capped)
    return capped


def split_rows(rows: list[dict[str, str]], split_ratio: float, rng: random.Random) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["ctype"]].append(row)
    train = []
    test = []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        split_at = int(round(len(label_rows) * split_ratio))
        if len(label_rows) > 1:
            split_at = min(max(split_at, 1), len(label_rows) - 1)
        train.extend(label_rows[:split_at])
        test.extend(label_rows[split_at:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def process_file(
    path: Path,
    label: str,
    dmrs_by_chrom: dict[str, list[Dmr]],
    fasta: IndexedFasta,
    mode: str,
    min_informative: int,
    max_reads_per_sample: int,
    stop_after_output_rows_per_sample: int,
    include_snp_cpgs: bool,
    k: int,
    rng: random.Random,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows = []
    stats: dict[str, int] = defaultdict(int)
    sample = path.name.replace(".per-read.bed.gz", "").replace(".per-read.bed", "").replace(".gz", "")
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames and "#chr" in reader.fieldnames:
            reader.fieldnames = ["chr" if field == "#chr" else field for field in reader.fieldnames]
        required = {"chr", "start", "end", "read_id", "orientation", "read_length", "mod_cps", "unmod_cpgs", "snp_cpgs"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} missing columns: {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            stats["input_rows"] += 1
            chrom = row["chr"]
            start0 = int(row["start"])
            end0 = int(row["end"])
            read_length = int(row["read_length"])
            if end0 - start0 != read_length:
                stats["span_read_length_mismatch"] += 1
                continue
            matched = matching_dmrs(dmrs_by_chrom.get(chrom, []), start0, end0, mode)
            if not matched:
                continue
            mod_offsets = parse_offsets(row.get("mod_cps", ""))
            unmod_offsets = parse_offsets(row.get("unmod_cpgs", ""))
            snp_offsets = parse_offsets(row.get("snp_cpgs", "")) if include_snp_cpgs else []
            informative = len(mod_offsets) + len(unmod_offsets) + len(snp_offsets)
            if informative < min_informative:
                stats["too_few_informative"] += len(matched)
                continue
            if any(offset < 0 or offset >= read_length for offset in mod_offsets + unmod_offsets + snp_offsets):
                stats["offset_out_of_read"] += 1
                continue
            seq = fasta.fetch0(chrom, start0, end0)
            if len(seq) != read_length:
                stats["reference_length_mismatch"] += 1
                continue
            methyl = [2] * len(seq)
            for offset in unmod_offsets:
                methyl[offset] = 0
            for offset in mod_offsets:
                methyl[offset] = 1
            for offset in snp_offsets:
                methyl[offset] = 1
            tokens = make_tokens(seq, methyl, k)
            if tokens is None:
                stats["token_failure"] += 1
                continue
            dna_seq, methyl_seq = tokens
            for dmr in matched:
                rows.append(
                    {
                        "name": f"{sample}:{line_no}:{dmr.dmr_id}",
                        "filename": sample,
                        "dna_seq": dna_seq,
                        "methyl_seq": methyl_seq,
                        "ctype": label,
                        "dmr_ctype": dmr.ctype,
                        "dmr_label": str(dmr.dmr_id),
                        "non_null_col": "",
                    }
                )
                stats["output_rows"] += 1
            if stop_after_output_rows_per_sample > 0 and len(rows) >= stop_after_output_rows_per_sample:
                stats["stopped_after_output_rows"] = 1
                break
    if max_reads_per_sample > 0 and len(rows) > max_reads_per_sample:
        rows = rng.sample(rows, max_reads_per_sample)
    return rows, dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sheet")
    parser.add_argument("--tumour-list")
    parser.add_argument("--normal-list")
    parser.add_argument("--dmrs", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--dmr-start-base", type=int, choices=[0, 1], default=1)
    parser.add_argument("--mode", choices=["contained", "overlap"], default="overlap")
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--max-reads-per-sample", type=int, default=200000)
    parser.add_argument(
        "--stop-after-output-rows-per-sample",
        type=int,
        default=0,
        help="Stop streaming each input file after this many generated rows; useful for quick smoke tests.",
    )
    parser.add_argument("--max-reads-per-label", type=int, default=500000)
    parser.add_argument("--min-informative", type=int, default=2)
    parser.add_argument("--include-snp-cpgs", action="store_true")
    parser.add_argument("--seed", type=int, default=950410)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    samples: list[tuple[Path, str]] = []
    if args.sample_sheet:
        samples.extend(read_sample_sheet(Path(args.sample_sheet)))
    if args.tumour_list:
        samples.extend(read_list(Path(args.tumour_list), "T"))
    if args.normal_list:
        samples.extend(read_list(Path(args.normal_list), "N"))
    if not samples:
        raise SystemExit("provide --sample-sheet or per-label lists")

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dmrs = read_dmrs(Path(args.dmrs), args.top_n, args.dmr_start_base)
    dmrs_by_chrom = build_dmr_index(dmrs)
    fasta = IndexedFasta(Path(args.reference))

    all_rows = []
    summary = []
    for path, label in samples:
        rows, stats = process_file(
            path=path,
            label=label,
            dmrs_by_chrom=dmrs_by_chrom,
            fasta=fasta,
            mode=args.mode,
            min_informative=args.min_informative,
            max_reads_per_sample=args.max_reads_per_sample,
            stop_after_output_rows_per_sample=args.stop_after_output_rows_per_sample,
            include_snp_cpgs=args.include_snp_cpgs,
            k=args.k,
            rng=rng,
        )
        all_rows.extend(rows)
        stats.update({"path": str(path), "label": label, "sampled_rows": len(rows)})
        summary.append(stats)
        print(f"{path}: label={label} rows={len(rows)}")

    if not all_rows:
        raise SystemExit("no rows generated")
    all_rows = cap_rows_per_label(all_rows, args.max_reads_per_label, rng)
    train, test = split_rows(all_rows, args.split_ratio, rng)
    write_rows(output_dir / "train_seq.csv", train)
    write_rows(output_dir / "test_seq.csv", test)
    with (output_dir / "read_call_preprocess_summary.tsv").open("w", newline="") as handle:
        fields = sorted({key for row in summary for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {len(train)} train reads to {output_dir / 'train_seq.csv'}")
    print(f"wrote {len(test)} test reads to {output_dir / 'test_seq.csv'}")


if __name__ == "__main__":
    main()
