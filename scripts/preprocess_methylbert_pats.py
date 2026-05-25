#!/usr/bin/env python3
"""Create upstream MethylBERT fine-tuning tables from PAT files.

The upstream MethylBERT preprocessor expects methylation-tagged BAMs.  The OAC
inputs available here are PAT files, so this script reconstructs the equivalent
fine-tuning rows from PAT read patterns, the CpG index, and the reference FASTA.
It writes the same tab-separated files consumed by MethylBERT fine-tuning:

    train_seq.csv
    test_seq.csv
    dmrs.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterable


@dataclass
class Dmr:
    chrom: str
    start: int
    end: int
    dmr_id: str
    ctype: str
    cpg_start: int | None = None
    cpg_end: int | None = None


@dataclass
class Candidate:
    weight: int
    row: dict[str, str]


class IndexedFasta:
    def __init__(self, fasta_path: Path):
        self.fasta_path = fasta_path
        self.index_path = Path(f"{fasta_path}.fai")
        self.index: dict[str, tuple[int, int, int, int]] = {}
        self.loaded: dict[str, str] = {}
        if self.index_path.exists():
            with self.index_path.open() as handle:
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 5:
                        continue
                    name = parts[0]
                    self.index[name] = tuple(int(value) for value in parts[1:5])

    def resolve_chrom(self, chrom: str) -> str:
        if chrom in self.index or chrom in self.loaded:
            return chrom
        if chrom.startswith("chr"):
            alt = chrom[3:]
        else:
            alt = f"chr{chrom}"
        if alt in self.index or alt in self.loaded:
            return alt
        return chrom

    def fetch(self, chrom: str, start_1: int, end_1: int) -> str:
        if end_1 < start_1:
            return ""
        chrom = self.resolve_chrom(chrom)
        if self.index:
            if chrom not in self.index:
                raise KeyError(f"{chrom} not found in FASTA index {self.index_path}")
            return self._fetch_indexed(chrom, start_1, end_1)
        if not self.loaded:
            self._load_all()
            chrom = self.resolve_chrom(chrom)
        if chrom not in self.loaded:
            raise KeyError(f"{chrom} not found in FASTA {self.fasta_path}")
        seq = self.loaded[chrom]
        start_0 = max(start_1 - 1, 0)
        end_0 = min(end_1, len(seq))
        return seq[start_0:end_0]

    def _fetch_indexed(self, chrom: str, start_1: int, end_1: int) -> str:
        length, offset, line_bases, line_width = self.index[chrom]
        start_0 = max(start_1 - 1, 0)
        end_0 = min(end_1, length)
        if end_0 <= start_0:
            return ""
        first_byte = offset + (start_0 // line_bases) * line_width + (start_0 % line_bases)
        last_base = end_0 - 1
        last_byte = offset + (last_base // line_bases) * line_width + (last_base % line_bases)
        n_bytes = last_byte - first_byte + 1
        with self.fasta_path.open("rb") as handle:
            handle.seek(first_byte)
            raw = handle.read(n_bytes + line_width)
        return raw.decode("ascii").replace("\n", "").replace("\r", "")[: end_0 - start_0].upper()

    def _load_all(self) -> None:
        current = None
        chunks: list[str] = []
        with self.fasta_path.open() as handle:
            for line in handle:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    if current is not None:
                        self.loaded[current] = "".join(chunks).upper()
                    current = line[1:].split()[0]
                    chunks = []
                elif current is not None:
                    chunks.append(line)
        if current is not None:
            self.loaded[current] = "".join(chunks).upper()


def open_text(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def read_list(path: Path, label: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append((line.split()[0], label))
    return rows


def read_sample_sheet(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open() as handle:
        reader = csv.reader(handle, delimiter="\t")
        for parts in reader:
            if not parts or not parts[0] or parts[0].startswith("#"):
                continue
            if len(parts) < 2:
                raise SystemExit(f"sample sheet row needs PAT path and label: {parts!r}")
            rows.append((parts[0], parts[1]))
    return rows


def read_dmrs(path: Path, top_n: int) -> list[Dmr]:
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise SystemExit(f"{path} has no header")
        missing = {"chr", "start", "end"} - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
        dmrs: list[Dmr] = []
        for idx, row in enumerate(reader):
            if top_n > 0 and len(dmrs) >= top_n:
                break
            start = int(float(row["start"]))
            end = int(float(row["end"]))
            if end < start:
                continue
            dmrs.append(
                Dmr(
                    chrom=row["chr"],
                    start=start,
                    end=end,
                    dmr_id=row.get("dmr_id") or str(idx),
                    ctype=row.get("ctype") or "T",
                )
            )
    if not dmrs:
        raise SystemExit(f"no DMRs loaded from {path}")
    return dmrs


def write_dmrs(path: Path, dmrs: list[Dmr]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["chr", "start", "end", "dmr_id", "ctype", "cpg_start", "cpg_end"])
        for dmr in dmrs:
            writer.writerow([dmr.chrom, dmr.start, dmr.end, dmr.dmr_id, dmr.ctype, dmr.cpg_start, dmr.cpg_end])


def annotate_dmrs_with_cpgs(
    dmrs: list[Dmr],
    cpg_path: Path,
    cpg_position_base: int,
) -> dict[tuple[str, int], int]:
    dmrs_by_chrom: dict[str, list[Dmr]] = defaultdict(list)
    for dmr in dmrs:
        dmrs_by_chrom[dmr.chrom].append(dmr)

    cpg_positions: dict[tuple[str, int], int] = {}
    with open_text(cpg_path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise SystemExit(f"CpG row has fewer than 3 columns: {line[:120]!r}")
            chrom = parts[0]
            try:
                pos = int(parts[1])
                cpg_idx = int(parts[2])
            except ValueError:
                continue
            pos_1 = pos + 1 if cpg_position_base == 0 else pos
            for dmr in dmrs_by_chrom.get(chrom, []):
                if dmr.start <= pos_1 <= dmr.end:
                    cpg_positions[(chrom, cpg_idx)] = pos_1
                    dmr.cpg_start = cpg_idx if dmr.cpg_start is None else min(dmr.cpg_start, cpg_idx)
                    dmr.cpg_end = cpg_idx if dmr.cpg_end is None else max(dmr.cpg_end, cpg_idx)

    missing = [dmr.dmr_id for dmr in dmrs if dmr.cpg_start is None or dmr.cpg_end is None]
    if missing:
        raise SystemExit(f"{len(missing)} DMRs had no CpGs in {cpg_path}; first missing IDs: {missing[:5]}")
    return cpg_positions


def build_dmr_lookup(dmrs: list[Dmr]) -> dict[str, list[Dmr]]:
    lookup: dict[str, list[Dmr]] = defaultdict(list)
    for dmr in dmrs:
        lookup[dmr.chrom].append(dmr)
    for chrom in lookup:
        lookup[chrom].sort(key=lambda dmr: (dmr.cpg_start or -1, dmr.cpg_end or -1))
    return lookup


def find_containing_dmr(dmrs: list[Dmr], start_cpg: int, end_cpg: int) -> Dmr | None:
    for dmr in dmrs:
        if dmr.cpg_start is None or dmr.cpg_end is None:
            continue
        if dmr.cpg_start <= start_cpg and end_cpg <= dmr.cpg_end:
            return dmr
    return None


def make_tokens(seq: str, methyl: list[int], k: int) -> tuple[str, str] | None:
    if k % 2 == 0:
        raise ValueError("k must be odd")
    if len(seq) != len(methyl) or len(seq) <= k:
        return None
    mid = k // 2
    dna_tokens: list[str] = []
    methyl_tokens: list[str] = []
    # Match upstream MethylBERT's range(len(seq) - k) behavior.
    for idx in range(len(seq) - k):
        token = seq[idx : idx + k].upper()
        if any(base not in "ACGTN" for base in token):
            token = "".join(base if base in "ACGT" else "N" for base in token)
        dna_tokens.append(token)
        methyl_tokens.append(str(methyl[idx + mid]))
    if not dna_tokens or all(value == "2" for value in methyl_tokens):
        return None
    return " ".join(dna_tokens), "".join(methyl_tokens)


def pat_candidates_for_sample(
    pat_path: Path,
    label: str,
    dmrs_by_chrom: dict[str, list[Dmr]],
    cpg_positions: dict[tuple[str, int], int],
    fasta: IndexedFasta,
    methylated_char: str,
    unmethylated_char: str,
    k: int,
    flank: int,
    min_cpgs: int,
    max_seq_bases: int,
) -> tuple[list[Candidate], dict[str, int]]:
    candidates: list[Candidate] = []
    stats = defaultdict(int)
    sample = pat_path.name.replace(".pat.gz", "").replace(".pat", "")
    with open_text(pat_path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                raise SystemExit(f"PAT row has fewer than 4 columns in {pat_path}: {line[:120]!r}")
            chrom, start_raw, pattern, count_raw = parts[:4]
            start_cpg = int(start_raw)
            count = int(count_raw)
            if count <= 0:
                continue
            stats["pat_rows"] += 1
            informative = sum(1 for state in pattern if state in (methylated_char, unmethylated_char))
            if informative < min_cpgs:
                continue
            end_cpg = start_cpg + len(pattern) - 1
            dmr = find_containing_dmr(dmrs_by_chrom.get(chrom, []), start_cpg, end_cpg)
            if dmr is None:
                continue
            first_pos = cpg_positions.get((chrom, start_cpg))
            last_pos = cpg_positions.get((chrom, end_cpg))
            if first_pos is None or last_pos is None:
                stats["missing_cpg_positions"] += 1
                continue
            seq_start = max(first_pos - flank, 1)
            seq_end = last_pos + 1 + flank
            if max_seq_bases > 0 and seq_end - seq_start + 1 > max_seq_bases:
                stats["too_long"] += 1
                continue
            seq = fasta.fetch(chrom, seq_start, seq_end)
            if not seq:
                stats["missing_ref"] += 1
                continue
            methyl = [2] * len(seq)
            for offset, state in enumerate(pattern):
                if state not in (methylated_char, unmethylated_char):
                    continue
                cpg_idx = start_cpg + offset
                pos = cpg_positions.get((chrom, cpg_idx))
                if pos is None:
                    continue
                seq_offset = pos - seq_start
                if 0 <= seq_offset < len(methyl):
                    methyl[seq_offset] = 1 if state == methylated_char else 0
            tokens = make_tokens(seq, methyl, k)
            if tokens is None:
                continue
            dna_seq, methyl_seq = tokens
            row_idx = len(candidates)
            candidates.append(
                Candidate(
                    weight=count,
                    row={
                        "name": f"{sample}:{row_idx}",
                        "filename": sample,
                        "dna_seq": dna_seq,
                        "methyl_seq": methyl_seq,
                        "ctype": label,
                        "dmr_ctype": dmr.ctype,
                        "dmr_label": str(dmr.dmr_id),
                        "non_null_col": "",
                    },
                )
            )
            stats["candidate_rows"] += 1
            stats["candidate_reads"] += count
    return candidates, dict(stats)


def expand_or_sample(candidates: list[Candidate], max_reads: int, rng: random.Random) -> list[dict[str, str]]:
    total = sum(candidate.weight for candidate in candidates)
    if not candidates or total == 0:
        return []
    if max_reads > 0 and total > max_reads:
        selected = rng.choices(candidates, weights=[candidate.weight for candidate in candidates], k=max_reads)
        rows = []
        for idx, candidate in enumerate(selected):
            row = dict(candidate.row)
            row["name"] = f"{row['name']}:{idx}"
            rows.append(row)
        return rows
    rows = []
    for candidate in candidates:
        for idx in range(candidate.weight):
            row = dict(candidate.row)
            row["name"] = f"{row['name']}:{idx}"
            rows.append(row)
    rng.shuffle(rows)
    return rows


def split_rows(rows: list[dict[str, str]], split_ratio: float, rng: random.Random) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["ctype"]].append(row)
    train: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
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


def cap_rows_per_label(rows: list[dict[str, str]], max_reads: int, rng: random.Random) -> list[dict[str, str]]:
    if max_reads <= 0:
        return rows
    capped: list[dict[str, str]] = []
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


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["name", "filename", "dna_seq", "methyl_seq", "ctype", "dmr_ctype", "dmr_label", "non_null_col"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sheet", help="TSV with PAT path and label columns")
    parser.add_argument("--tumour-pat-list", help="PAT list labelled T")
    parser.add_argument("--normal-pat-list", help="PAT list labelled N")
    parser.add_argument("--dmrs", required=True, help="Selected DMR TSV")
    parser.add_argument("--cpg-file", required=True, help="CpG index: chr, position, cpg_index")
    parser.add_argument("--reference", required=True, help="Reference FASTA, preferably with .fai")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--max-reads-per-sample", type=int, default=200000)
    parser.add_argument("--max-reads-per-label", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=950410)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--flank", type=int, default=1)
    parser.add_argument("--min-cpgs", type=int, default=1)
    parser.add_argument("--max-seq-bases", type=int, default=500)
    parser.add_argument("--methylated-char", default="C")
    parser.add_argument("--unmethylated-char", default="T")
    parser.add_argument("--cpg-position-base", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    if not 0 < args.split_ratio < 1:
        raise SystemExit("--split-ratio must be between 0 and 1")

    samples: list[tuple[str, str]] = []
    if args.sample_sheet:
        samples.extend(read_sample_sheet(Path(args.sample_sheet)))
    if args.tumour_pat_list:
        samples.extend(read_list(Path(args.tumour_pat_list), "T"))
    if args.normal_pat_list:
        samples.extend(read_list(Path(args.normal_pat_list), "N"))
    if not samples:
        raise SystemExit("provide --sample-sheet or PAT list arguments")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    dmrs = read_dmrs(Path(args.dmrs), args.top_n)
    cpg_positions = annotate_dmrs_with_cpgs(dmrs, Path(args.cpg_file), args.cpg_position_base)
    dmrs_by_chrom = build_dmr_lookup(dmrs)
    fasta = IndexedFasta(Path(args.reference))

    all_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, object]] = []
    for pat_raw, label in samples:
        pat_path = Path(pat_raw)
        candidates, stats = pat_candidates_for_sample(
            pat_path=pat_path,
            label=label,
            dmrs_by_chrom=dmrs_by_chrom,
            cpg_positions=cpg_positions,
            fasta=fasta,
            methylated_char=args.methylated_char,
            unmethylated_char=args.unmethylated_char,
            k=args.k,
            flank=args.flank,
            min_cpgs=args.min_cpgs,
            max_seq_bases=args.max_seq_bases,
        )
        rows = expand_or_sample(candidates, args.max_reads_per_sample, rng)
        all_rows.extend(rows)
        summary = {
            "pat": str(pat_path),
            "label": label,
            "sampled_reads": len(rows),
        }
        summary.update(stats)
        summary_rows.append(summary)
        print(f"{pat_path}: candidates={stats.get('candidate_rows', 0)} sampled_reads={len(rows)}")

    if not all_rows:
        raise SystemExit("no MethylBERT rows generated from PAT inputs")
    all_rows = cap_rows_per_label(all_rows, args.max_reads_per_label, rng)
    labels = {row["ctype"] for row in all_rows}
    if len(labels) < 2:
        raise SystemExit(f"need at least two labels for fine-tuning; observed {sorted(labels)}")

    train, test = split_rows(all_rows, args.split_ratio, rng)
    if not train or not test:
        raise SystemExit("train/test split produced an empty set")
    write_rows(output_dir / "train_seq.csv", train)
    write_rows(output_dir / "test_seq.csv", test)
    write_dmrs(output_dir / "dmrs.csv", dmrs)
    write_summary(output_dir / "pat_preprocess_summary.tsv", summary_rows)
    print(f"wrote {len(train)} train reads to {output_dir / 'train_seq.csv'}")
    print(f"wrote {len(test)} test reads to {output_dir / 'test_seq.csv'}")
    print(f"wrote DMRs to {output_dir / 'dmrs.csv'}")


if __name__ == "__main__":
    main()
