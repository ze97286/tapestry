#!/usr/bin/env python3
"""Merge per-sample MethylBERT read-call preprocessing shards."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


FIELDS = ["name", "filename", "dna_seq", "methyl_seq", "ctype", "dmr_ctype", "dmr_label", "non_null_col"]


def reservoir_add(
    reservoirs: dict[str, list[dict[str, str]]],
    seen: dict[str, int],
    row: dict[str, str],
    max_per_label: int,
    rng: random.Random,
) -> None:
    label = row["ctype"]
    seen[label] += 1
    reservoir = reservoirs[label]
    if max_per_label <= 0 or len(reservoir) < max_per_label:
        reservoir.append(row)
        return
    idx = rng.randrange(seen[label])
    if idx < max_per_label:
        reservoir[idx] = row


def read_rows(path: Path):
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            yield row


def split_rows_by_row(
    rows: list[dict[str, str]], split_ratio: float, rng: random.Random
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
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


def split_rows_by_sample(
    rows: list[dict[str, str]], split_ratio: float, rng: random.Random
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    sample_labels: dict[str, str] = {}
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        sample = row["filename"]
        label = row["ctype"]
        if sample in sample_labels and sample_labels[sample] != label:
            raise SystemExit(f"sample {sample!r} has multiple labels: {sample_labels[sample]!r}, {label!r}")
        sample_labels[sample] = label
        by_sample[sample].append(row)

    samples_by_label: dict[str, list[str]] = defaultdict(list)
    for sample, label in sample_labels.items():
        samples_by_label[label].append(sample)

    train_samples: set[str] = set()
    test_samples: set[str] = set()
    for samples in samples_by_label.values():
        rng.shuffle(samples)
        split_at = int(round(len(samples) * split_ratio))
        if len(samples) > 1:
            split_at = min(max(split_at, 1), len(samples) - 1)
        train_samples.update(samples[:split_at])
        test_samples.update(samples[split_at:])

    train = [row for sample in sorted(train_samples) for row in by_sample[sample]]
    test = [row for sample in sorted(test_samples) for row in by_sample[sample]]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def split_rows(
    rows: list[dict[str, str]], split_ratio: float, split_by: str, rng: random.Random
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if split_by == "row":
        return split_rows_by_row(rows, split_ratio, rng)
    if split_by == "sample":
        return split_rows_by_sample(rows, split_ratio, rng)
    raise SystemExit(f"unknown split mode: {split_by}")


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def concat_summaries(shard_dir: Path, output: Path) -> None:
    summaries = sorted(shard_dir.glob("*/read_call_preprocess_summary.tsv"))
    wrote_header = False
    with output.open("w") as out:
        for summary in summaries:
            with summary.open() as handle:
                header = handle.readline()
                if not header:
                    continue
                if not wrote_header:
                    out.write(header)
                    wrote_header = True
                for line in handle:
                    out.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-reads-per-label", type=int, default=500000)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument(
        "--split-by",
        choices=["sample", "row"],
        default="sample",
        help="Split train/test by sample to avoid read-level leakage; use row only for compatibility.",
    )
    parser.add_argument("--seed", type=int, default=950410)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    shard_dir = Path(args.shard_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reservoirs: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)
    row_files = sorted(shard_dir.glob("*/rows.tsv"))
    if not row_files:
        raise SystemExit(f"no shard rows found under {shard_dir}")

    for row_file in row_files:
        for row in read_rows(row_file):
            reservoir_add(reservoirs, seen, row, args.max_reads_per_label, rng)

    rows = [row for label_rows in reservoirs.values() for row in label_rows]
    if not rows:
        raise SystemExit("no rows retained from shards")
    rng.shuffle(rows)
    train, test = split_rows(rows, args.split_ratio, args.split_by, rng)
    write_rows(output_dir / "train_seq.csv", train)
    write_rows(output_dir / "test_seq.csv", test)
    concat_summaries(shard_dir, output_dir / "read_call_preprocess_summary.tsv")

    print(f"read shard rows by label: {dict(seen)}")
    print(f"retained rows by label: { {label: len(rows) for label, rows in reservoirs.items()} }")
    print(f"split_by: {args.split_by}")
    print(f"wrote {len(train)} train reads to {output_dir / 'train_seq.csv'}")
    print(f"wrote {len(test)} test reads to {output_dir / 'test_seq.csv'}")


if __name__ == "__main__":
    main()
