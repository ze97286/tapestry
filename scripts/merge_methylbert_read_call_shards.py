#!/usr/bin/env python3
"""Merge per-sample MethylBERT read-call preprocessing shards."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


FIELDS = [
    "name",
    "filename",
    "dna_seq",
    "methyl_seq",
    "ctype",
    "dmr_ctype",
    "dmr_label",
    "non_null_col",
    "read_length",
    "n_cpg",
]


def reservoir_add(
    reservoirs: dict[str, list[dict[str, str]]],
    seen: dict[str, int],
    row: dict[str, str],
    key: str,
    max_per_key: int,
    rng: random.Random,
) -> None:
    seen[key] += 1
    reservoir = reservoirs[key]
    if max_per_key <= 0 or len(reservoir) < max_per_key:
        reservoir.append(row)
        return
    idx = rng.randrange(seen[key])
    if idx < max_per_key:
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


def sample_cohort(sample: str) -> str:
    """Map a sample/filename to its source cohort (used for leave-one-batch-out)."""
    if "_tumour" in sample:
        return "tumour_tissue"
    if "_Ctrl_plasma" in sample:
        return "AB_plasma"
    if sample.startswith(("GI", "SCAN")):
        return "CD_plasma"
    return "other"


def shuffle_labels_by_sample(
    rows: list[dict[str, str]], rng: random.Random
) -> list[dict[str, str]]:
    """Permutation control: reassign the per-sample ctype labels at random while
    preserving how many samples carry each label. Every read of a sample keeps a
    single (permuted) label, so the by-sample split stays consistent. If held-out
    accuracy stays high under this shuffle, the head is reading per-sample identity
    or batch, not biology."""
    sample_label: dict[str, str] = {}
    for row in rows:
        sample_label[row["filename"]] = row["ctype"]
    samples = list(sample_label)
    labels = [sample_label[s] for s in samples]
    rng.shuffle(labels)
    permuted = dict(zip(samples, labels))
    for row in rows:
        row["ctype"] = permuted[row["filename"]]
    return rows


def length_match(
    rows: list[dict[str, str]], rng: random.Random, bin_bp: int
) -> list[dict[str, str]]:
    """Read-length-matched control: downsample so every read-length bin holds an
    equal number of reads from each class. Removes fragment length as a class proxy.
    Returns rows unchanged if read_length is unavailable."""
    by_label_bin: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    labels: set[str] = set()
    for row in rows:
        labels.add(row["ctype"])
        try:
            length = int(row["read_length"])
        except (KeyError, ValueError):
            print("length-match skipped: rows lack a usable read_length column")
            return rows
        by_label_bin[(row["ctype"], length // bin_bp)].append(row)

    bins = {b for (_, b) in by_label_bin}
    kept: list[dict[str, str]] = []
    for b in bins:
        per_class = [by_label_bin.get((label, b), []) for label in labels]
        smallest = min(len(pool) for pool in per_class)
        if smallest == 0:
            continue  # bin missing in a class -> drop it so marginals stay matched
        for pool in per_class:
            kept.extend(pool if len(pool) <= smallest else rng.sample(pool, smallest))
    rng.shuffle(kept)
    return kept


def filter_read_length(
    rows: list[dict[str, str]], min_length: int | None, max_length: int | None
) -> list[dict[str, str]]:
    """Keep only rows whose original read length falls inside the requested bounds."""
    if min_length is None and max_length is None:
        return rows
    kept: list[dict[str, str]] = []
    missing = 0
    for row in rows:
        try:
            length = int(row["read_length"])
        except (KeyError, ValueError):
            missing += 1
            continue
        if min_length is not None and length < min_length:
            continue
        if max_length is not None and length > max_length:
            continue
        kept.append(row)
    if missing:
        print(f"read-length-filter: dropped {missing} rows without usable read_length")
    return kept


def balance_labels(rows: list[dict[str, str]], rng: random.Random) -> list[dict[str, str]]:
    """Downsample every class label to the smallest retained class size."""
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["ctype"]].append(row)
    sizes = {label: len(label_rows) for label, label_rows in by_label.items()}
    if not sizes:
        return rows
    target = min(sizes.values())
    kept: list[dict[str, str]] = []
    for label_rows in by_label.values():
        kept.extend(label_rows if len(label_rows) <= target else rng.sample(label_rows, target))
    rng.shuffle(kept)
    print(f"balance-labels: class sizes {sizes} -> {target}/label, total {len(kept)}")
    return kept


def split_rows_by_holdout(
    rows: list[dict[str, str]],
    holdout_samples: set[str],
    holdout_cohorts: set[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Leave-one-sample-out / leave-one-batch-out: force the named samples (or whole
    cohorts) into the test set, everything else into train."""
    train: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for row in rows:
        sample = row["filename"]
        if sample in holdout_samples or sample_cohort(sample) in holdout_cohorts:
            test.append(row)
        else:
            train.append(row)
    return train, test


def balance_cohorts(
    rows: list[dict[str, str]], rng: random.Random, max_per_label: int
) -> list[dict[str, str]]:
    """Equalise source cohorts within each label so no single cohort dominates the
    class. When one normal cohort (e.g. CD) supplies far more reads than another
    (e.g. AB), the model learns 'normal = that cohort' and scores the other cohort as
    tumour. This downsamples each cohort within a label to the smallest cohort's read
    count, then caps the label total at max_per_label. NOTE: a tiny 'other' cohort
    would drag the per-label minimum down — check the printed sizes."""
    by_label_cohort: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_label_cohort[row["ctype"]][sample_cohort(row["filename"])].append(row)
    out: list[dict[str, str]] = []
    for label, cohorts in by_label_cohort.items():
        sizes = {c: len(v) for c, v in cohorts.items()}
        target = min(sizes.values())
        balanced: list[dict[str, str]] = []
        for pool in cohorts.values():
            balanced.extend(pool if len(pool) <= target else rng.sample(pool, target))
        if max_per_label > 0 and len(balanced) > max_per_label:
            balanced = rng.sample(balanced, max_per_label)
        out.extend(balanced)
        print(f"balance-cohorts: label {label} cohorts {sizes} -> {target}/cohort, total {len(balanced)}")
    rng.shuffle(out)
    return out


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
    parser.add_argument(
        "--shuffle-labels",
        action="store_true",
        help="Permutation control: randomly reassign per-sample T/N labels (preserving "
        "class sizes) before splitting. Held-out accuracy should fall to chance.",
    )
    parser.add_argument(
        "--length-match",
        action="store_true",
        help="Downsample so each read-length bin holds equal reads per class, removing "
        "fragment length as a class proxy.",
    )
    parser.add_argument(
        "--length-match-bin",
        type=int,
        default=10,
        help="Bin width in bp for --length-match (default: 10).",
    )
    parser.add_argument(
        "--min-read-length",
        type=int,
        default=None,
        help="Optional lower bound on original read_length before splitting/training.",
    )
    parser.add_argument(
        "--max-read-length",
        type=int,
        default=None,
        help="Optional upper bound on original read_length before splitting/training.",
    )
    parser.add_argument(
        "--holdout-samples",
        default="",
        help="Comma-separated sample names forced into the test set (leave-one-sample-out).",
    )
    parser.add_argument(
        "--holdout-cohort",
        default="",
        help="Comma-separated cohorts (tumour_tissue, AB_plasma, CD_plasma) forced into "
        "the test set (leave-one-batch-out).",
    )
    parser.add_argument(
        "--balance-cohorts",
        action="store_true",
        help="Equalise source cohorts within each label (e.g. AB vs CD plasma in the "
        "normal class) by downsampling each cohort to the smallest cohort's read count, "
        "so the model does not learn 'normal = the majority cohort'. Reservoirs per "
        "cohort instead of per label.",
    )
    parser.add_argument(
        "--balance-labels",
        action="store_true",
        help="After optional length filtering/matching, downsample T/N labels to the "
        "same read count before train/test splitting.",
    )
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

    # When balancing cohorts, reservoir per cohort (so a minority cohort like AB is
    # retained up to the same cap as CD); otherwise per label, as before.
    if args.balance_cohorts:
        def reservoir_key(r: dict[str, str]) -> str:
            return sample_cohort(r["filename"])
    else:
        def reservoir_key(r: dict[str, str]) -> str:
            return r["ctype"]

    for row_file in row_files:
        for row in read_rows(row_file):
            reservoir_add(reservoirs, seen, row, reservoir_key(row), args.max_reads_per_label, rng)

    rows = [row for key_rows in reservoirs.values() for row in key_rows]
    if not rows:
        raise SystemExit("no rows retained from shards")
    rng.shuffle(rows)

    if args.balance_cohorts:
        before = len(rows)
        rows = balance_cohorts(rows, rng, args.max_reads_per_label)
        print(f"balance-cohorts: {before} -> {len(rows)} reads")

    if args.shuffle_labels:
        rows = shuffle_labels_by_sample(rows, rng)
    if args.min_read_length is not None or args.max_read_length is not None:
        before = len(rows)
        rows = filter_read_length(rows, args.min_read_length, args.max_read_length)
        print(
            "read-length-filter: "
            f"{before} -> {len(rows)} rows "
            f"(min={args.min_read_length}, max={args.max_read_length})"
        )
        if not rows:
            raise SystemExit("read-length-filter removed all rows")
    if args.length_match:
        before = len(rows)
        rows = length_match(rows, rng, args.length_match_bin)
        print(f"length-match: {before} -> {len(rows)} reads")
    if args.balance_labels:
        before = len(rows)
        rows = balance_labels(rows, rng)
        print(f"balance-labels: {before} -> {len(rows)} reads")

    holdout_samples = {s for s in args.holdout_samples.split(",") if s}
    holdout_cohorts = {c for c in args.holdout_cohort.split(",") if c}
    if holdout_samples or holdout_cohorts:
        train, test = split_rows_by_holdout(rows, holdout_samples, holdout_cohorts)
        print(f"holdout split: samples={sorted(holdout_samples)} cohorts={sorted(holdout_cohorts)}")
    else:
        train, test = split_rows(rows, args.split_ratio, args.split_by, rng)
    write_rows(output_dir / "train_seq.csv", train)
    write_rows(output_dir / "test_seq.csv", test)
    concat_summaries(shard_dir, output_dir / "read_call_preprocess_summary.tsv")

    key_kind = "cohort" if args.balance_cohorts else "label"
    print(f"read shard rows by {key_kind}: {dict(seen)}")
    print(f"retained rows by {key_kind}: { {k: len(v) for k, v in reservoirs.items()} }")
    print(f"split_by: {args.split_by}")
    print(f"wrote {len(train)} train reads to {output_dir / 'train_seq.csv'}")
    print(f"wrote {len(test)} test reads to {output_dir / 'test_seq.csv'}")


if __name__ == "__main__":
    main()
