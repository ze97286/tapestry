#!/usr/bin/env python3
"""Select alternative MethylBERT DMR top-N sets from merged DSS calls.

The literal DSS top-N by abs(areaStat) can be dominated by nearby broad regions.
This script keeps that literal set, and also writes simple post-processing
variants that are useful for diagnosing whether one locus or chromosome is
dominating the final MethylBERT region filter.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


HG38_CHROM_ORDER = {
    "chr1": 1,
    "chr2": 2,
    "chr3": 3,
    "chr4": 4,
    "chr5": 5,
    "chr6": 6,
    "chr7": 7,
    "chr8": 8,
    "chr9": 9,
    "chr10": 10,
    "chr11": 11,
    "chr12": 12,
    "chr13": 13,
    "chr14": 14,
    "chr15": 15,
    "chr16": 16,
    "chr17": 17,
    "chr18": 18,
    "chr19": 19,
    "chr20": 20,
    "chr21": 21,
    "chr22": 22,
    "chrX": 23,
    "chrY": 24,
}


EXTRA_FIELDS = [
    "source_dmr_id",
    "source_rank",
    "selection_variant",
    "cluster_id",
    "cluster_start",
    "cluster_end",
    "cluster_size",
    "cluster_span",
]


def chrom_key(chrom: str) -> tuple[int, str]:
    return (HG38_CHROM_ORDER.get(chrom, 10_000), chrom)


def parse_int(value: str, field: str, row_num: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"row {row_num} has non-numeric {field}: {value!r}") from exc


def parse_float(value: str | None) -> float | None:
    if value is None or value in ("", "NA", "NaN", "nan"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def score(row: dict[str, str]) -> float:
    abs_area = parse_float(row.get("abs_areaStat"))
    if abs_area is not None:
        return abs(abs_area)
    for field in ("areaStat", "stat"):
        raw = parse_float(row.get(field))
        if raw is not None:
            return abs(raw)
    return 0.0


def interval_length(row: dict[str, str]) -> int:
    start = int(row.get("_start_int") or float(row["start"]))
    end = int(row.get("_end_int") or float(row["end"]))
    return end - start + 1


def read_dmrs(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise SystemExit(f"{path} has no header")
        missing = {"chr", "start", "end"} - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")

        rows: list[dict[str, str]] = []
        for row_num, row in enumerate(reader, start=2):
            if not row:
                continue
            start = parse_int(row["start"], "start", row_num)
            end = parse_int(row["end"], "end", row_num)
            if end < start:
                continue
            row["_source_order"] = str(len(rows))
            row["_start_int"] = str(start)
            row["_end_int"] = str(end)
            row["_score"] = f"{score(row):.17g}"
            rows.append(row)

    rows.sort(
        key=lambda row: (
            -float(row["_score"]),
            chrom_key(row["chr"]),
            int(row["_start_int"]),
            int(row["_end_int"]),
        )
    )
    for idx, row in enumerate(rows):
        row["_rank"] = str(idx + 1)
    return rows, list(reader.fieldnames)


def output_fields(input_fields: list[str]) -> list[str]:
    fields = list(input_fields)
    for required in ("abs_areaStat", "dmr_id"):
        if required not in fields:
            fields.append(required)
    for field in EXTRA_FIELDS:
        if field not in fields:
            fields.append(field)
    return fields


def prepare_output_rows(rows: list[dict[str, str]], variant: str) -> list[dict[str, str]]:
    prepared: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        out = {key: value for key, value in row.items() if not key.startswith("_")}
        out["source_dmr_id"] = row.get("source_dmr_id") or row.get("dmr_id", "")
        out["source_rank"] = row.get("_rank", "")
        out["selection_variant"] = variant
        out["dmr_id"] = str(idx)
        out["abs_areaStat"] = f"{score(row):.17g}"
        prepared.append(out)
    return prepared


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_bed(path: Path, rows: list[dict[str, str]], start_base: int) -> None:
    with path.open("w") as handle:
        for idx, row in enumerate(rows):
            chrom = row["chr"]
            start = int(float(row["start"]))
            end = int(float(row["end"]))
            bed_start = start - 1 if start_base == 1 else start
            if bed_start < 0:
                bed_start = 0
            if end <= bed_start:
                continue
            name = row.get("dmr_id") or str(idx)
            handle.write(f"{chrom}\t{bed_start}\t{end}\t{name}\n")


def write_variant(
    output_dir: Path,
    name: str,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    start_base: int,
) -> list[dict[str, str]]:
    prepared = prepare_output_rows(rows, name)
    write_tsv(output_dir / f"{name}.tsv", prepared, fieldnames)
    write_bed(output_dir / f"{name}.bed", prepared, start_base)
    return prepared


def capped_rows(rows: list[dict[str, str]], top_n: int, max_per_chrom: int) -> list[dict[str, str]]:
    counts: dict[str, int] = defaultdict(int)
    selected: list[dict[str, str]] = []
    for row in rows:
        chrom = row["chr"]
        if counts[chrom] >= max_per_chrom:
            continue
        selected.append(dict(row))
        counts[chrom] += 1
        if len(selected) >= top_n:
            break
    return selected


def collapsed_rows(rows: list[dict[str, str]], top_n: int, distance: int) -> list[dict[str, str]]:
    sorted_by_position = sorted(
        rows,
        key=lambda row: (chrom_key(row["chr"]), int(row["_start_int"]), int(row["_end_int"])),
    )
    representatives: list[dict[str, str]] = []
    cluster: list[dict[str, str]] = []
    cluster_chrom = ""
    cluster_start = 0
    cluster_end = 0
    cluster_id = 0

    def flush() -> None:
        nonlocal cluster_id
        if not cluster:
            return
        best = max(
            cluster,
            key=lambda row: (
                float(row["_score"]),
                -int(row["_start_int"]),
                -int(row["_end_int"]),
            ),
        )
        representative = dict(best)
        representative["cluster_id"] = str(cluster_id)
        representative["cluster_start"] = str(cluster_start)
        representative["cluster_end"] = str(cluster_end)
        representative["cluster_size"] = str(len(cluster))
        representative["cluster_span"] = str(cluster_end - cluster_start + 1)
        representatives.append(representative)
        cluster_id += 1

    for row in sorted_by_position:
        chrom = row["chr"]
        start = int(row["_start_int"])
        end = int(row["_end_int"])
        if not cluster:
            cluster = [row]
            cluster_chrom = chrom
            cluster_start = start
            cluster_end = end
            continue
        if chrom == cluster_chrom and start <= cluster_end + distance:
            cluster.append(row)
            cluster_end = max(cluster_end, end)
            continue
        flush()
        cluster = [row]
        cluster_chrom = chrom
        cluster_start = start
        cluster_end = end
    flush()

    representatives.sort(
        key=lambda row: (
            -float(row["_score"]),
            chrom_key(row["chr"]),
            int(row["_start_int"]),
            int(row["_end_int"]),
        )
    )
    return representatives[:top_n]


def write_summary(path: Path, variants: dict[str, list[dict[str, str]]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["variant", "chrom", "n_dmrs", "total_bp", "median_bp", "min_bp", "max_bp"])
        for variant, rows in variants.items():
            by_chrom: dict[str, list[int]] = defaultdict(list)
            for row in rows:
                by_chrom[row["chr"]].append(interval_length(row))
            all_lengths = [length for lengths in by_chrom.values() for length in lengths]
            if all_lengths:
                writer.writerow(
                    [
                        variant,
                        "all",
                        len(all_lengths),
                        sum(all_lengths),
                        statistics.median(all_lengths),
                        min(all_lengths),
                        max(all_lengths),
                    ]
                )
            for chrom in sorted(by_chrom, key=chrom_key):
                lengths = by_chrom[chrom]
                writer.writerow(
                    [
                        variant,
                        chrom,
                        len(lengths),
                        sum(lengths),
                        statistics.median(lengths),
                        min(lengths),
                        max(lengths),
                    ]
                )


def parse_distances(raw_values: list[str]) -> list[int]:
    distances: list[int] = []
    for raw_value in raw_values:
        for part in raw_value.split(","):
            if not part:
                continue
            value = int(part)
            if value < 0:
                raise SystemExit("--collapse-distances must be non-negative")
            if value not in distances:
                distances.append(value)
    return distances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Merged DSS DMR TSV")
    parser.add_argument("--output-dir", required=True, help="Directory for variant BED/TSV outputs")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument(
        "--collapse-distances",
        nargs="+",
        default=["500000", "1000000"],
        help="One or more bp distances for locus-collapsed variants; comma-separated values are accepted.",
    )
    parser.add_argument("--max-per-chrom", type=int, default=10, help="Maximum DMRs per chromosome for capped variant")
    parser.add_argument(
        "--start-base",
        type=int,
        choices=[0, 1],
        default=1,
        help="Coordinate base of the DSS start column. Default: 1.",
    )
    args = parser.parse_args()

    if args.top_n < 1:
        raise SystemExit("--top-n must be positive")
    if args.max_per_chrom < 1:
        raise SystemExit("--max-per-chrom must be positive")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, input_fields = read_dmrs(input_path)
    if not rows:
        raise SystemExit(f"no DMR rows found in {input_path}")
    fieldnames = output_fields(input_fields)

    variants: dict[str, list[dict[str, str]]] = {}
    literal_name = f"literal_top{args.top_n}"
    variants[literal_name] = write_variant(
        output_dir,
        literal_name,
        rows[: args.top_n],
        fieldnames,
        args.start_base,
    )

    capped_name = f"capped{args.max_per_chrom}_top{args.top_n}"
    variants[capped_name] = write_variant(
        output_dir,
        capped_name,
        capped_rows(rows, args.top_n, args.max_per_chrom),
        fieldnames,
        args.start_base,
    )

    for distance in parse_distances(args.collapse_distances):
        if distance % 1000 == 0:
            collapsed_name = f"collapsed_{distance // 1000}kb_top{args.top_n}"
        else:
            collapsed_name = f"collapsed_{distance}bp_top{args.top_n}"
        variants[collapsed_name] = write_variant(
            output_dir,
            collapsed_name,
            collapsed_rows(rows, args.top_n, distance),
            fieldnames,
            args.start_base,
        )

    write_summary(output_dir / "variant_summary.tsv", variants)
    for variant, selected in variants.items():
        print(f"{variant}: wrote {len(selected)} DMRs")
    print(f"wrote summary to {output_dir / 'variant_summary.tsv'}")


if __name__ == "__main__":
    main()
