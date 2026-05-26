#!/usr/bin/env python3
"""Diagnose TAPS per-read methylation calls for MethylBERT-style preprocessing.

This script is intentionally TAPS-specific: ``mod_cps`` are treated as
methylated CpGs and ``unmod_cpgs`` as unmethylated CpGs.  Coordinates are
expected to be hg38 BED-style intervals: 0-based, half-open.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterable

import numpy as np
import pandas as pd


REQUIRED_CALL_COLUMNS = {
    "#chr",
    "chr",
    "start",
    "end",
    "read_id",
    "orientation",
    "read_length",
    "num_cpg",
    "num_mod",
    "mod_cps",
    "unmod_cpgs",
}


@dataclass
class Dmr:
    chrom: str
    start0: int
    end0: int
    dmr_id: str


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
        if not item:
            continue
        try:
            offsets.append(int(item))
        except ValueError as exc:
            raise ValueError(f"non-integer offset {item!r} in {text!r}") from exc
    return offsets


def read_list(path: Path, label: str) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append((Path(line.split()[0]), label))
    return rows


def read_sample_sheet(path: Path) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
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
    dmrs: list[Dmr] = []
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise SystemExit(f"{path} has no header")
        missing = {"chr", "start", "end"} - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"{path} missing DMR columns: {sorted(missing)}")
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
            dmrs.append(Dmr(row["chr"], start0, end, row.get("dmr_id") or str(idx)))
    if not dmrs:
        raise SystemExit(f"no DMRs loaded from {path}")
    return dmrs


def chrom_index(dmrs: list[Dmr]) -> dict[str, list[Dmr]]:
    indexed: dict[str, list[Dmr]] = defaultdict(list)
    for dmr in dmrs:
        indexed[dmr.chrom].append(dmr)
    for chrom in indexed:
        indexed[chrom].sort(key=lambda item: (item.start0, item.end0))
    return indexed


def matching_dmrs(
    dmrs: list[Dmr],
    start0: int,
    end0: int,
    mode: str,
) -> list[Dmr]:
    if mode == "contained":
        return [dmr for dmr in dmrs if dmr.start0 <= start0 and end0 <= dmr.end0]
    if mode == "overlap":
        return [dmr for dmr in dmrs if start0 < dmr.end0 and end0 > dmr.start0]
    raise ValueError(f"unknown mode: {mode}")


def describe(values: pd.Series) -> dict[str, float]:
    values = values.dropna()
    if values.empty:
        return {}
    qs = values.quantile([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])
    return {f"q{int(q * 100):02d}": float(v) for q, v in qs.items()}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "#chr" in df.columns and "chr" not in df.columns:
        df = df.rename(columns={"#chr": "chr"})
    return df


def read_calls(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    kwargs = {"sep": "\t", "comment": None, "dtype": str, "keep_default_na": False}
    if chunksize > 0:
        yield from (normalise_columns(chunk) for chunk in pd.read_csv(path, chunksize=chunksize, **kwargs))
    else:
        yield normalise_columns(pd.read_csv(path, **kwargs))


def diagnose_file(
    path: Path,
    label: str,
    dmrs_by_chrom: dict[str, list[Dmr]],
    mode: str,
    chunksize: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    stats: dict[str, object] = defaultdict(int)
    dmr_rows: dict[tuple[str, str], dict[str, object]] = {}

    for chunk in read_calls(path, chunksize):
        missing = {"chr", "start", "end", "read_id", "orientation", "read_length", "num_cpg", "num_mod", "mod_cps", "unmod_cpgs"} - set(chunk.columns)
        if missing:
            raise SystemExit(f"{path} missing call columns: {sorted(missing)}")

        chunk["start"] = chunk["start"].astype(int)
        chunk["end"] = chunk["end"].astype(int)
        chunk["read_length"] = chunk["read_length"].astype(int)
        chunk["num_cpg"] = chunk["num_cpg"].astype(int)
        chunk["num_mod"] = chunk["num_mod"].astype(int)
        chunk["span_len"] = chunk["end"] - chunk["start"]
        chunk["span_matches_read_length"] = chunk["span_len"] == chunk["read_length"]

        stats["rows"] += int(len(chunk))
        stats["span_matches_read_length"] += int(chunk["span_matches_read_length"].sum())
        stats["span_mismatch"] += int((~chunk["span_matches_read_length"]).sum())
        stats["num_cpg_sum"] += int(chunk["num_cpg"].sum())
        stats["num_mod_sum"] += int(chunk["num_mod"].sum())

        for row in chunk.itertuples(index=False):
            chrom = getattr(row, "chr")
            start0 = int(row.start)
            end0 = int(row.end)
            span_len = int(row.span_len)
            read_len = int(row.read_length)
            mod_offsets = parse_offsets(row.mod_cps)
            unmod_offsets = parse_offsets(row.unmod_cpgs)
            informative = len(mod_offsets) + len(unmod_offsets)
            out_of_span = sum(offset < 0 or offset >= span_len for offset in mod_offsets + unmod_offsets)
            out_of_read = sum(offset < 0 or offset >= read_len for offset in mod_offsets + unmod_offsets)
            stats["informative_offsets"] += informative
            stats["offsets_out_of_span"] += out_of_span
            stats["offsets_out_of_read_length"] += out_of_read

            matched = matching_dmrs(dmrs_by_chrom.get(chrom, []), start0, end0, mode)
            if not matched:
                continue
            stats["rows_with_dmr"] += 1
            stats["dmr_hits"] += len(matched)
            stats["dmr_hit_informative_offsets"] += informative
            stats["dmr_hit_mod_offsets"] += len(mod_offsets)
            stats["dmr_hit_unmod_offsets"] += len(unmod_offsets)
            if informative <= 1:
                stats["dmr_hits_one_or_fewer_informative"] += len(matched)
            if informative <= 2:
                stats["dmr_hits_two_or_fewer_informative"] += len(matched)
            if informative >= 5:
                stats["dmr_hits_five_or_more_informative"] += len(matched)

            frac_methylated = len(mod_offsets) / informative if informative else np.nan
            for dmr in matched:
                key = (dmr.dmr_id, label)
                dmr_row = dmr_rows.setdefault(
                    key,
                    {
                        "dmr_id": dmr.dmr_id,
                        "label": label,
                        "n_rows": 0,
                        "n_samples": 1,
                        "informative_sum": 0,
                        "mod_sum": 0,
                        "unmod_sum": 0,
                        "frac_methylated_values": [],
                    },
                )
                dmr_row["n_rows"] += 1
                dmr_row["informative_sum"] += informative
                dmr_row["mod_sum"] += len(mod_offsets)
                dmr_row["unmod_sum"] += len(unmod_offsets)
                dmr_row["frac_methylated_values"].append(frac_methylated)

    stats = dict(stats)
    stats["path"] = str(path)
    stats["label"] = label
    stats["span_match_rate"] = stats.get("span_matches_read_length", 0) / stats["rows"] if stats.get("rows") else np.nan
    stats["mean_num_cpg"] = stats.get("num_cpg_sum", 0) / stats["rows"] if stats.get("rows") else np.nan
    stats["mean_num_mod"] = stats.get("num_mod_sum", 0) / stats["rows"] if stats.get("rows") else np.nan
    stats["mean_informative_offsets"] = stats.get("informative_offsets", 0) / stats["rows"] if stats.get("rows") else np.nan
    stats["mean_dmr_hit_informative_offsets"] = (
        stats.get("dmr_hit_informative_offsets", 0) / stats["dmr_hits"] if stats.get("dmr_hits") else np.nan
    )
    stats["dmr_hit_frac_methylated"] = (
        stats.get("dmr_hit_mod_offsets", 0) / stats.get("dmr_hit_informative_offsets", 0)
        if stats.get("dmr_hit_informative_offsets")
        else np.nan
    )

    dmr_summary = []
    for row in dmr_rows.values():
        values = pd.Series(row.pop("frac_methylated_values"), dtype=float)
        row["mean_informative"] = row["informative_sum"] / row["n_rows"] if row["n_rows"] else np.nan
        row["frac_methylated"] = row["mod_sum"] / row["informative_sum"] if row["informative_sum"] else np.nan
        row["median_read_frac_methylated"] = float(values.median()) if not values.dropna().empty else np.nan
        dmr_summary.append(row)

    return stats, dmr_summary


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sheet", help="TSV with per-read call path and label columns")
    parser.add_argument("--tumour-list", help="Per-read call files labelled T")
    parser.add_argument("--normal-list", help="Per-read call files labelled N")
    parser.add_argument("--dmrs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--dmr-start-base", type=int, choices=[0, 1], default=1)
    parser.add_argument("--mode", choices=["contained", "overlap"], default="contained")
    parser.add_argument("--chunksize", type=int, default=250000)
    args = parser.parse_args()

    samples: list[tuple[Path, str]] = []
    if args.sample_sheet:
        samples.extend(read_sample_sheet(Path(args.sample_sheet)))
    if args.tumour_list:
        samples.extend(read_list(Path(args.tumour_list), "T"))
    if args.normal_list:
        samples.extend(read_list(Path(args.normal_list), "N"))
    if not samples:
        raise SystemExit("provide --sample-sheet or call file lists")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dmrs = read_dmrs(Path(args.dmrs), args.top_n, args.dmr_start_base)
    dmrs_by_chrom = chrom_index(dmrs)

    sample_rows = []
    dmr_rows = []
    for path, label in samples:
        stats, per_dmr = diagnose_file(path, label, dmrs_by_chrom, args.mode, args.chunksize)
        sample_rows.append(stats)
        dmr_rows.extend(per_dmr)
        print(
            f"{path}\tlabel={label}\trows={stats.get('rows', 0)}\t"
            f"dmr_hits={stats.get('dmr_hits', 0)}\t"
            f"mean_dmr_hit_informative={stats.get('mean_dmr_hit_informative_offsets', np.nan):.3f}\t"
            f"dmr_hit_frac_methylated={stats.get('dmr_hit_frac_methylated', np.nan):.3f}"
        )

    write_tsv(output_dir / "taps_read_call_diagnostics_by_sample.tsv", sample_rows)
    write_tsv(output_dir / "taps_read_call_diagnostics_by_dmr.tsv", dmr_rows)

    sample_df = pd.DataFrame(sample_rows)
    overview = {
        "n_files": int(len(sample_rows)),
        "labels": sample_df["label"].value_counts().to_dict() if not sample_df.empty else {},
        "mode": args.mode,
        "dmr_start_base": args.dmr_start_base,
        "total_rows": int(sample_df["rows"].sum()) if "rows" in sample_df else 0,
        "total_dmr_hits": int(sample_df["dmr_hits"].sum()) if "dmr_hits" in sample_df else 0,
        "span_match_rate_by_file": describe(sample_df["span_match_rate"]) if "span_match_rate" in sample_df else {},
        "dmr_hits_by_file": describe(sample_df["dmr_hits"]) if "dmr_hits" in sample_df else {},
        "mean_dmr_hit_informative_by_file": describe(sample_df["mean_dmr_hit_informative_offsets"]) if "mean_dmr_hit_informative_offsets" in sample_df else {},
        "dmr_hit_frac_methylated_by_file": describe(sample_df["dmr_hit_frac_methylated"]) if "dmr_hit_frac_methylated" in sample_df else {},
        "offsets_out_of_span": int(sample_df["offsets_out_of_span"].sum()) if "offsets_out_of_span" in sample_df else 0,
        "offsets_out_of_read_length": int(sample_df["offsets_out_of_read_length"].sum()) if "offsets_out_of_read_length" in sample_df else 0,
    }
    with (output_dir / "taps_read_call_diagnostics_summary.json").open("w") as handle:
        json.dump(overview, handle, indent=2)
    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
