#!/usr/bin/env python3
"""Build MethylBERT TAPS per-read call lists from sample-selection lists.

The source lists are used only to recover sample IDs.  They may contain PAT
paths from the earlier DSS step or plain sample IDs; the PAT files themselves
are not read and are not inputs to the per-read-call workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path


EMPTY_CD_CONTROLS = {"SCAN3144", "SCAN3783"}


def sample_name_from_source(value: str) -> str:
    name = Path(value.split()[0]).name
    if name.endswith(".pat.gz"):
        name = name[: -len(".pat.gz")]
    elif name.endswith(".per-read.bed.gz"):
        name = name[: -len(".per-read.bed.gz")]
    if name.startswith("OAC_"):
        name = name[len("OAC_") :]
    return name


def read_source_samples(path: Path) -> list[str]:
    samples = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                samples.append(sample_name_from_source(line))
    return samples


def tumour_read_call_name(sample: str) -> str:
    if sample.endswith("_md") or sample.endswith("_F_md"):
        return sample
    return f"{sample}_md"


def resolve_bulk_path(entry: str, bulk_dir: Path | None) -> Path:
    """Resolve a bulk cfDNA entry to a per-read.bed.gz path. Accepts a full path or a
    bare sample ID (resolved against --bulk-read-call-dir)."""
    text = entry.split()[0]
    if "/" in text or text.endswith(".gz"):
        return Path(text)
    name = text if text.endswith(".per-read.bed.gz") else f"{text}.per-read.bed.gz"
    if bulk_dir is None:
        raise SystemExit("--bulk-read-call-dir is required when bulk entries are bare sample IDs")
    return bulk_dir / name


def require_nonempty(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"missing per-read call file: {path}")
    if path.stat().st_size <= 28:
        raise SystemExit(f"empty-looking per-read call file: {path}")


def write_list(path: Path, rows: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(f"{row}\n")


def write_sample_sheet(path: Path, tumour_rows: list[Path], normal_rows: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in tumour_rows:
            handle.write(f"{row}\tT\n")
        for row in normal_rows:
            handle.write(f"{row}\tN\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tumour-sample-list", "--tumour-pat-list", dest="tumour_sample_list", required=True)
    parser.add_argument("--normal-sample-list", "--normal-pat-list", dest="normal_sample_list", required=True)
    parser.add_argument("--tumour-read-call-dir", required=True)
    parser.add_argument("--ab-read-call-dir", required=True)
    parser.add_argument("--cd-read-call-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument(
        "--bulk-sample-list",
        help="Optional list of cfDNA samples (full paths or bare IDs) to deconvolute; "
        "writes oac_bulk_read_calls.sample_sheet.tsv for the deconvolution step.",
    )
    parser.add_argument(
        "--bulk-read-call-dir",
        help="Directory holding bulk per-read.bed.gz files when --bulk-sample-list "
        "entries are bare sample IDs.",
    )
    args = parser.parse_args()

    tumour_dir = Path(args.tumour_read_call_dir)
    ab_dir = Path(args.ab_read_call_dir)
    cd_dir = Path(args.cd_read_call_dir)
    output_dir = Path(args.output_dir)

    tumour_rows = [
        tumour_dir / f"{tumour_read_call_name(sample)}.per-read.bed.gz"
        for sample in read_source_samples(Path(args.tumour_sample_list))
    ]
    normal_rows = []
    excluded = []
    for sample in read_source_samples(Path(args.normal_sample_list)):
        if sample in EMPTY_CD_CONTROLS:
            excluded.append(sample)
            continue
        if sample.endswith("_Ctrl_plasma_md"):
            normal_rows.append(ab_dir / f"{sample}.per-read.bed.gz")
        elif sample.startswith(("GI", "SCAN")):
            normal_rows.append(cd_dir / f"{sample}.per-read.bed.gz")
        else:
            raise SystemExit(f"cannot place normal sample {sample!r} into AB or CD per-read roots")

    if not args.allow_empty:
        for row in tumour_rows + normal_rows:
            require_nonempty(row)

    tumour_out = output_dir / "oac_dmr_tumour_read_calls.list"
    normal_out = output_dir / "oac_dmr_normal_read_calls.list"
    sample_sheet_out = output_dir / "oac_dmr_read_calls.sample_sheet.tsv"
    write_list(tumour_out, tumour_rows)
    write_list(normal_out, normal_rows)
    write_sample_sheet(sample_sheet_out, tumour_rows, normal_rows)

    print(f"wrote {len(tumour_rows)} tumour read-call paths to {tumour_out}")
    print(f"wrote {len(normal_rows)} normal read-call paths to {normal_out}")
    print(f"wrote {len(tumour_rows) + len(normal_rows)} sample-sheet rows to {sample_sheet_out}")

    if args.bulk_sample_list:
        bulk_dir = Path(args.bulk_read_call_dir) if args.bulk_read_call_dir else None
        bulk_rows = []
        with open(args.bulk_sample_list) as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#"):
                    bulk_rows.append(resolve_bulk_path(line, bulk_dir))
        if not args.allow_empty:
            for row in bulk_rows:
                require_nonempty(row)
        bulk_out = output_dir / "oac_bulk_read_calls.sample_sheet.tsv"
        write_list(bulk_out, bulk_rows)
        print(f"wrote {len(bulk_rows)} bulk read-call paths to {bulk_out}")

    if excluded:
        print("excluded empty CD controls: " + ", ".join(excluded))


if __name__ == "__main__":
    main()
