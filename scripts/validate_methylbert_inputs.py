#!/usr/bin/env python3
"""Validate source data required by the paper-style MethylBERT pipeline."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path


REQUIRED_VARS = [
    "PROJECT_DIR",
    "OUTPUT_DIR",
    "RUN_LABEL",
    "METHYLBERT_DIR",
    "METHYLBERT_REF_FASTA",
    "METHYLBERT_REF_FASTA_GZ",
    "METHYLBERT_REF_STAGE_DIR",
    "METHYLBERT_METHYLCALLER",
    "METHYLBERT_DMR_TUMOUR_BAM_LIST",
    "METHYLBERT_DMR_NORMAL_BAM_LIST",
    "METHYLBERT_TUMOUR_BAM_LIST",
    "METHYLBERT_CONTROL_BAM_LIST",
    "METHYLBERT_BULK_BAM_LIST",
]


LIST_VARS = [
    "METHYLBERT_DMR_TUMOUR_BAM_LIST",
    "METHYLBERT_DMR_NORMAL_BAM_LIST",
    "METHYLBERT_TUMOUR_BAM_LIST",
    "METHYLBERT_CONTROL_BAM_LIST",
    "METHYLBERT_BULK_BAM_LIST",
]


def load_config(config: Path) -> dict[str, str]:
    command = (
        "set -a; "
        "PROJECT_DIR=${PROJECT_DIR:-$(pwd)}; "
        "export PROJECT_DIR; "
        f"source {shlex.quote(str(config))}; "
        "for key in " + " ".join(REQUIRED_VARS) + "; do "
        "printf '%s=%s\\n' \"$key\" \"${!key:-}\"; "
        "done"
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        check=True,
        text=True,
        capture_output=True,
    )
    env: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env[key] = value
    return env


def read_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip().split()[0]
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def inspect_first_alignment(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["samtools", "view", str(path)],
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    for line in result.stdout.splitlines():
        if line:
            return line
    return None


def has_expected_tags(alignment_line: str, methylcaller: str) -> bool:
    if methylcaller == "bismark":
        return bool(re.search(r"\tXM:Z:", alignment_line))
    if methylcaller == "dorado":
        return bool(re.search(r"\tMM:Z:|\tML:B:", alignment_line))
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Sourceable MethylBERT env config")
    parser.add_argument(
        "--check-tags",
        action="store_true",
        help="Use samtools to inspect one BAM from each list for methylation tags",
    )
    args = parser.parse_args()

    config = Path(args.config)
    errors: list[str] = []
    warnings: list[str] = []
    if not config.exists():
        raise SystemExit(f"missing config: {config}")

    env = load_config(config)
    optional_vars = {"METHYLBERT_REF_FASTA", "METHYLBERT_REF_FASTA_GZ", "METHYLBERT_REF_STAGE_DIR"}
    for key in REQUIRED_VARS:
        if key in optional_vars:
            continue
        if not env.get(key):
            errors.append(f"{key} is not set")
    if not env.get("METHYLBERT_REF_FASTA") and not env.get("METHYLBERT_REF_FASTA_GZ"):
        errors.append("set METHYLBERT_REF_FASTA or METHYLBERT_REF_FASTA_GZ")

    methylcaller = env.get("METHYLBERT_METHYLCALLER", "")
    if methylcaller not in {"bismark", "dorado"}:
        errors.append("METHYLBERT_METHYLCALLER must be bismark or dorado")

    ref_value = env.get("METHYLBERT_REF_FASTA", "")
    ref_gz_value = env.get("METHYLBERT_REF_FASTA_GZ", "")
    if ref_value:
        ref = Path(ref_value)
        if ref.suffix == ".gz":
            warnings.append("METHYLBERT_REF_FASTA points to .gz; treating it as compressed source")
        elif not ref.exists():
            errors.append(f"METHYLBERT_REF_FASTA does not exist: {ref}")
        elif not Path(str(ref) + ".fai").exists():
            warnings.append(f"reference FASTA index not found: {ref}.fai")
    if ref_gz_value:
        ref_gz = Path(ref_gz_value)
        if not ref_gz.exists():
            errors.append(f"METHYLBERT_REF_FASTA_GZ does not exist: {ref_gz}")

    methylbert_dir = Path(env.get("METHYLBERT_DIR", ""))
    if env.get("METHYLBERT_DIR") and not (methylbert_dir / "src/methylbert").exists():
        errors.append(f"upstream MethylBERT source not found under {methylbert_dir}")

    list_counts: dict[str, int] = {}
    first_bams: list[tuple[str, Path]] = []
    for key in LIST_VARS:
        list_path = Path(env.get(key, ""))
        if not env.get(key):
            continue
        if not list_path.exists():
            errors.append(f"{key} list does not exist: {list_path}")
            continue
        entries = read_list(list_path)
        list_counts[key] = len(entries)
        if not entries:
            errors.append(f"{key} list is empty: {list_path}")
            continue
        missing = [entry for entry in entries if not Path(entry).exists()]
        if missing:
            errors.append(f"{key} has {len(missing)} missing BAM/CRAM paths; first missing: {missing[0]}")
        else:
            first_bams.append((key, Path(entries[0])))

    if args.check_tags and not errors:
        for key, bam in first_bams:
            line = inspect_first_alignment(bam)
            if line is None:
                warnings.append(f"could not inspect alignments with samtools for {key}: {bam}")
            elif not has_expected_tags(line, methylcaller):
                errors.append(f"{key} first BAM does not show {methylcaller} methylation tags: {bam}")

    print("MethylBERT input validation")
    for key in REQUIRED_VARS:
        print(f"{key}={env.get(key, '')}")
    print()
    for key in LIST_VARS:
        if key in list_counts:
            print(f"{key}: {list_counts[key]} entries")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("\nOK")


if __name__ == "__main__":
    main()
