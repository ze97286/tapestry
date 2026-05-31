#!/usr/bin/env python3
"""Generate leave-one-sample-out (LOSO) folds over tumour biopsies for MethylBERT.

With only a handful of tumour-tissue samples, the standard 0.8 by-sample split leaves
the entire positive held-out class as one biopsy, so a high tumour accuracy can be
single-biopsy memorisation. This emits one fold per tumour sample (that sample held out
to the test set) as ready-to-run merge commands, plus a note on the finetune/eval env
per fold. Run the merge commands, then fine-tune + evaluate each fold and compare the
per-fold tumour accuracy: a genuine signal is consistent across folds, memorisation is
high-variance with one biopsy carrying it.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def sample_name(path: str) -> str:
    name = Path(path.split()[0]).name
    for suffix in (".per-read.bed.gz", ".per-read.bed", ".bed.gz", ".gz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def read_label_samples(sample_sheet: Path, positive_label: str) -> list[str]:
    samples: list[str] = []
    with sample_sheet.open() as handle:
        for parts in csv.reader(handle, delimiter="\t"):
            if not parts or not parts[0] or parts[0].startswith("#"):
                continue
            if len(parts) >= 2 and parts[1] == positive_label:
                samples.append(sample_name(parts[0]))
    seen: set[str] = set()
    unique: list[str] = []
    for sample in samples:
        if sample not in seen:
            seen.add(sample)
            unique.append(sample)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sheet", required=True, help="Read-call sample sheet (path<TAB>label)")
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--output-root", required=True, help="Root for per-fold preprocess dirs")
    parser.add_argument("--positive-label", default="T")
    parser.add_argument("--max-reads-per-label", type=int, default=500000)
    parser.add_argument("--split-by", default="sample")
    parser.add_argument("--output-script", required=True, help="Bash script to write the fold commands to")
    args = parser.parse_args()

    tumour_samples = read_label_samples(Path(args.sample_sheet), args.positive_label)
    if not tumour_samples:
        raise SystemExit(f"no '{args.positive_label}'-labelled samples found in {args.sample_sheet}")

    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        f"# {len(tumour_samples)} leave-one-sample-out folds over '{args.positive_label}' samples.",
        "# Each fold holds out one tumour sample to the test set; the rest train.",
        "",
    ]
    for sample in tumour_samples:
        fold_dir = f"{args.output_root}/fold_{sample}"
        lines += [
            f'echo "=== LOSO fold: holdout {sample} ==="',
            "python3 scripts/merge_methylbert_read_call_shards.py \\",
            f'    --shard-dir "{args.shard_dir}" \\',
            f'    --output-dir "{fold_dir}" \\',
            f"    --max-reads-per-label {args.max_reads_per_label} \\",
            f"    --split-by {args.split_by} \\",
            f'    --holdout-samples "{sample}"',
            f"# then fine-tune + evaluate with PREPROCESS_DIR={fold_dir},",
            "# a per-fold MODEL_DIR and EVAL_DIR (see methylbert_bmrc_standalone/03 and 04),",
            "# and record summary.json accuracy / roc_auc for the held-out tumour sample.",
            "",
        ]

    out = Path(args.output_script)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.chmod(0o755)
    print(f"wrote {len(tumour_samples)} LOSO folds to {out}")
    for sample in tumour_samples:
        print(f"  fold holdout: {sample}")


if __name__ == "__main__":
    main()
