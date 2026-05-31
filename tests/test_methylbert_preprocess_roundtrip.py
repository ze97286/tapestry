#!/usr/bin/env python3
"""Synthetic round-trip test for the MethylBERT TAPS preprocessing changes.

Confirms that:
  * the new covariate columns (read_length, n_cpg) are written and correct;
  * the ablation flags (--blank-methyl, --blank-dna, --collapse-dmr-label) behave;
  * the resulting CSV still parses through the unmodified upstream dataset.py
    (_parse_line / __getitem__) and carries the covariates to the model layer.

Runnable directly: ``python3 tests/test_methylbert_preprocess_roundtrip.py``.
Exits non-zero on the first failure. The dataset-load section is skipped if torch /
tqdm are unavailable (e.g. off-cluster), but the preprocessing checks always run.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREPROCESS = REPO / "scripts" / "preprocess_methylbert_taps_read_calls.py"
EXPECTED_HEADER = [
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


def write_inputs(tmp: Path) -> tuple[Path, Path]:
    seq = ("ACGT" * 75)  # 300 bp, single line
    fasta = tmp / "ref.fa"
    fasta.write_text(f">chr1\n{seq}\n")
    # .fai: name, length, byte-offset of first base, bases-per-line, bytes-per-line
    offset = len(">chr1\n")
    (tmp / "ref.fa.fai").write_text(f"chr1\t{len(seq)}\t{offset}\t{len(seq)}\t{len(seq) + 1}\n")

    dmrs = tmp / "dmrs.tsv"
    dmrs.write_text("chr\tstart\tend\tctype\tdmr_id\nchr1\t1\t300\tT\t0\n")

    header = "#chr\tstart\tend\tread_id\torientation\tread_length\tmod_cps\tunmod_cpgs\tsnp_cpgs\n"
    tumour = tmp / "tumour_sample.per-read.bed"
    tumour.write_text(
        header
        + "chr1\t20\t120\tt1\t+\t100\t5,30\t10,60,90\t\n"
        + "chr1\t50\t130\tt2\t+\t80\t10\t20,40\t\n"
    )
    normal = tmp / "normal_sample.per-read.bed"
    normal.write_text(
        header
        + "chr1\t0\t90\tn1\t+\t90\t8\t12,40\t\n"
        + "chr1\t40\t160\tn2\t+\t120\t15,70\t25,55\t\n"
    )

    sheet = tmp / "sheet.tsv"
    sheet.write_text(f"{tumour}\tT\n{normal}\tN\n")
    return sheet, dmrs


def run_preprocess(tmp: Path, sheet: Path, dmrs: Path, rows_out: Path, *extra: str) -> None:
    cmd = [
        sys.executable,
        str(PREPROCESS),
        "--sample-sheet",
        str(sheet),
        "--dmrs",
        str(dmrs),
        "--reference",
        str(tmp / "ref.fa"),
        "--output-dir",
        str(tmp / "out"),
        "--rows-output",
        str(rows_out),
        "--mode",
        "contained",
        "--min-informative",
        "2",
        *extra,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def check(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}")
        sys.exit(1)
    print(f"  ok: {message}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        sheet, dmrs = write_inputs(tmp)

        # --- baseline schema + covariate correctness ---------------------------
        rows_path = tmp / "rows.tsv"
        run_preprocess(tmp, sheet, dmrs, rows_path)
        header, rows = read_rows(rows_path)
        check(header == EXPECTED_HEADER, f"header is the 10-column schema (got {header})")
        check(len(rows) > 0, "rows were generated")
        for row in rows:
            n_cpg_actual = row["methyl_seq"].count("0") + row["methyl_seq"].count("1")
            check(int(row["n_cpg"]) == n_cpg_actual, f"n_cpg matches methyl_seq for {row['name']}")
            check(int(row["read_length"]) > 0, f"read_length populated for {row['name']}")
            check(len(row["dna_seq"].split(" ")) == len(row["methyl_seq"]), "token counts align")
        labels = {row["ctype"] for row in rows}
        check(labels == {"T", "N"}, f"both classes present ({labels})")

        # --- ablation flags ----------------------------------------------------
        bm_path = tmp / "blank_methyl.tsv"
        run_preprocess(tmp, sheet, dmrs, bm_path, "--blank-methyl")
        _, bm_rows = read_rows(bm_path)
        check(all("0" not in r["methyl_seq"] for r in bm_rows), "--blank-methyl removes unmethylated state")
        check(all(r["methyl_seq"].count("1") > 0 for r in bm_rows), "--blank-methyl keeps CpG positions")

        bd_path = tmp / "blank_dna.tsv"
        run_preprocess(tmp, sheet, dmrs, bd_path, "--blank-dna")
        _, bd_rows = read_rows(bd_path)
        check(
            all(set(r["dna_seq"].split(" ")) == {"NNN"} for r in bd_rows),
            "--blank-dna replaces all DNA tokens with a constant",
        )

        cd_path = tmp / "collapse_dmr.tsv"
        run_preprocess(tmp, sheet, dmrs, cd_path, "--collapse-dmr-label")
        _, cd_rows = read_rows(cd_path)
        check(all(r["dmr_label"] == "0" for r in cd_rows), "--collapse-dmr-label sets dmr_label to 0")

        # --- upstream dataset compatibility (skipped if core deps unavailable) --
        # vocab.py has a transitive `import tqdm` that the parsing path never uses;
        # stub it so the schema check runs even where tqdm is absent.
        if "tqdm" not in sys.modules:
            try:
                import tqdm  # noqa: F401
            except ModuleNotFoundError:
                import types

                sys.modules["tqdm"] = types.ModuleType("tqdm")
        try:
            sys.path.insert(0, str(REPO / "external" / "methylbert" / "src"))
            from methylbert.data.dataset import _line2tokens_finetune, _parse_line
            from methylbert.data.vocab import MethylVocab
        except Exception as exc:  # noqa: BLE001 - core deps may be absent off-cluster
            print(f"SKIP dataset round-trip (deps unavailable: {type(exc).__name__}: {exc})")
            print("PASS (preprocessing checks)")
            return

        vocab = MethylVocab(k=3)
        raw_lines = rows_path.read_text().splitlines()
        headers = raw_lines[0].split("\t")
        parsed = _parse_line(raw_lines[1], headers)  # raises if the schema is incompatible
        check("ctype_label" in parsed, "upstream _parse_line computed ctype_label from the new schema")
        check("read_length" in parsed, "covariate read_length survives into the parsed line")
        check("n_cpg" in parsed, "covariate n_cpg survives into the parsed line")
        item = _line2tokens_finetune(dict(parsed), tokenizer=vocab, max_len=150, headers=headers)
        check(len(item["dna_seq"]) == 150, "tokeniser pads dna_seq to seq_len")
        check(len(item["methyl_seq"]) == 150, "tokeniser pads methyl_seq to seq_len")
        print("PASS (preprocessing + upstream dataset round-trip)")


if __name__ == "__main__":
    main()
