"""Materialised manifest schema (one row per per-read-call file).

Tab-separated, header. Columns:

    sample_id   basename of the per-read.bed.gz
    role        reference | query
    group       tumour | healthy | cfdna
    cohort      OAC_tissue | AB | CD   (the batch label)
    file_path   path to the .per-read.bed.gz
    is_cancer   0/1 for query rows (blank for reference)
    patient_id  patient identifier (query patients only)
    timepoint   e.g. ScrBsl, C1W3 (query patients only)
    subtype     EAC / ESCC (query patients only)
    tf          ichorCNA tumour fraction — CLINICAL VALIDATION ONLY, never a model input

The manifest is materialised from the config (`rltf.config.build_manifest_rows`)
and never hand-authored. No `convention` column: per-read calls are explicit
mod/unmod, so there is no flip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REQUIRED = ("sample_id", "role", "group", "cohort", "file_path")
OPTIONAL = ("is_cancer", "patient_id", "timepoint", "subtype", "tf")


def _na(x: str) -> bool:
    return x is None or x.strip() == "" or x.strip().upper() in ("NA", "NAN", "NONE")


def read_manifest(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = {n: i for i, n in enumerate(header)}
        for req in REQUIRED:
            if req not in col:
                raise ValueError(f"manifest missing required column '{req}'")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            f = line.split("\t")

            def val(name, default=""):
                return f[col[name]] if name in col and col[name] < len(f) else default

            rows.append({
                "sample_id": val("sample_id").strip(),
                "role": val("role").strip().lower(),
                "group": val("group").strip().lower(),
                "cohort": val("cohort").strip(),
                "file_path": val("file_path").strip(),
                "is_cancer": (np.nan if _na(val("is_cancer")) else int(float(val("is_cancer")))),
                "patient_id": val("patient_id").strip(),
                "timepoint": val("timepoint").strip(),
                "subtype": val("subtype").strip(),
                "tf": (np.nan if _na(val("tf")) else float(val("tf"))),
            })
    return rows
