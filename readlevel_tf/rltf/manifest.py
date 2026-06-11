"""Single self-describing sample manifest for the standalone pipeline.

Tab-separated, header row. Columns:

    sample_id    unique id
    role         reference | query
    group        tumour | healthy   (reference rows: tumour=tissue, healthy=cfDNA controls)
    cohort       batch label (e.g. OAC_tissue / AB / CD) — the LOGO grouping
    convention   bisulfite | taps   (per-sample sequencing convention)
    file_path    path to the .pat(.gz)
    is_cancer    0/1 for query rows (blank/NA for reference)
    tf           optional ichorCNA tumour fraction for query rows (blank/NA otherwise)

Reference-healthy samples (build profiles) must be DISJOINT from query-healthy
samples (evaluated as negatives), or specificity is optimistic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REQUIRED = ("sample_id", "role", "group", "cohort", "convention", "file_path")


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
        has_cancer, has_tf = "is_cancer" in col, "tf" in col
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            f = line.split("\t")
            role = f[col["role"]].strip().lower()
            group = f[col["group"]].strip().lower()
            group = "tumour" if group in ("tumour", "tumor") else group
            rows.append({
                "sample_id": f[col["sample_id"]].strip(),
                "role": role,
                "group": group,
                "cohort": f[col["cohort"]].strip(),
                "convention": f[col["convention"]].strip().lower(),
                "file_path": f[col["file_path"]].strip(),
                "is_cancer": (np.nan if not has_cancer or _na(f[col["is_cancer"]]) else int(float(f[col["is_cancer"]]))),
                "tf": (np.nan if not has_tf or _na(f[col["tf"]]) else float(f[col["tf"]])),
            })
    return rows
