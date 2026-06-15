"""Single-source-of-truth configuration + manifest materialisation.

Everything (per-read-call directories, naming rules, baseline/EAC filters,
control split, parameters) lives in one committed TOML file. The manifest is
materialised from it — patients vs controls by filename rule, baseline-only,
EAC-only via the clinical CSV, one row per patient, controls split into the
reference (build p_H) vs query (negatives). ichorCNA `tf` is attached for
clinical validation only, never used by the model.

Loaded with stdlib ``tomllib`` (Python >= 3.11).
"""

from __future__ import annotations

import fnmatch
import glob
import json
import logging
import math
import os
import tomllib
from pathlib import Path
from typing import Any

logger = logging.getLogger("rltf.config")


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def basename(path: str) -> str:
    n = Path(path).name
    for suf in (".per-read.bed.gz", ".per-read.bed", ".bed.gz", ".pat.gz", ".gz"):
        if n.endswith(suf):
            return n[: -len(suf)]
    return n


def numeric_prefix(patient_id: str) -> str:
    """First two '-'-delimited tokens, e.g. '071-001' or '069-006-NBY' -> '069-006'."""
    return "-".join(patient_id.split("-")[:2])


def parse_patient(bn: str, style: str) -> tuple[str, str]:
    """Return (patient_id, timepoint) from a basename per the cohort's naming style."""
    if style == "ab_underscore":          # 071-001_ScrBsl_plasma_md
        parts = bn.split("_")
        return parts[0], (parts[1] if len(parts) > 1 else "")
    if style == "cd_hyphen":              # 069-006-NBY-C1W3
        parts = bn.split("-")
        return "-".join(parts[:-1]), parts[-1]
    raise ValueError(f"unknown patient_style '{style}'")


def _matches(bn: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(bn, p) for p in patterns)


def _to_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def load_survival(cfg: dict) -> dict[tuple[str, str], dict]:
    """Per-cohort clinical survival join, keyed by ``(cohort, numeric_prefix)``.

    The AB and CD summaries use DIFFERENT schemas — in AB the event flag is ``OS`` and in
    CD it is ``OS_ind`` (CD's ``OS`` is a date). Clinical benefit also differs: AB has a
    categorical ``Clinical_Benefit`` (Y/N), CD encodes it via the ``Mandard`` regression
    grade (TRG 1-3 = responder). Each cohort declares its own columns + benefit rule; a
    single global name would silently NaN-drop one cohort. Returns
    ``{(cohort, prefix): {os_days, os_event, benefit}}`` with benefit harmonised to 1/0/None.

    Benefit rule per cohort: ``benefit_good_values`` (benefit if the cell is in this set,
    categorical) takes precedence; else ``benefit_max`` (benefit if numeric value <= it,
    e.g. Mandard <= 3); else benefit is None.
    """
    import csv as _csv
    out: dict[tuple[str, str], dict] = {}
    for src in get(cfg, "cohorts", []):
        path = src.get("clinical_csv", "")
        if not path or not Path(path).exists():
            continue
        tcol, ecol, bcol = src.get("os_time_col", "OS_days"), src.get("os_event_col", "OS"), src.get("benefit_col", "")
        good = {str(v) for v in src.get("benefit_good_values", [])}
        bmax = src.get("benefit_max", None)
        with open(path) as fh:
            for row in _csv.DictReader(fh):
                subj = (row.get("subject") or "").strip()
                if not subj:
                    continue
                raw = (row.get(bcol) or "").strip() if bcol else ""
                if not bcol or raw in ("", "NA", "Unknown", "NaN"):
                    benefit = None
                elif good:
                    benefit = 1.0 if raw in good else 0.0
                elif bmax is not None:
                    v = _to_float(raw)
                    benefit = None if math.isnan(v) else (1.0 if v <= float(bmax) else 0.0)
                else:
                    benefit = None
                out[(src["name"], numeric_prefix(subj))] = {
                    "os_days": _to_float(row.get(tcol)),
                    "os_event": _to_float(row.get(ecol)),
                    "benefit": benefit,
                }
    return out


def _load_ichorcna(json_path: str) -> dict[str, float]:
    data = json.loads(Path(json_path).read_text())
    out = {}
    for k, v in data.items():
        out[k] = float(v["tumour_fraction"]) if isinstance(v, dict) else float(v)
    return out


def _row(sample_id, role, group, cohort, file_path, *, is_cancer=math.nan,
         patient_id="", timepoint="", subtype="", tf=math.nan) -> dict:
    return dict(sample_id=sample_id, role=role, group=group, cohort=cohort, file_path=file_path,
               is_cancer=is_cancer, patient_id=patient_id, timepoint=timepoint, subtype=subtype, tf=tf)


# ---------------------------------------------------------------------------
# Manifest materialisation
# ---------------------------------------------------------------------------

def build_manifest_rows(cfg: dict) -> list[dict]:
    ichor = _load_ichorcna(get(cfg, "paths.ichorcna_json", "")) if get(cfg, "paths.ichorcna_json", "") else {}
    baseline = get(cfg, "labels.baseline_timepoint", "ScrBsl")
    # The trial is EAC-dominated; non-EAC (ESCC) patients are excluded explicitly by numeric
    # prefix per cohort (see each cohort's `exclude_prefixes`), NOT by requiring a positive
    # match against a clinical table — a missing clinical row must never silently drop a
    # patient (that bug dropped every CD patient when only an AB clinical table was loaded).
    disease = get(cfg, "labels.disease_filter", "EAC")          # stamped as the kept-patient subtype
    min_bytes = int(get(cfg, "labels.min_file_bytes", 0))       # skip empty/stub per-read files (0 = no guard)

    rows: list[dict] = []
    for path in get(cfg, "reference.tumour.files", []):
        rows.append(_row(basename(path), "reference", "tumour", "OAC_tissue", path))

    seen_patient: set[str] = set()
    for src in get(cfg, "cohorts", []):
        name = src["name"]
        files = sorted(glob.glob(src["dir"].rstrip("/") + "/*.per-read.bed.gz"))
        ctrl_globs = src.get("control_globs", [])
        style = src.get("patient_style")
        exclude = set(src.get("exclude_prefixes", []))          # non-EAC (ESCC) patient numeric prefixes
        followup_tps = set(get(cfg, "labels.followup_timepoints", []))  # on-treatment timepoints to score (monitoring)
        controls: list[tuple[str, str]] = []
        followups: list[dict] = []
        n_kept = n_escc = n_other = n_empty = n_dup = n_followup = 0
        for f in files:
            bn = basename(f)
            if min_bytes:
                try:
                    if os.path.getsize(f) < min_bytes:
                        n_empty += 1
                        continue
                except OSError:
                    pass
            if _matches(bn, ctrl_globs):
                controls.append((bn, f))
                continue
            pid, tp = parse_patient(bn, style)
            if numeric_prefix(pid) in exclude:                  # ESCC: drop EVERY timepoint, logged, never silent
                n_escc += 1
                continue
            if tp == baseline:
                if pid in seen_patient:
                    n_dup += 1
                    continue
                seen_patient.add(pid)
                rows.append(_row(bn, "query", "cfdna", name, f, is_cancer=1,
                                patient_id=pid, timepoint=tp, subtype=disease, tf=ichor.get(bn, math.nan)))
                n_kept += 1
            elif tp in followup_tps:                            # on-treatment: scored-only, never trained on
                followups.append(_row(bn, "followup", "cfdna", name, f, is_cancer=math.nan,
                                      patient_id=pid, timepoint=tp, subtype=disease, tf=ichor.get(bn, math.nan)))
                n_followup += 1
            else:
                n_other += 1
        rows.extend(followups)
        logger.info("cohort %s: kept %d baseline %s patients; %d followup [%s]; excluded %d ESCC, "
                    "%d other-timepoint, %d empty, %d duplicate-patient", name, n_kept, disease, n_followup,
                    ",".join(sorted(followup_tps)) or "-", n_escc, n_other, n_empty, n_dup)

        controls.sort()
        crole = src.get("controls_role", "query")
        if crole == "reference":
            for bn, f in controls:
                rows.append(_row(bn, "reference", "healthy", name, f))
        elif crole == "split":
            frac = float(src.get("controls_reference_fraction", 0.5))
            k = int(round(frac * len(controls)))
            for bn, f in controls[:k]:
                rows.append(_row(bn, "reference", "healthy", name, f))
            for bn, f in controls[k:]:
                rows.append(_row(bn, "query", "cfdna", name, f, is_cancer=0, tf=0.0))
        else:  # query
            for bn, f in controls:
                rows.append(_row(bn, "query", "cfdna", name, f, is_cancer=0, tf=0.0))

    if not any(r["role"] == "reference" and r["group"] == "tumour" for r in rows):
        raise ValueError("no tumour reference files (reference.tumour.files)")
    if not any(r["role"] == "reference" and r["group"] == "healthy" for r in rows):
        raise ValueError("no reference-healthy controls (set a cohort controls_role to reference/split)")
    return rows


def write_manifest_tsv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ("sample_id", "role", "group", "cohort", "file_path",
            "is_cancer", "patient_id", "timepoint", "subtype", "tf")

    def fmt(v):
        if isinstance(v, float) and math.isnan(v):
            return ""
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    with open(path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(fmt(r[c]) for c in cols) + "\n")
