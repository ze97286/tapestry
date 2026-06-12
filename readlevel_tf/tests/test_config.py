"""Label-builder tests: naming rules, EAC/baseline filters, control split."""

from __future__ import annotations

import tempfile
from pathlib import Path

from rltf.config import build_manifest_rows, numeric_prefix, parse_patient, write_manifest_tsv
from rltf.manifest import read_manifest


def test_naming_helpers():
    assert parse_patient("071-001_ScrBsl_plasma_md", "ab_underscore") == ("071-001", "ScrBsl")
    assert parse_patient("069-006-NBY-C1W3", "cd_hyphen") == ("069-006-NBY", "C1W3")
    assert numeric_prefix("069-006-NBY") == "069-006"
    assert numeric_prefix("071-001") == "071-001"


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")


def test_manifest_materialisation():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for f in ["tissue/069-009_ScrBsl_tumour_md", "tissue/071-021_ScrBsl_tumour_md",
                  "AB/X2881_Ctrl_plasma_md", "AB/071-001_ScrBsl_plasma_md",
                  "AB/071-001_C1W3_plasma_md", "AB/069-002_ScrBsl_plasma_md",
                  "CD/GI10888", "CD/SCAN3149", "CD/SCAN3160", "CD/SCAN3768",
                  "CD/069-006-NBY-ScrBsl", "CD/069-006-NBY-C1W3"]:
            _touch(root / (f + ".per-read.bed.gz"))
        (root / "clinical.csv").write_text(
            'subject,subject_recode\n"071-001-RON","EAC-ILEK"\n"069-002-ATH","ESCC-FLJG"\n"069-006-NBY","EAC-XXXX"\n')
        (root / "ichor.json").write_text('{"071-001_ScrBsl_plasma_md": {"tumour_fraction": 0.0073}}')
        cfg = {
            "paths": {"clinical_csv": str(root / "clinical.csv"), "ichorcna_json": str(root / "ichor.json")},
            "labels": {"baseline_timepoint": "ScrBsl", "disease_filter": "EAC"},
            "reference": {"tumour": {"files": [str(root / "tissue/069-009_ScrBsl_tumour_md.per-read.bed.gz"),
                                               str(root / "tissue/071-021_ScrBsl_tumour_md.per-read.bed.gz")]}},
            "cohorts": [
                {"name": "AB", "dir": str(root / "AB"), "control_globs": ["*Ctrl*"],
                 "patient_style": "ab_underscore", "controls_role": "query"},
                {"name": "CD", "dir": str(root / "CD"), "control_globs": ["GI*", "SCAN*"],
                 "patient_style": "cd_hyphen", "controls_role": "split", "controls_reference_fraction": 0.5},
            ],
        }
        rows = build_manifest_rows(cfg)
        ref_t = [r for r in rows if r["role"] == "reference" and r["group"] == "tumour"]
        ref_h = [r for r in rows if r["role"] == "reference" and r["group"] == "healthy"]
        pos = [r for r in rows if r["role"] == "query" and r["is_cancer"] == 1]
        neg = [r for r in rows if r["role"] == "query" and r["is_cancer"] == 0]
        assert len(ref_t) == 2
        assert len(ref_h) == 2 and all(r["cohort"] == "CD" for r in ref_h)
        assert {r["patient_id"] for r in pos} == {"071-001", "069-006-NBY"}
        assert all(r["subtype"] == "EAC" for r in pos)            # ESCC excluded
        assert not any("C1W3" in r["sample_id"] for r in rows)    # on-treatment excluded
        assert len(neg) == 3                                       # AB control + 2 CD controls
        assert abs([r for r in pos if r["cohort"] == "AB"][0]["tf"] - 0.0073) < 1e-9

        write_manifest_tsv(rows, root / "m.tsv")
        assert len(read_manifest(root / "m.tsv")) == len(rows)


if __name__ == "__main__":
    test_naming_helpers()
    test_manifest_materialisation()
    print("rltf config/label builder: all checks passed")
