"""Step-5 monitoring: followup emission, per-cohort survival harmonisation, trajectory summary."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from rltf.config import build_manifest_rows, load_survival

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_monitor import _benefit_vs_delta, summarise_trajectory  # noqa: E402


def _touch(p: Path, content: str = "x" * 4000):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def test_manifest_emits_followup_and_excludes_escc():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for f in ["tissue/069-009_ScrBsl_tumour_md", "tissue/071-021_ScrBsl_tumour_md",
                  "CD/SCAN1", "CD/SCAN2",
                  "CD/069-006-NBY-ScrBsl", "CD/069-006-NBY-Immonly", "CD/069-006-NBY-C1W3",
                  "CD/069-015-TWS-ScrBsl", "CD/069-015-TWS-Immonly"]:  # 069-015 = ESCC, drop all timepoints
            _touch(root / (f + ".per-read.bed.gz"))
        cfg = {
            "paths": {},
            "labels": {"baseline_timepoint": "ScrBsl", "followup_timepoints": ["Immonly"], "disease_filter": "EAC"},
            "reference": {"tumour": {"files": [str(root / "tissue/069-009_ScrBsl_tumour_md.per-read.bed.gz"),
                                               str(root / "tissue/071-021_ScrBsl_tumour_md.per-read.bed.gz")]}},
            "cohorts": [{"name": "CD", "dir": str(root / "CD"), "control_globs": ["SCAN*"],
                         "patient_style": "cd_hyphen", "controls_role": "split",
                         "controls_reference_fraction": 0.5, "exclude_prefixes": ["069-015"]}],
        }
        rows = build_manifest_rows(cfg)
        fu = [r for r in rows if r["role"] == "followup"]
        assert len(fu) == 1                                   # only 069-006-NBY Immonly (ESCC 069-015 dropped)
        assert fu[0]["patient_id"] == "069-006-NBY" and fu[0]["timepoint"] == "Immonly"
        assert np.isnan(fu[0]["is_cancer"])                   # followup is unlabelled (never trained on)
        assert not any(r["timepoint"] == "C1W3" for r in rows)        # non-followup timepoint not emitted
        assert not any("069-015" in str(r["patient_id"]) for r in rows)  # ESCC excluded at every timepoint


def test_load_survival_harmonises_per_cohort_event_column():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        # AB: event flag is OS, benefit categorical Y/N. CD: OS is a DATE (event=OS_ind),
        # benefit from Mandard TRG (<=3 responder). NA Mandard -> unknown benefit.
        (root / "ab.csv").write_text(
            'subject,OS,OS_days,Clinical_Benefit\n"071-001-RON",1,257,Y\n"071-002-XXX",0,800,N\n')
        (root / "cd.csv").write_text(
            'subject,OS,OS_ind,OS_days,Mandard\n"069-006-NBY",2020-08-24,0,857,2\n'
            '"069-012-TNS",2022-01-17,1,1166,5\n"069-014-KEN",2022-01-17,0,1125,NA\n')
        cfg = {"cohorts": [
            {"name": "AB", "clinical_csv": str(root / "ab.csv"), "os_time_col": "OS_days",
             "os_event_col": "OS", "benefit_col": "Clinical_Benefit", "benefit_good_values": ["Y"]},
            {"name": "CD", "clinical_csv": str(root / "cd.csv"), "os_time_col": "OS_days",
             "os_event_col": "OS_ind", "benefit_col": "Mandard", "benefit_max": 3},
        ]}
        surv = load_survival(cfg)
        assert surv[("AB", "071-001")] == {"os_days": 257.0, "os_event": 1.0, "benefit": 1.0}
        assert surv[("AB", "071-002")]["benefit"] == 0.0             # N -> no benefit
        cd = surv[("CD", "069-006")]
        assert cd["os_days"] == 857.0 and cd["os_event"] == 0.0      # OS_ind, not the date in OS
        assert cd["benefit"] == 1.0                                  # Mandard 2 <= 3 -> responder
        assert surv[("CD", "069-012")]["benefit"] == 0.0            # Mandard 5 -> no benefit
        assert surv[("CD", "069-014")]["benefit"] is None          # Mandard NA -> unknown


def test_summarise_trajectory_counts():
    traj = pd.DataFrame({
        "delta":          [-0.3, -0.1, 0.2, np.nan],
        "followup_score": [0.10, 0.80, 0.95, 0.97],
    })
    out = summarise_trajectory(traj, threshold=0.90)
    assert out["n_followup_samples"] == 4 and out["n_with_paired_scores"] == 3
    assert out["n_decreased"] == 2 and abs(out["frac_decreased"] - 2 / 3) < 1e-9
    assert out["n_residual_at_followup"] == 2 and out["n_cleared_at_followup"] == 2  # >=0.90: 0.95,0.97


def test_benefit_vs_delta_directionality():
    # Benefit patients drop (Δ<0); no-benefit patients rise (Δ>0) → −Δ ranks benefit perfectly.
    traj = pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(8)],
        "delta":   [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4],
        "benefit": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    })
    out = _benefit_vs_delta(traj)
    assert out["n_benefit"] == 4 and out["n_no_benefit"] == 4
    assert out["auc_neg_delta_predicts_benefit"] == 1.0
    assert out["contingency_downup_x_benefit"] == [[4, 0], [0, 4]]   # all down=benefit, all up=no-benefit
    assert out["median_delta_benefit"] < out["median_delta_no_benefit"]
    assert out["mannwhitney_p_benefit_more_negative"] < 0.05


if __name__ == "__main__":
    test_manifest_emits_followup_and_excludes_escc()
    test_load_survival_harmonises_per_cohort_event_column()
    test_summarise_trajectory_counts()
    test_benefit_vs_delta_directionality()
    print("rltf monitor layer: all checks passed")
