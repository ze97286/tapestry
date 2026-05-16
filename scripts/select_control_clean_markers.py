#!/usr/bin/env python3
"""Select UXM markers with healthy cfDNA controls in candidate definition.

This is upstream marker selection, not post-hoc trimming.  It consumes:

  * reference homog files over all segmentation blocks
  * a reference manifest mapping samples to cell types
  * healthy cfDNA control homog files over the same blocks

For OAC markers, a block must be both reference-discriminative and clean in
healthy cfDNA controls:

  OAC reference signal high vs non-OAC background
  healthy-control U-signal p95 below an absolute ceiling
  OAC reference signal separated from healthy-control p95 by a margin

For non-OAC markers, selection remains reference-driven by default so the
atlas keeps a deconvolution backbone.  An optional backbone control ceiling can
be set if desired.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from select_markers import BLOCK_COLS, load_and_aggregate


logger = logging.getLogger(__name__)


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def read_uxm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=BLOCK_COLS + ["U", "X", "M"],
        dtype={
            "chr": str,
            "start": np.int64,
            "end": np.int64,
            "startCpG": np.int32,
            "endCpG": np.int32,
            "U": np.int32,
            "X": np.int32,
            "M": np.int32,
        },
    )
    df["total"] = df["U"] + df["X"] + df["M"]
    return df.set_index(BLOCK_COLS)


def signal(df: pd.DataFrame, direction: str) -> pd.Series:
    return df[direction] / df["total"].clip(lower=1)


def load_control_summary(
    control_homog_dirs: list[Path],
    control_pattern: str,
    direction: str,
    min_control_cov: int,
) -> tuple[pd.DataFrame, list[str]]:
    control_re = re.compile(control_pattern)
    signal_parts = []
    observed_parts = []
    sample_ids: list[str] = []

    for control_dir in control_homog_dirs:
        for path in sorted(control_dir.glob("*.uxm.bed.gz")):
            sid = path.name.replace(".uxm.bed.gz", "")
            if not control_re.search(sid):
                continue
            logger.info("  Loading control %s", sid)
            df = read_uxm(path)
            sig = signal(df, direction)
            observed = df["total"] >= min_control_cov
            signal_parts.append(sig.where(observed).rename(sid))
            observed_parts.append(observed.rename(sid))
            sample_ids.append(sid)

    if not signal_parts:
        raise ValueError(
            f"no control homog files matched pattern {control_pattern!r} in "
            f"{', '.join(str(p) for p in control_homog_dirs)}"
        )

    signals = pd.concat(signal_parts, axis=1)
    observed = pd.concat(observed_parts, axis=1).reindex(signals.index).fillna(False)

    summary = pd.DataFrame(index=signals.index)
    summary["control_signal_mean"] = signals.mean(axis=1, skipna=True)
    summary["control_signal_median"] = signals.median(axis=1, skipna=True)
    summary["control_signal_p95"] = signals.quantile(0.95, axis=1, interpolation="linear")
    summary["control_signal_max"] = signals.max(axis=1, skipna=True)
    summary["control_observed_frac"] = observed.mean(axis=1)
    summary["control_n_observed"] = observed.sum(axis=1)
    return summary, sample_ids


def reference_metrics_for_target(
    ct_counts: dict[str, pd.DataFrame],
    sample_counts: dict[str, dict[str, pd.DataFrame]],
    target_ct: str,
    direction: str,
    min_cov_per_sample: int,
    min_consistency: float,
) -> pd.DataFrame:
    cell_types = sorted(ct_counts.keys())
    target = ct_counts[target_ct].copy()
    target["target_signal"] = signal(target, direction)

    bg_parts = [
        ct_counts[ct][["U", "X", "M", "total"]]
        for ct in cell_types
        if ct != target_ct
    ]
    bg = bg_parts[0].copy()
    for part in bg_parts[1:]:
        bg += part
    bg["bg_signal"] = signal(bg, direction)

    consistent = pd.Series(True, index=target.index)
    for _, sdf in sample_counts[target_ct].items():
        sample_signal = signal(sdf, direction)
        consistent &= sdf["total"] >= min_cov_per_sample
        consistent &= sample_signal >= min_consistency

    max_single_bg = pd.Series(0.0, index=target.index)
    for other_ct in cell_types:
        if other_ct == target_ct:
            continue
        other_signal = signal(ct_counts[other_ct], direction)
        max_single_bg = pd.concat([max_single_bg, other_signal], axis=1).max(axis=1)

    noise_floor = 0.01
    out = pd.DataFrame({
        "target_signal": target["target_signal"],
        "bg_signal": bg["bg_signal"],
        "snr": target["target_signal"] / (bg["bg_signal"] + noise_floor),
        "target_total": target["total"],
        "bg_total": bg["total"],
        "consistent": consistent,
        "max_single_bg": max_single_bg,
    })
    return out


def build_atlas_columns(
    markers: pd.DataFrame,
    ct_counts: dict[str, pd.DataFrame],
    direction: str,
) -> pd.DataFrame:
    out = markers.reset_index()
    for ct in sorted(ct_counts.keys()):
        ct_signal = signal(ct_counts[ct], direction).rename(ct)
        out = out.merge(ct_signal.reset_index(), on=BLOCK_COLS, how="left")
    return out


def select_control_clean_markers(
    ct_counts: dict[str, pd.DataFrame],
    sample_counts: dict[str, dict[str, pd.DataFrame]],
    control_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cell_types = sorted(ct_counts.keys())
    all_selected = []
    diagnostics_rows = []
    per_target = {}

    for target_ct in cell_types:
        logger.info("Processing %s", target_ct)
        ref = reference_metrics_for_target(
            ct_counts,
            sample_counts,
            target_ct=target_ct,
            direction=args.direction,
            min_cov_per_sample=args.min_cov_per_sample,
            min_consistency=args.min_consistency,
        )
        table = ref.join(control_summary, how="left")
        table["target"] = target_ct
        table["direction"] = args.direction
        table["n_cpgs"] = (
            table.index.get_level_values("endCpG")
            - table.index.get_level_values("startCpG")
        )
        table["ref_delta"] = table["target_signal"] - table["bg_signal"]
        table["target_control_delta"] = (
            table["target_signal"] - table["control_signal_p95"]
        )

        base_mask = (
            table["consistent"]
            & (table["snr"] >= args.min_snr)
            & (table["target_signal"] >= args.min_signal)
            & (table["bg_signal"] <= args.max_bg)
            & (table["max_single_bg"] <= args.max_single_bg)
        )

        if target_ct == args.target_cell_type:
            mask = (
                base_mask
                & (table["control_observed_frac"] >= args.min_control_observed_frac)
                & (table["control_signal_p95"] <= args.max_target_control_p95)
                & (table["target_control_delta"] >= args.min_target_control_delta)
            )
            table["selection_score"] = (
                args.ref_delta_weight * table["ref_delta"]
                + args.target_control_delta_weight * table["target_control_delta"]
                + args.snr_weight * np.log1p(table["snr"])
                - args.control_signal_weight * table["control_signal_p95"].fillna(1.0)
            )
        else:
            mask = base_mask
            if np.isfinite(args.max_backbone_control_p95):
                mask &= (
                    table["control_observed_frac"] >= args.min_control_observed_frac
                )
                mask &= table["control_signal_p95"] <= args.max_backbone_control_p95
            table["selection_score"] = (
                args.ref_delta_weight * table["ref_delta"]
                + args.snr_weight * np.log1p(table["snr"])
            )

        candidates = table[mask].copy()
        selected = candidates.nlargest(args.top_n, "selection_score").copy()
        all_selected.append(selected)

        per_target[target_ct] = {
            "n_reference_base_candidates": int(base_mask.sum()),
            "n_control_clean_candidates": int(mask.sum()),
            "n_selected": int(len(selected)),
            "selected_control_p95_mean": (
                float(selected["control_signal_p95"].mean()) if len(selected) else None
            ),
            "selected_target_control_delta_mean": (
                float(selected["target_control_delta"].mean()) if len(selected) else None
            ),
        }
        diag = table.reset_index()
        diag["selected"] = False
        if len(selected):
            selected_keys = selected.index
            diag["selected"] = pd.MultiIndex.from_frame(diag[BLOCK_COLS]).isin(selected_keys)
        diagnostics_rows.append(diag)
        logger.info(
            "  %s: base=%d control-clean=%d selected=%d",
            target_ct,
            per_target[target_ct]["n_reference_base_candidates"],
            per_target[target_ct]["n_control_clean_candidates"],
            per_target[target_ct]["n_selected"],
        )

    markers = pd.concat(all_selected)
    markers = build_atlas_columns(markers, ct_counts, args.direction)
    out_cols = BLOCK_COLS + [
        "n_cpgs", "target", "direction",
        "target_signal", "bg_signal", "snr",
        "target_total", "bg_total",
        "consistent", "max_single_bg",
        "control_signal_mean", "control_signal_median",
        "control_signal_p95", "control_signal_max",
        "control_observed_frac", "control_n_observed",
        "ref_delta", "target_control_delta", "selection_score",
    ] + cell_types
    markers = markers[out_cols].sort_values(
        ["target", "selection_score"],
        ascending=[True, False],
        kind="mergesort",
    )

    diagnostics = pd.concat(diagnostics_rows, ignore_index=True)
    summary = {
        "target_cell_type": args.target_cell_type,
        "cell_types": cell_types,
        "direction": args.direction,
        "top_n": args.top_n,
        "max_target_control_p95": args.max_target_control_p95,
        "min_target_control_delta": args.min_target_control_delta,
        "min_control_observed_frac": args.min_control_observed_frac,
        "max_backbone_control_p95": args.max_backbone_control_p95,
        "n_final_markers": int(len(markers)),
        "per_target": per_target,
    }
    return markers, diagnostics, summary


def write_bed(markers: pd.DataFrame, output: Path) -> None:
    bed = markers[BLOCK_COLS].sort_values(["startCpG", "chr", "start"], kind="mergesort")
    output.parent.mkdir(parents=True, exist_ok=True)
    bed.to_csv(output, sep="\t", header=False, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select markers using reference specificity and healthy cfDNA control cleanliness.",
    )
    parser.add_argument("--ref-homog-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--control-homog-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bed-output", default=None)
    parser.add_argument("--diagnostics-output", default=None)
    parser.add_argument("--candidate-output", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--target-cell-type", default="OAC")
    parser.add_argument("--control-pattern", default=r"Ctrl|healthy|^(GI|SCAN)")
    parser.add_argument("--direction", default="U", choices=["U", "M"])
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--min-snr", type=float, default=2.0)
    parser.add_argument("--min-signal", type=float, default=0.2)
    parser.add_argument("--max-bg", type=float, default=0.2)
    parser.add_argument("--max-single-bg", type=float, default=0.35)
    parser.add_argument("--min-consistency", type=float, default=0.1)
    parser.add_argument("--min-cov-per-sample", type=int, default=5)
    parser.add_argument("--min-control-cov", type=int, default=1)
    parser.add_argument(
        "--min-controls",
        type=int,
        default=1,
        help="Fail if fewer matched healthy controls are available.",
    )
    parser.add_argument("--min-control-observed-frac", type=float, default=0.6)
    parser.add_argument("--max-target-control-p95", type=float, default=0.02)
    parser.add_argument("--min-target-control-delta", type=float, default=0.2)
    parser.add_argument("--max-backbone-control-p95", type=float, default=float("inf"))
    parser.add_argument("--ref-delta-weight", type=float, default=1.0)
    parser.add_argument("--target-control-delta-weight", type=float, default=1.0)
    parser.add_argument("--snr-weight", type=float, default=0.25)
    parser.add_argument("--control-signal-weight", type=float, default=1.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    logger.info("Loading reference homog files")
    ct_counts, sample_counts = load_and_aggregate(
        Path(args.ref_homog_dir), Path(args.manifest)
    )
    if args.target_cell_type not in ct_counts:
        raise ValueError(f"target cell type {args.target_cell_type!r} not found in manifest")

    logger.info("Loading healthy-control homog files")
    control_summary, controls = load_control_summary(
        [Path(p) for p in args.control_homog_dir],
        control_pattern=args.control_pattern,
        direction=args.direction,
        min_control_cov=args.min_control_cov,
    )
    logger.info("Loaded %d controls", len(controls))
    if len(controls) < args.min_controls:
        raise ValueError(
            f"only {len(controls)} controls matched {args.control_pattern!r}; "
            f"expected at least {args.min_controls}"
        )

    markers, diagnostics, summary = select_control_clean_markers(
        ct_counts, sample_counts, control_summary, args
    )
    summary["n_controls"] = len(controls)
    summary["controls"] = controls

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    markers.to_csv(output, sep="\t", index=False)

    bed_output = Path(args.bed_output) if args.bed_output else output.with_suffix(".bed")
    write_bed(markers, bed_output)

    diagnostics_output = (
        Path(args.diagnostics_output)
        if args.diagnostics_output
        else output.with_name(output.stem + ".diagnostics.tsv")
    )
    diagnostics.to_csv(diagnostics_output, sep="\t", index=False)

    candidate_output = (
        Path(args.candidate_output)
        if args.candidate_output
        else output.with_name(output.stem + ".candidates.tsv")
    )
    diagnostics.sort_values(["target", "selection_score"], ascending=[True, False]).to_csv(
        candidate_output, sep="\t", index=False
    )

    summary_output = (
        Path(args.summary_output)
        if args.summary_output
        else output.with_name(output.stem + ".summary.json")
    )
    summary_output.write_text(json.dumps(summary, indent=2) + "\n")

    logger.info("Saved markers to %s", output)
    logger.info("Saved marker BED to %s", bed_output)
    logger.info("Saved diagnostics to %s", diagnostics_output)
    logger.info("Saved candidates to %s", candidate_output)
    logger.info("Saved summary to %s", summary_output)


if __name__ == "__main__":
    main()
