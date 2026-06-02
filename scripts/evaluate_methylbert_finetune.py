#!/usr/bin/env python3
"""Evaluate a fine-tuned MethylBERT read classifier on a held-out CSV."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from methylbert.data.dataset import MethylBertFinetuneDataset
from methylbert.data.vocab import MethylVocab
from methylbert.trainer import MethylBertFinetuneTrainer

# Apply the non-invasive runtime patches so eval uses the same masked forward the model
# was fine-tuned with. See scripts/methylbert_patches.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from methylbert_patches import apply_patches

apply_patches()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretrain", default=None)
    parser.add_argument("--n-encoder", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=150)
    parser.add_argument("--n-mers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--with-cuda", action="store_true")
    parser.add_argument("--loss", default="bce")
    return parser.parse_args()


def metric_or_nan(func, y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(func(y_true, y_score))
    except ValueError:
        return float("nan")


def cohort_of(name: str) -> str:
    """Map a sample/filename to its source cohort (for per-batch stratification)."""
    name = str(name)
    if "_tumour" in name:
        return "tumour_tissue"
    if "_Ctrl_plasma" in name:
        return "AB_plasma"
    if name.startswith(("GI", "SCAN")):
        return "CD_plasma"
    return "other"


def compute_strata(result: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-stratum accuracy tables for shortcut diagnosis (sample, cohort, read
    length, CpG count). A model learning genuine methylation should be roughly flat
    across length and n_cpg; a shortcut shows accuracy tracking those covariates."""

    def grouped(df: pd.DataFrame, key: str) -> pd.DataFrame:
        return (
            df.groupby(key, dropna=False, observed=True)
            .agg(
                n=("is_correct", "size"),
                accuracy=("is_correct", "mean"),
                mean_prob_matching=("prob_matching_dmr_ctype", "mean"),
                mean_ctype_label=("ctype_label", "mean"),
            )
            .reset_index()
        )

    strata: dict[str, pd.DataFrame] = {}

    if "filename" in result.columns:
        strata["by_sample"] = grouped(result, "filename")
        cohorts = result.assign(cohort=result["filename"].map(cohort_of))
        by_cohort = (
            cohorts.groupby("cohort", dropna=False, observed=True)
            .agg(
                n_reads=("is_correct", "size"),
                n_samples=("filename", "nunique"),
                accuracy=("is_correct", "mean"),
                mean_prob_matching=("prob_matching_dmr_ctype", "mean"),
                mean_ctype_label=("ctype_label", "mean"),
            )
            .reset_index()
        )
        by_cohort["reads_per_sample"] = (
            by_cohort["n_reads"] / by_cohort["n_samples"].replace(0, pd.NA)
        ).round(0)
        strata["by_cohort"] = by_cohort

    if "read_length" in result.columns:
        read_length_num = pd.to_numeric(result["read_length"], errors="coerce")
        if read_length_num.notna().sum() >= 10:
            try:
                length_bin = pd.qcut(read_length_num, q=10, duplicates="drop")
            except ValueError:
                length_bin = pd.cut(read_length_num, bins=10)
            strata["by_length_decile"] = grouped(
                result.assign(length_bin=length_bin.astype(str)), "length_bin"
            )

    if "n_cpg" in result.columns:
        n_cpg_num = pd.to_numeric(result["n_cpg"], errors="coerce")
        if n_cpg_num.notna().sum() >= 10:
            n_cpg_bin = n_cpg_num.clip(upper=10).astype("Int64").astype(str)
            strata["by_n_cpg"] = grouped(result.assign(n_cpg_bin=n_cpg_bin), "n_cpg_bin")

    return strata


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir)
    bert_dir = model_dir / "bert.model"
    if not bert_dir.is_dir():
        raise SystemExit(f"Missing model directory: {bert_dir}")

    tokenizer = MethylVocab(k=args.n_mers)
    dataset = MethylBertFinetuneDataset(args.test_dataset, tokenizer, seq_len=args.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=bool(args.with_cuda),
        shuffle=False,
    )

    trainer = MethylBertFinetuneTrainer(
        len(tokenizer),
        save_path=str(bert_dir),
        train_dataloader=loader,
        test_dataloader=loader,
        with_cuda=args.with_cuda,
        loss=args.loss,
    )

    has_saved_base = any((bert_dir / name).exists() for name in ("pytorch_model.bin", "model.safetensors"))
    if has_saved_base:
        trainer.load(str(bert_dir), n_dmrs=dataset.num_dmrs(), load_fine_tune=True)
    else:
        pretrain = args.pretrain or f"hanyangii/methylbert_hg19_{args.n_encoder}l"
        trainer.load(pretrain, n_dmrs=dataset.num_dmrs(), load_fine_tune=False)
        dmr_encoder = model_dir / "dmr_encoder.pickle"
        read_classifier = model_dir / "read_classification_model.pickle"
        if not dmr_encoder.exists() or not read_classifier.exists():
            raise SystemExit(
                "No saved BERT base was found, and classifier artifacts are incomplete: "
                f"{dmr_encoder}, {read_classifier}"
            )
        trainer.bert.from_pretrained_dmr_encoder(str(dmr_encoder), trainer.device)
        trainer.bert.from_pretrained_read_classifier(str(read_classifier), trainer.device)
        trainer.model = trainer.bert.to(trainer.device)

    result, probs = trainer.read_classification(loader, tokenizer=tokenizer, logit=True)
    result["prob_not_matching_dmr_ctype"] = probs[:, 0]
    result["prob_matching_dmr_ctype"] = probs[:, 1]
    result["is_correct"] = result["pred"].astype(int) == result["ctype_label"].astype(int)

    predictions_path = output_dir / "test_predictions.tsv"
    result.to_csv(predictions_path, sep="\t", index=False)

    y_true = result["ctype_label"].to_numpy(dtype=int)
    y_pred = result["pred"].to_numpy(dtype=int)
    y_score = result["prob_matching_dmr_ctype"].to_numpy(dtype=float)

    summary = {
        "n_reads": int(len(result)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": metric_or_nan(roc_auc_score, y_true, y_score),
        "average_precision": metric_or_nan(average_precision_score, y_true, y_score),
        "mean_prob_matching_true_0": float(np.mean(y_score[y_true == 0])) if np.any(y_true == 0) else float("nan"),
        "mean_prob_matching_true_1": float(np.mean(y_score[y_true == 1])) if np.any(y_true == 1) else float("nan"),
        "predictions": str(predictions_path),
    }

    by_label = (
        result.groupby(["ctype", "ctype_label"], dropna=False)
        .agg(
            n=("pred", "size"),
            accuracy=("is_correct", "mean"),
            mean_prob_matching=("prob_matching_dmr_ctype", "mean"),
            median_prob_matching=("prob_matching_dmr_ctype", "median"),
        )
        .reset_index()
    )

    # ---- Shortcut diagnostics --------------------------------------------------
    # Stratify accuracy by covariates that should NOT, on their own, carry tumour
    # signal: sample identity, source cohort, read length and CpG count. If accuracy
    # tracks any of these (or P(tumour) correlates with read length / n_cpg), the
    # classifier is leaning on a source/batch/length shortcut rather than methylation.
    strata = compute_strata(result)
    for name, table in strata.items():
        table.to_csv(output_dir / f"summary_{name}.tsv", sep="\t", index=False)

    if "read_length" in result.columns:
        read_length_num = pd.to_numeric(result["read_length"], errors="coerce")
        if read_length_num.notna().sum() >= 3:
            summary["corr_prob_vs_read_length"] = float(
                result["prob_matching_dmr_ctype"].corr(read_length_num)
            )
    if "n_cpg" in result.columns:
        n_cpg_num = pd.to_numeric(result["n_cpg"], errors="coerce")
        if n_cpg_num.notna().sum() >= 3:
            summary["corr_prob_vs_n_cpg"] = float(
                result["prob_matching_dmr_ctype"].corr(n_cpg_num)
            )
    # Explicit AB-vs-CD normal-cohort check: both are control (N) cohorts, so a healthy
    # model should score them similarly. A large gap means the classifier is keyed on
    # source/batch (the cohort-imbalance shortcut). --balance-cohorts should shrink it.
    if "by_cohort" in strata:
        cohort_prob = (
            strata["by_cohort"].set_index("cohort")["mean_prob_matching"].astype(float).to_dict()
        )
        summary["cohort_mean_prob_matching"] = {k: float(v) for k, v in cohort_prob.items()}
        if "AB_plasma" in cohort_prob and "CD_plasma" in cohort_prob:
            summary["normal_cohort_gap_AB_minus_CD"] = float(
                cohort_prob["AB_plasma"] - cohort_prob["CD_plasma"]
            )
    summary["strata_files"] = sorted(strata)

    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    by_label.to_csv(output_dir / "summary_by_label.tsv", sep="\t", index=False)

    print(json.dumps(summary, indent=2))
    print(by_label.to_csv(sep="\t", index=False))
    for name in ("by_cohort", "by_length_decile", "by_n_cpg"):
        if name in strata:
            print(f"# {name}")
            print(strata[name].to_csv(sep="\t", index=False))


if __name__ == "__main__":
    main()
