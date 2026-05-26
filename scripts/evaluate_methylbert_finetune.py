#!/usr/bin/env python3
"""Evaluate a fine-tuned MethylBERT read classifier on a held-out CSV."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from methylbert.data.dataset import MethylBertFinetuneDataset
from methylbert.data.vocab import MethylVocab
from methylbert.trainer import MethylBertFinetuneTrainer


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

    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    by_label.to_csv(output_dir / "summary_by_label.tsv", sep="\t", index=False)

    print(json.dumps(summary, indent=2))
    print(by_label.to_csv(sep="\t", index=False))


if __name__ == "__main__":
    main()
