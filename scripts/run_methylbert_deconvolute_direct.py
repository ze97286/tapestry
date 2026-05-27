#!/usr/bin/env python3
"""Run MethylBERT deconvolution without importing BAM preprocessing dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from methylbert.data.dataset import MethylBertFinetuneDataset
from methylbert.data.vocab import MethylVocab
from methylbert.deconvolute import deconvolute
from methylbert.trainer import MethylBertFinetuneTrainer


def read_train_params(model_dir: Path) -> dict[str, str]:
    param_path = model_dir / "train_param.txt"
    if not param_path.exists():
        raise SystemExit(f"{param_path} does not exist. Check the fine-tuned model directory.")

    params: dict[str, str] = {}
    with param_path.open() as handle:
        for line in handle:
            key, value = line.rstrip("\n").split("\t", 1)
            params[key] = value
    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_grid", type=int, default=10000)
    parser.add_argument("--adjustment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    params = read_train_params(model_dir)
    print(f"Restored parameters: {params}")

    tokenizer = MethylVocab(k=int(params["n_mers"]))
    dataset = MethylBertFinetuneDataset(args.input_data, tokenizer, seq_len=int(params["seq_len"]))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"Bulk data ({args.input_data}) is loaded")

    train_dataset = params["train_dataset"]
    df_train = pd.read_csv(train_dataset, sep="\t")
    n_dmrs = max(
        len(df_train["dmr_label"].unique()),
        int(df_train["dmr_label"].max()) + 1,
    )

    torch.set_num_threads(max(1, min(args.num_workers, torch.get_num_threads())))
    trainer = MethylBertFinetuneTrainer(
        len(tokenizer),
        save_path="./test",
        train_dataloader=data_loader,
        test_dataloader=data_loader,
    )

    restore_dir = model_dir / "bert.model"
    trainer.load(str(restore_dir), load_fine_tune=True, n_dmrs=int(n_dmrs))
    print(f"Trained model ({restore_dir}) is restored")

    deconvolute(
        trainer=trainer,
        data_loader=data_loader,
        df_train=df_train,
        tokenizer=tokenizer,
        output_path=str(output_path),
        n_grid=args.n_grid,
        adjustment=args.adjustment,
    )


if __name__ == "__main__":
    main()
