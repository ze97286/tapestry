#!/usr/bin/env python3
"""Run MethylBERT fine-tuning without importing BAM preprocessing dependencies."""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

from methylbert.data.dataset import MethylBertFinetuneDataset
from methylbert.data.vocab import MethylVocab
from methylbert.trainer import MethylBertFinetuneTrainer
from methylbert.utils import set_seed

# Apply the non-invasive runtime patches (attention_mask default + pad masking) so the
# vendored external/methylbert stays pristine. See scripts/methylbert_patches.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from methylbert_patches import apply_patches

apply_patches()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("run_methylbert_finetune_direct.py")
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--test_dataset", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--pretrain", default=None)
    parser.add_argument("--n_encoder", type=int, default=None)
    parser.add_argument("--without_pretrain", action="store_true")
    parser.add_argument("--n_mers", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--valid_batch", type=int, default=-1)
    parser.add_argument("--loss", default="bce")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--save_freq", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--with_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.98)
    parser.add_argument("--warm_up", type=int, default=100)
    parser.add_argument("--decrease_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=950410)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    with open(os.path.join(args.output_path, "finetune_config.json"), "w") as handle:
        json.dump(vars(args), handle)
    with open(os.path.join(args.output_path, "train_param.txt"), "w") as handle:
        for key, value in vars(args).items():
            handle.write(f"{key}\t{value}\n")

    set_seed(args.seed)

    print(f"Create a tokenizer for {args.n_mers}-mers")
    tokenizer = MethylVocab(k=args.n_mers)
    print("Vocab Size: ", len(tokenizer))

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    print("CPU info:", torch.get_num_threads(), torch.get_num_interop_threads())

    print("Loading Train Dataset:", args.train_dataset)
    train_dataset = MethylBertFinetuneDataset(args.train_dataset, tokenizer, seq_len=args.seq_len)
    print(f"{len(train_dataset)} seqs with {train_dataset.num_dmrs()} labels ")

    test_dataset = None
    if args.test_dataset is not None:
        print("Loading Test Dataset:", args.test_dataset)
        test_dataset = MethylBertFinetuneDataset(args.test_dataset, tokenizer, seq_len=args.seq_len)

    print("Creating Dataloader")
    local_step_batch_size = int(args.batch_size / args.gradient_accumulation_steps)
    print("Local step batch size : ", local_step_batch_size)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=local_step_batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        shuffle=True,
    )

    if args.valid_batch < 0:
        args.valid_batch = args.batch_size

    test_data_loader = (
        DataLoader(
            test_dataset,
            batch_size=args.valid_batch,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        if test_dataset is not None
        else None
    )

    print("Creating BERT Trainer")
    model_path = os.path.join(args.output_path, "bert.model")
    trainer = MethylBertFinetuneTrainer(
        len(tokenizer),
        save_path=model_path,
        train_dataloader=train_data_loader,
        test_dataloader=test_data_loader,
        lr=args.lr,
        beta=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        with_cuda=args.with_cuda,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        warmup_step=args.warm_up,
        decrease_steps=args.decrease_steps,
        save_freq=args.save_freq,
        loss=args.loss,
    )

    if not args.without_pretrain:
        if args.pretrain is None and args.n_encoder is None:
            raise ValueError("Either --pretrain or --n_encoder is required.")
        if args.pretrain is None:
            print(f"Pre-trained MethylBERT model for {args.n_encoder} encoder blocks is selected.")
            args.pretrain = f"hanyangii/methylbert_hg19_{args.n_encoder}l"
        trainer.load(args.pretrain)
    else:
        if args.n_encoder is None:
            raise ValueError("--n_encoder is required with --without_pretrain.")
        trainer.create_model(config_file=f"hanyangii/methylbert_hg19_{args.n_encoder}l")

    print("Training Start")
    trainer.train(args.steps)


if __name__ == "__main__":
    main()
