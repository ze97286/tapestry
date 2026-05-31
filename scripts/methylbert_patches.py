#!/usr/bin/env python3
"""Runtime patches for the vendored CompEpigen/methylbert package (tapestry).

`external/methylbert` is kept byte-identical to upstream HEAD f82f83b. Two behaviours
we want for the OAC shortcut work are applied here at runtime via monkeypatching, so the
vendored model code stays pristine:

1. ``MethylBertEmbeddedDMR.forward`` defaults the ``attention_mask`` from the pad token
   (``MethylVocab.pad_index == 0``) so BERT no longer attends to padding, and zeroes
   padded positions before the flatten read classifier. Upstream calls ``forward`` with
   ``attention_mask=None``, which lets the dense per-position classifier read fragment
   length off the padding boundary. This reduces that leakage. It does not fully remove
   it: the EOS-at-read-end plus the flatten head still encode some length.

2. ``deconvolute`` accepts ``margins_override`` so the deploy prior can be set explicitly
   instead of always using the training class frequencies (which, for a tissue-vs-plasma
   fine-tune, encode that source balance rather than the cfDNA bulk being deconvolved).

Call :func:`apply_patches` once, after importing methylbert and before building/using the
model. Both patches faithfully mirror the upstream bodies with only the additions above.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

import methylbert.deconvolute as _deconv
from methylbert.network import MethylBertEmbeddedDMR

_PATCHED = False


def _patched_forward(
    self,
    step,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    ctype_label=None,
):
    # tapestry patch: default the attention mask from the pad token so padding is
    # ignored, then zero padded positions before the flatten read classifier.
    if attention_mask is None and input_ids is not None:
        attention_mask = (input_ids != 0).long()

    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
    )

    sequence_output = outputs[0]
    sequence_output = self.dropout(sequence_output)

    if attention_mask is not None:
        sequence_output = sequence_output * attention_mask.unsqueeze(-1).to(sequence_output.dtype)

    encoded_dmr = self.dmr_encoder(labels.view(-1))
    sequence_output = torch.cat((sequence_output, encoded_dmr.unsqueeze(-1)), axis=-1)
    ctype_logits = self.read_classifier(sequence_output.view(-1, (self.seq_len + 1) * 769))

    loss = self.classification_loss_fct(
        ctype_logits.view(-1, 2),
        F.one_hot(ctype_label, num_classes=2).to(torch.float32).view(-1, 2),
    )
    ctype_logits = ctype_logits.softmax(dim=1)

    return {"loss": loss, "dmr_logits": sequence_output, "classification_logits": ctype_logits}


def _patched_deconvolute(
    trainer,
    data_loader,
    df_train,
    tokenizer,
    output_path="./",
    n_grid=10000,
    adjustment=False,
    margins_override=None,
):
    import pandas as pd

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    total_res, logits = trainer.read_classification(
        data_loader=data_loader, tokenizer=tokenizer, logit=True
    )
    total_res = total_res.drop(columns=["ctype_label"])

    total_res["n_cpg"] = total_res["methyl_seq"].apply(lambda x: x.count("0") + x.count("1"))
    total_res["P_ctype"] = logits[:, 1]
    total_res.to_csv(output_path + "/res.csv", sep="\t", header=True, index=False)
    total_res["P_N"] = logits[:, 0]

    total_res = total_res[total_res["n_cpg"] > 0]
    assert total_res.shape[0] != 0, (
        "There are no reads selected for deconvolution. It may mean all of the reads do "
        "not have CpG methylation."
    )

    margins = df_train.value_counts("ctype", normalize=True)
    if margins_override is not None:
        # tapestry patch: override the training prior with an explicit deploy prior.
        margins = pd.Series(margins_override, dtype=float)

    print("Margins : ", margins)
    print(total_res.head())

    if len(margins.keys()) == 2:
        deconv_res, fi_res = _deconv.purity_estimation(
            reads=total_res, margins=margins, n_grid=n_grid, adjustment=adjustment
        )
        deconv_res.to_csv(output_path + "/deconvolution.csv", sep="\t", header=True, index=False)
        fi_res.to_csv(output_path + "/FI.csv", sep="\t", header=True, index=False)
    elif len(margins.keys()) > 2:
        deconv_res = _deconv.optimise_nll_deconvolute(reads=total_res, margins=margins)
        deconv_res.to_csv(output_path + "/deconvolution.csv", sep="\t", header=True, index=False)
    else:
        raise RuntimeError(
            f"There are less than two cell types in the training data set. {margins.keys()} "
            "Neither purity estimation nor deconvolution can be performed."
        )


def apply_patches() -> None:
    """Idempotently install the forward / deconvolute patches onto the vendored package."""
    global _PATCHED
    if _PATCHED:
        return
    MethylBertEmbeddedDMR.forward = _patched_forward
    _deconv.deconvolute = _patched_deconvolute
    _PATCHED = True
    print("methylbert_patches: applied attention_mask + margins_override patches")
