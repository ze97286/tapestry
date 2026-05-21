#!/usr/bin/env python3
"""Thin entrypoint for the upstream CompEpigen/methylbert CLI.

This repository intentionally does not reimplement MethylBERT. Set
PYTHONPATH to include external/methylbert/src and pass the upstream CLI
arguments unchanged, for example:

    PYTHONPATH=external/methylbert/src python scripts/run_upstream_methylbert.py preprocess_finetune ...
"""

from methylbert.cli import main


if __name__ == "__main__":
    main()
