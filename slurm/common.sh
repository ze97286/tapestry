#!/bin/bash
# slurm/common.sh — sourced by all job scripts
export PROJECT_DIR="${PROJECT_DIR:-/users/ludwig/uii408/sharedscratch/tapestry}"
export CONFIG="${CONFIG:-${PROJECT_DIR}/configs/main.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-/users/ludwig/uii408/sharedscratch/tapestry/runs/run_v0.3}"
export PAT_DIR="${PAT_DIR:-${PROJECT_DIR}/data/pats}"
export MANIFEST="${MANIFEST:-${PROJECT_DIR}/data/manifest_ben_atlas.tsv}"
export WGBSTOOLS="${WGBSTOOLS:-/users/ludwig/uii408/sharedscratch/wgbs_tools/wgbstools}"
export PATTOOLS="${PATTOOLS:-/gpfs3/users/ludwig/uii408/sharedscratch/pattools}"

# Activate environment
module load GCC/12.3.0
cd ${PROJECT_DIR}
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONHASHSEED=42
