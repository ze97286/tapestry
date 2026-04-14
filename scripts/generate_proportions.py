#!/usr/bin/env python3
"""Generate proportion tables for synthetic mixture training data.

Produces CSV files specifying, for each synthetic mixture:
  - proportions for each of 13 cell types
  - target sequencing depth
  - which reference sample to use for each cell type

Five proportion strategies are blended:
  1. Blood-dominated realistic (40%)
  2. Rare-type emphasis (20%)
  3. Broad Dirichlet (20%)
  4. Zero-forcing (10%)
  5. Near-pure / edge cases (10%)

Usage:
    python scripts/generate_proportions.py \
        --manifest data/manifest_atlas.tsv \
        --output-dir runs/run_002/training \
        --n-train 100000 \
        --n-eval 20000 \
        --seed 42
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cell type configuration
# ---------------------------------------------------------------------------

# Maximum realistic proportion for each cell type in cfDNA.
# Keys are normalised: looked up case-insensitively and with common aliases.
_MAX_CONCENTRATIONS = {
    "b": 0.20, "b-cells": 0.20,
    "colon": 0.10,
    "erythrocyte_progenitors": 0.15, "cd34-erythroblasts": 0.15,
    "esophagus": 0.10,
    "gastric": 0.10,
    "granulocytes": 0.65,
    "hepatocytes": 0.15,
    "monocytes": 0.35,
    "nk": 0.20, "nk-cells": 0.20,
    "oac": 0.45,
    "small_int": 0.10, "small-intestine": 0.10,
    "t-cells": 0.30, "t-cd4": 0.20, "t-cd8": 0.15,
    "cd34-megakaryocytes": 0.35,
    "pancreas": 0.10, "duodenum": 0.10,
}

# Typical blood composition means (for blood-dominated strategy).
# Same normalised lookup.
_TYPICAL_BLOOD = {
    "granulocytes": 0.50,
    "monocytes": 0.15,
    "t-cells": 0.12, "t-cd4": 0.08, "t-cd8": 0.04,
    "b": 0.05, "b-cells": 0.05,
    "nk": 0.04, "nk-cells": 0.04,
    "erythrocyte_progenitors": 0.03, "cd34-erythroblasts": 0.03,
    "cd34-megakaryocytes": 0.03,
}


def _lookup(mapping: dict, key: str, default: float) -> float:
    """Case-insensitive lookup with fallback."""
    k = key.lower().strip()
    return mapping.get(k, default)


def get_max_concentration(ct: str) -> float:
    return _lookup(_MAX_CONCENTRATIONS, ct, 0.20)


def get_typical_blood(ct: str) -> float:
    return _lookup(_TYPICAL_BLOOD, ct, 0.01)


# ---------------------------------------------------------------------------
# Dirichlet helper
# ---------------------------------------------------------------------------

def sample_dirichlet(rng: np.random.Generator, alpha: np.ndarray) -> np.ndarray:
    """Sample from Dirichlet, handling near-zero alphas gracefully."""
    gamma = rng.gamma(alpha)
    total = gamma.sum()
    if total == 0:
        # Fallback to uniform
        return np.ones_like(alpha) / len(alpha)
    return gamma / total


# ---------------------------------------------------------------------------
# Proportion strategies
# ---------------------------------------------------------------------------

def blood_dominated(rng: np.random.Generator, cell_types: list[str]) -> np.ndarray:
    """Realistic cfDNA: blood types dominate, tissue types are rare."""
    C = len(cell_types)
    conc = np.zeros(C)

    for i, ct in enumerate(cell_types):
        blood_mean = get_typical_blood(ct)
        if blood_mean > 0.02:  # blood type
            conc[i] = max(0, rng.normal(blood_mean, blood_mean * 0.3))
        else:
            # Tissue types: usually absent or very low
            if rng.random() < 0.7:
                conc[i] = 0.0
            else:
                conc[i] = rng.uniform(0.001, 0.03)

    # Sometimes add a small OAC component
    oac_idx = cell_types.index("OAC")
    if rng.random() < 0.3:
        conc[oac_idx] = rng.uniform(0.001, 0.05)

    total = conc.sum()
    if total > 0:
        conc /= total
    return conc


def rare_emphasis(rng: np.random.Generator, cell_types: list[str]) -> np.ndarray:
    """Force one rare type into 0.1-10%, rest is realistic background."""
    C = len(cell_types)

    # Start with blood-dominated background
    conc = blood_dominated(rng, cell_types)

    # Pick a rare type to emphasise (only from types actually in this atlas)
    # Non-blood types + any T-cell subtypes are candidates
    rare_candidates = [ct for ct in cell_types
                       if get_typical_blood(ct) < 0.02 or ct.lower().startswith("t-")]
    if not rare_candidates:
        rare_candidates = cell_types  # fallback
    target_type = rng.choice(rare_candidates)
    target_idx = cell_types.index(target_type)

    # OAC and T-cell types get special treatment (lower ranges)
    if target_type.lower() == "oac":
        ranges = [(0.001, 0.005, 0.3), (0.005, 0.05, 0.4), (0.05, 0.15, 0.3)]
    elif target_type.lower().startswith("t-"):
        ranges = [(0.0005, 0.005, 0.3), (0.005, 0.01, 0.4), (0.01, 0.05, 0.3)]
    else:
        ranges = [(0.005, 0.02, 0.3), (0.02, 0.05, 0.4), (0.05, 0.10, 0.3)]

    # Pick a sub-range
    probs = [r[2] for r in ranges]
    probs = np.array(probs) / sum(probs)
    idx = rng.choice(len(ranges), p=probs)
    lo, hi, _ = ranges[idx]
    target_conc = rng.uniform(lo, hi)

    # Set the target and renormalise
    scale = (1 - target_conc) / (conc.sum() - conc[target_idx] + 1e-12)
    conc *= scale
    conc[target_idx] = target_conc

    return conc


def broad_dirichlet(rng: np.random.Generator, cell_types: list[str]) -> np.ndarray:
    """Explore the full simplex with a sparse Dirichlet."""
    C = len(cell_types)
    max_conc = np.array([get_max_concentration(ct) for ct in cell_types])

    # Keep sampling until within max concentration bounds
    for _ in range(100):
        alpha = np.full(C, 0.5)
        conc = sample_dirichlet(rng, alpha)
        if np.all(conc <= max_conc):
            return conc

    # Fallback: clip and renormalise
    conc = np.minimum(conc, max_conc)
    return conc / conc.sum()


def zero_forcing(rng: np.random.Generator, cell_types: list[str]) -> np.ndarray:
    """Start from any distribution, then force 2-5 cell types to zero."""
    C = len(cell_types)

    # Base distribution
    if rng.random() < 0.5:
        conc = blood_dominated(rng, cell_types)
    else:
        conc = broad_dirichlet(rng, cell_types)

    # Force 2-5 types to zero, balanced across types
    n_zero = rng.integers(2, 6)
    zero_indices = rng.choice(C, size=min(n_zero, C), replace=False)
    conc[zero_indices] = 0.0

    total = conc.sum()
    if total > 0:
        conc /= total
    else:
        # All zeroed — assign everything to granulocytes
        conc[cell_types.index("Granulocytes")] = 1.0
    return conc


def near_pure(rng: np.random.Generator, cell_types: list[str]) -> np.ndarray:
    """One type at 80-95%, rest is trace/noise."""
    C = len(cell_types)
    dominant_idx = rng.integers(C)
    dominant_frac = rng.uniform(0.80, 0.95)

    conc = np.zeros(C)
    conc[dominant_idx] = dominant_frac

    # Distribute remaining mass
    remaining = 1.0 - dominant_frac
    for i in range(C):
        if i != dominant_idx:
            if rng.random() < 0.5:
                conc[i] = rng.uniform(0, remaining / max(C - 1, 1))
    total = conc.sum()
    if total > 0:
        conc /= total
    return conc


STRATEGIES = {
    "blood_dominated": (blood_dominated, 0.40),
    "rare_emphasis": (rare_emphasis, 0.20),
    "broad_dirichlet": (broad_dirichlet, 0.20),
    "zero_forcing": (zero_forcing, 0.10),
    "near_pure": (near_pure, 0.10),
}


# ---------------------------------------------------------------------------
# Dilution series (structured evaluation)
# ---------------------------------------------------------------------------

def generate_dilution_series(
    rng: np.random.Generator,
    cell_types: list[str],
    target_type: str,
    dilutions: list[float],
    n_per_dilution: int,
    n_depths: int = 4,
) -> pd.DataFrame:
    """Generate structured dilution series for a target cell type.

    For each dilution level × depth, generates n_per_dilution mixtures
    with the target type at the specified fraction and a realistic blood
    background.
    """
    target_idx = cell_types.index(target_type)
    rows = []

    depths = np.geomspace(5_000, 200_000, n_depths).astype(int)

    for dilution in dilutions:
        for depth in depths:
            for _ in range(n_per_dilution):
                # Blood background
                conc = blood_dominated(rng, cell_types)
                # Force target to exact dilution
                scale = (1 - dilution) / (conc.sum() - conc[target_idx] + 1e-12)
                conc *= scale
                conc[target_idx] = dilution

                row = {ct: conc[i] for i, ct in enumerate(cell_types)}
                row["depth"] = int(depth)
                row["strategy"] = f"{target_type}_dilution"
                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reference sample assignment
# ---------------------------------------------------------------------------

def assign_reference_samples(
    rng: np.random.Generator,
    proportions_df: pd.DataFrame,
    manifest: pd.DataFrame,
    cell_types: list[str],
    held_out: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Assign a reference sample to each cell type for each mixture.

    For each mixture × cell type, randomly picks one of the available
    reference samples. If held_out is provided, those samples are excluded.
    """
    # Build per-cell-type sample lists
    available = {}
    for ct in cell_types:
        samples = manifest[manifest["cell_type"] == ct]["sample_id"].tolist()
        if held_out and ct in held_out:
            samples = [s for s in samples if s != held_out[ct]]
        if not samples:
            raise ValueError(f"No reference samples available for {ct} (after LOO exclusion)")
        available[ct] = samples

    # Assign one sample per cell type per mixture
    n = len(proportions_df)
    for ct in cell_types:
        col = f"ref_{ct}"
        samples = available[ct]
        proportions_df[col] = rng.choice(samples, size=n)

    return proportions_df


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_proportion_table(
    rng: np.random.Generator,
    cell_types: list[str],
    n_samples: int,
) -> pd.DataFrame:
    """Generate a table of proportions with strategies and depths."""
    strategy_names = list(STRATEGIES.keys())
    strategy_fns = [STRATEGIES[s][0] for s in strategy_names]
    strategy_weights = np.array([STRATEGIES[s][1] for s in strategy_names])
    strategy_weights /= strategy_weights.sum()

    rows = []
    for i in range(n_samples):
        # Pick strategy
        s_idx = rng.choice(len(strategy_names), p=strategy_weights)
        conc = strategy_fns[s_idx](rng, cell_types)

        # Target depth: log-uniform from 5K to 200K
        depth = int(np.exp(rng.uniform(np.log(5_000), np.log(200_000))))

        row = {ct: conc[j] for j, ct in enumerate(cell_types)}
        row["depth"] = depth
        row["strategy"] = strategy_names[s_idx]
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate proportion tables for training data.")
    parser.add_argument("--manifest", required=True, help="Path to manifest_atlas.tsv")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-eval", type=int, default=20_000)
    parser.add_argument("--n-dilution-per-level", type=int, default=50,
                        help="Samples per dilution level per depth tier")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest = pd.read_csv(args.manifest, sep="\t")
    cell_types = sorted(manifest["cell_type"].unique())
    logger.info("Cell types (%d): %s", len(cell_types), cell_types)

    # Log max concentrations for each cell type
    for ct in cell_types:
        logger.info("  %s: max_conc=%.2f, blood_mean=%.3f",
                     ct, get_max_concentration(ct), get_typical_blood(ct))

    # --- Training proportions ---
    logger.info("Generating %d training proportions...", args.n_train)
    train_df = generate_proportion_table(rng, cell_types, args.n_train)
    train_df = assign_reference_samples(rng, train_df, manifest, cell_types)
    train_df.to_csv(out_dir / "train_proportions.csv", index=False)
    logger.info("Saved train_proportions.csv")

    # Log strategy distribution
    logger.info("Training strategy distribution:\n%s",
                train_df["strategy"].value_counts().to_string())

    # --- Evaluation proportions ---
    logger.info("Generating %d evaluation proportions...", args.n_eval)
    eval_df = generate_proportion_table(rng, cell_types, args.n_eval)
    eval_df = assign_reference_samples(rng, eval_df, manifest, cell_types)
    eval_df.to_csv(out_dir / "eval_proportions.csv", index=False)
    logger.info("Saved eval_proportions.csv")

    # --- OAC dilution series ---
    oac_dilutions = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
    logger.info("Generating OAC dilution series (%d levels × %d depths × %d per)...",
                len(oac_dilutions), 4, args.n_dilution_per_level)
    oac_df = generate_dilution_series(
        rng, cell_types, "OAC", oac_dilutions, args.n_dilution_per_level
    )
    oac_df = assign_reference_samples(rng, oac_df, manifest, cell_types)
    oac_df.to_csv(out_dir / "oac_dilution_proportions.csv", index=False)
    logger.info("Saved oac_dilution_proportions.csv (%d samples)", len(oac_df))

    # --- T-cell dilution series ---
    # Use T-CD4 if available (Ben's atlas), else T-cells
    tcell_type = "T-CD4" if "T-CD4" in cell_types else "T-cells"
    tcell_dilutions = [0.0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.10]
    logger.info("Generating %s dilution series...", tcell_type)
    tcell_df = generate_dilution_series(
        rng, cell_types, tcell_type, tcell_dilutions, args.n_dilution_per_level
    )
    tcell_df = assign_reference_samples(rng, tcell_df, manifest, cell_types)
    tcell_df.to_csv(out_dir / "tcell_dilution_proportions.csv", index=False)
    logger.info("Saved tcell_dilution_proportions.csv (%d samples)", len(tcell_df))

    # --- Summary ---
    total = len(train_df) + len(eval_df) + len(oac_df) + len(tcell_df)
    logger.info("Total samples to generate: %d", total)
    logger.info("  Training: %d", len(train_df))
    logger.info("  Evaluation: %d", len(eval_df))
    logger.info("  OAC dilution: %d", len(oac_df))
    logger.info("  T-cell dilution: %d", len(tcell_df))


if __name__ == "__main__":
    main()
