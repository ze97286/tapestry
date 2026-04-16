"""Loss functions for tapestry models.

Main loss: ``tapestry_loss`` — combines proportion regression, beta-binomial
observation model NLL, detection BCE, and sparsity regularisation.

Also retains ``detector_loss`` and ``aggregator_loss`` for the single-cell-type
OAC detector models.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Beta-binomial log-likelihood
# ---------------------------------------------------------------------------

def beta_binomial_nll(
    u: torch.Tensor,
    c: torch.Tensor,
    q: torch.Tensor,
    phi: torch.Tensor | float = 50.0,
) -> torch.Tensor:
    """Negative log-likelihood of observed counts under a beta-binomial model.

    Parameters
    ----------
    u : (B, M) — observed unmethylated counts.
    c : (B, M) — total coverage.
    q : (B, M) — expected unmethylated probability (from decoder).
    phi : scalar or (1,) — concentration parameter. Higher = less overdispersion.
          phi → ∞ recovers the binomial.

    Returns
    -------
    Scalar NLL, averaged over valid (c > 0) markers.
    """
    if isinstance(phi, (int, float)):
        phi = torch.tensor(phi, device=u.device, dtype=u.dtype)

    m = c - u  # methylated counts
    alpha = q * phi          # (B, M)
    beta = (1 - q) * phi     # (B, M)

    # Log beta-binomial PMF (unnormalised — C(n,k) cancels in the gradient):
    # log P(u | c, α, β) = lgamma(c+1) - lgamma(u+1) - lgamma(m+1)
    #                     + lgamma(u+α) + lgamma(m+β) - lgamma(c+α+β)
    #                     + lgamma(α+β) - lgamma(α) - lgamma(β)
    log_prob = (
        torch.lgamma(c + 1) - torch.lgamma(u + 1) - torch.lgamma(m + 1)
        + torch.lgamma(u + alpha) + torch.lgamma(m + beta) - torch.lgamma(c + alpha + beta)
        + torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    )

    # Only count markers with coverage > 0
    valid = c > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=u.device)

    return -log_prob[valid].mean()


# ---------------------------------------------------------------------------
# Main tapestry deconvolution loss
# ---------------------------------------------------------------------------

def tapestry_loss(
    model_output: dict[str, torch.Tensor],
    true_props: torch.Tensor,
    u: torch.Tensor,
    m: torch.Tensor,
    c: torch.Tensor,
    phi: torch.Tensor | float = 50.0,
    proportion_weight: float = 0.5,
    log_proportion_weight: float = 5.0,
    nll_weight: float = 0.1,
    detection_weight: float = 0.1,
    sparsity_weight: float = 0.01,
    detection_threshold: float = 0.001,
    concentration_weighting: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined loss for the tapestry deconvolution model.

    Components
    ----------
    1. Proportion loss — Huber loss on predicted vs true proportions, optionally
       with concentration-dependent weighting (low concentrations get more weight).
    2. Observation model NLL — beta-binomial NLL linking predicted proportions
       back to observed counts through the atlas.
    3. Detection BCE — binary cross-entropy on presence predictions.
    4. Sparsity — L1 penalty on unnormalised masses.

    Parameters
    ----------
    model_output : dict from TapestryModel.forward()
    true_props : (B, C) — ground-truth proportions.
    u, m, c : (B, M) — observed counts.
    phi : beta-binomial concentration parameter.
    proportion_weight, nll_weight, detection_weight, sparsity_weight : floats.
    detection_threshold : float — threshold for presence labels.
    concentration_weighting : bool — upweight low concentrations in proportion loss.

    Returns
    -------
    total_loss : scalar
    details : dict of component values for logging
    """
    proportions = model_output["proportions"]
    detection = model_output["detection"]
    expected_q = model_output["expected_q"]

    # --- Proportion loss (MSE on linear scale) ---
    prop_loss = ((proportions - true_props) ** 2).mean()

    # --- Log-space proportion loss ---
    # Equal weight to errors at 0.1% and 10%.
    # Only computed where both pred and true are non-negligible.
    eps = 1e-6
    log_mask = (true_props > 0.001) & (proportions > eps)
    if log_mask.sum() > 0:
        log_pred = torch.log10(proportions[log_mask] + eps)
        log_true = torch.log10(true_props[log_mask] + eps)
        log_prop_loss = ((log_pred - log_true) ** 2).mean()
    else:
        log_prop_loss = torch.tensor(0.0, device=proportions.device)

    # --- Observation model NLL ---
    nll = beta_binomial_nll(u, c, expected_q, phi)

    # --- Detection BCE ---
    presence_labels = (true_props > detection_threshold).float()
    det_loss = F.binary_cross_entropy(
        detection.clamp(1e-7, 1 - 1e-7), presence_labels, reduction="mean"
    )

    # --- Sparsity (entropy penalty — lower entropy = sparser) ---
    log_p = torch.log(proportions + 1e-8)
    sparsity = -(proportions * log_p).sum(dim=1).mean()

    # --- Combine ---
    total = (
        proportion_weight * prop_loss
        + log_proportion_weight * log_prop_loss
        + nll_weight * nll
        + detection_weight * det_loss
        + sparsity_weight * sparsity
    )

    details = {
        "total_loss": total.item(),
        "proportion_loss": prop_loss.item(),
        "log_proportion_loss": log_prop_loss.item(),
        "nll": nll.item(),
        "detection_loss": det_loss.item(),
        "sparsity": sparsity.item(),
    }

    return total, details


# ---------------------------------------------------------------------------
# Detector loss (TransformerDetector / DeepSetsDetector)
# ---------------------------------------------------------------------------

def detector_loss(
    mu: torch.Tensor,
    uncertainty: torch.Tensor,
    y_true: torch.Tensor,
    control_mask: torch.Tensor | None = None,
    zero_prob: torch.Tensor | None = None,
    control_weight: float = 100.0,
    zero_weight: float = 5.0,
    relative_weight: float = 2.0,
    calibration_weight: float = 0.2,
) -> torch.Tensor:
    """Loss for the single-cell-type detector models."""
    epsilon = 1e-6

    mse = F.mse_loss(mu, y_true, reduction="none")

    non_zero = y_true > epsilon
    if non_zero.sum() > 0:
        rel_error = torch.abs(mu[non_zero] - y_true[non_zero]) / (y_true[non_zero] + epsilon)
        rel_loss = rel_error.mean()
    else:
        rel_loss = torch.tensor(0.0, device=mu.device)

    control_loss = torch.tensor(0.0, device=mu.device)
    if control_mask is not None and control_mask.sum() > 0:
        control_loss = control_weight * mu[control_mask].mean()

    zero_loss = torch.tensor(0.0, device=mu.device)
    if zero_prob is not None:
        zero_target = (y_true < epsilon).float()
        zero_loss = F.binary_cross_entropy(
            zero_prob.squeeze(), zero_target.squeeze()
        ) * zero_weight

    cal_loss = torch.tensor(0.0, device=mu.device)
    if uncertainty is not None:
        z_scores = torch.abs(mu - y_true) / (uncertainty + epsilon)
        cal_loss = F.smooth_l1_loss(z_scores, torch.ones_like(z_scores) * 1.96)

    return (
        mse.mean()
        + relative_weight * rel_loss
        + control_loss
        + zero_loss
        + calibration_weight * cal_loss
    )


def aggregator_loss(
    mu: torch.Tensor,
    y_true: torch.Tensor,
    control_mask: torch.Tensor | None = None,
    control_weight: float = 0.1,
    max_loss: float = 10.0,
) -> torch.Tensor:
    """Simplified MSE loss for the DeepSets aggregator."""
    mu = torch.clamp(mu, 0.0, 1.0)
    mse = F.mse_loss(mu, y_true, reduction="mean")

    control_loss = torch.tensor(0.0, device=mu.device)
    if control_mask is not None and control_mask.sum() > 0:
        control_loss = control_weight * mu[control_mask].mean()

    return torch.clamp(mse + control_loss, 0.0, max_loss)
