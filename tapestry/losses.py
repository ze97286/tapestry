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
    log_proportion_weight: float = 15.0,
    nll_weight: float = 0.1,
    detection_weight: float = 1.0,
    sparsity_weight: float = 0.01,
    detection_threshold: float = 0.001,
    # Legacy kwargs, kept so existing call sites don't break.
    proportion_weight: float | None = None,
    concentration_weighting: bool | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined loss for the tapestry deconvolution model.

    Components
    ----------
    1. Log-space proportion loss — MSE on log10(predicted) vs log10(true) on
       every true-nonzero position. Predictions are clamped at 1e-6 inside the
       log so a rare type that has been pushed toward zero still receives
       gradient (otherwise the loss silently abandons it).
    2. Observation model NLL — beta-binomial NLL linking predicted proportions
       back to observed counts through the atlas.
    3. Detection BCE — binary cross-entropy on presence predictions, with
       weight large enough that the detection head learns to differentiate
       gates toward {0, 1} rather than sitting at ~0.5–0.7 (where soft gating
       has no functional effect after renormalisation).
    4. Sparsity — entropy penalty on predicted proportions.

    Returns
    -------
    total_loss : scalar
    details : dict of component values for logging
    """
    del proportion_weight, concentration_weighting  # unused legacy kwargs

    proportions = model_output["proportions"]
    detection = model_output["detection"]
    expected_q = model_output["expected_q"]

    # --- Log-space proportion loss ---
    # Include every true-nonzero position. Clamping predictions (not masking
    # them out) keeps gradient flowing when a prediction has collapsed near
    # zero — this is what stops the model from pulling low-abundance types
    # back up once it starts shrinking them.
    eps = 1e-6
    log_mask = true_props > 0.001
    if log_mask.any():
        log_pred = torch.log10(proportions[log_mask].clamp(min=eps))
        log_true = torch.log10(true_props[log_mask])
        log_prop_loss = ((log_pred - log_true) ** 2).mean()
    else:
        log_prop_loss = torch.tensor(0.0, device=proportions.device)

    # --- Observation model NLL ---
    nll = beta_binomial_nll(u, c, expected_q, phi)

    # --- Detection BCE ---
    presence_labels = (true_props > detection_threshold).float()
    det_loss = F.binary_cross_entropy(
        detection, presence_labels, reduction="mean"
    )

    # --- Sparsity (entropy penalty — lower entropy = sparser) ---
    log_p = torch.log(proportions + 1e-8)
    sparsity = -(proportions * log_p).sum(dim=1).mean()

    # --- Combine ---
    total = (
        log_proportion_weight * log_prop_loss
        + nll_weight * nll
        + detection_weight * det_loss
        + sparsity_weight * sparsity
    )

    details = {
        "total_loss": total.item(),
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
