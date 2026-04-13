"""Coverage-matched data augmentation for deconvolution training.

Simulates clinical-like coverage profiles by redistributing marker coverage
into three bands (zero, low 1-4, high 5-max) based on target distribution
parameters derived from real clinical cfDNA data.
"""

import numpy as np
import torch


def coverage_matched_augmentation(
    marker_values,
    coverage,
    target_dist_params: dict,
    augmentation_prob: float = 1.0,
):
    """Augment marker values and coverage to match a target clinical distribution.

    Works with both numpy arrays and PyTorch tensors.

    Parameters
    ----------
    marker_values : ndarray or Tensor
        Shape (N, M) — methylation fractions.
    coverage : ndarray or Tensor
        Shape (N, M) — read counts.
    target_dist_params : dict
        Must contain ``zero_rate`` (float) and ``quantiles`` (dict with
        '5%', '25%', '50%', '75%', '95%' keys).
    augmentation_prob : float
        Probability of augmenting each sample (0-1).

    Returns
    -------
    augmented_values, augmented_coverage : same type as inputs
    """
    is_torch = isinstance(marker_values, torch.Tensor)

    if is_torch:
        aug_values = marker_values.clone()
        aug_coverage = coverage.clone()
        device = marker_values.device
        num_samples, num_markers = marker_values.shape
        zeros_fn = lambda shape: torch.zeros(shape, device=device)
        rand_fn = lambda shape: torch.rand(shape, device=device)
        normal_fn = lambda mean, std, shape: torch.normal(mean=mean, std=std, size=shape, device=device)
        clamp_fn = torch.clamp
        nan_to_num_fn = torch.nan_to_num
        binomial_fn = lambda n, p: torch.distributions.binomial.Binomial(n, p).sample()
        where_fn = torch.where
    else:
        aug_values = marker_values.copy()
        aug_coverage = coverage.copy()
        num_samples, num_markers = marker_values.shape
        zeros_fn = np.zeros
        rand_fn = np.random.random
        normal_fn = np.random.normal
        clamp_fn = np.clip
        nan_to_num_fn = np.nan_to_num
        binomial_fn = np.random.binomial
        where_fn = np.where

    # Sanitise: where coverage == 0, set values to 0
    zero_cov_mask = aug_coverage == 0
    aug_values[zero_cov_mask] = zeros_fn(zero_cov_mask.sum().item() if is_torch else zero_cov_mask.sum())
    non_zero = ~zero_cov_mask
    aug_values[non_zero] = clamp_fn(aug_values[non_zero], 0.0, 1.0)
    aug_values[non_zero] = nan_to_num_fn(aug_values[non_zero], nan=0.0)

    # Target distribution parameters
    zero_fraction = target_dist_params["zero_rate"]
    if is_torch:
        zero_fraction = torch.tensor(zero_fraction, device=device)

    max_coverage = 22.25  # Based on observed clinical tail
    high_cov_fraction = 0.62
    low_cov_fraction = 1.0 - zero_fraction - high_cov_fraction
    if is_torch:
        high_cov_fraction = torch.tensor(high_cov_fraction, device=device)
        low_cov_fraction = torch.tensor(low_cov_fraction, device=device)

    # Per-sample augmentation
    augment_mask = rand_fn((num_samples,)) < augmentation_prob

    if augment_mask.any():
        delta = normal_fn(0, 0.6, (num_samples,))
        delta = clamp_fn(delta, -0.6, 0.6)
        zero_frac = clamp_fn(zero_fraction + delta, 0.0, 0.15)
        low_cov_frac = clamp_fn(low_cov_fraction - delta / 2, 0.0, 0.77)
        high_cov_frac = 1.0 - zero_frac - low_cov_frac

        rand_vals = rand_fn((num_samples, num_markers))

        zero_thresh = zero_frac.unsqueeze(1) if is_torch else zero_frac[:, np.newaxis]
        low_thresh = (zero_frac + low_cov_frac).unsqueeze(1) if is_torch else (zero_frac + low_cov_frac)[:, np.newaxis]

        zero_mask = rand_vals < zero_thresh
        low_mask = (rand_vals >= zero_thresh) & (rand_vals < low_thresh)
        high_mask = rand_vals >= (1.0 - high_cov_frac.unsqueeze(1) if is_torch else high_cov_frac[:, np.newaxis])

        samples_mask = where_fn(augment_mask, True, False)
        samples_mask = samples_mask.unsqueeze(1) if is_torch else samples_mask[:, np.newaxis]

        # Zero coverage band
        aug_coverage = where_fn(zero_mask & samples_mask, zeros_fn((num_samples, num_markers)), aug_coverage)
        aug_values = where_fn(zero_mask & samples_mask, zeros_fn((num_samples, num_markers)), aug_values)

        # Low coverage band (1-4)
        new_cov_low = rand_fn((num_samples, num_markers)) * 3 + 1
        aug_coverage = where_fn(low_mask & samples_mask, new_cov_low, aug_coverage)
        n_low = new_cov_low.to(torch.int) if is_torch else new_cov_low.astype(int)
        p_low = where_fn(low_mask & samples_mask, marker_values, zeros_fn((num_samples, num_markers)))
        p_low = nan_to_num_fn(p_low, nan=0.0)
        p_low = clamp_fn(p_low, 0.0, 1.0)
        nan_mask_low = p_low == 0
        if nan_mask_low.any():
            p_low = where_fn(nan_mask_low, rand_fn((num_samples, num_markers)), p_low)
        successes_low = binomial_fn(n_low, p_low)
        aug_values = where_fn(low_mask & samples_mask, successes_low / new_cov_low, aug_values)

        # High coverage band (5-max)
        new_cov_high = rand_fn((num_samples, num_markers)) * (max_coverage - 5) + 5
        aug_coverage = where_fn(high_mask & samples_mask, new_cov_high, aug_coverage)
        n_high = new_cov_high.to(torch.int) if is_torch else new_cov_high.astype(int)
        p_high = where_fn(high_mask & samples_mask, marker_values, zeros_fn((num_samples, num_markers)))
        p_high = nan_to_num_fn(p_high, nan=0.0)
        p_high = clamp_fn(p_high, 0.0, 1.0)
        nan_mask_high = p_high == 0
        if nan_mask_high.any():
            p_high = where_fn(nan_mask_high, rand_fn((num_samples, num_markers)), p_high)
        successes_high = binomial_fn(n_high, p_high)
        aug_values = where_fn(high_mask & samples_mask, successes_high / new_cov_high, aug_values)

    # Final consistency
    aug_values = where_fn(aug_coverage == 0, zeros_fn(aug_coverage.shape), aug_values)

    return aug_values, aug_coverage
