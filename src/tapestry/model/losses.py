"""
Loss functions for TAPESTRY.
"""

import torch
import torch.nn.functional as F
import numpy as np


def binomial_nll_loss(
    reconstructed_beta: torch.Tensor,
    methylation_counts: torch.Tensor,
    coverage: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Binomial negative log-likelihood loss.
    
    This is THE KEY INNOVATION over NNLS - models the correct generative process.
    
    Args:
        reconstructed_beta: Predicted methylation rates [batch × n_regions]
        methylation_counts: Observed methylated read counts [batch × n_regions]
        coverage: Total read coverage [batch × n_regions]
        eps: Small constant for numerical stability
        
    Returns:
        Scalar loss value
    """
    # Clamp to avoid log(0)
    p = torch.clamp(reconstructed_beta, eps, 1 - eps)
    
    # Binomial log-likelihood: k*log(p) + (n-k)*log(1-p)
    # We drop the binomial coefficient (constant w.r.t. parameters)
    log_likelihood = (
        methylation_counts * torch.log(p) +
        (coverage - methylation_counts) * torch.log(1 - p)
    )
    
    # Return negative log-likelihood (we want to maximize likelihood = minimize -LL)
    return -torch.sum(log_likelihood)


def kl_divergence_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor
) -> torch.Tensor:
    """
    KL divergence between learned latent distribution and standard normal.
    
    KL(N(mu, var) || N(0, 1)) = -0.5 * sum(1 + log(var) - mu^2 - var)
    
    Args:
        mu: Mean of latent distribution [batch × latent_dim]
        logvar: Log variance of latent distribution [batch × latent_dim]
        
    Returns:
        Scalar KL divergence
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def sparsity_loss(
    proportions: torch.Tensor,
    loss_type: str = 'l1'
) -> torch.Tensor:
    """
    Sparsity penalty on component proportions.
    
    Encourages most samples to use few components (i.e., most proportions → 0).
    
    Args:
        proportions: Component proportions [batch × n_components]
        loss_type: 'l1' for L1 penalty, 'l2' for L2, 'hoyer' for Hoyer sparsity
        
    Returns:
        Scalar sparsity loss
    """
    if loss_type == 'l1':
        # L1: Sum of absolute values
        return torch.sum(torch.abs(proportions))
    
    elif loss_type == 'l2':
        # L2: Sum of squares (less aggressive)
        return torch.sum(proportions ** 2)
    
    elif loss_type == 'hoyer':
        # Hoyer sparsity: (√n - ||x||_1 / ||x||_2) / (√n - 1)
        # Measures how close vector is to having single non-zero element
        n = proportions.shape[1]
        l1_norm = torch.sum(torch.abs(proportions), dim=1)
        l2_norm = torch.sqrt(torch.sum(proportions ** 2, dim=1) + 1e-8)
        hoyer = (np.sqrt(n) - l1_norm / l2_norm) / (np.sqrt(n) - 1)
        return -torch.mean(hoyer)  # Negative because we want to maximize sparsity
    
    else:
        raise ValueError(f"Unknown sparsity loss type: {loss_type}")


def tumor_fraction_loss(
    proportions: torch.Tensor,
    cancer_components: list,
    true_tf: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Supervised loss matching predicted TF to ichorCNA estimates.
    
    Args:
        proportions: Component proportions [batch × n_components]
        cancer_components: List of indices for cancer-related components
        true_tf: Ground truth tumor fractions from ichorCNA [batch]
        mask: Boolean mask for samples with reliable TF estimates [batch]
        
    Returns:
        Scalar MSE loss (only for masked samples)
    """
    # Sum cancer components to get total cancer signal
    predicted_tf = proportions[:, cancer_components].sum(dim=1)
    
    if mask is not None:
        # Only compute loss for reliable samples
        predicted_tf = predicted_tf[mask]
        true_tf = true_tf[mask]
        
        if len(predicted_tf) == 0:
            return torch.tensor(0.0, device=proportions.device)
    
    # MSE loss
    return F.mse_loss(predicted_tf, true_tf)


def signature_regularization_loss(
    signatures: torch.Tensor,
    reg_type: str = 'bimodal'
) -> torch.Tensor:
    """
    Regularization on learned signature matrix.
    
    Encourages signatures to look like real methylation patterns.
    
    Args:
        signatures: Signature matrix [n_components × n_regions]
        reg_type: Type of regularization
            - 'bimodal': Prefer values near 0 or 1 (realistic methylation)
            - 'smooth': Prefer moderate values near 0.5
            - 'l2': Simple L2 penalty
            
    Returns:
        Scalar regularization loss
    """
    if reg_type == 'bimodal':
        # Penalize values near 0.5 (prefer 0 or 1)
        # Loss is minimal when sig ∈ {0, 1}, maximal when sig = 0.5
        distance_from_extreme = torch.abs(signatures - 0.5)
        return -torch.mean(distance_from_extreme)  # Negative to maximize distance
    
    elif reg_type == 'smooth':
        # Prefer moderate values (opposite of bimodal)
        distance_from_middle = (signatures - 0.5) ** 2
        return torch.mean(distance_from_middle)
    
    elif reg_type == 'l2':
        # Simple L2 penalty
        return torch.sum(signatures ** 2)
    
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")


class TAPESTRYLoss:
    """
    Combined loss function for TAPESTRY training.
    """
    
    def __init__(
        self,
        kl_weight: float = 0.1,
        sparsity_weight: float = 0.01,
        tf_weight: float = 0.3,
        sig_reg_weight: float = 0.001,
        sparsity_type: str = 'l1',
        sig_reg_type: str = 'bimodal'
    ):
        """
        Initialize combined loss.
        
        Args:
            kl_weight: Weight for KL divergence
            sparsity_weight: Weight for sparsity penalty
            tf_weight: Weight for tumor fraction supervision
            sig_reg_weight: Weight for signature regularization
            sparsity_type: Type of sparsity ('l1', 'l2', 'hoyer')
            sig_reg_type: Type of signature regularization
        """
        self.kl_weight = kl_weight
        self.sparsity_weight = sparsity_weight
        self.tf_weight = tf_weight
        self.sig_reg_weight = sig_reg_weight
        self.sparsity_type = sparsity_type
        self.sig_reg_type = sig_reg_type
    
    def __call__(
        self,
        reconstructed_beta: torch.Tensor,
        methylation_counts: torch.Tensor,
        coverage: torch.Tensor,
        proportions: torch.Tensor,
        signatures: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        true_tf: torch.Tensor = None,
        tf_mask: torch.Tensor = None,
        cancer_components: list = None
    ) -> dict:
        """
        Compute total loss and individual components.
        
        Returns:
            Dictionary with 'total' loss and individual components
        """
        # 1. Reconstruction loss (primary objective)
        recon_loss = binomial_nll_loss(
            reconstructed_beta, 
            methylation_counts, 
            coverage
        )
        
        # 2. KL divergence (VAE regularization)
        kl_loss = kl_divergence_loss(mu, logvar)
        
        # 3. Sparsity penalty
        sparse_loss = sparsity_loss(proportions, self.sparsity_type)
        
        # 4. Signature regularization
        sig_reg = signature_regularization_loss(signatures, self.sig_reg_type)
        
        # 5. Tumor fraction supervision (if available)
        if true_tf is not None and cancer_components is not None:
            tf_loss = tumor_fraction_loss(
                proportions, 
                cancer_components, 
                true_tf, 
                tf_mask
            )
        else:
            tf_loss = torch.tensor(0.0, device=reconstructed_beta.device)
        
        # Total loss
        total_loss = (
            recon_loss +
            self.kl_weight * kl_loss +
            self.sparsity_weight * sparse_loss +
            self.sig_reg_weight * sig_reg +
            self.tf_weight * tf_loss
        )
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss.item(),
            'kl_divergence': kl_loss.item(),
            'sparsity': sparse_loss.item(),
            'signature_reg': sig_reg.item(),
            'tumor_fraction': tf_loss.item() if isinstance(tf_loss, torch.Tensor) else tf_loss,
        }