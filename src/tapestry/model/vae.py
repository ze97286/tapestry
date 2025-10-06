"""
Blind Deconvolution VAE for TAPESTRY.
"""

import torch
import torch.nn as nn
import numpy as np


class BlindDeconvolutionVAE(nn.Module):
    """
    Variational Autoencoder for blind deconvolution of cfDNA methylation.
    
    Architecture:
        Input: (methylation_counts, coverage) → beta values
        Encoder: beta values → latent distribution (mu, logvar)
        Decoder: latent → proportions → reconstructed beta values
        
    Key features:
        - Decoder weights ARE the learned signature matrix
        - Proportions sum to 1 (biological constraint)
        - Binomial likelihood (correct noise model)
    """
    
    def __init__(
        self,
        n_regions: int,
        n_components: int = 12,
        latent_dim: int = 32,
        hidden_dims: list = [1024, 512, 256],
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Initialize the VAE.
        
        Args:
            n_regions: Number of genomic regions (features)
            n_components: Number of latent components to discover
            latent_dim: Dimensionality of VAE latent space
            hidden_dims: List of hidden layer dimensions for encoder
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.n_regions = n_regions
        self.n_components = n_components
        self.latent_dim = latent_dim
        
        # Encoder: (beta values, log coverage) → latent distribution
        encoder_layers = []
        input_dim = n_regions * 2  # beta + log(coverage)
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder: latent → proportions
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], n_components),
        )
        
        # Softmax to ensure proportions sum to 1
        self.softmax = nn.Softmax(dim=-1)
        
        # Signature matrix: proportions → methylation rates
        # THIS IS THE KEY: decoder weights are the learned signatures!
        self.signature_decoder = nn.Linear(n_components, n_regions, bias=False)
        
        # Initialize signatures to reasonable methylation values [0.2, 0.8]
        nn.init.uniform_(self.signature_decoder.weight, 0.2, 0.8)
        
        # Activation to keep methylation rates in [0, 1]
        self.sigmoid = nn.Sigmoid()
    
    def encode(
        self, 
        methylation_counts: torch.Tensor, 
        coverage: torch.Tensor
    ) -> tuple:
        """
        Encode observations to latent distribution.
        
        Args:
            methylation_counts: [batch × n_regions]
            coverage: [batch × n_regions]
            
        Returns:
            (mu, logvar): Parameters of latent distribution
        """
        # Calculate beta values (methylation rates)
        beta = methylation_counts / (coverage + 1e-6)
        
        # Use log(coverage) as additional feature (coverage is informative)
        log_cov = torch.log1p(coverage)
        
        # Concatenate features
        x = torch.cat([beta, log_cov], dim=-1)
        
        # Encode
        h = self.encoder(x)
        
        # Get distribution parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(
        self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon
        
        Args:
            mu: Mean of latent distribution [batch × latent_dim]
            logvar: Log variance [batch × latent_dim]
            
        Returns:
            Sampled latent vector [batch × latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(
        self, 
        z: torch.Tensor
    ) -> tuple:
        """
        Decode latent vector to proportions and methylation rates.
        
        Args:
            z: Latent vector [batch × latent_dim]
            
        Returns:
            (proportions, reconstructed_beta): 
                - proportions: [batch × n_components]
                - reconstructed_beta: [batch × n_regions]
        """
        # Decode to proportions
        logits = self.decoder_fc(z)
        proportions = self.softmax(logits)
        
        # Multiply by signature matrix
        # This is matrix deconvolution: reconstructed = proportions @ signatures
        reconstructed_beta = torch.matmul(proportions, self.signature_decoder.weight)
        
        # Ensure in [0, 1] range
        reconstructed_beta = self.sigmoid(reconstructed_beta)
        
        return proportions, reconstructed_beta
    
    def forward(
        self, 
        methylation_counts: torch.Tensor, 
        coverage: torch.Tensor
    ) -> tuple:
        """
        Full forward pass.
        
        Args:
            methylation_counts: [batch × n_regions]
            coverage: [batch × n_regions]
            
        Returns:
            (reconstructed_beta, proportions, mu, logvar)
        """
        # Encode
        mu, logvar = self.encode(methylation_counts, coverage)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        proportions, reconstructed_beta = self.decode(z)
        
        return reconstructed_beta, proportions, mu, logvar
    
    def get_proportions(
        self,
        methylation_counts: torch.Tensor,
        coverage: torch.Tensor
    ) -> torch.Tensor:
        """
        Get component proportions (inference mode).
        
        Args:
            methylation_counts: [batch × n_regions]
            coverage: [batch × n_regions]
            
        Returns:
            proportions: [batch × n_components]
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(methylation_counts, coverage)
            # Use mean (no sampling) for deterministic inference
            proportions, _ = self.decode(mu)
        return proportions
    
    @property
    def signatures(self) -> torch.Tensor:
        """
        Get learned signature matrix.
        
        Returns:
            Signature matrix [n_components × n_regions]
        """
        return self.signature_decoder.weight.detach()
    
    def get_signature_methylation(
        self, 
        component_idx: int
    ) -> np.ndarray:
        """
        Get methylation pattern for a specific component.
        
        Args:
            component_idx: Index of component (0 to n_components-1)
            
        Returns:
            Methylation rates across regions [n_regions]
        """
        sig = self.signatures[component_idx].cpu().numpy()
        return sig